###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The cached quantized weights of the Primus-Turbo linears (multi-micro-batch FP8 / FP4 paths): the previous step's
caches are dropped before re-quantizing, the cached copies are not autograd leaves, the weight-gradient bridge hands
backward no materialized zero gradient for the transpose copy, and the manual fallback quantizes with the Turbo
config. Covers FP8 tensorwise (current scaling), MXFP8 and MXFP4; the GEMMs and quantizers are stand-ins, so no
particular GPU architecture is needed (the extension itself imports Primus-Turbo, which needs a GPU)."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus.backends.megatron.core.extensions.primus_turbo as ext  # noqa: E402  isort:skip
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboQuantConfig,
    PrimusTurboQuantizedTensorPair,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    _bridge_weight_grad,
    _maybe_create_quantized_weight_buffers,
    _weight_scaling_recipe,
    float8_e4m3,
)

CONFIGS = {
    "fp8_tensorwise": PrimusTurboQuantConfig(),
    "mxfp8": PrimusTurboQuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    ),
    "mxfp4": PrimusTurboQuantConfig(
        format=Format.E2M1_X2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    ),
}
BLOCKWISE = PrimusTurboQuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)
RAW_2D = ScalingRecipe(use_2d_block=True)


def _turbo_state(monkeypatch, fmt):
    """The Turbo autocast state a linear's forward sees for ``fmt``."""
    manager = ext.PrimusTurboLowPrecisionGlobalStateManager
    fp4 = fmt == "mxfp4"
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", not fp4)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", fp4)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", CONFIGS[fmt])


# ------------------------------------------------------------------------- old caches dropped before re-quantizing


class _Stub:
    """A Primus-Turbo linear stand-in: the attributes their forward paths read."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _linear_stub(n, k):
    weight = torch.nn.Parameter(torch.randn(n, k))
    return _Stub(
        _parameters={"weight": weight},
        weights=weight,
        use_bias=False,
        bias_names=[],
        offload=False,
        config=SimpleNamespace(gradient_accumulation_fusion=False, normalization="RMSNorm"),
        disable_parameter_transpose_cache=False,
        quantized_weight_buffer=None,
        quantized_weight_t_buffer=None,
        layer_norm_weight=torch.ones(k),
        layer_norm_bias=None,
        zero_centered_gamma=False,
        eps=1e-5,
        _skip_fused_norm=True,
        is_first_microbatch=True,
        _ensure_weight_views=lambda: None,
    )


@pytest.fixture
def requantize_probe(monkeypatch):
    """Stand-ins for the GEMMs, the bridge and the quantizer; returns what each re-quantization found on the stub."""
    seen = []
    stub = {}

    def create(weight, dest_dtype, quant_config, disable_parameter_transpose_cache):
        module = stub["module"]
        seen.append((module.quantized_weight_buffer, module.quantized_weight_t_buffer))
        return torch.zeros(1), torch.zeros(1)

    def gemm(a, b, **kwargs):
        return torch.zeros(a.shape[0], stub["out_features"])

    monkeypatch.setattr(ext, "_maybe_create_quantized_weight_buffers", create)
    monkeypatch.setattr(ext, "_bridge_weight_grad", lambda x, w, pair, fuse_wgrad_accum=False: (x, pair))
    monkeypatch.setattr(ext, "_fuse_wgrad_accum_pattern", lambda config, weight: None)
    monkeypatch.setattr(ext, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(ext, "_use_split_wgrad_op", lambda: False)
    for name in ("gemm_fp8", "gemm_fp4"):
        monkeypatch.setattr(ext.primus_turbo_torch.ops, name, gemm)
    return stub, seen


LINEARS = [
    "PrimusTurboLinear",
    "PrimusTurboRowParallelLinear",
    "PrimusTurboColumnParallelLinear",
    "PrimusTurboLayerNormColumnParallelLinear",
]


@pytest.mark.parametrize("fmt", list(CONFIGS))
@pytest.mark.parametrize("cls", LINEARS)
def test_linear_drops_old_caches_before_requantizing(monkeypatch, requantize_probe, cls, fmt):
    """Each step's first micro-batch re-quantizes the weight: the previous step's copies are released first, so two
    sets of caches are never alive together (and the caching allocator can reuse their blocks)."""
    _turbo_state(monkeypatch, fmt)
    stub, seen = requantize_probe
    module = stub["module"] = _linear_stub(16, 64)
    stub["out_features"] = 16
    for _ in range(2):  # two steps
        getattr(ext, cls).forward_internal(module, torch.randn(4, 2, 64), is_first_microbatch=True)
    assert seen == [(None, None), (None, None)]


@pytest.mark.parametrize("fmt", list(CONFIGS))
def test_grouped_linear_drops_old_caches_before_requantizing(monkeypatch, requantize_probe, fmt):
    _turbo_state(monkeypatch, fmt)
    stub, seen = requantize_probe
    module = stub["module"] = _linear_stub(16, 64)
    for _ in range(2):
        module.is_first_microbatch = True
        ext.PrimusTurboGroupedLinear.prepare_weights(module, torch.randn(8, 64))
    assert seen == [(None, None), (None, None)]


# ------------------------------------------------------------------------------- cached copies are not autograd leaves


@pytest.mark.parametrize("fmt", list(CONFIGS))
@pytest.mark.parametrize("with_transpose", [True, False])
@pytest.mark.parametrize("transpose_is_a_view", [False, True])
def test_cached_copies_are_not_autograd_leaves(monkeypatch, fmt, with_transpose, transpose_is_a_view):
    """Primus-Turbo hands back copies with the weight's requires_grad; as autograd leaves they would sit in
    AccumulateGrad nodes of every micro-batch's graph (kept alive across steps by the token dispatcher's probs).
    The cache returns them detached (same storage, no requires_grad); a missing transpose copy stays None. A copy
    that is a view with a grad_fn (a TENSORWISE transpose copy) is detached too, not refused."""
    weight = torch.zeros(2, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    made = []

    def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False, **kwargs):
        rowwise = torch.zeros(weight.shape, dtype=torch.bfloat16, requires_grad=weight.requires_grad)
        colwise = None
        if need_weight_transpose_cache:
            colwise = (
                rowwise.transpose(-2, -1)
                if transpose_is_a_view
                else torch.zeros(weight.shape, dtype=torch.bfloat16, requires_grad=weight.requires_grad)
            )
        made.extend((rowwise, colwise))
        return rowwise, colwise

    monkeypatch.setattr(ext, "create_quantized_weight", create)
    rowwise, colwise = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, CONFIGS[fmt], disable_parameter_transpose_cache=not with_transpose
    )
    assert rowwise.data_ptr() == made[0].data_ptr() and not rowwise.requires_grad and rowwise.grad_fn is None
    if with_transpose:
        assert colwise.data_ptr() == made[1].data_ptr() and colwise.shape == made[1].shape
        assert not colwise.requires_grad and colwise.grad_fn is None
    else:
        assert colwise is None


# --------------------------------------------------------------------------- the manual fallback (no create_quantized_weight)


def test_weight_scaling_recipe_is_turbos_default():
    assert _weight_scaling_recipe(CONFIGS["mxfp8"].data()) == RAW_2D
    assert _weight_scaling_recipe(BLOCKWISE.data()) == RAW_2D
    assert _weight_scaling_recipe(CONFIGS["fp8_tensorwise"].data()) == ScalingRecipe()
    fp4 = CONFIGS["mxfp4"].data()
    preshuffle = getattr(fp4, "use_preshuffle", False)
    assert _weight_scaling_recipe(fp4) == ScalingRecipe(
        use_2d_block=True, shuffle_scale=preshuffle, shuffle_out=preshuffle
    )


@pytest.mark.parametrize("fmt", list(CONFIGS) + ["blockwise"])
def test_fallback_quantizes_with_the_turbo_config(monkeypatch, fmt):
    """A Primus-Turbo without create_quantized_weight: the manual path reads granularity / block size / recipe off
    the Turbo config (the wrapper has none of them), and its copies are not autograd leaves either."""
    config = BLOCKWISE if fmt == "blockwise" else CONFIGS[fmt]
    calls = []

    class Quantized:
        @staticmethod
        def quantize(t, dest_dtype, granularity, block_size, scaling_recipe, axis):
            calls.append((granularity, block_size, scaling_recipe, axis))
            return torch.zeros(t.shape, requires_grad=t.requires_grad)

    monkeypatch.setattr(ext, "create_quantized_weight", None)
    monkeypatch.setattr(ext, "PrimusTurboQuantizedTensor", Quantized)
    weight = torch.nn.Parameter(torch.randn(32, 64))
    rowwise, colwise = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, config, disable_parameter_transpose_cache=False
    )
    data = config.data()
    recipe = _weight_scaling_recipe(data)
    assert calls[0] == (data.granularity, data.block_size, recipe, -1)
    if data.granularity == ScalingGranularity.TENSORWISE:
        assert len(calls) == 1 and colwise.shape == (64, 32)  # the transpose of the row copy
    else:
        assert calls[1] == (data.granularity, data.block_size, recipe, -2)
    for copy in (rowwise, colwise):
        assert not copy.requires_grad and copy.grad_fn is None


# ----------------------------------------------------------------- the bridge: no materialized zero transpose gradient


class _UsesRowCopyOnly(torch.autograd.Function):
    """A GEMM stand-in: y = x @ w^T; returns the wgrad for the row copy and none for the transpose copy."""

    @staticmethod
    def forward(ctx, x, w, w_t):
        ctx.save_for_backward(x, w)
        return x @ w.t()

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        return grad_y @ w, grad_y.t() @ x, None


@pytest.mark.parametrize("fused", [False, True])
def test_bridge_routes_the_wgrad_without_a_transpose_gradient(monkeypatch, fused):
    """Through _bridge_weight_grad with non-leaf cached copies: unfused, the row copy's wgrad lands in main_grad
    on every micro-batch; fused (the GEMM accumulated already), main_grad is left alone. The transpose copy gets
    no gradient, which backward must take as None (no materialized zero tensor of the weight's shape)."""
    monkeypatch.setattr(ext, "_is_gfx1250", lambda: False)
    # the dummy wgrad handed to autograd for the parameter, on the test's (CPU) device
    monkeypatch.setattr(
        ext, "_get_dummy_wgrad", lambda shape, dtype, zero=False: torch.empty(shape, dtype=dtype)
    )
    materialize = []
    original = torch.autograd.function.FunctionCtx.set_materialize_grads

    def record(ctx, value):
        materialize.append(value)
        return original(ctx, value)

    monkeypatch.setattr(torch.autograd.function.FunctionCtx, "set_materialize_grads", record)
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(8, 16))
    weight.main_grad = torch.zeros(8, 16)
    weight.grad_added_to_main_grad = False
    rowwise = weight.detach().clone()
    colwise = weight.detach().t().contiguous()
    expected = torch.zeros(8, 16)
    for _ in range(2):  # two micro-batches
        x = torch.randn(4, 16, requires_grad=True)
        grad_y = torch.randn(4, 8)
        bridged_x, pair = _bridge_weight_grad(
            x, weight, PrimusTurboQuantizedTensorPair(data=rowwise, data_t=colwise), fuse_wgrad_accum=fused
        )
        _UsesRowCopyOnly.apply(bridged_x, pair.data, pair.data_t).backward(grad_y)
        torch.testing.assert_close(x.grad, grad_y @ rowwise)
        expected += grad_y.t() @ x.detach()
        assert weight.grad_added_to_main_grad
    torch.testing.assert_close(weight.main_grad, torch.zeros(8, 16) if fused else expected)
    assert materialize == [False, False]  # the bridge opts out of zero-filled gradients, once per micro-batch
