###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The dense MXFP8 SwiGLU MLP patch (``dense_mlp_mxfp8_patches``): which MLPs it routes to Primus-Turbo's
``mlp_fp8`` (CPU), and on gfx1250 the routed forward / backward against the unrouted path (the Turbo MX linears around
Megatron's SwiGLU): bitwise when that SwiGLU computes the fused kernels' bf16 values, within the activation's
rounding otherwise; cached weights and the wgrads fused into a bf16 main_grad, one and two micro-batches."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import primus.backends.megatron.patches.turbo.dense_mlp_mxfp8_patches as patch

# ----------------------------------------------------------------------------------------------- selection (CPU)


class _Fc1:
    pass


class _Fc2:
    pass


def _cfg(**kw):
    base = dict(
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        use_te_activation_func=False,
        glu_linear_offset=0.0,
        activation_func_clamp_value=None,
    )
    return SimpleNamespace(**{**base, **kw})


def _args(**kw):
    return SimpleNamespace(
        **{**dict(turbo_fused_gemm=True, fp8="e4m3", fp8_recipe="mxfp8", use_turbo_gemm=True), **kw}
    )


@pytest.fixture
def stub_types(monkeypatch):
    monkeypatch.delenv("PRIMUS_TURBO_FUSED_DENSE_MLP", raising=False)
    monkeypatch.delenv("PRIMUS_TURBO_FUSED_GEMM", raising=False)
    monkeypatch.setattr(patch, "_turbo_linear_types", lambda: ((_Fc1,), (_Fc2,)))
    monkeypatch.setattr(patch, "_mlp_fp8_op", lambda: object())
    monkeypatch.setattr(patch, "is_gfx1250", lambda: True)


def _mlp(cfg=None, fc1=None, fc2=None):
    return SimpleNamespace(config=cfg or _cfg(), linear_fc1=fc1 or _Fc1(), linear_fc2=fc2 or _Fc2())


def test_enabled_for_a_shared_expert_mlp(stub_types):
    assert patch._should_enable(_mlp(), _args(), is_expert=False)


@pytest.mark.parametrize(
    "args, cfg, is_expert, linears",
    [
        (dict(turbo_fused_gemm=False), {}, False, {}),  # switch off (the default)
        (dict(fp8_recipe="tensorwise"), {}, False, {}),  # not the MXFP8 recipe
        (dict(fp8=None), {}, False, {}),
        (dict(use_turbo_gemm=False), {}, False, {}),  # TE linears
        ({}, {}, True, {}),  # a routed / sequential expert
        ({}, dict(activation_func=F.gelu), False, {}),
        ({}, dict(gated_linear_unit=False), False, {}),
        ({}, dict(add_bias_linear=True), False, {}),
        ({}, dict(use_te_activation_func=True), False, {}),
        ({}, dict(glu_linear_offset=1.0), False, {}),
        ({}, dict(activation_func_clamp_value=7.0), False, {}),
        ({}, dict(layernorm_zero_centered_gamma=True), False, {}),  # the fc1-fused norm would miss the +1
        ({}, {}, False, dict(fc1=torch.nn.Linear(2, 2))),  # not a Turbo linear
        ({}, {}, False, dict(fc2=torch.nn.Linear(2, 2))),
    ],
)
def test_not_enabled(stub_types, args, cfg, is_expert, linears):
    assert not patch._should_enable(_mlp(_cfg(**cfg), **linears), _args(**args), is_expert=is_expert)


def test_env_switch_overrides_the_arg(stub_types, monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_DENSE_MLP", "1")
    assert patch._should_enable(_mlp(), _args(turbo_fused_gemm=False), is_expert=False)
    monkeypatch.setenv("PRIMUS_TURBO_FUSED_DENSE_MLP", "0")
    assert not patch._should_enable(_mlp(), _args(), is_expert=False)


def test_gfx1250_only(stub_types, monkeypatch):
    """mlp_fp8 runs on gfx1250 alone: elsewhere neither the patch nor a module takes it, whatever the switch."""
    from primus.core.patches import PatchContext

    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=_args())}
    )
    assert patch._patch_condition(ctx)
    monkeypatch.setattr(patch, "is_gfx1250", lambda: False)
    assert not patch._patch_condition(ctx)
    assert not patch._should_enable(_mlp(), _args(), is_expert=False)


def test_no_op_without_mlp_fp8(stub_types, monkeypatch):
    monkeypatch.setattr(patch, "_mlp_fp8_op", lambda: None)
    assert not patch._should_enable(_mlp(), _args(), is_expert=False)


class _Manager:
    FP8, TURBO_FP8, TURBO_FP4, QUANT = False, True, False, "mxfp8"

    @classmethod
    def is_fp8_enabled(cls):
        return cls.FP8

    @classmethod
    def is_turbo_fp8_enabled(cls):
        return cls.TURBO_FP8

    @classmethod
    def is_turbo_fp4_enabled(cls):
        return cls.TURBO_FP4

    @classmethod
    def get_turbo_quant_config(cls):
        return cls.QUANT


def _import_ext_or_skip():
    """The Primus-Turbo extension: its import needs Primus-Turbo, which initializes CUDA on import."""
    try:
        import primus.backends.megatron.core.extensions.primus_turbo as ext
    except Exception as e:  # noqa: BLE001 - RuntimeError on a GPU-less host, ImportError without Turbo
        pytest.skip(f"the Primus-Turbo extension does not import here: {type(e).__name__}")
    return ext


@pytest.mark.parametrize(
    "fc1_recipe, fc2_recipe, want",
    [("mx", "mx", True), (None, "mx", False), ("mx", None, False), ("mx", "mx2", False)],
)
def test_route_per_forward(monkeypatch, fc1_recipe, fc2_recipe, want):
    """Both linears under the same Turbo MXFP8 recipe here (else a BF16 match): memoized per autocast state, the
    recipe by value (a new but equal config object per forward hits the memo)."""
    ext = _import_ext_or_skip()
    monkeypatch.setattr(ext, "PrimusTurboLowPrecisionGlobalStateManager", _Manager)
    monkeypatch.setattr(ext, "_use_split_wgrad_op", lambda: False)
    monkeypatch.setattr(ext, "_quant_recipe_key", lambda q: q)
    asked = []
    recipes = {"fc1": fc1_recipe, "fc2": fc2_recipe}
    monkeypatch.setattr(ext, "_turbo_mxfp8_quant", lambda lin: asked.append(lin) or recipes[lin])
    mlp = SimpleNamespace(training=True, linear_fc1="fc1", linear_fc2="fc2")
    assert patch._routes_here(mlp) is want and patch._routes_here(mlp) is want
    assert asked == ["fc1", "fc2"], "the decision is memoized per autocast state"
    monkeypatch.setattr(_Manager, "TURBO_FP8", False)  # another autocast state: decided again
    patch._routes_here(mlp)
    assert len(asked) == 4
    monkeypatch.setattr(ext, "_use_split_wgrad_op", lambda: True)
    assert not patch._routes_here(mlp)


# --------------------------------------------------------------------------------------------- module test (GPU)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="gfx1250 MXFP8 module test")


def _ext_or_skip():
    from primus_turbo.pytorch.core.utils import is_gfx1250

    import primus.backends.megatron.core.extensions.primus_turbo as ext

    if not is_gfx1250() or patch._mlp_fp8_op() is None:
        pytest.skip("needs gfx1250 and Primus-Turbo's mlp_fp8")
    return ext


@pytest.fixture
def mx_state(monkeypatch):
    ext = _ext_or_skip()
    manager = ext.PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", True)
    quant = ext.PrimusTurboQuantConfig(
        format=ext.Format.E4M3,
        granularity=ext.ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ext.ScaleDtype.E8M0,
    )
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", quant)
    monkeypatch.setattr(ext, "_use_split_wgrad_op", lambda: False)
    return ext


class _Stub:
    """A Primus-Turbo linear / MLP stand-in: the attributes their forward paths read."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _randn(*shape, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()


def _linear(n, k, seed, main_grad_dtype):
    w = torch.nn.Parameter(_randn(n, k, scale=k**-0.5, seed=seed))
    w.main_grad = (0.01 * _randn(n, k, seed=seed + 100).float()).to(main_grad_dtype)
    w.grad_added_to_main_grad = False
    config = SimpleNamespace(gradient_accumulation_fusion=True, normalization="RMSNorm")
    return _Stub(
        _parameters={"weight": w},
        use_bias=False,
        offload=False,
        config=config,
        disable_parameter_transpose_cache=False,
        quantized_weight_buffer=None,
        quantized_weight_t_buffer=None,
        te_quant_params=None,
        training=True,
        is_first_microbatch=True,
        _forward_pre_hooks={},
        _forward_pre_hooks_with_kwargs={},
    )


class _KernelSwiGLU(torch.autograd.Function):
    """SwiGLU with the fused kernels' own bf16 values (their test outputs)."""

    @staticmethod
    def forward(ctx, l1):
        from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q

        flat = l1.reshape(-1, l1.shape[-1])
        act = torch.empty(flat.shape[0], flat.shape[1] // 2, dtype=l1.dtype, device=l1.device)
        Q.glu_quant_mxfp8(flat, pad=True, act_out=act)
        ctx.save_for_backward(flat)
        return act.view(*l1.shape[:-1], -1)

    @staticmethod
    def backward(ctx, d_act):
        from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q

        (flat,) = ctx.saved_tensors
        grad = torch.empty_like(flat)
        Q.dglu_quant_mxfp8(d_act.reshape(flat.shape[0], -1).contiguous(), flat, pad=True, grad_out=grad)
        return grad.view(*d_act.shape[:-1], -1)


def _megatron_swiglu(l1):
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    return bias_swiglu_impl(l1, None, False, False)


def _run(ext, routed, swiglu, micro_batches, main_grad_dtype, s, b, h, i):
    """(out, dx) per micro-batch, then both main_grads: the patched forward, or the unrouted path (the Turbo linears'
    forward_internal around ``swiglu``)."""
    fc1, fc2 = _linear(2 * i, h, 1, main_grad_dtype), _linear(h, i, 2, main_grad_dtype)
    mlp = _Stub(linear_fc1=fc1, linear_fc2=fc2, config=fc1.config, training=True)
    res = []
    for mb in range(micro_batches):
        x = _randn(s, b, h, seed=10 + mb).requires_grad_()
        if routed:
            assert patch._routes_here(mlp)
            out, _ = patch._forward_turbo_dense_mlp_mxfp8(mlp, x)
        else:
            l1, _ = ext.PrimusTurboColumnParallelLinear.forward_internal(fc1, x, fc1.is_first_microbatch)
            out, _ = ext.PrimusTurboRowParallelLinear.forward_internal(
                fc2, swiglu(l1), fc2.is_first_microbatch
            )
            fc1.is_first_microbatch = fc2.is_first_microbatch = False
        out.backward(_randn(s, b, h, seed=20 + mb))
        res += [out.detach(), x.grad]
    for fc in (fc1, fc2):
        assert fc._parameters["weight"].grad_added_to_main_grad, "the wgrad was not fused into main_grad"
        res.append(fc._parameters["weight"].main_grad)
    return res


def _snr(a, ref):
    a, ref = a.double(), ref.double()
    err = (a - ref).pow(2).sum().item()
    return math.inf if err == 0 else 10 * math.log10(ref.pow(2).sum().item() / err)


# the DSv3 shared expert at T = 8192 (s 4096, b 2, H 7168, I 2048); M / H tails (T 1000, H 576 % 128 = 64)
SHAPES = [(4096, 2, 7168, 2048), (500, 2, 576, 256)]


@gpu
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16])
@pytest.mark.parametrize("micro_batches", [1, 2])
@pytest.mark.parametrize("s, b, h, i", SHAPES)
def test_matches_the_unrouted_path_bitwise(mx_state, monkeypatch, s, b, h, i, micro_batches, main_grad_dtype):
    """With the SwiGLU computing the fused kernels' bf16 values, the routed MLP is the unrouted path bit for bit: the same
    GEMMs, weight caches and MXFP8 casts (out, dx per micro-batch; both main_grads)."""
    monkeypatch.setattr(mx_state, "get_num_microbatches", lambda: micro_batches)
    got = _run(mx_state, True, None, micro_batches, main_grad_dtype, s, b, h, i)
    ref = _run(mx_state, False, _KernelSwiGLU.apply, micro_batches, main_grad_dtype, s, b, h, i)
    for k, (a, r) in enumerate(zip(got, ref)):
        assert torch.equal(a, r), f"result {k}: max |diff| {(a.float() - r.float()).abs().max().item()}"


@gpu
@pytest.mark.parametrize("s, b, h, i", SHAPES[:1])
def test_close_to_megatron_swiglu(mx_state, monkeypatch, s, b, h, i):
    """Against the unrouted path with Megatron's SwiGLU: its bf16 activation (and dGLU) differ from the fused kernels' by
    rounding (an ulp in a few elements), which the MXFP8 casts turn into an occasional one-code change: far below
    the recipe's own error (>= 30 dB between the paths; both ~ the same SNR vs the fp32 math)."""
    monkeypatch.setattr(mx_state, "get_num_microbatches", lambda: 2)
    got = _run(mx_state, True, None, 2, torch.bfloat16, s, b, h, i)
    ref = _run(mx_state, False, _megatron_swiglu, 2, torch.bfloat16, s, b, h, i)
    for k, (a, r) in enumerate(zip(got, ref)):
        assert torch.isfinite(a).all() and _snr(a, r) >= 30, (k, _snr(a, r))
    x = _randn(s, b, h, seed=10).float()
    w1, w2 = _randn(2 * i, h, scale=h**-0.5, seed=1).float(), _randn(h, i, scale=i**-0.5, seed=2).float()
    gate, up = (x @ w1.t()).chunk(2, dim=-1)
    ref32 = (F.silu(gate) * up) @ w2.t()  # micro-batch 0's output in fp32
    assert abs(_snr(got[0], ref32) - _snr(ref[0], ref32)) <= 0.5, (_snr(got[0], ref32), _snr(ref[0], ref32))


def test_install_wraps_mlp(stub_types, monkeypatch):
    """The patch wraps Megatron's MLP: constructor decides (an expert never), forward routes per call (else the
    original forward with per_token_scale), backward_dw stands down only after a routed forward."""
    import sys

    calls = []

    class FakeMLP:
        def __init__(self, config, submodules, is_expert=False, **kw):
            self.config, self.linear_fc1, self.linear_fc2, self.training = config, _Fc1(), _Fc2(), True

        def forward(self, hidden_states, per_token_scale=None, **kwargs):
            calls.append(("orig", per_token_scale))
            return "orig", None

        def backward_dw(self):
            calls.append("orig_dw")

    # stand-ins for the two modules the install reads (the real ones need a GPU to import)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.mlp", SimpleNamespace(MLP=FakeMLP))
    monkeypatch.setitem(sys.modules, "megatron.training", SimpleNamespace(get_args=lambda: _args()))
    route = [True]
    monkeypatch.setattr(patch, "_routes_here", lambda m: route[0])
    monkeypatch.setattr(
        patch, "_forward_turbo_dense_mlp_mxfp8", lambda m, h: (calls.append("fused"), ("fused", None))[1]
    )
    patch._install_dense_mlp_mxfp8_patch()
    shared, expert = FakeMLP(_cfg(), None), FakeMLP(_cfg(), None, is_expert=True)
    assert shared._turbo_dense_mlp_mxfp8 and not expert._turbo_dense_mlp_mxfp8
    assert shared.forward("h") == ("fused", None)
    shared.backward_dw()
    assert calls == ["fused"], "backward_dw must not run the linears' wgrad after the fused op"
    route[0] = False  # e.g. a BF16 matcher / autocast here
    assert shared.forward("h") == ("orig", None) and expert.forward("h", "s") == ("orig", None)
    shared.backward_dw()
    assert calls[1:] == [("orig", None), ("orig", "s"), "orig_dw"]
