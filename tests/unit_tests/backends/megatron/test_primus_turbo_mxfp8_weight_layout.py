###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cached MXFP8 weights on gfx1250 take the layout Primus-Turbo's ``mx_weight_layout_recipe(grouped)`` reports for
the op kind (dense for a 2-D weight, grouped for the 3-D expert stack); other architectures and recipes, and a
Primus-Turbo without the query, keep Turbo's default recipe."""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as primus_turbo_torch  # noqa: E402  isort:skip

import primus.backends.megatron.core.extensions.primus_turbo as primus_turbo_ext  # noqa: E402  isort:skip
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

MXFP8 = PrimusTurboQuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)
BLOCKWISE = PrimusTurboQuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)
TENSORWISE = PrimusTurboQuantConfig()
GEMM_LAYOUT = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)
RAW = ScalingRecipe(use_2d_block=True)


def test_weight_scaling_recipe_is_turbos_default():
    assert _weight_scaling_recipe(MXFP8.data()) == RAW
    assert _weight_scaling_recipe(BLOCKWISE.data()) == RAW
    assert _weight_scaling_recipe(TENSORWISE.data()) == ScalingRecipe()


def _recording_create(calls, *, takes_recipe):
    if takes_recipe:

        def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False, scaling_recipe=None):
            calls.append(scaling_recipe)
            return "rowwise", "colwise"

    else:

        def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False):
            calls.append("no scaling_recipe argument")
            return "rowwise", "colwise"

    return create


def _turbo_layout_query(monkeypatch, *, present):
    """A Turbo whose dense MX GEMM reads the GEMM layout and whose grouped one does not (yet)."""
    asked = []
    if present:

        def mx_weight_layout_recipe(grouped, shape=None):
            asked.append((grouped, shape))
            return RAW if grouped else GEMM_LAYOUT

        monkeypatch.setattr(
            primus_turbo_torch.ops, "mx_weight_layout_recipe", mx_weight_layout_recipe, raising=False
        )
    else:
        monkeypatch.delattr(primus_turbo_torch.ops, "mx_weight_layout_recipe", raising=False)
    return asked


@pytest.mark.parametrize(
    "gfx1250, config, query, takes_recipe, shape, expected, asked",
    [
        (True, MXFP8, True, True, (64, 64), GEMM_LAYOUT, [(False, (64, 64))]),  # dense linear
        (True, MXFP8, True, True, (2, 64, 64), RAW, [(True, (2, 64, 64))]),  # grouped linear's expert stack
        (True, MXFP8, False, True, (64, 64), None, []),  # a Turbo without the query: its default layout
        (True, MXFP8, True, False, (64, 64), "no scaling_recipe argument", [(False, (64, 64))]),
        (False, MXFP8, True, True, (64, 64), None, []),  # other archs: unchanged
        (True, BLOCKWISE, True, True, (64, 64), None, []),
        (True, TENSORWISE, True, True, (64, 64), None, []),
    ],
)
def test_cached_weight_layout_follows_turbos_query(
    monkeypatch, gfx1250, config, query, takes_recipe, shape, expected, asked
):
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: gfx1250)
    asked_now = _turbo_layout_query(monkeypatch, present=query)
    calls = []
    monkeypatch.setattr(
        primus_turbo_ext, "create_quantized_weight", _recording_create(calls, takes_recipe=takes_recipe)
    )
    weight = torch.zeros(shape, dtype=torch.bfloat16)
    out = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, config, disable_parameter_transpose_cache=False
    )
    assert out == ("rowwise", "colwise")
    assert calls == [expected]
    assert asked_now == asked


@pytest.mark.parametrize("with_transpose", [True, False])
@pytest.mark.parametrize("transpose_is_a_view", [False, True])
def test_cached_copies_are_not_autograd_leaves(monkeypatch, with_transpose, transpose_is_a_view):
    """Primus-Turbo hands back copies with the weight's requires_grad; as autograd leaves they would sit in
    AccumulateGrad nodes of every micro-batch's graph (kept alive across steps by the token dispatcher's probs).
    The cache returns them detached (same storage, no requires_grad); a missing transpose copy stays None. A copy
    that is a view with a grad_fn (a TENSORWISE transpose copy) is detached too, not refused."""
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: False)
    weight = torch.zeros(2, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    made = []

    def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False, scaling_recipe=None):
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

    monkeypatch.setattr(primus_turbo_ext, "create_quantized_weight", create)
    rowwise, colwise = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, MXFP8, disable_parameter_transpose_cache=not with_transpose
    )
    assert rowwise.data_ptr() == made[0].data_ptr() and not rowwise.requires_grad and rowwise.grad_fn is None
    if with_transpose:
        assert colwise.data_ptr() == made[1].data_ptr() and colwise.shape == made[1].shape
        assert not colwise.requires_grad and colwise.grad_fn is None
    else:
        assert colwise is None


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
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: False)
    # the dummy wgrad handed to autograd for the parameter, on the test's (CPU) device
    monkeypatch.setattr(
        primus_turbo_ext, "_get_dummy_wgrad", lambda shape, dtype, zero=False: torch.empty(shape, dtype=dtype)
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
