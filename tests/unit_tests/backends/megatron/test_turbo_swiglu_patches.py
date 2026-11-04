###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Primus-Turbo dense SwiGLU patch (``use_turbo_swiglu``).

Verifies:
  1. Only bias-free, non-FP8-store, non-offloaded float CUDA calls with a unit last-dim stride and
     an even last dim reach the Turbo op; every other call keeps Megatron's implementation.
  2. The patch rebinds ``megatron.core.transformer.mlp.bias_swiglu_impl`` once (idempotent).
  3. With a Primus-Turbo that has the op, forward and backward match Megatron's compiled SwiGLU.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import megatron.core.transformer.mlp as mlp_module

import primus.backends.megatron.patches.turbo.swiglu_patches as patch_mod
from primus.core.patches import PatchContext

CTX = PatchContext(
    backend="megatron",
    phase="before_train",
    extra={"module_config": SimpleNamespace(params=SimpleNamespace(use_turbo_swiglu=True))},
)


def _recorder(tag, calls):
    def fn(*args, **kwargs):
        calls.append(tag)
        return tag

    return fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_routing():
    calls = []
    impl = patch_mod._make_impl(_recorder("megatron", calls), _recorder("turbo", calls))
    x = torch.zeros(8, 2, 64, device="cuda", dtype=torch.bfloat16)
    assert impl(x, None) == "turbo"
    assert impl(x.view(16, 64), None) == "turbo"
    # Megatron's path: bias, FP8 input store, CPU offload, odd last dim, strided last dim, CPU,
    # integer input, 4-D input
    assert impl(x, torch.zeros(64, device="cuda", dtype=torch.bfloat16)) == "megatron"
    assert impl(x, None, True) == "megatron"
    assert impl(x, None, False, True) == "megatron"
    assert impl(torch.zeros(4, 63, device="cuda"), None) == "megatron"
    assert impl(torch.zeros(64, 4, device="cuda").t(), None) == "megatron"
    assert impl(torch.zeros(4, 64), None) == "megatron"
    assert impl(torch.zeros(4, 64, device="cuda", dtype=torch.int32), None) == "megatron"
    assert impl(torch.zeros(2, 2, 2, 64, device="cuda"), None) == "megatron"
    assert calls == ["turbo"] * 2 + ["megatron"] * 8


def test_patch_installs_once(monkeypatch):
    pytest.importorskip("primus_turbo")
    try:
        from primus_turbo.pytorch.ops.activation import swiglu  # noqa: F401
    except ImportError:
        pytest.skip("this Primus-Turbo has no dense swiglu op")
    original = mlp_module.bias_swiglu_impl
    monkeypatch.setattr(mlp_module, "bias_swiglu_impl", original)
    patch_mod.patch_turbo_swiglu(CTX)
    patched = mlp_module.bias_swiglu_impl
    assert patched is not original and getattr(patched, "_primus_turbo_swiglu", False)
    patch_mod.patch_turbo_swiglu(CTX)
    assert mlp_module.bias_swiglu_impl is patched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(64, 2 * 1408), (16, 4, 2 * 4096)])
def test_matches_megatron(shape, monkeypatch):
    try:
        from primus_turbo.pytorch.ops.activation import swiglu
    except ImportError:
        pytest.skip("this Primus-Turbo has no dense swiglu op")
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    impl = patch_mod._make_impl(bias_swiglu_impl, swiglu)
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 3).to(torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_()
    x_tur = x.detach().clone().requires_grad_()
    out_ref = bias_swiglu_impl(x_ref, None)
    out_tur = impl(x_tur, None)
    assert out_tur.shape == out_ref.shape
    torch.testing.assert_close(out_tur, out_ref, rtol=1e-2, atol=1e-2)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_tur.backward(grad)
    torch.testing.assert_close(x_tur.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    # the same fp32 math rounded once: at most a few last-bit differences of the sigmoid
    assert (out_tur != out_ref).float().mean().item() < 1e-3
    assert (x_tur.grad != x_ref.grad).float().mean().item() < 1e-3
