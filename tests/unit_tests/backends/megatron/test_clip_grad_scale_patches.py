###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gradient clipping's Triton scale pass (``clip_grad_triton_scale``) against Megatron's
``clip_grad_by_total_norm_fp32`` and TE's ``multi_tensor_scale``: bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_SIZES = [1, 7, 8192, 8193, 65536 * 3 + 5, 1 << 20, 7168 * 4096]


def _tensors(dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", dtype=torch.float32, generator=g).to(dtype) for n in _SIZES]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("on_device", [False, True])
def test_matches_te_multi_tensor_scale(dtype, on_device):
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_scale,
        multi_tensor_scale_tensor,
    )

    from primus.backends.megatron.core.extensions._triton.multi_tensor_scale import (
        multi_tensor_scale_,
    )

    ref, out = _tensors(dtype, 0), _tensors(dtype, 0)
    coeff = 0.1234567
    flag = torch.zeros(1, dtype=torch.int, device="cuda")
    if on_device:
        c = torch.tensor([coeff], dtype=torch.float32, device="cuda")
        multi_tensor_applier(multi_tensor_scale_tensor, flag, [ref, ref], c)
        multi_tensor_scale_(out, c)
    else:
        multi_tensor_applier(multi_tensor_scale, flag, [ref, ref], coeff)
        multi_tensor_scale_(out, coeff)
    for r, o in zip(ref, out):
        assert torch.equal(r, o)
    # the cached table is reused, and a second call scales again
    multi_tensor_scale_(out, 2.0)
    for r, o in zip(ref, out):
        assert torch.equal((r.float() * 2.0).to(dtype), o)


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("norm_on_device", [False, True])
def test_patched_clip_matches_megatron(decoupled, norm_on_device):
    from megatron.core.optimizer import clip_grads

    from primus.backends.megatron.patches.clip_grad_scale_patches import _make_clip

    original = clip_grads.clip_grad_by_total_norm_fp32
    patched = _make_clip(original, clip_grads)
    dtype = torch.bfloat16 if decoupled else torch.float32

    def params(seed):
        out = []
        for i, t in enumerate(_tensors(dtype, seed)):
            p = torch.nn.Parameter(torch.zeros(t.numel(), device="cuda"))
            if decoupled:
                p.decoupled_grad = t
            elif i != 2:  # one parameter without a gradient
                p.grad = t
            out.append(p)
        return out

    norm = 12.5
    total = torch.tensor([norm], device="cuda") if norm_on_device else norm
    ref, out = params(1), params(1)
    original(ref, 1.0, total.clone() if norm_on_device else total, use_decoupled_grad=decoupled)
    patched(out, 1.0, total.clone() if norm_on_device else total, use_decoupled_grad=decoupled)
    for r, o in zip(ref, out):
        rg = r.decoupled_grad if decoupled else r.grad
        og = o.decoupled_grad if decoupled else o.grad
        assert (rg is None) == (og is None)
        if rg is not None:
            assert torch.equal(rg, og)


def test_patch_installs_once(monkeypatch):
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as optimizer_module

    from primus.backends.megatron.patches import clip_grad_scale_patches as mod

    monkeypatch.setattr(clip_grads, "clip_grad_by_total_norm_fp32", clip_grads.clip_grad_by_total_norm_fp32)
    monkeypatch.setattr(
        optimizer_module, "clip_grad_by_total_norm_fp32", optimizer_module.clip_grad_by_total_norm_fp32
    )
    mod.patch_clip_grad_triton_scale(None)
    patched = clip_grads.clip_grad_by_total_norm_fp32
    assert patched._primus_triton_scale and optimizer_module.clip_grad_by_total_norm_fp32 is patched
    mod.patch_clip_grad_triton_scale(None)
    assert clip_grads.clip_grad_by_total_norm_fp32 is patched


def test_device_coefficient_of_one_moves_nothing():
    """Clipping under the limit hands the kernel a device coefficient of exactly 1: the values stay bit for bit
    and the pass reads and writes nothing (much faster than a real scale over the same tensors)."""
    from primus.backends.megatron.core.extensions._triton.multi_tensor_scale import multi_tensor_scale_

    g = torch.Generator(device="cuda").manual_seed(7)
    tensors = [torch.randn(n, device="cuda", generator=g).to(torch.bfloat16) for n in (1 << 26, 3 << 24, 12345)]
    ref = [t.clone() for t in tensors]
    one = torch.ones((), device="cuda")
    multi_tensor_scale_(tensors, one)
    assert all(torch.equal(t, r) for t, r in zip(tensors, ref))

    def timed(coeff):
        multi_tensor_scale_(tensors, coeff)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(10):
            multi_tensor_scale_(tensors, coeff)
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en)

    assert timed(one) < 0.5 * timed(torch.full((), 0.999, device="cuda"))
    # a NaN coefficient still poisons the gradients, as a multiply would
    multi_tensor_scale_(tensors, torch.full((), float("nan"), device="cuda"))
    assert all(torch.isnan(t.float()).all() for t in tensors)
