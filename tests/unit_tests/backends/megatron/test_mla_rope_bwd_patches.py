###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MLA RoPE query backward with contiguous stores gives Megatron's gradient up to the rounding of a
differently contracted multiply-add (one BF16 ulp on a few elements in a million)."""

import pytest
import torch


def _assert_same_up_to_one_ulp(out, ref):
    """A few elements in ten thousand may differ, each by a rounding step of the operands' scale (a rotated
    pair can nearly cancel, so the step is relative to the inputs, not to the small result)."""
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
    diff = (out.float() - ref.float()).abs()
    assert float(diff.max()) <= float(ref.float().abs().max()) * 2.0**-7
    assert int((diff > 0).sum()) <= max(1, out.numel() // 10000)  # a handful of elements, not a layout error

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _grad(seq, batch, heads, qk, emb, seed, cu_seqlens=None):
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_apply_mla_rope_for_q

    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (seq, batch, heads, qk + emb) if cu_seqlens is None else (seq, heads, qk + emb)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(True)
    ang = torch.rand(seq, 1, 1, emb, device="cuda", generator=g) * 6.28
    cos, sin = ang.cos(), ang.sin()
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
    y = fused_apply_mla_rope_for_q(x * 1, cos, sin, qk, emb, cu_seqlens_q=cu_seqlens)
    y.backward(dy)
    return x.grad


@pytest.mark.parametrize("seq,batch,heads", [(512, 2, 128), (257, 1, 16)])
def test_matches_megatron_kernel(monkeypatch, seq, batch, heads):
    from megatron.core.fusions import fused_mla_yarn_rope_apply as rope

    from primus.backends.megatron.patches.mla_rope_bwd_patches import _make_kernel

    ref = _grad(seq, batch, heads, 128, 64, seed=seq)
    monkeypatch.setattr(rope, "rotary_bwd_q_kernel", _make_kernel())
    out = _grad(seq, batch, heads, 128, 64, seed=seq)
    _assert_same_up_to_one_ulp(out, ref)


def test_thd_format(monkeypatch):
    from megatron.core.fusions import fused_mla_yarn_rope_apply as rope

    from primus.backends.megatron.patches.mla_rope_bwd_patches import _make_kernel

    cu = torch.tensor([0, 100, 300, 512], dtype=torch.int32, device="cuda")
    ref = _grad(512, 1, 16, 128, 64, seed=7, cu_seqlens=cu)
    monkeypatch.setattr(rope, "rotary_bwd_q_kernel", _make_kernel())
    out = _grad(512, 1, 16, 128, 64, seed=7, cu_seqlens=cu)
    _assert_same_up_to_one_ulp(out, ref)
