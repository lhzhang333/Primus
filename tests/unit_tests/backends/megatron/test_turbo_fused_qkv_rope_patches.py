###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``use_turbo_fused_qkv_rope`` (``fused_single_qkv_rope`` on Primus-Turbo's kernels).

Verifies:
  1. Calls the Turbo kernels take go to Turbo; extra arguments or an unsupported shape keep
     Transformer Engine's function.
  2. The patch needs ``fused_single_qkv_rope`` and installs once, making Megatron see the fused op.
  3. With a Primus-Turbo that has the op, q / k / v and the packed dQKV match Transformer Engine's
     fused QKV RoPE and the unfused split + RoPE + concatenation Megatron runs otherwise.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import megatron.core.transformer.attention as attention_module

import primus.backends.megatron.patches.turbo.fused_qkv_rope_patches as patch_mod
from primus.core.patches import PatchContext


def _ctx(fused_single_qkv_rope=True):
    params = SimpleNamespace(use_turbo_fused_qkv_rope=True, fused_single_qkv_rope=fused_single_qkv_rope)
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


def test_routing():
    calls = []

    def original(*args, **kwargs):
        calls.append(("te", len(args), tuple(kwargs)))
        return "te"

    def turbo(*args):
        calls.append(("turbo", len(args), ()))
        return "turbo"

    supported = {"ok": True}
    apply = patch_mod._make_apply(original, turbo, lambda *a: None if supported["ok"] else "unsupported")
    assert apply("qkv", "qf", "kf", [512, 128, 128]) == "turbo"
    assert apply("qkv", "qf", "kf", [512, 128, 128], "bshd") == "te"
    assert apply("qkv", "qf", "kf", [512, 128, 128], cu_seqlens="c") == "te"
    supported["ok"] = False
    assert apply("qkv", "qf", "kf", [512, 128, 128]) == "te"
    assert [c[0] for c in calls] == ["turbo", "te", "te", "te"]
    with pytest.raises(RuntimeError):
        patch_mod._make_apply(None, turbo, lambda *a: "unsupported")(torch.empty(1), "qf", "kf", [1, 2, 3])


def test_patch_needs_fused_single_qkv_rope(monkeypatch):
    pytest.importorskip("primus_turbo")
    try:
        from primus_turbo.pytorch.ops.rope import fused_qkv_rope  # noqa: F401
    except ImportError:
        pytest.skip("this Primus-Turbo has no fused_qkv_rope")
    original = getattr(attention_module, "apply_fused_qkv_rotary_pos_emb", None)
    monkeypatch.setattr(attention_module, "apply_fused_qkv_rotary_pos_emb", original, raising=False)
    monkeypatch.setattr(attention_module, "HAVE_FUSED_QKV_ROPE", attention_module.HAVE_FUSED_QKV_ROPE)
    patch_mod.patch_turbo_fused_qkv_rope(_ctx(fused_single_qkv_rope=False))
    assert getattr(attention_module, "apply_fused_qkv_rotary_pos_emb", None) is original
    patch_mod.patch_turbo_fused_qkv_rope(_ctx())
    patched = attention_module.apply_fused_qkv_rotary_pos_emb
    assert getattr(patched, "_primus_turbo", False) and attention_module.HAVE_FUSED_QKV_ROPE
    patch_mod.patch_turbo_fused_qkv_rope(_ctx())
    assert attention_module.apply_fused_qkv_rotary_pos_emb is patched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_matches_te_and_unfused():
    try:
        from primus_turbo.pytorch.ops.rope import fused_qkv_rope
    except ImportError:
        pytest.skip("this Primus-Turbo has no fused_qkv_rope")
    te_rope = pytest.importorskip("transformer_engine.pytorch.attention.rope")
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd,
    )

    torch.manual_seed(0)
    S, B, G, Q, D = 64, 2, 8, 4, 128  # Llama-3.1-8B: 8 KV groups, 4 query heads per group
    split = [Q * D, D, D]
    inv = 1.0 / (500000.0 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    ang = torch.outer(torch.arange(S, device="cuda", dtype=torch.float32), inv)
    freqs = torch.cat((ang, ang), dim=-1)[:, None, None, :].contiguous()  # [S, 1, 1, D]
    qkv = torch.randn(S, B, G, (Q + 2) * D, device="cuda", dtype=torch.bfloat16)
    grads = (
        torch.randn(S, B, G * Q, D, device="cuda", dtype=torch.bfloat16),
        torch.randn(S, B, G, D, device="cuda", dtype=torch.bfloat16),
        torch.randn(S, B, G, D, device="cuda", dtype=torch.bfloat16),
    )

    def run(fn):
        x = qkv.detach().clone().requires_grad_()
        outs = fn(x)
        torch.autograd.backward(outs, grads)
        return [o.detach().float() for o in outs], x.grad.float()

    def unfused(x):
        q, k, v = torch.split(x, split, dim=3)
        q = q.reshape(S, B, G * Q, D)
        return _apply_rotary_pos_emb_bshd(q, freqs), _apply_rotary_pos_emb_bshd(k, freqs), v

    outs_t, dqkv_t = run(
        lambda x: patch_mod._make_apply(None, fused_qkv_rope, lambda *a: None)(x, freqs, freqs, split)
    )
    outs_te, dqkv_te = run(lambda x: te_rope.apply_fused_qkv_rotary_pos_emb(x, freqs, freqs, split))
    outs_u, dqkv_u = run(unfused)
    # Turbo and Transformer Engine both compute in fp32 and round once: at most one bf16 ulp apart on
    # a few elements. Megatron's unfused path rounds cos / sin and every intermediate to bf16.
    for a, b in zip(outs_t, outs_te):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=2**-8, atol=2**-6)
        assert (a != b).float().mean().item() < 1e-3
    torch.testing.assert_close(dqkv_t, dqkv_te, rtol=2**-8, atol=2**-6)
    assert (dqkv_t != dqkv_te).float().mean().item() < 1e-3
    for a, b in zip(outs_t, outs_u):
        torch.testing.assert_close(a, b, rtol=2**-6, atol=2**-4)
    torch.testing.assert_close(dqkv_t, dqkv_u, rtol=2**-6, atol=2**-4)
    # v passes through unchanged
    assert torch.equal(outs_t[2], qkv[..., (Q + 1) * D :].float())
