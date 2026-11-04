###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The word-embedding weight gradient added straight into main_grad matches the default path (dense
gradient, then DDP's main_grad.add_) bit for bit, including heavily repeated token ids."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _function():
    from primus.backends.megatron.patches.embedding_wgrad_main_grad_patches import _make_function

    return _make_function()


# More than 3072 looked-up rows: PyTorch's dense backward then sorts the ids and sums each id's rows in
# FP32 before one rounding, the path every training micro-batch takes (below that it switches to a
# per-feature kernel that rounds differently, so the comparison would test PyTorch's switch, not this).
@pytest.mark.parametrize("vocab,hidden,tokens,id_range", [(4096, 1024, 4096, 64), (32000, 2048, 8192, 32000)])
def test_matches_dense_gradient_plus_ddp_add(vocab, hidden, tokens, id_range):
    fn = _function()
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.nn.Parameter(torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    ids = torch.randint(0, id_range, (2, tokens // 2), device="cuda", generator=g)
    grad = torch.randn(2, tokens // 2, hidden, device="cuda", dtype=torch.bfloat16, generator=g)
    base = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, generator=g) * 0.01

    F.embedding(ids, w).backward(grad)
    ref = base.clone()
    ref.add_(w.grad)

    w.grad = None
    w.main_grad = base.clone()
    w.grad_added_to_main_grad = False
    out = fn.apply(ids, w)
    torch.testing.assert_close(out, F.embedding(ids, w), rtol=0, atol=0)
    out.backward(grad)
    assert w.grad_added_to_main_grad
    torch.testing.assert_close(w.main_grad, ref, rtol=0, atol=0)


def test_without_main_grad_returns_the_dense_gradient():
    fn = _function()
    w = torch.nn.Parameter(torch.randn(1000, 256, device="cuda", dtype=torch.bfloat16))
    ids = torch.randint(0, 1000, (4, 64), device="cuda")
    grad = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
    F.embedding(ids, w).backward(grad)
    ref = w.grad.clone()
    w.grad = None
    fn.apply(ids, w).backward(grad)
    torch.testing.assert_close(w.grad, ref, rtol=0, atol=0)
