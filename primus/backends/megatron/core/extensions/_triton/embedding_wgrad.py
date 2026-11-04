###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Add a lookup's row gradients into a [vocab, hidden] main_grad, one FP32 sum per distinct id.

``embedding_wgrad_into_main_grad(main_grad, ids, grad)`` performs ``main_grad[i] += sum of the grad
rows looked up with id i`` for every id in ``ids``. The ids are sorted and each run's end is found
with ``searchsorted`` (fixed-size outputs: no host sync); one program per sorted position, the first
of its run, sums the run's rows in FP32, rounds the sum to ``main_grad``'s dtype and adds it to its
``main_grad`` row -- the same two roundings as a dense gradient added in, so the result matches that
path bit for bit.
Distinct ids touch distinct rows, so the result does not depend on scheduling.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_add_kernel(
    sorted_ids, order, run_end, grad, main_grad, hidden, grad_stride, mg_stride, BLOCK_H: tl.constexpr
):
    i = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < hidden
    idx = tl.load(sorted_ids + i)
    prev = tl.load(sorted_ids + tl.maximum(i - 1, 0))
    if (i == 0) | (prev != idx):  # the first position of this id's run
        end = tl.load(run_end + i)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for j in range(i, end):
            row = tl.load(order + j).to(tl.int64)
            acc += tl.load(grad + row * grad_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dst = main_grad + idx.to(tl.int64) * mg_stride + cols
        old = tl.load(dst, mask=mask, other=0.0).to(tl.float32)
        # round the sum to main_grad's dtype first, as the dense path's gradient is, so the add matches
        # it bit for bit
        add = acc.to(main_grad.dtype.element_ty).to(tl.float32)
        tl.store(dst, (old + add).to(main_grad.dtype.element_ty), mask=mask)


def embedding_wgrad_into_main_grad(main_grad: torch.Tensor, ids: torch.Tensor, grad: torch.Tensor) -> None:
    """``main_grad[ids[t]] += grad[t]`` for every t, summing repeated ids in FP32 (see module docstring)."""
    flat_ids = ids.reshape(-1)
    g = grad.reshape(-1, grad.shape[-1])
    if not g.is_contiguous():
        g = g.contiguous()
    assert g.shape[0] == flat_ids.numel() and g.shape[1] == main_grad.shape[1]
    assert main_grad.stride(1) == 1
    n = flat_ids.numel()
    if n == 0:
        return
    sorted_ids, order = torch.sort(flat_ids)
    run_end = torch.searchsorted(sorted_ids, sorted_ids, right=True)
    hidden = g.shape[1]
    block_h = 1024
    grid = (n, triton.cdiv(hidden, block_h))
    _segment_add_kernel[grid](
        sorted_ids, order, run_end, g, main_grad, hidden, g.stride(0), main_grad.stride(0),
        BLOCK_H=block_h, num_warps=4,
    )
