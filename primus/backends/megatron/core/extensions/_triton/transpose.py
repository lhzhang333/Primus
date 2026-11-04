###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A contiguous transpose of a 2-D tensor, for the Primus Megatron GEMM patches.

``x.t().contiguous()`` runs PyTorch's generic strided copy, far below the memory bandwidth on gfx1250;
this tiled copy reads and writes whole tiles. Values are copied bit for bit.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(src, dst, rows, cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """dst[c, r] = src[r, c] for a row-major [rows, cols] src and a row-major [cols, rows] dst."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = (pid_r * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    c = (pid_c * BLOCK_C + tl.arange(0, BLOCK_C)).to(tl.int64)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    tile = tl.load(src + r[:, None] * cols + c[None, :], mask=mask)
    tl.store(dst + c[None, :] * rows + r[:, None], tile, mask=mask)


def transpose_2d(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """``x.t().contiguous()`` for a contiguous 2-D CUDA tensor, into ``out`` when given."""
    assert x.dim() == 2 and x.is_contiguous(), "transpose_2d takes a contiguous 2-D tensor"
    rows, cols = x.shape
    if out is None:
        out = torch.empty((cols, rows), dtype=x.dtype, device=x.device)
    else:
        assert out.shape == (cols, rows) and out.dtype == x.dtype and out.is_contiguous()
    if x.numel() == 0:
        return out
    # taller tiles for the large weights, wider ones for the small (the faster of each on gfx1250)
    block_r, block_c = (128, 64) if x.numel() >= (1 << 25) else (64, 128)
    grid = (triton.cdiv(rows, block_r), triton.cdiv(cols, block_c))
    _transpose_kernel[grid](x, out, rows, cols, BLOCK_R=block_r, BLOCK_C=block_c, num_warps=4)
    return out


def iteration_transposer():
    """A function returning ``weight.t().contiguous()``, made once per training iteration.

    The weights change only in the optimizer step, so a copy is reused while ``args.curr_iteration``
    and the weight's storage stay the same. Each weight keeps one buffer that the next iteration's
    copy is written into, so the copies cost no allocations after the first iteration. The copy is
    only valid until the next iteration's first call for that weight: use it right away.
    """
    cache = {}

    def transposed(weight: torch.Tensor) -> torch.Tensor:
        from megatron.training import get_args

        it = getattr(get_args(), "curr_iteration", None)
        if it is None:
            return transpose_2d(weight.detach())
        storage = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
        hit = cache.get(id(weight))
        if hit is not None and hit[1] == storage:
            if hit[0] != it:
                transpose_2d(weight.detach(), out=hit[2])
                hit[0] = it
            return hit[2]
        wt = transpose_2d(weight.detach())
        cache[id(weight)] = [it, storage, wt]
        return wt

    return transposed
