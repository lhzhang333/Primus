###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place ``t *= coeff`` over a list of tensors in one launch, for the Primus Megatron gradient patches.

``multi_tensor_apply`` (TE / Apex ``multi_tensor_scale``) caps each launch at a few hundred workgroups, so
scaling every gradient of a large model takes hundreds of under-filled launches. Here one launch covers
all tensors of a dtype: program ``i`` scales ``BLOCK`` elements of one tensor, found from a per-program
table that is built once per tensor list (the gradient buffers do not move between steps) and cached.

Each product is taken in fp32 and rounded once to the tensor's dtype, as ``multi_tensor_scale`` does, so
the results are bit for bit the same. A coefficient of exactly 1, which gradient clipping passes as a device
tensor whenever the norm is under the limit, moves no memory (``x * 1`` is ``x``), without the host having to
read the coefficient.
"""

from typing import Dict, List, Tuple, Union

import torch
import triton
import triton.language as tl

_BLOCK = 8192
_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}
_tables: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _multi_tensor_scale_kernel(
    PTRS, NUMELS, BLK_TENSOR, BLK_START, COEFF_PTR, coeff, DTYPE: tl.constexpr, COEFF_ON_DEVICE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.load(BLK_TENSOR + pid)
    start = tl.load(BLK_START + pid)
    n = tl.load(NUMELS + t)
    base = tl.load(PTRS + t).to(tl.pointer_type(DTYPE))
    if COEFF_ON_DEVICE:
        c = tl.load(COEFF_PTR).to(tl.float32)
    else:
        c = coeff
    offs = start + tl.arange(0, BLOCK).to(tl.int64)
    # a coefficient of exactly 1 (clipping's clamp when the norm is under the limit) leaves every value as it
    # is: move nothing
    mask = (offs < n) & (c != 1.0)
    x = tl.load(base + offs, mask=mask).to(tl.float32)
    tl.store(base + offs, (x * c).to(DTYPE), mask=mask)


def supported(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype in _TL_DTYPES and t.is_contiguous() and t.numel() > 0


def _table(tensors: List[torch.Tensor]):
    key = tuple((t.data_ptr(), t.numel()) for t in tensors) + (tensors[0].dtype, tensors[0].device)
    hit = _tables.get(key)
    if hit is None:
        numels = torch.tensor([t.numel() for t in tensors], dtype=torch.int64)
        blocks = (numels + _BLOCK - 1) // _BLOCK
        blk_tensor = torch.repeat_interleave(torch.arange(len(tensors), dtype=torch.int32), blocks)
        first = torch.cumsum(blocks, 0) - blocks  # each tensor's first program
        blk_start = (torch.arange(int(blocks.sum()), dtype=torch.int64) - first.repeat_interleave(blocks)) * _BLOCK
        dev = tensors[0].device
        ptrs = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)
        hit = (ptrs.to(dev), numels.to(dev), blk_tensor.to(dev), blk_start.to(dev))
        if len(_tables) > 64:  # tensor lists are static in training; bound the cache anyway
            _tables.clear()
        _tables[key] = hit
    return hit


def multi_tensor_scale_(tensors: List[torch.Tensor], coeff: Union[float, torch.Tensor]) -> None:
    """``t.mul_(coeff)`` for every tensor (contiguous CUDA bf16 / fp16 / fp32); ``coeff`` a float or a
    one-element device tensor. One launch per dtype."""
    by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for t in tensors:
        if t.numel():
            by_dtype.setdefault(t.dtype, []).append(t)
    on_device = isinstance(coeff, torch.Tensor)
    for dtype, group in by_dtype.items():
        ptrs, numels, blk_tensor, blk_start = _table(group)
        _multi_tensor_scale_kernel[(blk_tensor.numel(),)](
            ptrs,
            numels,
            blk_tensor,
            blk_start,
            coeff if on_device else ptrs,
            0.0 if on_device else float(coeff),
            DTYPE=_TL_DTYPES[dtype],
            COEFF_ON_DEVICE=on_device,
            BLOCK=_BLOCK,
            num_warps=8,
        )
