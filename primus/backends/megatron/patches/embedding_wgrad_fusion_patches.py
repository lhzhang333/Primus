###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Embedding wgrad straight into main_grad
=======================================

Megatron's ``VocabParallelEmbedding`` backward is ``embedding_dense_backward``: every micro-batch it allocates a
zeroed gradient the size of the whole table, scatters the token rows into it, and Megatron's grad hook then adds all
of it into ``main_grad``. With this patch the backward computes the same per-row sums on a compacted index space --
``embedding_dense_backward`` over each token's slot (its id's rank among the batch's distinct ids, from a stable
sort: no host sync), so a row's tokens are summed exactly as in the dense gradient -- and a small Triton kernel adds
each touched row's sum onto ``main_grad`` as Megatron's grad hook does (``main_grad.add_(grad)``). The result is
bitwise the unfused one, without the table-sized gradient, its fill or its add. It sets ``grad_added_to_main_grad``
and hands autograd a placeholder from the dummy-wgrad pool; the forward is unchanged (``F.embedding``).
(``index_put_(accumulate=True)`` into ``main_grad`` would add every token row separately in ``main_grad``'s dtype,
a different rounding from the dense gradient's per-row sums.) Opt-in (default off):

    embedding_wgrad_fusion: true

Taken per forward only for TP = 1 without GTP rematerialization or reduce-scatter, under
``gradient_accumulation_fusion`` with a ``main_grad`` on the weight, while training with grad enabled outside
Megatron's ``deterministic_mode`` (which embeds with ``weight[ids]``), and never for a weight with
``zero_out_wgrad`` (shared with another layer of the stage: a tied output layer, MTP), whose other uses add real
gradients through autograd.
"""

from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.embedding_wgrad_fusion"


def _dummy_wgrad(shape, dtype, device):
    """The weight's placeholder gradient, never read or written (Megatron's grad hook sees ``grad_added_to_main_grad``
    and drops it): Primus-Turbo's public ``get_dummy_wgrad`` when it has one (its shared pool), else TE's
    ``get_dummy_wgrad`` -- looked up at call time, so the opt-in ``turbo_shared_dummy_wgrad`` patch's rebinding to
    Primus-Turbo's pool applies."""
    try:
        turbo_ops = importlib.import_module("primus_turbo.pytorch.ops")
    except ImportError:
        turbo_ops = None
    helper = getattr(turbo_ops, "get_dummy_wgrad", None)
    if callable(helper):
        return helper(list(shape), dtype, device)
    te_base = importlib.import_module("transformer_engine.pytorch.module.base")
    return te_base.get_dummy_wgrad(list(shape), dtype)


_ROW_ADD_KERNEL = None


def _row_add_kernel():
    """The Triton kernel (built on first use): ``main[row[u]] = main[row[u]] + sums[u]`` (fp32 add, one rounding to
    main's dtype: what ``main.add_(dense_grad)`` does to that row) for the slots u below the device-side count.
    """
    global _ROW_ADD_KERNEL
    if _ROW_ADD_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(main, sums, row_of_slot, nslots, H, ld_main, ld_sums, BLOCK: tl.constexpr):
            u = tl.program_id(0).to(tl.int64)  # u * ld_sums and row * ld_main can pass 2**31
            if u < tl.load(nslots):
                row = tl.load(row_of_slot + u)
                cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
                mask = cols < H
                old = tl.load(main + row * ld_main + cols, mask=mask, other=0.0).to(tl.float32)
                add = tl.load(sums + u * ld_sums + cols, mask=mask, other=0.0).to(tl.float32)
                tl.store(main + row * ld_main + cols, (old + add).to(main.dtype.element_ty), mask=mask)

        _ROW_ADD_KERNEL = kernel
    return _ROW_ADD_KERNEL


def accumulate_rows_(main: torch.Tensor, ids: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``main += embedding_dense_backward(rows, ids)`` without the table-sized gradient: the per-row sums of
    PyTorch's own kernel over the slots of the batch's distinct ids, added onto the touched rows of ``main`` as
    ``main.add_(dense)`` would. Bitwise that result; deterministic; no host sync (the slot count stays on device).
    """
    T, H = rows.shape
    sorted_ids, perm = torch.sort(ids, stable=True)
    first = torch.ones(T, dtype=torch.bool, device=ids.device)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    seg = torch.cumsum(first, 0) - 1  # slot of each sorted token: its id's rank among the distinct ids
    slot = torch.empty_like(seg).scatter_(0, perm, seg)  # slot of each token, in token order
    sums = torch.ops.aten.embedding_dense_backward(
        rows, slot, T, -1, False
    )  # [T, H]; slots past the count: 0
    row_of_slot = torch.zeros(T, dtype=torch.long, device=ids.device).scatter_(0, seg, sorted_ids)
    block = 1024
    _row_add_kernel()[(T, -(-H // block))](
        main, sums, row_of_slot, seg[-1:] + 1, H, main.stride(0), sums.stride(0), BLOCK=block
    )
    return main


class _EmbeddingWgradIntoMainGrad(torch.autograd.Function):
    """``F.embedding(ids, weight)`` whose backward adds the token rows into ``weight.main_grad``
    (``accumulate_rows_``) and returns a pooled dummy as the weight's gradient (Megatron's grad hook skips it:
    ``grad_added_to_main_grad``)."""

    @staticmethod
    def forward(ctx, ids, weight):
        ctx.save_for_backward(ids)
        ctx.weight = weight
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad_out):
        (ids,) = ctx.saved_tensors
        weight = ctx.weight
        main_grad = weight.main_grad
        accumulate_rows_(main_grad, ids.reshape(-1), grad_out.reshape(-1, grad_out.shape[-1]).contiguous())
        weight.grad_added_to_main_grad = True
        return None, _dummy_wgrad(weight.shape, weight.dtype, weight.device)


def _fusable(emb, ids) -> bool:
    """Whether this forward of ``emb`` (a VocabParallelEmbedding) may take the fused backward."""
    w = emb.weight
    return (
        emb.training
        and torch.is_grad_enabled()
        and not getattr(emb, "deterministic_mode", True)  # Megatron embeds with weight[ids] there
        and w.requires_grad
        and ids.dtype == torch.int64
        and emb.tp_group.size() == 1
        and getattr(emb, "gtp_remat_size", 1) <= 1
        and not getattr(emb, "reduce_scatter_embeddings", False)
        and bool(getattr(emb.config, "gradient_accumulation_fusion", False))
        and isinstance(getattr(w, "main_grad", None), torch.Tensor)
        and hasattr(w, "grad_added_to_main_grad")
        and not getattr(w, "zero_out_wgrad", False)
    )


def _install_embedding_wgrad_fusion_patch() -> None:
    from megatron.core.tensor_parallel import layers

    cls = layers.VocabParallelEmbedding
    if is_patched(cls, _PATCH_KEY):
        return
    orig_forward = cls.forward

    def _forward(self, input_):
        if _fusable(self, input_):
            return _EmbeddingWgradIntoMainGrad.apply(input_, self.weight)
        return orig_forward(self, input_)

    cls.forward = _forward
    mark_patched(cls, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] VocabParallelEmbedding's wgrad goes straight into main_grad (TP=1).")


def _patch_condition(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "embedding_wgrad_fusion", False))


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Embedding wgrad straight into main_grad (no dense table-sized gradient per micro-batch)",
    priority=60,
    condition=_patch_condition,
)
def patch_embedding_wgrad_fusion(ctx: PatchContext) -> None:
    _install_embedding_wgrad_fusion_patch()
