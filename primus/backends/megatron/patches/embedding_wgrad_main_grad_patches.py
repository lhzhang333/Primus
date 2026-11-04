###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Add the word-embedding gradient straight into ``main_grad`` instead of building a dense one.

Megatron's ``VocabParallelEmbedding`` looks the rows up with ``F.embedding``, whose backward builds a
DENSE ``[vocab, hidden]`` gradient: it zero-fills the whole table, scatters the batch's rows in, and
DDP then adds that dense tensor into ``main_grad``. For a 129280 x 7168 BF16 table that is about
1.9 GB written, then read twice and written again, every micro-batch, to move a few thousand rows.

``embedding_wgrad_into_main_grad: true`` routes the lookup (TP 1, no reduce-scatter of the
embeddings, no GTP, not deterministic mode, weight not shared with the output layer) through an
autograd function whose backward sums the gradient rows of each distinct token in FP32, adds them
into ``weight.main_grad`` with ``index_add_``, sets ``grad_added_to_main_grad`` and returns an
uninitialized dummy -- the contract Megatron's gradient-accumulation fusion uses, so DDP's hook skips
its add. The sums run in one Triton pass over the sorted ids (``_triton/embedding_wgrad.py``), with no
host sync. Without ``main_grad`` (the weight is not wrapped by DDP) the function returns the usual dense
gradient. The forward is unchanged; the gradient differs from the default path only in summation
order (rows of a repeated token are summed in FP32 before the one BF16 rounding of the add).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.embedding.wgrad_into_main_grad"


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "embedding_wgrad_into_main_grad", False))
        and bool(getattr(args, "untie_embeddings_and_output_weights", False))
        and not getattr(args, "mtp_num_layers", None)
        and not bool(getattr(args, "deterministic_mode", False))
    )


def _make_function():
    import torch
    import torch.nn.functional as F

    from primus.backends.megatron.core.extensions._triton.embedding_wgrad import (
        embedding_wgrad_into_main_grad,
    )

    class EmbeddingWgradIntoMainGrad(torch.autograd.Function):
        """``F.embedding(ids, weight)`` whose weight gradient goes straight into ``weight.main_grad``."""

        @staticmethod
        def forward(ctx, ids, weight):
            ctx.save_for_backward(ids)
            # The Parameter itself, for main_grad and the DDP flag; not saved for backward because the
            # backward does not read its values.
            ctx.weight = weight
            return F.embedding(ids, weight)

        @staticmethod
        def backward(ctx, grad_output):
            (ids,) = ctx.saved_tensors
            weight = ctx.weight
            main_grad = getattr(weight, "main_grad", None)
            if main_grad is None or not hasattr(weight, "grad_added_to_main_grad"):
                dense = torch.ops.aten.embedding_dense_backward(grad_output, ids, weight.shape[0], -1, False)
                return None, dense
            embedding_wgrad_into_main_grad(main_grad, ids, grad_output)
            weight.grad_added_to_main_grad = True
            # Megatron's gradient-accumulation fusion hands autograd a dummy of the weight's shape so the
            # DDP hook still runs (on the main backward thread) and, seeing the flag, skips its add.
            if getattr(weight, "zero_out_wgrad", False):
                dummy = torch.zeros(main_grad.shape, dtype=weight.dtype, device=weight.device)
            else:
                dummy = torch.empty(main_grad.shape, dtype=weight.dtype, device=weight.device)
            return None, dummy

    return EmbeddingWgradIntoMainGrad


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Word-embedding weight gradient added straight into main_grad (no dense gradient)",
    condition=_enabled,
)
def patch_embedding_wgrad_into_main_grad(ctx: PatchContext) -> None:
    import torch
    from megatron.core.tensor_parallel import layers

    cls = layers.VocabParallelEmbedding
    if getattr(cls.forward, "_primus_wgrad_into_main_grad", False):
        return
    original = cls.forward
    fn = _make_function()

    def forward(self, input_):
        weight = self.weight
        if (
            self.tp_group.size() == 1
            and not self.reduce_scatter_embeddings
            and getattr(self, "gtp_remat_size", 1) <= 1
            and not self.deterministic_mode
            and weight.requires_grad
            and torch.is_grad_enabled()
            and not getattr(weight, "shared", False)
            and not getattr(weight, "shared_embedding", False)
        ):
            return fn.apply(input_, weight)
        return original(self, input_)

    forward._primus_wgrad_into_main_grad = True
    cls.forward = forward
    log_rank_0(f"[Patch:{_PATCH_KEY}] word-embedding weight gradients go straight into main_grad")
