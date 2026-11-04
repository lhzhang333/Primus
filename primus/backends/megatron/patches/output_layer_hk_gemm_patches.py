###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The backward GEMMs of Megatron's non-TE linear layers (the output layer) on HipKittens (gfx1250).

Megatron's ``ColumnParallelLinear`` (the LM head of GPT models) computes, in its backward,
``grad_input = grad_output @ weight`` (an NN GEMM) and ``grad_weight = grad_output.t() @ input`` (NT).
On gfx1250 the hipBLASLt NN and NT kernels run well below its TN ones, and below Primus-Turbo's
HipKittens TN kernels at the LM-head shapes (vocabulary-sized K or N). With
``output_layer_gemm_hipkittens: true`` both run as HipKittens TN GEMMs where it takes the shape:

- ``grad_input = hk_gemm_tn(weight^T, grad_output)``, on a transposed copy of the weight made once per
  training iteration (the weight changes only in the optimizer step; the copy is keyed by
  ``args.curr_iteration`` and the weight's storage, and rewritten in place each iteration);
- ``grad_weight = hk_gemm_nt(input, grad_output)``, HipKittens' NT kernel on both operands as they are
  ([tokens, in] and [tokens, out]).

The forward stays on hipBLASLt (``torch.matmul``), which is as fast there. Only the plain case is
taken -- TP 1, no sequence parallelism, no bias, no gradient-accumulation fusion, no deferred or
buffered weight gradient, BF16 -- every other call keeps Megatron's function. The products are the
same in exact arithmetic; the kernel and its summation order change.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.output_layer.gemm_hipkittens"


def _enabled(ctx: PatchContext) -> bool:
    import torch

    return (
        bool(getattr(get_args(ctx), "output_layer_gemm_hipkittens", False))
        and torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _make_function(hk_tn, hk_nt, transposed):
    """``hk_tn`` / ``hk_nt``: (gemm, plan, supported) of the TN and NT HipKittens GEMMs."""
    import torch
    from megatron.core.tensor_parallel.layers import custom_bwd, custom_fwd

    class LinearHipKittensBackward(torch.autograd.Function):
        """``input @ weight.t()``; the backward GEMMs on HipKittens TN where it takes the shape."""

        @staticmethod
        @custom_fwd
        def forward(ctx, input, weight):
            ctx.save_for_backward(input, weight)
            return torch.matmul(input, weight.t())

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            g = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()  # [T, out]
            x = input.reshape(-1, input.shape[-1])  # [T, in]
            tokens, n_out = g.shape
            n_in = x.shape[1]

            # grad_input [T, in] = g @ W = TN(W^T [in, out], g [T, out])
            grad_input = None
            if hk_tn[1](n_in, tokens, n_out) is not None:
                wt = transposed(weight)
                if hk_tn[2](wt, g):
                    grad_input = hk_tn[0](wt, g)
            if grad_input is None:
                grad_input = g.matmul(weight)
            grad_input = grad_input.view(*grad_output.shape[:-1], n_in)

            # grad_weight [out, in] = g^T @ x = NT(x [T, in], g [T, out])
            grad_weight = None
            x = x.contiguous()
            if hk_nt[1](n_in, n_out, tokens) is not None and hk_nt[2](x, g):
                grad_weight = hk_nt[0](x, g)
            if grad_weight is None:
                grad_weight = g.t().matmul(x)
            return grad_input, grad_weight

    return LinearHipKittensBackward


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Backward GEMMs of Megatron's non-TE linear layers (the output layer) on HipKittens TN (gfx1250)",
    condition=_enabled,
)
def patch_output_layer_gemm_hipkittens(ctx: PatchContext) -> None:
    import torch

    try:
        from primus_turbo.hipkittens.gemm_gfx1250 import (
            hk_gemm_available,
            hk_gemm_nt,
            hk_gemm_nt_plan,
            hk_gemm_nt_supported,
            hk_gemm_tn,
            hk_gemm_tn_plan,
            hk_gemm_tn_supported,
        )
    except ImportError:
        log_rank_0(f"[Patch:{_PATCH_KEY}] Primus-Turbo has no HipKittens gfx1250 TN/NT GEMM; skipped")
        return
    if not hk_gemm_available():
        log_rank_0(f"[Patch:{_PATCH_KEY}] HipKittens gfx1250 GEMM unavailable; skipped")
        return

    from megatron.core.tensor_parallel import layers

    from primus.backends.megatron.core.extensions._triton.transpose import (
        iteration_transposer,
    )

    original = layers.linear_with_grad_accumulation_and_async_allreduce
    if getattr(original, "_primus_hk_backward", False):
        return
    fn = _make_function(
        (hk_gemm_tn, hk_gemm_tn_plan, hk_gemm_tn_supported),
        (hk_gemm_nt, hk_gemm_nt_plan, hk_gemm_nt_supported),
        iteration_transposer(),
    )

    def linear_with_grad_accumulation_and_async_allreduce(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        tp_group=None,
        gtp_remat_size=1,
    ):
        if (
            bias is None
            and not gradient_accumulation_fusion
            and not allreduce_dgrad
            and not sequence_parallel
            and grad_output_buffer is None
            and not wgrad_deferral_limit
            and gtp_remat_size == 1
            and (tp_group is None or tp_group.size() == 1)
            and input.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.dim() == 2
            and weight.requires_grad
        ):
            return fn.apply(input, weight)
        return original(
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
            gtp_remat_size,
        )

    linear_with_grad_accumulation_and_async_allreduce._primus_hk_backward = True
    linear_with_grad_accumulation_and_async_allreduce.warned = True
    layers.linear_with_grad_accumulation_and_async_allreduce = linear_with_grad_accumulation_and_async_allreduce
    log_rank_0(f"[Patch:{_PATCH_KEY}] output-layer dgrad/wgrad GEMMs on HipKittens TN where it takes the shape")
