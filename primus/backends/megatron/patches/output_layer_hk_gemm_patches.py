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

Under gradient-accumulation fusion (Megatron adds the weight gradient straight into a BF16
``main_grad`` and hands autograd a dummy), a Primus-Turbo whose NT kernel accumulates
(``hk_gemm_nt_out_ok``) takes that add too: ``hk_gemm_nt(input, grad_output, out=main_grad,
accumulate=True)``, one rounding of the FP32 sum as Megatron's GEMM with beta = 1; the dummy and the
``grad_added_to_main_grad`` flag follow Megatron's function.

The forward stays on hipBLASLt (``torch.matmul``), which is as fast there. Only the plain case is
taken -- TP 1, no sequence parallelism, no bias, no deferred or buffered weight gradient, BF16, and
under gradient-accumulation fusion a BF16 ``main_grad`` and a weight-gradient shape HipKittens takes
-- every other call keeps Megatron's function. The products are the same in exact arithmetic; the
kernel and its summation order change.
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


def _dummy_wgrad(weight, dtype):
    """What Megatron's linear hands autograd for a weight whose gradient went into ``main_grad``."""
    import torch
    from megatron.core.tensor_parallel import layers

    shape = list(weight.main_grad.shape)
    zero = getattr(weight, "zero_out_wgrad", False)
    if getattr(layers, "HAVE_TE", False):
        # TE's get_dummy_wgrad reuses one buffer across layers and micro-batches
        return layers.get_dummy_wgrad(shape, dtype, zero=True) if zero else layers.get_dummy_wgrad(shape, dtype)
    return (torch.zeros if zero else torch.empty)(shape, dtype=dtype, device=torch.cuda.current_device())


def _make_function(hk_tn, hk_nt, transposed, nt_out_ok=None):
    """``hk_tn`` / ``hk_nt``: (gemm, plan, supported) of the TN and NT HipKittens GEMMs; ``nt_out_ok``: the NT
    kernel's output check when it can accumulate (gradient-accumulation fusion), else None."""
    import torch
    from megatron.core.tensor_parallel.layers import custom_bwd, custom_fwd

    class LinearHipKittensBackward(torch.autograd.Function):
        """``input @ weight.t()``; the backward GEMMs on HipKittens TN where it takes the shape."""

        @staticmethod
        @custom_fwd
        def forward(ctx, input, weight, into_main_grad=False):
            ctx.save_for_backward(input, weight)
            ctx.into_main_grad = into_main_grad
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
            x = x.contiguous()
            if ctx.into_main_grad:
                main_grad = weight.main_grad
                if hk_nt[2](x, g) and nt_out_ok(main_grad, x, g, accumulate=True):
                    hk_nt[0](x, g, out=main_grad, accumulate=True)
                else:  # BF16 GEMM with beta = 1, as Megatron's wgrad_gemm_accum_fp16
                    main_grad.addmm_(g.t(), x)
                if not hasattr(weight, "grad_added_to_main_grad"):
                    return grad_input, None, None
                grad_weight = _dummy_wgrad(weight, input.dtype)
                weight.grad_added_to_main_grad = True
                return grad_input, grad_weight, None
            grad_weight = None
            if hk_nt[1](n_in, n_out, tokens) is not None and hk_nt[2](x, g):
                grad_weight = hk_nt[0](x, g)
            if grad_weight is None:
                grad_weight = g.t().matmul(x)
            return grad_input, grad_weight, None

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

    try:  # a Primus-Turbo whose NT kernel accumulates into its output
        from primus_turbo.hipkittens.gemm_gfx1250 import hk_gemm_nt_out_ok
    except ImportError:
        hk_gemm_nt_out_ok = None

    original = layers.linear_with_grad_accumulation_and_async_allreduce
    if getattr(original, "_primus_hk_backward", False):
        return
    fn = _make_function(
        (hk_gemm_tn, hk_gemm_tn_plan, hk_gemm_tn_supported),
        (hk_gemm_nt, hk_gemm_nt_plan, hk_gemm_nt_supported),
        iteration_transposer(),
        hk_gemm_nt_out_ok,
    )

    def into_main_grad(input, weight) -> bool:
        """Gradient-accumulation fusion that the accumulating NT kernel can take: a BF16 main_grad (Megatron
        DDP, not Megatron-FSDP) and a weight-gradient shape it takes."""
        main_grad = getattr(weight, "main_grad", None)
        return (
            hk_gemm_nt_out_ok is not None
            and isinstance(main_grad, torch.Tensor)
            and main_grad.dtype == torch.bfloat16
            and main_grad.shape == weight.shape
            and not hasattr(weight, "__fsdp_param__")
            and hk_gemm_nt_plan(weight.shape[1], weight.shape[0], input.numel() // input.shape[-1]) is not None
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
            if not gradient_accumulation_fusion:
                return fn.apply(input, weight, False)
            if into_main_grad(input, weight):
                return fn.apply(input, weight, True)
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
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] output-layer dgrad/wgrad GEMMs on HipKittens where it takes the shape "
        f"(wgrad into main_grad under gradient-accumulation fusion: {hk_gemm_nt_out_ok is not None})"
    )
