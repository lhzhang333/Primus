###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense MLP SwiGLU on Primus-Turbo's elementwise kernels.

With ``bias_swiglu_fusion`` (Megatron's default) the dense MLP runs SwiGLU through
``bias_swiglu_impl``, whose ``SwiGLUFunction`` forward and backward are ``torch.compile``
(Inductor) kernels. On gfx1250 at Llama-3.1-8B's MLP shape those pointwise kernels stay far
below the memory bandwidth, the backward (which also concatenates the two input-gradient halves)
the most.

``use_turbo_swiglu: true`` points ``megatron.core.transformer.mlp.bias_swiglu_impl`` at
``primus_turbo.pytorch.ops.swiglu`` for the calls it covers: no bias, no FP8 input store, no CPU
offload of the input, a floating-point CUDA input with a unit last-dim stride. Every other call
(and a Primus-Turbo without the op) keeps Megatron's implementation. Both compute in fp32 and round
once to the input dtype.
"""

import torch

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.turbo.swiglu"
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "use_turbo_swiglu", False)) and is_primus_turbo_can_patch(ctx)


def _make_impl(original, turbo_swiglu):
    def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        if (
            bias is not None
            or fp8_input_store
            or cpu_offload_input
            or input.dim() not in (2, 3)
            or input.dtype not in _DTYPES
            or not input.is_cuda
            or input.stride(-1) != 1
            or input.size(-1) % 2
        ):
            return original(input, bias, fp8_input_store, cpu_offload_input)
        return turbo_swiglu(input)

    bias_swiglu_impl._primus_turbo_swiglu = True
    return bias_swiglu_impl


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Dense MLP SwiGLU on Primus-Turbo's elementwise kernels (bias-free calls)",
    condition=_enabled,
)
def patch_turbo_swiglu(ctx: PatchContext) -> None:
    import megatron.core.transformer.mlp as mlp_module

    try:
        from primus_turbo.pytorch.ops.activation import swiglu as turbo_swiglu
    except ImportError:
        warning_rank_0(f"[Patch:{_PATCH_KEY}] this Primus-Turbo has no dense swiglu op; keeping Megatron's")
        return

    original = mlp_module.bias_swiglu_impl
    if getattr(original, "_primus_turbo_swiglu", False):
        return
    mlp_module.bias_swiglu_impl = _make_impl(original, turbo_swiglu)
    log_rank_0(f"[Patch:{_PATCH_KEY}] megatron.core.transformer.mlp.bias_swiglu_impl -> Primus-Turbo swiglu")
