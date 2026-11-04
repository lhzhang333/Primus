###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gradient clipping's scale pass in one Triton launch.

Megatron's ``clip_grad_by_total_norm_fp32`` multiplies every main gradient by the clip coefficient with
``multi_tensor_scale`` (TE / Apex ``multi_tensor_apply``), which caps each launch at a few hundred
workgroups: for a model with billions of parameters that is hundreds of under-filled launches, several
times slower than the memory allows. ``clip_grad_triton_scale: true`` replaces that pass with
``multi_tensor_scale_`` (one launch per gradient dtype, the same fp32 product and single rounding, so the
gradients come out bit for bit the same). The gradient selection, the coefficient (a device tensor when
TE's ``multi_tensor_scale_tensor`` is available, else a float applied only below 1) and everything else
are Megatron's; a gradient the kernel cannot take sends the whole call back to Megatron's function.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.clip_grad_triton_scale"


def _enabled(ctx: PatchContext) -> bool:
    import torch

    return bool(getattr(get_args(ctx), "clip_grad_triton_scale", False)) and torch.cuda.is_available()


def _make_clip(original, clip_grads):
    import torch

    from primus.backends.megatron.core.extensions._triton.multi_tensor_scale import (
        multi_tensor_scale_,
        supported,
    )

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = []
        for param in parameters:
            if use_decoupled_grad:
                g = getattr(param, "decoupled_grad", None)
                if g is not None:
                    assert g.dtype in (torch.float32, torch.bfloat16)
                    grads.append(clip_grads.to_local_if_dtensor(g).detach())
            elif param.grad is not None:
                assert param.grad.type() == "torch.cuda.FloatTensor"
                grads.append(clip_grads.to_local_if_dtensor(param.grad).detach())
        if not all(supported(g) for g in grads):
            return original(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if isinstance(clip_coeff, torch.Tensor):
            clip_coeff.clamp_max_(1.0)
            if grads:
                multi_tensor_scale_(grads, clip_coeff)
        elif clip_coeff < 1.0 and grads:
            multi_tensor_scale_(grads, clip_coeff)

    clip_grad_by_total_norm_fp32._primus_triton_scale = True
    return clip_grad_by_total_norm_fp32


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Gradient clipping's scale pass as one Triton launch per dtype",
    condition=_enabled,
)
def patch_clip_grad_triton_scale(ctx: PatchContext) -> None:
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as optimizer_module

    original = clip_grads.clip_grad_by_total_norm_fp32
    if getattr(original, "_primus_triton_scale", False):
        return
    clip = _make_clip(original, clip_grads)
    clip_grads.clip_grad_by_total_norm_fp32 = clip
    optimizer_module.clip_grad_by_total_norm_fp32 = clip
    log_rank_0(f"[Patch:{_PATCH_KEY}] gradient clipping scales the gradients in one Triton launch per dtype")
