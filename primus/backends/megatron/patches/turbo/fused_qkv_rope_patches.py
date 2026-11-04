###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron's single fused QKV RoPE (``fused_single_qkv_rope``) on Primus-Turbo's FlyDSL kernels.

With ``fused_single_qkv_rope: true`` Megatron's self attention keeps the packed QKV tensor unsplit
and calls ``apply_fused_qkv_rotary_pos_emb(mixed_qkv, q_freqs, k_freqs, split_sizes)`` (Transformer
Engine's), which returns q / k rotated and v, and whose backward writes the packed dQKV directly:
no separate split, query reshape, per-tensor RoPE and concatenation of dq / dk / dv.

``use_turbo_fused_qkv_rope: true`` points that call at ``primus_turbo.pytorch.ops.rope.fused_qkv_rope``
(the same contract) for the inputs its kernels take (bf16, contiguous, head dim 128, q a whole number
of heads per group, S * B a multiple of the kernel's row group) and keeps Transformer Engine's
function for every other call.
"""

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.turbo.fused_qkv_rope"


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "use_turbo_fused_qkv_rope", False)) and is_primus_turbo_can_patch(ctx)


def _make_apply(original, fused_qkv_rope, rope_shape_error):
    """``apply_fused_qkv_rotary_pos_emb`` on Turbo's kernels for the inputs they take, else ``original``."""

    def apply_fused_qkv_rotary_pos_emb(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs):
        # extra arguments (tensor format, start positions, interleaving, packed sequences, context
        # parallelism) are Transformer Engine's only
        if not args and not kwargs and rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list) is None:
            return fused_qkv_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        if original is None:
            raise RuntimeError(
                f"[Patch:{_PATCH_KEY}] unsupported fused QKV RoPE input {tuple(qkv.shape)} and no "
                "Transformer Engine fallback"
            )
        return original(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs)

    apply_fused_qkv_rotary_pos_emb._primus_turbo = True
    return apply_fused_qkv_rotary_pos_emb


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="fused_single_qkv_rope on Primus-Turbo's FlyDSL QKV RoPE kernels",
    condition=_enabled,
)
def patch_turbo_fused_qkv_rope(ctx: PatchContext) -> None:
    import megatron.core.transformer.attention as attention_module

    if not getattr(get_args(ctx), "fused_single_qkv_rope", False):
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] use_turbo_fused_qkv_rope needs fused_single_qkv_rope: true; not installed"
        )
        return
    try:
        from primus_turbo.pytorch.kernels.rope.rope_impl import rope_shape_error
        from primus_turbo.pytorch.ops.rope import fused_qkv_rope
    except ImportError as exc:
        warning_rank_0(f"[Patch:{_PATCH_KEY}] Primus-Turbo has no fused_qkv_rope ({exc}); not installed")
        return

    original = getattr(attention_module, "apply_fused_qkv_rotary_pos_emb", None)
    if getattr(original, "_primus_turbo", False):
        return
    attention_module.apply_fused_qkv_rotary_pos_emb = _make_apply(original, fused_qkv_rope, rope_shape_error)
    # Megatron gates fused_single_qkv_rope on Transformer Engine's function being importable
    attention_module.HAVE_FUSED_QKV_ROPE = True
    log_rank_0(f"[Patch:{_PATCH_KEY}] apply_fused_qkv_rotary_pos_emb -> Primus-Turbo fused_qkv_rope")
