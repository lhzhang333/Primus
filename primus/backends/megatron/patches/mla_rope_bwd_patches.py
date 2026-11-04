###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLA YaRN RoPE: the query backward with contiguous stores.

Megatron's fused MLA RoPE (``megatron.core.fusions.fused_mla_yarn_rope_apply``) un-rotates the query
gradient in place with ``rotary_bwd_q_kernel``, which writes the two halves of each rotated pair as two
stride-2 stores (even, then odd elements). On gfx1250 that kernel takes about twice as long as the forward
one, which moves the same data with contiguous stores. ``mla_rope_bwd_contiguous_store: true`` swaps in the
same kernel with the two halves interleaved in registers (``tl.interleave``) and written in one contiguous
store per row. The arithmetic is the same; the compiler may contract its multiply-adds differently, so a few
elements in a million can differ by one BF16 ulp.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.fusions.mla_rope_bwd_contiguous_store"


def _make_kernel():
    import triton
    import triton.language as tl
    from megatron.core.fusions.fused_mla_yarn_rope_apply import _get_thd_token_idx

    @triton.autotune(
        configs=[triton.Config({"BLOCK_H": b}) for b in (1, 2, 4, 8, 16, 32, 64, 128)],
        key=["emb_dim", "head_num"],
        restore_value=["DO"],
    )
    @triton.jit
    def rotary_bwd_q_kernel(
        DO,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        """Megatron's rotary_bwd_q_kernel (in place on DO) with one contiguous store per row."""
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)

        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

        DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
        mask = x_off < head_num * stride_x_nheads
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)

        x_1 = x_left * cos_left + x_right * sin_right
        x_2 = -x_left * sin_left + x_right * cos_right

        # [x_1[0], x_2[0], x_1[1], x_2[1], ...]: the stride-2 stores of the original, as one store
        out = tl.interleave(x_1, x_2)
        tl.store(DO + x_off + tl.arange(0, emb_dim)[None, :], out, mask=mask)

    return rotary_bwd_q_kernel


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="MLA YaRN RoPE query backward with contiguous stores",
    condition=lambda ctx: bool(getattr(get_args(ctx), "mla_rope_bwd_contiguous_store", False)),
)
def patch_mla_rope_bwd_contiguous_store(ctx: PatchContext) -> None:
    try:
        from megatron.core.fusions import fused_mla_yarn_rope_apply as rope
    except ImportError:
        log_rank_0(f"[Patch:{_PATCH_KEY}] megatron.core.fusions.fused_mla_yarn_rope_apply unavailable; skipped")
        return
    if getattr(rope, "_primus_bwd_q_contiguous", False):
        return
    rope.rotary_bwd_q_kernel = _make_kernel()
    rope._primus_bwd_q_contiguous = True
    log_rank_0(f"[Patch:{_PATCH_KEY}] MLA RoPE query backward writes each row with one contiguous store")
