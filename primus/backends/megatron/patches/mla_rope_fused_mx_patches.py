###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Opt-in (PRIMUS_TURBO_MLA_ROPE_FUSION=1): MLA self-attention with Megatron's fused MLA RoPE folded into Primus-Turbo's
MXFP8 attention (see primus/backends/megatron/core/transformer/mla_rope_fused_mx.py)."""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.transformer.mla_rope_fused_mx",
    backend="megatron",
    phase="before_train",
    description="MLA: RoPE fused into Primus-Turbo's MXFP8 attention (PRIMUS_TURBO_MLA_ROPE_FUSION=1)",
    condition=lambda ctx: os.environ.get("PRIMUS_TURBO_MLA_ROPE_FUSION", "0") == "1",
)
def patch_mla_rope_fused_mx(ctx: PatchContext):
    from megatron.core.transformer import multi_latent_attention

    from primus.backends.megatron.core.transformer.mla_rope_fused_mx import install

    install(multi_latent_attention)
    log_rank_0(
        "[Patch:megatron.transformer.mla_rope_fused_mx] MultiLatentAttention.forward -> fused-RoPE MXFP8 path"
    )
