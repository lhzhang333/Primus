###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in (PRIMUS_MXFP8_FUSED_NORM=input or 1): an MLA layer's input RMSNorm hands MXFP8 operands to its
two readers.

Megatron's MLA reads ``input_layernorm``'s output only through the q / kv down projections. With this patch
every TransformerLayer links that norm (a ``PrimusTurboRMSNorm``) to those two Primus-Turbo linears after it
is built (``link_mxfp8_norm_readers``); at run time the norm then emits Primus-Turbo's fused RMSNorm + MXFP8
pair for them instead of a bf16 output, when both run Turbo MXFP8 (see ``primus_turbo.py``,
``_FUSED_MXFP8_ACTIVATION_ATTR``). The LN-fused q_up / kv_up site (``up``) needs no patch.
"""

import functools

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _input_site_enabled(ctx: PatchContext) -> bool:
    from primus.backends.megatron.core.extensions.primus_turbo import (
        mxfp8_fused_norm_sites,
    )

    return "input" in mxfp8_fused_norm_sites()


@register_patch(
    "megatron.turbo.mxfp8_fused_norm",
    backend="megatron",
    phase="before_train",
    description="MLA input RMSNorm -> MXFP8 operands of q / kv down projections (PRIMUS_MXFP8_FUSED_NORM)",
    condition=_input_site_enabled,
    # after megatron.turbo.rms_norm (PrimusTurboRMSNorm in place)
    priority=60,
)
def patch_mxfp8_fused_norm(ctx: PatchContext):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.extensions.primus_turbo import (
        link_mxfp8_norm_readers,
    )

    init = TransformerLayer.__init__
    if getattr(init, "_primus_mxfp8_norm_link", False):
        return

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if link_mxfp8_norm_readers(self):
            log_rank_0(
                f"[Patch:megatron.turbo.mxfp8_fused_norm] layer {getattr(self, 'layer_number', '?')}: "
                "input_layernorm -> MXFP8 operands of linear_q_down_proj / linear_kv_down_proj"
            )

    __init__._primus_mxfp8_norm_link = True
    TransformerLayer.__init__ = __init__
    log_rank_0(
        "[Patch:megatron.turbo.mxfp8_fused_norm] TransformerLayer links MLA input norms to MXFP8 readers"
    )
