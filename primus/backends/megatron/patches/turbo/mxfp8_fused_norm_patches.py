###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in (PRIMUS_MXFP8_FUSED_NORM, default 0): RMSNorms hand Primus-Turbo MXFP8 operands to the linears that read
them, instead of a bf16 output each of those quantizes again (Primus-Turbo's ``rmsnorm_mxfp8``; see
``primus_turbo.py``, ``_FUSED_MXFP8_ACTIVATION_ATTR``). The sites:

- ``input``: an MLA layer's input RMSNorm and its only readers, the q / kv down projections
  (``link_mxfp8_norm_readers``);
- ``mlp``: a MoE layer's pre-MLP RMSNorm and its shared experts on Primus-Turbo's ``mlp_fp8``
  (``link_mxfp8_mlp_norm_reader``);
- ``up``: the LN-fused q_up / kv_up linears, which need no patch.

This patch links the first two once each TransformerLayer is built; at run time a norm takes the fused path only
when its readers run Turbo MXFP8. The condition reads the switch before anything else and imports neither
Primus-Turbo nor TransformerEngine, so it costs a job that leaves the switch off nothing.
"""

import functools

from primus.backends.megatron.core.extensions.mxfp8_fused_norm_sites import (
    mxfp8_fused_norm_sites,
)
from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_LINKED_SITES = frozenset({"input", "mlp"})


def _linked_site_requested(ctx: PatchContext) -> bool:
    """Whether PRIMUS_MXFP8_FUSED_NORM asks for a site this patch links (``input`` / ``mlp``), with Primus-Turbo on.
    Never raises: a malformed value links nothing here (the ``up`` site raises on it at its first forward)."""
    try:
        sites = mxfp8_fused_norm_sites()
    except ValueError as e:
        warning_rank_0(f"[Patch:megatron.turbo.mxfp8_fused_norm] {e}; no norm is linked")
        return False
    return bool(_LINKED_SITES & sites) and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.mxfp8_fused_norm",
    backend="megatron",
    phase="before_train",
    description="RMSNorm -> MXFP8 operands of the linears that read it (PRIMUS_MXFP8_FUSED_NORM=input / mlp)",
    condition=_linked_site_requested,
    # after megatron.turbo.rms_norm (PrimusTurboRMSNorm in place)
    priority=60,
)
def patch_mxfp8_fused_norm(ctx: PatchContext):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.extensions.primus_turbo import (
        link_mxfp8_mlp_norm_reader,
        link_mxfp8_norm_readers,
    )

    sites = mxfp8_fused_norm_sites()

    init = TransformerLayer.__init__
    if getattr(init, "_primus_mxfp8_norm_link", False):
        return

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        layer = getattr(self, "layer_number", "?")
        if "input" in sites and link_mxfp8_norm_readers(self):
            log_rank_0(
                f"[Patch:megatron.turbo.mxfp8_fused_norm] layer {layer}: "
                "input_layernorm -> MXFP8 operands of linear_q_down_proj / linear_kv_down_proj"
            )
        if "mlp" in sites and link_mxfp8_mlp_norm_reader(self):
            log_rank_0(
                f"[Patch:megatron.turbo.mxfp8_fused_norm] layer {layer}: "
                "pre_mlp_layernorm -> MXFP8 operands of the shared experts' mlp_fp8"
            )

    __init__._primus_mxfp8_norm_link = True
    TransformerLayer.__init__ = __init__
    log_rank_0(
        f"[Patch:megatron.turbo.mxfp8_fused_norm] TransformerLayer links RMSNorms to their MXFP8 readers "
        f"(sites {sorted(_LINKED_SITES & sites)})"
    )
