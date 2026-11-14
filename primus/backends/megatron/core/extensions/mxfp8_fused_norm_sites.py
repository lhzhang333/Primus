###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_MXFP8_FUSED_NORM: the sites where an RMSNorm hands Primus-Turbo linears MXFP8 operands (see
``primus_turbo.py``). Kept out of the Primus-Turbo extension so that the patch condition reading it imports
neither Primus-Turbo nor TransformerEngine."""

import os
from functools import lru_cache

MXFP8_FUSED_NORM_SITES = ("up", "input", "mlp")


def mxfp8_fused_norm_sites() -> frozenset:
    """The sites PRIMUS_MXFP8_FUSED_NORM switches on: 0 (default), up, input, mlp, a comma list, or 1 / all.
    Raises ValueError on an unknown site."""
    return _parse_mxfp8_fused_norm_sites(os.environ.get("PRIMUS_MXFP8_FUSED_NORM", "0"))


@lru_cache(maxsize=16)
def _parse_mxfp8_fused_norm_sites(raw: str) -> frozenset:
    value = raw.strip().lower()
    if value in ("", "0", "off", "false", "no"):
        return frozenset()
    if value in ("1", "all", "on", "true", "yes"):
        return frozenset(MXFP8_FUSED_NORM_SITES)
    sites = frozenset(s.strip() for s in value.split(",") if s.strip())
    unknown = sites.difference(MXFP8_FUSED_NORM_SITES)
    if unknown:
        raise ValueError(
            f"PRIMUS_MXFP8_FUSED_NORM: unknown site(s) {sorted(unknown)}; use 0, 1 or a comma list of up, input, mlp"
        )
    return sites
