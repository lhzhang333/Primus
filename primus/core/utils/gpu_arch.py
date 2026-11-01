###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The local GPU's architecture, for paths tuned and validated on one architecture only."""

from functools import lru_cache


@lru_cache(maxsize=1)
def is_gfx1250() -> bool:
    """Whether the current GPU is a gfx1250 (MI455X). False without a GPU; never raises."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        return arch.split(":")[0] == "gfx1250"
    except Exception:  # noqa: BLE001 - an arch probe must never abort a run
        return False
