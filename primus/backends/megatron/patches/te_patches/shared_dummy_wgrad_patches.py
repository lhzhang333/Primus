###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One set of dummy wgrad buffers per process: Transformer Engine's and Primus-Turbo's.

With wgrad accumulation fused into ``main_grad``, a linear's backward still hands autograd a tensor of the
weight's shape (never read: Megatron's grad hook sees ``grad_added_to_main_grad`` and drops it). TE
(``transformer_engine.pytorch.module.base.get_dummy_wgrad``: TE linears and Megatron's fused-accumulation
linears such as the LM head), Primus-Turbo (``primus_turbo.pytorch.ops.utils._get_dummy_wgrad``: its GEMMs)
and Primus (``_get_dummy_wgrad`` of the Turbo extensions, which uses Primus-Turbo's) each keep their own
persistent buffers. This rebinds TE's helper, everywhere it was imported, to Primus-Turbo's, so all of them
draw on one pool: one buffer per weight shape and dtype, or, with a Primus-Turbo whose helper hands out views
of one flat buffer per dtype, the largest weight once (DeepSeek-V3 proxy with MXFP8 experts: 7.4 GiB of
dummies -> 1.75 GiB).
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.te.shared_dummy_wgrad"


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(getattr(args, "enable_primus_turbo", False)) and bool(
        getattr(args, "gradient_accumulation_fusion", False)
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="TE's dummy wgrad buffers shared with Primus-Turbo's (one pool per process)",
    condition=_enabled,
)
def patch_shared_dummy_wgrad(ctx: PatchContext) -> None:
    del ctx
    try:
        import transformer_engine.pytorch.module.base as te_base
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad
    except ImportError:
        return
    if is_patched(te_base, _PATCH_KEY) or not hasattr(te_base, "get_dummy_wgrad"):
        return
    rebound = rebind_everywhere(te_base, "get_dummy_wgrad", _get_dummy_wgrad)
    mark_patched(te_base, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] get_dummy_wgrad -> Primus-Turbo's in {', '.join(sorted(set(rebound)))}")
