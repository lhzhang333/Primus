###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in (``turbo_shared_dummy_wgrad``, default off): one pool of dummy wgrad buffers per process, Primus-Turbo's.

With wgrad accumulation fused into ``main_grad``, a linear's backward still hands autograd a tensor of the
weight's shape: Megatron's grad hook sees ``grad_added_to_main_grad`` and drops it, or (a ``zero_out_wgrad``
weight, ``zero=True``) adds it, zeroed, into ``main_grad``. TE (``transformer_engine.pytorch.module.base.
get_dummy_wgrad``: TE linears and Megatron's fused-accumulation linears such as the LM head), Primus-Turbo (its
GEMMs) and the Primus-Turbo extension's linears (``_get_dummy_wgrad``) each keep their own persistent buffers.
This patch points TE's helper, everywhere it was imported, and the extension's at Primus-Turbo's public
``primus_turbo.pytorch.ops.get_dummy_wgrad``, so all of them draw on Primus-Turbo's pool. It applies only with
a Primus-Turbo whose pool is one flat buffer per dtype (``DUMMY_WGRAD_FLAT_POOL``), which costs the largest
weight once instead of one buffer per weight shape.

Every dummy is then a view of one buffer, which is correct while nothing writes a dummy it was handed: the
pool zeroes a ``zero=True`` dummy when handing it out, and DDP's hook reads it at once.
"""

import importlib

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.te.shared_dummy_wgrad"


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if not getattr(args, "turbo_shared_dummy_wgrad", False):
        return False
    return bool(getattr(args, "gradient_accumulation_fusion", False)) and is_primus_turbo_can_patch(ctx)


def turbo_flat_dummy_wgrad():
    """Primus-Turbo's public ``get_dummy_wgrad(shape, dtype, device, zero=False)`` when its pool is one flat buffer
    per dtype (``DUMMY_WGRAD_FLAT_POOL``), else None."""
    try:
        ops = importlib.import_module("primus_turbo.pytorch.ops")
    except ImportError:
        return None
    helper = getattr(ops, "get_dummy_wgrad", None)
    return helper if callable(helper) and getattr(ops, "DUMMY_WGRAD_FLAT_POOL", False) else None


def _te_signature(helper):
    """``helper`` in TE's ``get_dummy_wgrad(shape, dtype, zero=False)`` signature, on the current device."""

    def get_dummy_wgrad(shape, dtype, zero=False):
        return helper(list(shape), dtype, torch.device("cuda", torch.cuda.current_device()), zero=zero)

    return get_dummy_wgrad


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="TE's and the Primus-Turbo linears' dummy wgrads from Primus-Turbo's flat pool (turbo_shared_dummy_wgrad)",
    condition=_enabled,
)
def patch_shared_dummy_wgrad(ctx: PatchContext) -> None:
    del ctx
    helper = turbo_flat_dummy_wgrad()
    if helper is None:
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] turbo_shared_dummy_wgrad is set, but this Primus-Turbo has no flat dummy-wgrad "
            "pool (primus_turbo.pytorch.ops.get_dummy_wgrad with DUMMY_WGRAD_FLAT_POOL): TE and Primus keep their "
            "own dummy buffers."
        )
        return
    # importlib, not "import a.b as c": that resolves through the parent package's attributes, while
    # import_module returns the module sys.modules holds (the same one in a run; the stand-in in a test)
    te_base = importlib.import_module("transformer_engine.pytorch.module.base")
    primus_turbo_ext = importlib.import_module("primus.backends.megatron.core.extensions.primus_turbo")

    if is_patched(te_base, _PATCH_KEY) or not hasattr(te_base, "get_dummy_wgrad"):
        return
    shared = _te_signature(helper)
    rebound = rebind_everywhere(te_base, "get_dummy_wgrad", shared)
    primus_turbo_ext._shared_dummy_wgrad = shared
    mark_patched(te_base, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] get_dummy_wgrad -> Primus-Turbo's flat pool in "
        f"{', '.join(sorted(set(rebound)))} and the Primus-Turbo extension's linears"
    )
