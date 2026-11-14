###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Skip the all-to-all dispatcher's D2H event wait when nothing was copied to the host
====================================================================================

``MoEAlltoAllTokenDispatcher`` copies routing metadata to the host on a side stream at ``cuda_dtoh_point`` and
waits for those copies (``d2h_event.synchronize()``) at ``cuda_sync_point``. With EP = TP = 1, dropless routing, the
fused permute and PrimusTurbo's grouped GEMM consuming ``tokens_per_expert`` on the device
(``megatron.moe_alltoall_dtoh_turbo_grouped_gemm``), the D2H point copies nothing, yet the wait still blocks the
host once per MoE layer until the GPU has run everything launched before it. The host then cannot queue work ahead
of the GPU, and the launch time of the kernels after each wait becomes GPU idle time.

This patch wraps the installed ``_maybe_dtoh_and_synchronize``. At the D2H point it compares the dispatcher's
attributes before and after the call: a copy is an attribute that held a device tensor and holds something else
afterwards, or one newly set to a host tensor or numpy array; a returned ``tokens_per_expert`` that left the device
counts too. A later sync point of the same dispatch skips ``d2h_event.synchronize()`` only when the D2H point copied
nothing. Any copy -- EP or TP > 1, token dropping, router padding, an unfused permute, ``tokens_per_expert`` on the
host -- keeps the wait. No tensor or kernel changes.

Opt-in (default off):

    moe_skip_idle_dtoh_sync: true
"""

from __future__ import annotations

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.moe_alltoall_skip_idle_dtoh_sync"
_MISSING = object()


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_skip_idle_dtoh_sync", False))


def _on_device(x) -> bool:
    """A device tensor (anything with ``is_cuda`` True)."""
    return getattr(x, "is_cuda", False) is True


def _on_host(x) -> bool:
    """Host-side data a device copy could have produced: a CPU tensor (``is_cuda`` False) or a numpy array."""
    return getattr(x, "is_cuda", None) is False or (
        type(x).__name__ == "ndarray" and type(x).__module__ == "numpy"
    )


def copied_to_host(before: dict, after: dict) -> bool:
    """Whether the attribute snapshots ``before`` / ``after`` (``vars`` of the dispatcher around the D2H point)
    show a device-to-host copy: a device tensor replaced by anything else, or an attribute newly set to host data.
    """
    for name, new in after.items():
        old = before.get(name, _MISSING)
        if new is old:
            continue
        if _on_device(old) and not _on_device(new):
            return True
        if _on_host(new):
            return True
    return False


def wrap_maybe_dtoh_and_synchronize(original):
    """``_maybe_dtoh_and_synchronize`` that skips the event wait of a dispatch whose D2H point copied nothing
    to the host."""
    if getattr(original, "_primus_skip_idle_dtoh_sync", False):
        return original

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        if self.drop_and_pad:  # Megatron does neither copies nor a wait here
            return original(self, point, tokens_per_expert)
        if point == self.cuda_dtoh_point:
            before = dict(vars(self))
            out = original(self, point, tokens_per_expert)
            # When the sync point is this same point, the original has already waited (if it had to); a later
            # sync point of this dispatch reads the flag below.
            self._primus_dtoh_host_copies = copied_to_host(before, vars(self)) or (
                _on_device(tokens_per_expert) and not _on_device(out)
            )
            return out
        if point == self.cuda_sync_point and getattr(self, "_primus_dtoh_host_copies", True) is False:
            if not _maybe_dtoh_and_synchronize.skipped:
                log_rank_0(
                    f"[Patch:{_PATCH_KEY}] first skipped D2H event wait at {point}: nothing was copied "
                    f"to the host at {self.cuda_dtoh_point}"
                )
            _maybe_dtoh_and_synchronize.skipped += 1
            return tokens_per_expert
        return original(self, point, tokens_per_expert)

    _maybe_dtoh_and_synchronize._primus_skip_idle_dtoh_sync = True
    _maybe_dtoh_and_synchronize.skipped = 0
    _maybe_dtoh_and_synchronize.__wrapped__ = original
    _maybe_dtoh_and_synchronize.__doc__ = original.__doc__
    return _maybe_dtoh_and_synchronize


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Skip MoEAlltoAllTokenDispatcher's D2H event wait when its D2H point copied nothing to the host "
        "(moe_skip_idle_dtoh_sync)"
    ),
    # after megatron.moe_alltoall_dtoh_turbo_grouped_gemm (default priority 50), whose method it wraps
    priority=60,
    condition=_enabled,
)
def patch_moe_skip_idle_dtoh_sync(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import token_dispatcher

    cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    cls._maybe_dtoh_and_synchronize = wrap_maybe_dtoh_and_synchronize(cls._maybe_dtoh_and_synchronize)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Wrapped MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize: "
        "the D2H event wait is skipped when nothing was copied to the host"
    )
