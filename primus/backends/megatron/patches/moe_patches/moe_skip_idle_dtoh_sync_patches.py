###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Skip the all-to-all dispatcher's D2H event wait when nothing was copied to the host
====================================================================================

``MoEAlltoAllTokenDispatcher`` copies its routing metadata to the host on a side stream at
``cuda_dtoh_point`` and waits for those copies (``d2h_event.synchronize()``) at ``cuda_sync_point``.
With EP = TP = 1 the sync point is ``before_finish``: the CPU value it waits for is
``tokens_per_expert``. When PrimusTurbo's grouped GEMM consumes ``tokens_per_expert`` on the device
(``megatron.moe_alltoall_dtoh_turbo_grouped_gemm``), dropless routing keeps ``num_out_tokens`` a Python
int and the permute is fused (``num_global_tokens_per_local_expert`` stays on the device), the D2H point
copies nothing, yet every MoE layer's forward still blocks the CPU until the GPU has run everything
launched so far. That wait throws away the CPU's lead over the GPU: in the 1-GPU DSv3 proxy the first MoE
layer of every micro-batch waits ~17 ms for the previous micro-batch's backward, after which the forward
is launch-bound.

This patch wraps whatever ``_maybe_dtoh_and_synchronize`` is installed: at the D2H point it records whether
any of the dispatcher's metadata (or the returned ``tokens_per_expert``) went from a device tensor to a host
copy; at a later sync point of the same dispatch it skips ``d2h_event.synchronize()`` only if nothing did.
Whenever something is copied (EP or TP > 1, token dropping, router padding, unfused permute, or
``tokens_per_expert`` copied to the host) the wait happens exactly as before. No tensor and no kernel
changes: the training is bitwise the same.

Opt-in (default off):

    env   PRIMUS_MOE_SKIP_IDLE_DTOH_SYNC=1
"""

from __future__ import annotations

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.moe_alltoall_skip_idle_dtoh_sync"
_ENV = "PRIMUS_MOE_SKIP_IDLE_DTOH_SYNC"
# dispatcher attributes _maybe_dtoh_and_synchronize may copy to the host at cuda_dtoh_point
_HOST_COPY_ATTRS = (
    "input_splits",
    "output_splits",
    "output_splits_tp",
    "num_out_tokens",
    "num_global_tokens_per_local_expert",
)


def requested() -> bool:
    return os.environ.get(_ENV, "0") not in ("0", "", "false", "False")


def _on_device(x) -> bool:
    """A device (GPU) tensor: torch tensors have ``is_cuda``; host copies (CPU tensors, numpy), ints and
    None do not."""
    return getattr(x, "is_cuda", False) is True


def _device_flags(dispatcher, tokens_per_expert):
    return [_on_device(tokens_per_expert)] + [
        _on_device(getattr(dispatcher, a, None)) for a in _HOST_COPY_ATTRS
    ]


def wrap_maybe_dtoh_and_synchronize(original):
    """``_maybe_dtoh_and_synchronize`` that skips the event wait of a dispatch whose D2H point copied nothing
    to the host."""
    if getattr(original, "_primus_skip_idle_dtoh_sync", False):
        return original

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        if self.drop_and_pad:  # Megatron does neither copies nor a wait here
            return original(self, point, tokens_per_expert)
        if point == self.cuda_dtoh_point:
            before = _device_flags(self, tokens_per_expert)
            out = original(self, point, tokens_per_expert)
            # host copies made here: a device tensor before, not after. If the sync point is this same point,
            # the original has already waited (when it had to); a later sync point reads the flag below.
            after = _device_flags(self, out)
            self._primus_dtoh_host_copies = any(b and not a for b, a in zip(before, after))
            return out
        if point == self.cuda_sync_point and getattr(self, "_primus_dtoh_host_copies", True) is False:
            if not _maybe_dtoh_and_synchronize._logged:
                _maybe_dtoh_and_synchronize._logged = True
                log_rank_0(
                    f"[Patch:{_PATCH_KEY}] first skipped D2H event wait at {point}: nothing was copied "
                    f"to the host at {self.cuda_dtoh_point}"
                )
            _maybe_dtoh_and_synchronize.skipped += 1
            return tokens_per_expert
        return original(self, point, tokens_per_expert)

    _maybe_dtoh_and_synchronize._primus_skip_idle_dtoh_sync = True
    _maybe_dtoh_and_synchronize._logged = False
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
        f"(EP1/TP1 + PrimusTurbo grouped gemm on device; {_ENV}=1)"
    ),
    # after megatron.moe_alltoall_dtoh_turbo_grouped_gemm (default priority 50), whose method it wraps
    priority=60,
    condition=lambda ctx: requested(),
)
def patch_moe_skip_idle_dtoh_sync(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import token_dispatcher

    cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    cls._maybe_dtoh_and_synchronize = wrap_maybe_dtoh_and_synchronize(cls._maybe_dtoh_and_synchronize)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Wrapped MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize: "
        "the D2H event wait is skipped when nothing was copied to the host"
    )
