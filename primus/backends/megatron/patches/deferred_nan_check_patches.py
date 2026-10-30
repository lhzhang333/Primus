###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check a step's fatal NaN / Inf validations with one host sync, before the step's update.

``check_for_nan_in_loss_and_grad`` (on by default) validates every micro-batch's loss and every
gradient bucket's norm through ``RerunStateMachine.validate_result``. In the default ``disabled``
rerun mode that evaluates ``torch.isnan`` / ``torch.isinf`` on the host at once, a sync each, and the
GPU then waits for the host to queue work again. On the 4-layer DeepSeek-V3 proxy (EP4, 8
micro-batches) that is 16 syncs for the losses and 22 for the gradients, ~4.6 ms of GPU idle in a
~930 ms step.

With this patch those calls (fatal, ``torch.isnan`` / ``torch.isinf``, a one-element CUDA result)
are queued with their verdict computed on the device, and one sync checks the queue:

- in ``DistributedDataParallel.finish_grad_sync``, which runs once per step after the last
  backward, at its start and end (the end covers buckets it starts itself);
- at the entry of the optimizers' ``step``, and at the end of ``evaluate``, as backstops;
- whenever the queue reaches ``_MAX_PENDING`` entries.

A rejected entry is replayed through the original ``validate_result``, in queue order, so the error,
its message and the validation log are the ones upstream raises, in the same iteration and before
any parameter is updated. What changes is the moment: the step runs to the end of its backward
first. Other rerun modes, non-fatal checks and other rejection functions are untouched.

``PRIMUS_DEFERRED_NAN_CHECK=0`` turns it off.
"""

import os
from functools import wraps

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_MAX_PENDING = 4096
_PENDING: list = []
_ORIGINAL = None


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_DEFERRED_NAN_CHECK", "1") != "0"


def deferring() -> bool:
    """Whether ``validate_result`` currently queues the checks this patch defers."""
    from megatron.core.rerun_state_machine import RerunStateMachine

    return getattr(RerunStateMachine.validate_result, "_primus_deferred", False)


def flush() -> None:
    """Check every queued validation with one sync; replay them in order if any is rejected."""
    if not _PENDING:
        return
    pending = list(_PENDING)
    _PENDING.clear()
    if not bool(torch.cat([entry[0] for entry in pending]).any()):
        return
    for _, machine, args in pending:
        _ORIGINAL(machine, *args)


def _deferrable(machine, result, rejection_func, fatal) -> bool:
    from megatron.core.rerun_state_machine import RerunMode

    return (
        machine.mode == RerunMode.DISABLED
        and fatal
        and (rejection_func is torch.isnan or rejection_func is torch.isinf)
        and isinstance(result, torch.Tensor)
        and result.is_cuda
        and result.numel() == 1
    )


def _flushing(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        flush()
        out = fn(*args, **kwargs)
        flush()
        return out

    wrapper._primus_flushes = True
    return wrapper


@register_patch(
    "megatron.rerun.deferred_nan_check",
    backend="megatron",
    phase="before_train",
    description="One host sync per step for the fatal NaN/Inf checks of losses and gradient norms",
    condition=_enabled,
)
def patch_deferred_nan_check(ctx: PatchContext) -> None:
    global _ORIGINAL
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.optimizer import optimizer as mcore_optimizer
    from megatron.core.rerun_state_machine import RerunStateMachine

    if getattr(RerunStateMachine.validate_result, "_primus_deferred", False):
        return
    _ORIGINAL = RerunStateMachine.validate_result

    @wraps(_ORIGINAL)
    def validate_result(
        self, result, rejection_func, message, comparison_func=None, tolerance=0.0, fatal=True
    ):
        if not _deferrable(self, result, rejection_func, fatal):
            return _ORIGINAL(self, result, rejection_func, message, comparison_func, tolerance, fatal)
        result = result.detach()
        _PENDING.append(
            (
                rejection_func(result).reshape(1),
                self,
                (result, rejection_func, message, comparison_func, tolerance, fatal),
            )
        )
        if len(_PENDING) >= _MAX_PENDING:
            flush()

    validate_result._primus_deferred = True
    RerunStateMachine.validate_result = validate_result

    flushed = []
    for owner, name in (
        (DistributedDataParallel, "finish_grad_sync"),
        (getattr(mcore_optimizer, "MixedPrecisionOptimizer", None), "step"),
        (getattr(mcore_optimizer, "FP32Optimizer", None), "step"),
        (getattr(mcore_optimizer, "ChainedOptimizer", None), "step"),
    ):
        fn = getattr(owner, name, None) if owner is not None else None
        if fn is not None and not getattr(fn, "_primus_flushes", False):
            setattr(owner, name, _flushing(fn))
            flushed.append(f"{owner.__name__}.{name}")
    try:
        from megatron.training import training

        if not getattr(training.evaluate, "_primus_flushes", False):
            training.evaluate = _flushing(training.evaluate)
            flushed.append("training.evaluate")
    except ImportError:
        pass
    log_rank_0(
        "[Patch:megatron.rerun.deferred_nan_check] fatal NaN/Inf validations are checked with one "
        f"sync at: {', '.join(flushed)}"
    )
