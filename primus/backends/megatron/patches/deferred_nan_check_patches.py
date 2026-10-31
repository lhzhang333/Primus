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

The gradient checks keep one sync per bucket group (a flush right after ``check_grads``) unless
``grad_checks_deferrable``. With MegaMoE at EP > 1 and overlapped gradient reduction, the
reduce-scatter a bucket group starts must not queue behind the next layers' Mega kernels: their
workgroups spin until the other EP ranks reach the same kernel, and there are more of them than fit
on the GPU at once, so their dispatch does not finish while they wait. ROCm runs a process's streams
on ``GPU_MAX_HW_QUEUES`` hardware queues and dispatches a queue in order; with 2 (Primus' default)
a reduce-scatter, or the work it waits for, can land in a Mega kernel's queue behind it, and the
ranks deadlock (MI455X, DeepSeek-V3 proxy at EP4: the second step hangs, high-priority streams for
the reduce-scatters do not help). With 4 queues it runs. The group sync instead puts each
reduce-scatter in its queue ahead of those kernels. ``PRIMUS_DEFERRED_GRAD_CHECK=1`` / ``0`` defers
the gradient checks always / never instead.
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


_GRADS_DEFERRABLE = None
_MIN_HW_QUEUES = 4


def _grad_checks_deferrable():
    forced = os.environ.get("PRIMUS_DEFERRED_GRAD_CHECK", "auto")
    if forced in ("0", "1"):
        return forced == "1", f"PRIMUS_DEFERRED_GRAD_CHECK={forced}"
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
    except (ImportError, AssertionError):
        return True, "no Megatron training arguments"
    if not getattr(args, "use_turbo_mega_moe", False) or args.expert_model_parallel_size <= 1:
        return True, "no MegaMoE expert parallelism"
    if not getattr(args, "overlap_grad_reduce", False):
        return True, "the gradient reduce-scatters do not overlap the backward"
    queues = os.environ.get("GPU_MAX_HW_QUEUES") or "4"  # unset: ROCm's default, 4
    if int(queues) < _MIN_HW_QUEUES:
        return False, f"MegaMoE at EP > 1 with GPU_MAX_HW_QUEUES={queues} (needs {_MIN_HW_QUEUES})"
    return True, f"GPU_MAX_HW_QUEUES={queues}"


def grad_checks_deferrable() -> bool:
    """Whether the gradient checks can go without their per-bucket-group sync (module docstring)."""
    global _GRADS_DEFERRABLE
    if _GRADS_DEFERRABLE is None:
        _GRADS_DEFERRABLE, why = _grad_checks_deferrable()
        log_rank_0(
            "[Patch:megatron.rerun.deferred_nan_check] gradient checks "
            f"{'deferred' if _GRADS_DEFERRABLE else 'keep one sync per bucket group'}: {why}"
        )
    return _GRADS_DEFERRABLE


@register_patch(
    "megatron.rerun.deferred_nan_check",
    backend="megatron",
    phase="before_train",
    description="One host sync per step for the fatal NaN/Inf checks of losses and gradient norms",
    condition=_enabled,
    # after megatron.ddp.fused_grad_check_scaling (50): the guard below wraps its check_grads
    priority=60,
)
def patch_deferred_nan_check(ctx: PatchContext) -> None:
    global _ORIGINAL
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
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
    check_grads = _ParamAndGradBucketGroup.check_grads
    if not getattr(check_grads, "_primus_guarded", False):

        @wraps(check_grads)
        def guarded_check_grads(self, *args, **kwargs):
            out = check_grads(self, *args, **kwargs)
            if not grad_checks_deferrable():
                flush()  # one sync per bucket group, as without the deferral
            return out

        guarded_check_grads._primus_guarded = True
        _ParamAndGradBucketGroup.check_grads = guarded_check_grads
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
