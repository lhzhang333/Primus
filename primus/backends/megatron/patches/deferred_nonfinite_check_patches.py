###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Deferred NaN / Inf checks: one host sync per training step
==========================================================

With ``check_for_nan_in_loss_and_grad`` (Megatron's default) every micro-batch's loss is validated twice
(``pretrain_gpt.loss_func``: ``torch.isnan`` then ``torch.isinf``), and the micro-batch whose backward launches the
data-parallel gradient collectives validates every DDP bucket group's grad norm twice
(``_ParamAndGradBucketGroup.check_grads``). In ``RerunMode.DISABLED`` each check is ``bool(rejection_func(result))``,
a host sync: the loss checks block the CPU at the end of every forward until the GPU has run it (the CPU loses its
lead over the GPU), and the grad checks drain the GPU queue twice per bucket group.

With this patch these checks are recorded instead -- a copy of the checked value, launched where the check was,
no sync -- while a pipeline schedule (``forward_backward_func``) runs, and evaluated together when the schedule
returns: one fused reduction and one host sync per call, i.e. per training step, before ``train_step`` gets to
``optimizer.step()``. If a check fails, the first failing one in the original call order is re-run through
Megatron's ``validate_result``, which raises the same ``RuntimeError`` (same message with rank / node / device /
iteration, same tracker-file line). Differences, by design:

* timing: the error is raised when the schedule returns, after the remaining micro-batches' forward and backward of
  that step, instead of in the failing micro-batch's ``loss_func`` or in the backward of the last micro-batch before
  its data-parallel collective. The grad norms are still computed before the collective (their kernels are launched
  where they were); only reading them on the host moves.
* the loss in the message is a detached copy: its repr has no ``grad_fn=<...>``. Grad norms keep their dtype and
  have no ``grad_fn``: their repr is unchanged.
* no update with non-finite values: the evaluation runs inside ``forward_backward_func``, before ``train_step``
  calls ``optimizer.step()``. With more than one rank the local verdict is first all-reduced (MAX) over the data
  parallel group (with context parallel) and then the model parallel group -- every rank of the training job, and
  no rank outside it -- so every rank raises: a rank whose own checks passed raises a RuntimeError saying a deferred
  check failed on another rank (without the patch it would wait in the failed rank's next collective).

Only fatal ``torch.isnan`` / ``torch.isinf`` checks of single-element tensors with the messages above, in
``RerunMode.DISABLED``, called while a wrapped schedule runs, are deferred. Every other ``validate_result`` call
(non-fatal or stateful checks such as ``check_for_large_grads`` and the spiky-loss check, the rerun modes, calls
outside a schedule) runs as before; if the schedule is not wrapped, nothing is deferred. An exception inside the
schedule drops the recorded checks.

Opt-in (default off); installed only when Megatron's checks are on and reruns are disabled, the only case with
something to defer:

    defer_nonfinite_checks: true
    check_for_nan_in_loss_and_grad: true
    rerun_mode: disabled
"""

from __future__ import annotations

import functools
import os
import weakref
from collections import OrderedDict

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.deferred_nonfinite_checks"
_MARK = "_primus_deferred_nonfinite"
DEFERRED_MESSAGE_PREFIXES = (
    "found NaN in local forward loss calculation",  # pretrain_gpt.loss_func
    "found Inf in local forward loss calculation",
    "found NaN in local grad norm for bucket #",  # _ParamAndGradBucketGroup.check_grads
    "found Inf in local grad norm for bucket #",
)


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "defer_nonfinite_checks", False))
        and bool(getattr(args, "check_for_nan_in_loss_and_grad", False))
        and str(getattr(args, "rerun_mode", "disabled")) == "disabled"
    )


def _verdict_groups() -> list:
    """The process groups the verdict is all-reduced over: the data parallel group (with context parallel), then
    the model parallel group. A MAX over both reaches every rank of the training job and none outside it (WORLD can
    hold other ranks, e.g. RL inference workers). Outside Megatron's parallel state: WORLD."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return []
    from megatron.core import parallel_state as mpu

    if not mpu.model_parallel_is_initialized():
        return [None] if dist.get_world_size() > 1 else []
    groups = [mpu.get_data_parallel_group(with_context_parallel=True), mpu.get_model_parallel_group()]
    return [g for g in groups if dist.get_world_size(g) > 1]


class _Check:
    __slots__ = ("rsm", "value", "rejection_func", "message", "comparison_func", "tolerance", "fatal")

    def __init__(self, rsm, value, rejection_func, message, comparison_func, tolerance, fatal):
        self.rsm = rsm
        self.value = value
        self.rejection_func = rejection_func
        self.message = message
        self.comparison_func = comparison_func
        self.tolerance = tolerance
        self.fatal = fatal


class DeferredChecks:
    """The checks recorded while a schedule runs. Process-wide, not per thread: the DDP grad hooks validate from
    the autograd engine's thread while the training thread waits in backward()."""

    def __init__(self):
        self.depth = 0  # nesting of wrapped schedules; only the outermost evaluates
        self.checks = []
        self._last = None  # (weakref to the last checked tensor, its _version, its copy)
        self.host_syncs = 0
        self.evaluations = 0
        self.logged = False

    def reset(self):
        self.checks = []
        self._last = None

    def record(self, rsm, result, rejection_func, message, comparison_func, tolerance, fatal):
        last = self._last
        # the isnan and isinf checks of one value share one copy (same tensor, not modified in between)
        if last is not None and last[0]() is result and last[1] == result._version:
            value = last[2]
        else:
            # a copy: the schedule divides the loss in place after loss_func returns
            value = result.detach().clone()
            self._last = (weakref.ref(result), result._version, value)
        self.checks.append(_Check(rsm, value, rejection_func, message, comparison_func, tolerance, fatal))

    def evaluate(self, original_validate):
        """Evaluate the recorded checks with one host sync; raise like Megatron's ``validate_result``."""
        checks = self.checks
        self.reset()
        verdict_groups = _verdict_groups()
        if not checks and not verdict_groups:
            return
        groups = OrderedDict()
        for i, c in enumerate(checks):
            groups.setdefault((c.rejection_func, c.value.dtype, c.value.device), []).append(i)
        order = [i for idxs in groups.values() for i in idxs]
        if checks:
            rejected = torch.cat(
                [
                    fn(torch.stack([checks[i].value.reshape(()) for i in idxs]))
                    for (fn, _, _), idxs in groups.items()
                ]
            )
        else:  # nothing checked on this rank: it still joins the all-reduces
            cuda = torch.cuda.is_available()
            device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
            rejected = torch.zeros(0, dtype=torch.bool, device=device)
        flags = [rejected.to(torch.int32), rejected.any().reshape(1).to(torch.int32)]
        if verdict_groups:
            verdict = flags[1].clone()
            for group in verdict_groups:
                torch.distributed.all_reduce(verdict, op=torch.distributed.ReduceOp.MAX, group=group)
            flags.append(verdict)
        host = torch.cat(flags).tolist()  # the one host sync
        self.host_syncs += 1
        self.evaluations += 1
        if checks and not self.logged:
            self.logged = True
            n_loss = sum(1 for c in checks if "loss calculation" in c.message)
            log_rank_0(
                f"[Patch:{_PATCH_KEY}] first deferred evaluation: {len(checks)} NaN/Inf checks "
                f"({n_loss} loss, {len(checks) - n_loss} grad-norm) with one host sync"
            )
        n = len(order)
        if not host[-1]:
            return
        if host[n]:
            first = min(order[k] for k in range(n) if host[k])
            c = checks[first]
            original_validate(
                c.rsm,
                result=c.value,
                rejection_func=c.rejection_func,
                message=c.message,
                comparison_func=c.comparison_func,
                tolerance=c.tolerance,
                fatal=c.fatal,
            )
            raise RuntimeError(f"deferred check rejected a value its re-run accepted (message='{c.message}')")
        from megatron.core.rerun_state_machine import get_rerun_state_machine

        rsm = get_rerun_state_machine()
        rank = torch.distributed.get_rank()
        raise RuntimeError(
            f"Rank {rank}, node {os.uname()[1]}, device {torch.cuda.current_device()}, "
            f"iteration {rsm.current_iteration + 1}: a NaN / Inf loss or grad-norm check deferred to the end of the "
            "step failed on another rank"
        )


_STATE = DeferredChecks()


def wrap_validate_result(original):
    """``RerunStateMachine.validate_result`` that records the deferrable checks while a wrapped schedule runs."""
    if getattr(original, _MARK, False):
        return original

    @functools.wraps(original)
    def validate_result(
        self, result, rejection_func, message, comparison_func=None, tolerance=0.0, fatal=True
    ):
        if (
            _STATE.depth > 0
            and fatal
            and (rejection_func is torch.isnan or rejection_func is torch.isinf)
            and isinstance(message, str)
            and message.startswith(DEFERRED_MESSAGE_PREFIXES)
            and torch.is_tensor(result)
            and result.numel() == 1
            and _is_disabled(self)
        ):
            _STATE.record(self, result, rejection_func, message, comparison_func, tolerance, fatal)
            return None
        return original(self, result, rejection_func, message, comparison_func, tolerance, fatal)

    setattr(validate_result, _MARK, True)
    validate_result._primus_original = original
    return validate_result


def _is_disabled(rsm) -> bool:
    from megatron.core.rerun_state_machine import RerunMode

    return rsm.mode == RerunMode.DISABLED


def wrap_schedule(schedule, original_validate=None):
    """A pipeline schedule (forward_backward_func) that evaluates the checks deferred during it before returning."""
    if getattr(schedule, _MARK, False):
        return schedule

    @functools.wraps(schedule)
    def forward_backward_func(*args, **kwargs):
        _STATE.depth += 1
        outer = _STATE.depth == 1
        if outer:
            _STATE.reset()
        done = False
        try:
            out = schedule(*args, **kwargs)
            done = True
        finally:
            _STATE.depth -= 1
            if outer and not done:
                _STATE.reset()
        if outer:
            _STATE.evaluate(original_validate or _original_validate())
        return out

    setattr(forward_backward_func, _MARK, True)
    return forward_backward_func


def wrap_getter(get_forward_backward_func):
    """``get_forward_backward_func`` whose schedules evaluate their deferred checks before returning."""
    if getattr(get_forward_backward_func, _MARK, False):
        return get_forward_backward_func

    @functools.wraps(get_forward_backward_func)
    def getter(*args, **kwargs):
        return wrap_schedule(get_forward_backward_func(*args, **kwargs))

    setattr(getter, _MARK, True)
    return getter


def _original_validate():
    from megatron.core.rerun_state_machine import RerunStateMachine

    method = RerunStateMachine.validate_result
    return getattr(method, "_primus_original", method)


def install(rerun_state_machine_cls, *schedule_modules) -> None:
    """Wrap ``validate_result`` of the rerun state machine class and ``get_forward_backward_func`` of the
    modules that hold one (idempotent)."""
    rerun_state_machine_cls.validate_result = wrap_validate_result(rerun_state_machine_cls.validate_result)
    for mod in schedule_modules:
        get = getattr(mod, "get_forward_backward_func", None)
        if get is not None:
            mod.get_forward_backward_func = wrap_getter(get)


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Defer the NaN/Inf loss and DDP grad-norm checks to one fused evaluation (one host sync) at the end of "
        "each forward_backward_func, before optimizer.step (defer_nonfinite_checks)"
    ),
    priority=60,
    condition=_enabled,
)
def patch_deferred_nonfinite_checks(ctx: PatchContext) -> None:
    import megatron.core.pipeline_parallel as mpp
    import megatron.training.training as mt_training
    from megatron.core.rerun_state_machine import RerunStateMachine

    # Primus' trainer wraps mpp.get_forward_backward_func again in train() (dump_pp_data) and calls this one
    install(RerunStateMachine, mpp, mt_training)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   NaN/Inf loss and DDP grad-norm checks are recorded during forward_backward_func "
        "and evaluated once (one host sync) when it returns, before optimizer.step"
    )
