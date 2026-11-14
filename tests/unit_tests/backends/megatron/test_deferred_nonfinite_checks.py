###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU coverage of megatron.deferred_nonfinite_checks: the NaN / Inf loss and DDP grad-norm checks are recorded
during a schedule and evaluated with one host sync when it returns, raising what Megatron's validate_result raises
for the first failing check; everything else is untouched. Runs Megatron's RerunStateMachine (DISABLED mode) and
its real _ParamAndGradBucketGroup.check_grads on CPU tensors."""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.config.yaml_loader import parse_yaml
from primus.core.patches import run_patches
from primus.core.patches.patch_registry import PatchRegistry

rsm_mod = pytest.importorskip("megatron.core.rerun_state_machine")
pgb = pytest.importorskip("megatron.core.distributed.param_and_grad_buffer")

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "deferred_nonfinite_under_test",
    _ROOT / "primus/backends/megatron/patches/deferred_nonfinite_check_patches.py",
)
dn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dn)

RSM = rsm_mod.RerunStateMachine
ORIGINAL = RSM.validate_result
VERDICT_GROUPS = dn._verdict_groups  # the env fixture replaces it with "single rank"
LOSS_NAN = "found NaN in local forward loss calculation"
LOSS_INF = "found Inf in local forward loss calculation"


def grad_msg(kind, i):
    return (
        f"found {kind} in local grad norm for bucket #{i} "
        f"in backward pass before data-parallel communication collective"
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)  # message text on a CPU-only host
    calls = []

    def spy(self, *a, **k):
        calls.append((a, k))
        return ORIGINAL(self, *a, **k)

    monkeypatch.setattr(RSM, "validate_result", dn.wrap_validate_result(spy))
    RSM.validate_result._primus_original = ORIGINAL  # the failure path re-runs Megatron's method
    monkeypatch.setattr(dn, "_STATE", dn.DeferredChecks())
    monkeypatch.setattr(dn, "_verdict_groups", lambda: [])
    rsm = RSM(mode=rsm_mod.RerunMode.DISABLED)
    rsm.current_iteration = 41
    monkeypatch.setattr(pgb, "get_rerun_state_machine", lambda: rsm)
    return SimpleNamespace(rsm=rsm, immediate=calls)


def loss_checks(rsm, loss):
    """pretrain_gpt.loss_func's two checks, then the schedule's in-place division of the loss."""
    for fn, msg in ((torch.isnan, LOSS_NAN), (torch.isinf, LOSS_INF)):
        rsm.validate_result(result=loss, rejection_func=fn, message=msg, tolerance=0.0, fatal=True)
    loss /= 4.0


def bucket_group(*norm_inputs):
    """A DDP bucket group stand-in whose buckets' grad_data have the given values."""
    return SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.tensor(v)) for v in norm_inputs])


def step(rsm, losses, buckets=(1.0, 2.0, 3.0)):
    """A schedule: one loss per micro-batch, then the last micro-batch's DDP grad checks (Megatron's code)."""

    def schedule():
        for v in losses:
            loss_checks(rsm, torch.tensor(v, requires_grad=True) * 1.0)
        pgb._ParamAndGradBucketGroup.check_grads(bucket_group(*buckets), True, False)
        return "losses"

    return dn.wrap_schedule(schedule)


def expected_error(rsm, value, fn, msg):
    with pytest.raises(RuntimeError) as e:
        ORIGINAL(rsm, result=value, rejection_func=fn, message=msg, tolerance=0.0, fatal=True)
    return str(e.value)


def _ctx(**args):
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))})


@pytest.mark.parametrize(
    "args,want",
    [
        ({}, False),
        (
            dict(defer_nonfinite_checks=False, check_for_nan_in_loss_and_grad=True, rerun_mode="disabled"),
            False,
        ),
        (dict(defer_nonfinite_checks=True, check_for_nan_in_loss_and_grad=True, rerun_mode="disabled"), True),
        # nothing to defer: Megatron's checks off, or a rerun mode (validate_result is not a plain check there)
        (
            dict(defer_nonfinite_checks=True, check_for_nan_in_loss_and_grad=False, rerun_mode="disabled"),
            False,
        ),
        (
            dict(
                defer_nonfinite_checks=True,
                check_for_nan_in_loss_and_grad=True,
                rerun_mode="validate_results",
            ),
            False,
        ),
    ],
)
def test_installed_only_with_something_to_defer(args, want):
    assert dn._enabled(_ctx(**args)) is want


def test_finite_step_one_sync_no_immediate_checks(env):
    assert step(env.rsm, [2.0] * 8)() == "losses"
    assert env.immediate == []  # 16 loss + 6 grad-norm checks, none evaluated on the spot
    assert dn._STATE.host_syncs == 1 and dn._STATE.checks == []


def test_isnan_isinf_pair_shares_one_copy(env):
    seen = {}

    def schedule():
        loss = torch.tensor(3.0, requires_grad=True) * 1.0
        loss_checks(env.rsm, loss)  # isnan + isinf, then loss /= 4 (a new version)
        seen["values"] = [c.value for c in dn._STATE.checks]
        env.rsm.validate_result(result=loss, rejection_func=torch.isnan, message=LOSS_NAN, fatal=True)
        seen["after"] = dn._STATE.checks[-1].value

    dn.wrap_schedule(schedule)()
    a, b = seen["values"]
    assert a is b and a.item() == 3.0 and not a.requires_grad
    assert seen["after"] is not a and seen["after"].item() == 0.75


@pytest.mark.parametrize(
    "losses,buckets,fn,msg,value",
    [
        (
            [1.0, 1.0, math.nan, math.inf],
            (1.0,),
            torch.isnan,
            LOSS_NAN,
            math.nan,
        ),  # first failure in call order
        ([1.0, math.inf, math.nan], (1.0,), torch.isinf, LOSS_INF, math.inf),
        ([1.0, 2.0], (1.0, math.inf, math.nan), torch.isinf, grad_msg("Inf", 1), math.inf),
        ([1.0, 2.0], (1.0, 2.0, math.nan), torch.isnan, grad_msg("NaN", 2), math.nan),
    ],
)
def test_failure_raises_megatrons_error_for_the_first_failing_check(env, losses, buckets, fn, msg, value):
    with pytest.raises(RuntimeError) as e:
        step(env.rsm, losses, buckets)()
    if "loss" in msg:
        ref = torch.tensor(value)  # the recorded copy, before the schedule's in-place division
    else:
        ref = torch.tensor(value).norm(p=2)
    assert str(e.value) == expected_error(env.rsm, ref, fn, msg)
    assert f"iteration {env.rsm.current_iteration + 1}:" in str(e.value)
    assert dn._STATE.host_syncs == 1 and dn._STATE.checks == []


def test_failure_stops_train_step_before_the_optimizer(env):
    """train_step calls optimizer.step() only after forward_backward_func returns."""
    applied = []

    def train_step(schedule):
        schedule()
        applied.append("optimizer.step")

    with pytest.raises(RuntimeError, match="found NaN in local forward loss calculation"):
        train_step(step(env.rsm, [1.0, math.nan]))
    with pytest.raises(RuntimeError, match="grad norm for bucket #0"):
        train_step(step(env.rsm, [1.0], buckets=(math.inf,)))
    assert applied == []
    train_step(step(env.rsm, [1.0]))
    assert applied == ["optimizer.step"]


def test_value_is_taken_at_check_time(env):
    """A loss that becomes non-finite after its check passed is not reported (as with the immediate check)."""

    def schedule():
        loss = torch.tensor(1.0, requires_grad=True) * 1.0
        loss_checks(env.rsm, loss)
        loss.data.fill_(math.nan)

    dn.wrap_schedule(schedule)()


def test_not_deferred_outside_a_schedule_or_for_other_checks(env):
    with pytest.raises(RuntimeError):  # immediate, as Megatron
        env.rsm.validate_result(result=torch.tensor(math.nan), rejection_func=torch.isnan, message=LOSS_NAN)

    def schedule():
        other = [
            dict(result=torch.tensor(1.0), rejection_func=torch.isnan, message="spiky loss", fatal=True),
            dict(result=torch.tensor(1.0), rejection_func=torch.isnan, message=LOSS_NAN, fatal=False),
            dict(result=torch.tensor(1.0), rejection_func=lambda x: x > 10, message=LOSS_NAN, fatal=True),
            dict(result=torch.ones(2), rejection_func=torch.isnan, message=LOSS_NAN, fatal=True),
        ]
        for kw in other:
            with pytest.raises(RuntimeError) if kw["result"].numel() > 1 else _nullcontext():
                env.rsm.validate_result(**kw)
        assert dn._STATE.checks == []

    dn.wrap_schedule(schedule)()
    assert len(env.immediate) == 5


def test_rerun_modes_are_not_deferred(env):
    env.rsm.mode = rsm_mod.RerunMode.REPORT_DETERMINISM_STATS

    def schedule():
        try:
            env.rsm.validate_result(result=torch.tensor(1.0), rejection_func=torch.isnan, message=LOSS_NAN)
        except AssertionError:
            pass  # Megatron's rerun path asserts outside a real forward-backward; what matters is the routing
        assert dn._STATE.checks == []

    dn.wrap_schedule(schedule)()
    assert len(env.immediate) == 1


def test_exception_inside_the_schedule_drops_the_checks(env):
    def schedule():
        loss_checks(env.rsm, torch.tensor(math.nan))
        raise ValueError("boom")

    with pytest.raises(ValueError):
        dn.wrap_schedule(schedule)()
    assert dn._STATE.checks == [] and dn._STATE.depth == 0
    step(env.rsm, [1.0])()  # the next step starts clean


def test_nested_wrappers_evaluate_once(env):
    inner = step(env.rsm, [1.0, 2.0])
    outer = dn.wrap_schedule(lambda: inner())
    assert dn.wrap_schedule(inner) is inner
    outer()
    assert dn._STATE.host_syncs == 1


def test_other_rank_failure_stops_this_rank(env, monkeypatch):
    calls = []

    def all_reduce(t, op=None, group=None):
        calls.append((op, group))
        if group == "mp":
            t.fill_(1)  # a rank of the model parallel group saw a non-finite value

    monkeypatch.setattr(dn, "_verdict_groups", lambda: ["dp", "mp"])
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(rsm_mod, "get_rerun_state_machine", lambda: env.rsm)
    with pytest.raises(RuntimeError, match="failed on another rank"):
        step(env.rsm, [1.0, 2.0])()
    assert calls == [(torch.distributed.ReduceOp.MAX, "dp"), (torch.distributed.ReduceOp.MAX, "mp")]
    assert dn._STATE.host_syncs == 1


def test_all_ranks_finite_no_error(env, monkeypatch):
    monkeypatch.setattr(dn, "_verdict_groups", lambda: ["dp"])
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op=None, group=None: None)
    step(env.rsm, [1.0])()
    dn.wrap_schedule(lambda: None)()  # no checks on this rank: it still joins the all-reduce
    assert dn._STATE.host_syncs == 2


def test_single_rank_without_checks_does_not_sync(env):
    dn.wrap_schedule(lambda: None)()
    assert dn._STATE.host_syncs == 0


def test_verdict_groups_are_the_training_groups(monkeypatch):
    """Data parallel (with context parallel) and model parallel, not WORLD; size-1 groups are skipped."""
    mpu = pytest.importorskip("megatron.core.parallel_state")
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    sizes = {"dp+cp": 4, "tp+pp": 1, None: 16}
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: sizes[group])
    monkeypatch.setattr(mpu, "model_parallel_is_initialized", lambda: True)
    seen = {}

    def dp_group(**kw):
        seen.update(kw)
        return "dp+cp"

    monkeypatch.setattr(mpu, "get_data_parallel_group", dp_group)
    monkeypatch.setattr(mpu, "get_model_parallel_group", lambda: "tp+pp")
    assert VERDICT_GROUPS() == ["dp+cp"] and seen == {"with_context_parallel": True}
    sizes["tp+pp"] = 2
    assert VERDICT_GROUPS() == ["dp+cp", "tp+pp"]
    monkeypatch.setattr(mpu, "model_parallel_is_initialized", lambda: False)
    assert VERDICT_GROUPS() == [None]  # outside Megatron's parallel state: WORLD
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert VERDICT_GROUPS() == []


def test_install_wraps_getters_and_validate_result_once(monkeypatch):
    sched = lambda: "ran"  # noqa: E731

    def get():
        return sched

    mods = [SimpleNamespace(get_forward_backward_func=get), SimpleNamespace(get_forward_backward_func=get)]
    monkeypatch.setattr(RSM, "validate_result", ORIGINAL)
    dn.install(RSM, *mods)
    dn.install(RSM, *mods)  # idempotent
    assert RSM.validate_result._primus_original is ORIGINAL
    assert getattr(mods[0].get_forward_backward_func, dn._MARK)
    assert mods[0].get_forward_backward_func.__wrapped__ is get
    f = mods[1].get_forward_backward_func()
    assert getattr(f, dn._MARK) and f() == "ran"


def test_inert_with_the_shipped_defaults(monkeypatch):
    """The pretrain module's shipped defaults: the flag is off, the condition is false without importing
    Primus-Turbo, and run_patches leaves validate_result and the schedule getter alone."""
    importlib.import_module("primus.backends.megatron.patches")  # registers every Megatron patch
    mpp = pytest.importorskip("megatron.core.pipeline_parallel")
    params = SimpleNamespace(**parse_yaml(str(_ROOT / "primus/configs/modules/megatron/pre_trainer.yaml")))
    assert params.defer_nonfinite_checks is False
    monkeypatch.setitem(sys.modules, "primus_turbo", None)  # any import of it raises
    extra = {"module_config": SimpleNamespace(params=params)}
    assert PatchRegistry.get(dn._PATCH_KEY).condition(SimpleNamespace(extra=extra)) is False
    validate, getter = RSM.validate_result, mpp.get_forward_backward_func
    applied = run_patches(
        backend="megatron", phase="before_train", extra=extra, enabled_ids=[dn._PATCH_KEY], stop_on_error=True
    )
    assert applied == 0
    assert RSM.validate_result is validate and mpp.get_forward_backward_func is getter


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
