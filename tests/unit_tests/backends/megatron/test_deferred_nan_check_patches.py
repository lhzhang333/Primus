###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: deferred NaN / Inf validations raise upstream's error, in order, at the flush."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
rsm = pytest.importorskip("megatron.core.rerun_state_machine")
ddp_mod = pytest.importorskip("megatron.core.distributed.distributed_data_parallel")
pgb = pytest.importorskip("megatron.core.distributed.param_and_grad_buffer")
mcore_optimizer = pytest.importorskip("megatron.core.optimizer.optimizer")


def _load(name, rel):
    path = Path(__file__).resolve().parents[4] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


deferred = _load(
    "deferred_nan_check_under_test", "primus/backends/megatron/patches/deferred_nan_check_patches.py"
)
fused = _load(
    "fused_grad_check_under_test", "primus/backends/megatron/patches/ddp_grad_check_fused_scaling_patches.py"
)
CTX = PatchContext(backend="megatron", phase="before_train")


@pytest.fixture
def patched():
    owners = [
        (rsm.RerunStateMachine, "validate_result"),
        (ddp_mod.DistributedDataParallel, "finish_grad_sync"),
        (mcore_optimizer.MixedPrecisionOptimizer, "step"),
        (mcore_optimizer.FP32Optimizer, "step"),
        (mcore_optimizer.ChainedOptimizer, "step"),
        (pgb._ParamAndGradBucketGroup, "check_grads"),
    ]
    saved = [(o, n, o.__dict__[n]) for o, n in owners]
    try:
        from megatron.training import training

        saved_eval = training.evaluate
    except ImportError:
        training, saved_eval = None, None
    deferred.patch_deferred_nan_check(CTX)
    yield
    deferred._PENDING.clear()
    for o, n, fn in saved:
        setattr(o, n, fn)
    if training is not None:
        training.evaluate = saved_eval


def _machine():
    return rsm.get_rerun_state_machine()


def _message(fn):
    with pytest.raises(RuntimeError) as err:
        fn()
    return str(err.value).split(": ", 1)[1]


def _check(value, func=torch.isnan, message="found NaN in local forward loss calculation", fatal=True):
    _machine().validate_result(
        result=torch.tensor(value, device="cuda"),
        rejection_func=func,
        message=message,
        tolerance=0.0,
        fatal=fatal,
    )


def test_upstream_reference_raises_at_once():
    assert _message(lambda: _check(float("nan")))


def test_nan_raises_upstream_error_at_flush(patched):
    _check(1.0)
    _check(float("nan"))  # queued, not raised
    assert len(deferred._PENDING) == 2
    got = _message(deferred.flush)
    assert not deferred._PENDING
    want = _message(
        lambda: deferred._ORIGINAL(
            _machine(),
            torch.tensor(float("nan"), device="cuda"),
            torch.isnan,
            "found NaN in local forward loss calculation",
            None,
            0.0,
            True,
        )
    )
    assert got == want


def test_first_queued_failure_wins(patched):
    _check(float("inf"), torch.isinf, "found Inf in local forward loss calculation")
    _check(float("nan"), torch.isnan, "found NaN in local grad norm for bucket #0")
    assert "found Inf in local forward loss calculation" in _message(deferred.flush)


def test_finite_flush_is_quiet_and_clears(patched):
    for v in (1.0, 2.0, 0.0):
        _check(v)
        _check(v, torch.isinf)
    deferred.flush()
    assert not deferred._PENDING


def test_other_checks_run_at_once(patched):
    with pytest.raises(RuntimeError):  # a host result
        _machine().validate_result(
            result=torch.tensor(float("nan")), rejection_func=torch.isnan, message="cpu", fatal=True
        )
    _check(float("nan"), torch.isnan, "non-fatal", fatal=False)  # warns, never queued
    with pytest.raises(RuntimeError):  # another rejection function
        _machine().validate_result(
            result=torch.tensor(5.0, device="cuda"),
            rejection_func=lambda r: bool(r > 1),
            message="custom",
            fatal=True,
        )
    assert not deferred._PENDING


def test_flush_points_are_wrapped(patched):
    assert ddp_mod.DistributedDataParallel.finish_grad_sync._primus_flushes
    assert mcore_optimizer.ChainedOptimizer.step._primus_flushes
    calls = []
    wrapped = deferred._flushing(lambda: calls.append(len(deferred._PENDING)))
    _check(1.0)
    wrapped()
    assert calls == [0] and not deferred._PENDING


def test_fused_grad_check_defers(patched):
    fused.patch_fused_grad_check_scaling(CTX)
    g = torch.full((1000,), 0.5, dtype=torch.bfloat16, device="cuda")
    g[7] = float("nan")
    group = SimpleNamespace(buckets=[SimpleNamespace(grad_data=g, gradient_scaling_factor=0.25)])
    pgb._ParamAndGradBucketGroup.check_grads(group, True, False)  # no host sync, nothing raised
    assert len(deferred._PENDING) == 2
    assert "found NaN in local grad norm for bucket #0" in _message(deferred.flush)
