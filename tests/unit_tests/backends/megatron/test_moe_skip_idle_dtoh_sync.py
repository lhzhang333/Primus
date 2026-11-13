# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU coverage of megatron.moe_alltoall_skip_idle_dtoh_sync: MoEAlltoAllTokenDispatcher's D2H event wait is
skipped only for a dispatch whose D2H point copied nothing to the host -- on top of Megatron's own
``_maybe_dtoh_and_synchronize`` and of the one Primus installs for PrimusTurbo grouped GEMM (tokens_per_expert
kept on the device). Device tensors are stand-ins (``is_cuda``), streams and events are recorders (no GPU).
"""

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

td = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")

_ROOT = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


skip = _load(
    "skip_idle_dtoh_under_test",
    "primus/backends/megatron/patches/moe_patches/moe_skip_idle_dtoh_sync_patches.py",
)
CLS = td.MoEAlltoAllTokenDispatcher
UPSTREAM = CLS._maybe_dtoh_and_synchronize  # Megatron's (no Primus patch has run in this process)
POINTS = ("before_permutation_1", "before_ep_alltoall", "before_permutation_2", "before_finish")


class Dev:
    """A device tensor stand-in."""

    is_cuda = True

    def __init__(self, tag):
        self.tag = tag


class Host:
    """A host copy (CPU tensor / numpy) of a device tensor."""

    is_cuda = False

    def __init__(self, src):
        self.src = src


def fake_move(tensor, as_numpy=False, record_stream=False):
    return Host(tensor) if getattr(tensor, "is_cuda", False) is True else tensor


class Event:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


class Stream:
    def __init__(self):
        self.event = Event()

    def wait_stream(self, other):
        pass

    def record_event(self):
        return self.event


@pytest.fixture(autouse=True)
def cpu_cuda(monkeypatch):
    main = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(td, "maybe_move_tensor_to_cpu", fake_move)
    # every test starts from Megatron's method; the Primus patches below replace it on the class
    monkeypatch.setattr(CLS, "_maybe_dtoh_and_synchronize", UPSTREAM)


@pytest.fixture
def primus_dtoh(monkeypatch):
    """The method megatron.moe_alltoall_dtoh_turbo_grouped_gemm installs (the gate runs moe_patches' copy)."""
    mod = _load(
        "dtoh_turbo_under_test", "primus/backends/megatron/patches/moe_patches/moe_alltoall_dtoh_patches.py"
    )
    mod.patch_moe_alltoall_dtoh(SimpleNamespace())
    method = CLS._maybe_dtoh_and_synchronize
    assert method is not UPSTREAM
    return method


def dispatcher(*, ep_tp=False, permute_fusion=True, dropping=False, drop_and_pad=False, local_experts=32):
    d = object.__new__(CLS)
    d.drop_and_pad = drop_and_pad
    d.cuda_dtoh_point = "before_permutation_1"
    d.cuda_dtoh_stream = Stream()
    d.num_local_experts = local_experts
    d.config = SimpleNamespace(moe_permute_fusion=permute_fusion)
    d.input_splits = Dev("input_splits") if ep_tp else None
    d.output_splits = Dev("output_splits") if ep_tp else None
    d.output_splits_tp = Dev("output_splits_tp") if ep_tp else None
    d.num_out_tokens = Dev("num_out_tokens") if dropping else 8192 * 8
    d.num_global_tokens_per_local_expert = Dev("ngtple")
    # Megatron's preprocess: the earliest point that needs host values
    if dropping:
        d.cuda_sync_point = "before_permutation_1"
    elif ep_tp:
        d.cuda_sync_point = "before_ep_alltoall"
    elif local_experts > 1 and not permute_fusion:
        d.cuda_sync_point = "before_permutation_2"
    else:
        d.cuda_sync_point = "before_finish"
    return d


def run_dispatch(d, method):
    """The four calls of one dispatch (dispatch_preprocess, token_dispatch, dispatch_postprocess x2)."""
    tpe = Dev("tokens_per_expert")
    for p in POINTS:
        tpe = method(d, p, tpe)
    return tpe, d.cuda_dtoh_stream.event.waits


def _desc(x):
    if isinstance(x, Dev):
        return ("dev", x.tag)
    if isinstance(x, Host):
        return ("host", _desc(x.src))
    return x


STATE = (
    "input_splits",
    "output_splits",
    "output_splits_tp",
    "num_out_tokens",
    "num_global_tokens_per_local_expert",
)


def state(d):
    return {a: _desc(getattr(d, a)) for a in STATE}


def test_env_switch_default_off(monkeypatch):
    monkeypatch.delenv("PRIMUS_MOE_SKIP_IDLE_DTOH_SYNC", raising=False)
    assert not skip.requested()
    for v in ("0", "", "false"):
        monkeypatch.setenv("PRIMUS_MOE_SKIP_IDLE_DTOH_SYNC", v)
        assert not skip.requested()
    monkeypatch.setenv("PRIMUS_MOE_SKIP_IDLE_DTOH_SYNC", "1")
    assert skip.requested()


def test_ep1_turbo_grouped_gemm_skips_the_wait(primus_dtoh):
    """The gate's case: EP = TP = 1, dropless, fused permute, tokens_per_expert on the device: no copies."""
    base, wrapped = dispatcher(), dispatcher()
    tpe0, waits0 = run_dispatch(base, primus_dtoh)
    tpe1, waits1 = run_dispatch(wrapped, skip.wrap_maybe_dtoh_and_synchronize(primus_dtoh))
    assert waits0 == 1 and waits1 == 0
    assert isinstance(tpe1, Dev) and tpe1.tag == tpe0.tag == "tokens_per_expert"
    assert state(base) == state(wrapped)


def test_megatron_copy_of_tokens_per_expert_keeps_the_wait():
    """Without the Primus D2H patch tokens_per_expert goes to the host: the wait must stay."""
    tpe, waits = run_dispatch(dispatcher(), skip.wrap_maybe_dtoh_and_synchronize(UPSTREAM))
    assert waits == 1 and isinstance(tpe, Host)


@pytest.mark.parametrize(
    "kw,point",
    [
        (dict(ep_tp=True), "before_ep_alltoall"),  # input / output splits to the host
        (dict(permute_fusion=False), "before_permutation_2"),  # unfused sort_chunks needs the host counts
        (dict(dropping=True), "before_permutation_1"),  # token dropping: num_out_tokens to the host
    ],
)
def test_host_copies_keep_the_wait(primus_dtoh, kw, point):
    base, wrapped = dispatcher(**kw), dispatcher(**kw)
    assert base.cuda_sync_point == point
    tpe0, waits0 = run_dispatch(base, primus_dtoh)
    tpe1, waits1 = run_dispatch(wrapped, skip.wrap_maybe_dtoh_and_synchronize(primus_dtoh))
    assert waits0 == waits1 == 1
    assert _desc(tpe0) == _desc(tpe1)
    assert state(base) == state(wrapped)
    assert any(isinstance(v, tuple) and v[0] == "host" for v in state(wrapped).values())


def test_drop_and_pad_passes_through(primus_dtoh):
    d = dispatcher(drop_and_pad=True)
    _, waits = run_dispatch(d, skip.wrap_maybe_dtoh_and_synchronize(primus_dtoh))
    assert waits == 0 and not hasattr(d, "_primus_dtoh_host_copies")


def test_flag_is_per_dispatch(primus_dtoh):
    """A dispatch that copies waits even if the previous one did not, and the other way round."""
    method = skip.wrap_maybe_dtoh_and_synchronize(primus_dtoh)
    d = dispatcher()
    assert run_dispatch(d, method)[1] == 0
    d.num_global_tokens_per_local_expert = Dev("ngtple")
    d.config.moe_permute_fusion = False  # this dispatch copies the per-expert counts
    assert run_dispatch(d, method)[1] == 1
    d.config.moe_permute_fusion = True
    d.cuda_dtoh_stream = Stream()
    assert run_dispatch(d, method)[1] == 0


def test_sync_point_without_a_dtoh_record_waits(primus_dtoh):
    """No flag yet (sync point reached without the D2H point of this object): Megatron's behaviour."""
    d = dispatcher()
    d.d2h_event = d.cuda_dtoh_stream.record_event()
    skip.wrap_maybe_dtoh_and_synchronize(primus_dtoh)(d, "before_finish", Dev("t"))
    assert d.cuda_dtoh_stream.event.waits == 1


def test_patch_wraps_the_installed_method_once(primus_dtoh):
    skip.patch_moe_skip_idle_dtoh_sync(SimpleNamespace())
    m = CLS._maybe_dtoh_and_synchronize
    assert m._primus_skip_idle_dtoh_sync and m.__wrapped__ is primus_dtoh
    assert skip.wrap_maybe_dtoh_and_synchronize(m) is m
    d = dispatcher()
    _, waits = run_dispatch(d, CLS._maybe_dtoh_and_synchronize)
    assert waits == 0 and m.skipped == 1
