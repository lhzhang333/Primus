###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The host-sync patches and the embedding wgrad fusion are inert with the shipped module defaults: their
conditions are false without importing Primus-Turbo, run_patches applies none of them, and the Megatron functions
they would wrap stay Megatron's."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import primus.backends.megatron.patches  # noqa: F401  (registers every Megatron patch)
from primus.core.config.yaml_loader import parse_yaml
from primus.core.patches import run_patches
from primus.core.patches.patch_registry import PatchRegistry

_CONFIGS = Path(__file__).resolve().parents[4] / "primus/configs/modules/megatron"
PATCH_IDS = (
    "megatron.moe_alltoall_skip_idle_dtoh_sync",
    "megatron.deferred_nonfinite_checks",
    "megatron.te.cross_entropy_cg_capturable",
    "megatron.embedding_wgrad_fusion",
)
FLAGS = (
    "moe_skip_idle_dtoh_sync",
    "defer_nonfinite_checks",
    "te_cross_entropy_cg_capturable",
    "embedding_wgrad_fusion",
)


def _default_params() -> SimpleNamespace:
    params = {}
    for name in ("trainer_base.yaml", "primus_megatron_module.yaml", "primus_turbo.yaml"):
        params.update(parse_yaml(str(_CONFIGS / name)))
    return SimpleNamespace(**params)


def _ctx(params):
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=params)})


def test_flags_ship_off():
    params = _default_params()
    assert all(getattr(params, f) is False for f in FLAGS)


def test_conditions_false_by_default_without_primus_turbo(monkeypatch):
    monkeypatch.setitem(sys.modules, "primus_turbo", None)  # any import of it would raise
    ctx = _ctx(_default_params())
    for pid in PATCH_IDS:
        patch = PatchRegistry.get(pid)
        assert patch is not None, pid
        assert patch.condition(ctx) is False, pid


def test_run_patches_leaves_megatron_untouched():
    rsm = pytest.importorskip("megatron.core.rerun_state_machine")
    td = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")
    mpp = pytest.importorskip("megatron.core.pipeline_parallel")
    layers = pytest.importorskip("megatron.core.tensor_parallel.layers")
    targets = {
        "validate_result": (rsm.RerunStateMachine, "validate_result"),
        "dtoh": (td.MoEAlltoAllTokenDispatcher, "_maybe_dtoh_and_synchronize"),
        "schedule getter": (mpp, "get_forward_backward_func"),
        "embedding forward": (layers.VocabParallelEmbedding, "forward"),
    }
    before = {k: getattr(owner, name) for k, (owner, name) in targets.items()}
    applied = run_patches(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=_default_params())},
        enabled_ids=list(PATCH_IDS),
        stop_on_error=True,
    )
    assert applied == 0
    assert all(getattr(owner, name) is before[k] for k, (owner, name) in targets.items())
