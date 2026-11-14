###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MXFP8 patches are inert by default: every condition, evaluated the way the patch runner does for every
Megatron job, is False on the shipped defaults, raises nothing and imports neither Primus-Turbo nor
TransformerEngine (they are made unimportable here). Runs without a GPU."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.core.config.yaml_loader import parse_yaml
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

# patch id -> the module that registers it
_PATCHES = {
    "megatron.mlp.turbo_dense_mlp_mxfp8": "primus.backends.megatron.patches.turbo.dense_mlp_mxfp8_patches",
    "megatron.te.shared_dummy_wgrad": "primus.backends.megatron.patches.te_patches.shared_dummy_wgrad_patches",
    "megatron.optimizer.precision_aware_turbo_mxfp8": "primus.backends.megatron.patches.optimizer_patches",
    "megatron.moe.permute_fusion": "primus.backends.megatron.patches.moe_patches.permute_fusion_patches",
}

_EXTENSION = "primus.backends.megatron.core.extensions.primus_turbo"
_SWITCH_ENVS = ("PRIMUS_TURBO_FUSED_DENSE_MLP", "PRIMUS_TURBO_FUSED_GEMM")


def _ctx(params) -> PatchContext:
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


_CONFIGS = Path(__file__).resolve().parents[4] / "primus/configs"


def _default_params(**overrides) -> SimpleNamespace:
    """The shipped defaults: the Megatron module configs and the language-model defaults (e.g.
    gradient_accumulation_fusion: true), as a job without any of these switches gets them."""
    params = {}
    for name in (
        "models/megatron/language_model.yaml",
        "modules/megatron/trainer_base.yaml",
        "modules/megatron/primus_megatron_module.yaml",
        "modules/megatron/primus_turbo.yaml",
    ):
        params.update(parse_yaml(str(_CONFIGS / name)))
    params.update(overrides)
    return SimpleNamespace(**params)


@pytest.fixture
def no_turbo(monkeypatch):
    """Primus-Turbo, TransformerEngine and the Primus-Turbo extension unimportable; no switch in the environment."""
    for name in ("primus_turbo", "transformer_engine", _EXTENSION):
        monkeypatch.setitem(sys.modules, name, None)
    for name in _SWITCH_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("enable_primus_turbo", [False, True])
@pytest.mark.parametrize("patch_id", sorted(_PATCHES))
def test_inert_by_default(no_turbo, patch_id, enable_primus_turbo):
    importlib.import_module(_PATCHES[patch_id])
    patch = PatchRegistry.get(patch_id)
    assert patch is not None, f"{patch_id} is not registered"
    assert not patch.applies_to(_ctx(_default_params(enable_primus_turbo=enable_primus_turbo)))
