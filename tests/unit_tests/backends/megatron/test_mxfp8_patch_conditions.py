###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MXFP8 patches are inert by default: every condition, evaluated the way the patch runner does for every
Megatron job, is False on default args, raises nothing and imports neither Primus-Turbo nor TransformerEngine (they
are made unimportable here). Runs without a GPU."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

# patch id -> the module that registers it
_PATCHES = {
    "megatron.turbo.mxfp8_fused_norm": "primus.backends.megatron.patches.turbo.mxfp8_fused_norm_patches",
}

_EXTENSION = "primus.backends.megatron.core.extensions.primus_turbo"
_SWITCH_ENVS = ("PRIMUS_MXFP8_FUSED_NORM",)


def _ctx(**params) -> PatchContext:
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )


# The flags these conditions read, at their YAML defaults (primus_turbo.yaml, primus_megatron_module.yaml,
# language_model.yaml); enable_primus_turbo is set by most example configs, so both values are covered.
_DEFAULTS = dict(
    fp8=None,
    fp8_recipe="delayed",
    fp4=None,
    use_turbo_gemm=False,
    use_turbo_grouped_gemm=False,
    turbo_fused_gemm=False,
    turbo_fused_grouped_gemm=False,
    use_turbo_mega_moe=False,
    gradient_accumulation_fusion=True,
    use_precision_aware_optimizer=False,
    moe_permute_fusion=False,
    moe_permute_quant_fusion=False,
    moe_backward_permute_quant_fusion=False,
)


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
    assert not patch.applies_to(_ctx(**_DEFAULTS, enable_primus_turbo=enable_primus_turbo))


# ------------------------------------------------------------------------------------------ fused RMSNorm -> MXFP8


@pytest.mark.parametrize(
    "value, sites",
    [("0", set()), ("", set()), ("off", set()), ("1", {"up", "input", "mlp"}), ("all", {"up", "input", "mlp"}),
     ("up", {"up"}), ("input", {"input"}), ("mlp", {"mlp"}), ("up, input", {"up", "input"})],  # fmt: skip
)
def test_fused_norm_sites(monkeypatch, value, sites):
    from primus.backends.megatron.core.extensions.mxfp8_fused_norm_sites import (
        mxfp8_fused_norm_sites,
    )

    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", value)
    assert mxfp8_fused_norm_sites() == sites


def test_fused_norm_unknown_site_raises(monkeypatch):
    from primus.backends.megatron.core.extensions.mxfp8_fused_norm_sites import (
        mxfp8_fused_norm_sites,
    )

    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", "up,pre_mlp")
    with pytest.raises(ValueError, match="pre_mlp"):
        mxfp8_fused_norm_sites()


@pytest.mark.parametrize(
    "value, enable_primus_turbo, turbo_found, want",
    [
        ("input", True, True, True),
        ("mlp", True, True, True),
        ("1", True, True, True),
        ("up", True, True, False),  # needs no link
        ("input", False, True, False),  # Turbo off
        ("input", True, False, False),  # Turbo not installed
        ("input,bogus", True, True, False),  # malformed: links nothing, raises nothing
    ],
)
def test_fused_norm_condition(monkeypatch, value, enable_primus_turbo, turbo_found, want):
    from primus.backends.megatron.patches.turbo import mxfp8_fused_norm_patches as P
    from primus.backends.megatron.patches.turbo import utils

    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", value)
    monkeypatch.setattr(utils.importlib.util, "find_spec", lambda name: object() if turbo_found else None)
    assert P._linked_site_requested(_ctx(enable_primus_turbo=enable_primus_turbo)) is want
