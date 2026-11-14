###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron.te.shared_dummy_wgrad (``turbo_shared_dummy_wgrad``): off by default; when set, it points TE's dummy-wgrad
helper and the Primus-Turbo extension's at Primus-Turbo's public flat pool, and only with a Primus-Turbo that has one.
TransformerEngine, Primus-Turbo and the extension are stand-in modules here, so it runs without a GPU."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from primus.backends.megatron.patches.te_patches import shared_dummy_wgrad_patches as P
from primus.core.patches import PatchContext

_TE_BASE = "transformer_engine.pytorch.module.base"
_EXTENSION = "primus.backends.megatron.core.extensions.primus_turbo"


def _ctx(**params):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )


@pytest.mark.parametrize(
    "params, turbo_can_patch, want",
    [
        (dict(enable_primus_turbo=True, gradient_accumulation_fusion=True), True, False),  # off by default
        (dict(turbo_shared_dummy_wgrad=True, gradient_accumulation_fusion=True), True, True),
        (dict(turbo_shared_dummy_wgrad=True, gradient_accumulation_fusion=False), True, False),  # no dummies
        (dict(turbo_shared_dummy_wgrad=True, gradient_accumulation_fusion=True), False, False),  # no Turbo
    ],
)
def test_condition(monkeypatch, params, turbo_can_patch, want):
    asked = []
    monkeypatch.setattr(P, "is_primus_turbo_can_patch", lambda ctx: asked.append(1) or turbo_can_patch)
    assert P._enabled(_ctx(**params)) is want
    if not params.get("turbo_shared_dummy_wgrad"):
        assert not asked, "the flag is read before Primus-Turbo is looked up"


def _turbo_ops(monkeypatch, **attrs):
    ops = ModuleType("primus_turbo.pytorch.ops")
    ops.__dict__.update(attrs)
    for name in ("primus_turbo", "primus_turbo.pytorch"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    return ops


def _helper(shape, dtype, device, zero=False):
    return ("pool", tuple(shape), dtype, device.type, zero)


@pytest.mark.parametrize(
    "attrs, has_pool",
    [
        ({}, False),
        (dict(get_dummy_wgrad=_helper), False),
        (dict(DUMMY_WGRAD_FLAT_POOL=True), False),
        (dict(get_dummy_wgrad=_helper, DUMMY_WGRAD_FLAT_POOL=True), True),
    ],
)
def test_needs_turbos_public_flat_pool(monkeypatch, attrs, has_pool):
    _turbo_ops(monkeypatch, **attrs)
    assert (P.turbo_flat_dummy_wgrad() is _helper) is has_pool


@pytest.fixture
def stand_ins(monkeypatch):
    """TE's base module (and a module that imported its helper by name) and the extension, as stand-ins. The real
    extension, if this process imported it, must come out untouched."""
    import primus.backends.megatron.core.extensions as extensions

    real = getattr(extensions, "primus_turbo", None)
    real_before = getattr(real, "_shared_dummy_wgrad", None)

    def te_helper(shape, dtype, zero=False):
        return "te"

    te_base, user = ModuleType(_TE_BASE), ModuleType("te_user_under_test")
    te_base.get_dummy_wgrad = user.get_dummy_wgrad = te_helper
    ext = ModuleType(_EXTENSION)
    ext._shared_dummy_wgrad = None
    for name in ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.pytorch.module"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for module in (te_base, user, ext):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(P.torch.cuda, "current_device", lambda: 0)
    yield te_base, user, ext, te_helper
    assert getattr(real, "_shared_dummy_wgrad", None) is real_before, "the patch reached the real extension"


def test_patch_points_te_and_the_extension_at_the_pool(monkeypatch, stand_ins):
    te_base, user, ext, _ = stand_ins
    _turbo_ops(monkeypatch, get_dummy_wgrad=_helper, DUMMY_WGRAD_FLAT_POOL=True)
    P.patch_shared_dummy_wgrad(_ctx())
    for get in (te_base.get_dummy_wgrad, user.get_dummy_wgrad, ext._shared_dummy_wgrad):
        assert get([4, 8], "bf16", zero=True) == ("pool", (4, 8), "bf16", "cuda", True)


def test_patch_without_the_pool_changes_nothing(monkeypatch, stand_ins):
    te_base, user, ext, te_helper = stand_ins
    _turbo_ops(monkeypatch, get_dummy_wgrad=_helper)  # per-shape buffers: no flat pool to share
    P.patch_shared_dummy_wgrad(_ctx())
    assert te_base.get_dummy_wgrad is te_helper and user.get_dummy_wgrad is te_helper
    assert ext._shared_dummy_wgrad is None
