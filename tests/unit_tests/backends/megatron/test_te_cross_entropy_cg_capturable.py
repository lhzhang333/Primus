###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron.te.cross_entropy_cg_capturable on CPU: Megatron's te_parallel_cross_entropy (or Primus' overwrite_input
wrapper) called with is_cg_capturable=True; its order with the deferred cross entropy, which must keep its own path;
and the TE assumption that makes the flag bitwise."""

import inspect
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import primus.backends.megatron.patches.te_patches.cross_entropy_deferred_patches as deferred
import primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches as mod
from primus.core.patches.patch_registry import PatchRegistry


def _ctx(**args):
    defaults = dict(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
    defaults.update(args)
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**defaults))})


@pytest.mark.parametrize(
    "args,want",
    [
        ({}, False),
        (dict(te_cross_entropy_cg_capturable=True), True),
        (dict(te_cross_entropy_cg_capturable=True, cross_entropy_loss_fusion=False), False),
        (dict(te_cross_entropy_cg_capturable=True, cross_entropy_fusion_impl="native"), False),
    ],
)
def test_yaml_flag_and_te_fused_cross_entropy_only(args, want):
    assert mod._cg_capturable_enabled(_ctx(**args)) is want


def test_wrapper_passes_is_cg_capturable_true():
    calls = []

    def original(logits, labels, tp_group, is_cg_capturable=False):
        calls.append((logits, labels, tp_group, is_cg_capturable))
        return "loss"

    wrapped = mod.make_cg_capturable_cross_entropy(original)
    logits, labels, group = torch.zeros(4, 2, 8), torch.zeros(4, 2, dtype=torch.long), object()
    assert wrapped(logits, labels, group) == "loss"
    assert wrapped(logits, labels, group, False) == "loss"
    assert calls == [(logits, labels, group, True)] * 2
    assert wrapped._primus_cg_capturable and wrapped.__wrapped__ is original
    assert mod.make_cg_capturable_cross_entropy(wrapped) is wrapped


def test_wraps_the_overwrite_input_wrapper():
    seen = []

    def fake_pce(*args, **kwargs):
        seen.append((args, kwargs))
        return "te"

    overwrite = mod.make_overwrite_input_cross_entropy(lambda *a: "original", fake_pce)
    wrapped = mod.make_cg_capturable_cross_entropy(overwrite)
    logits, labels, group = torch.zeros(4, 2, 8), torch.zeros(4, 2, dtype=torch.long), object()
    assert wrapped(logits, labels, group, False) == "te"
    ((args, kwargs),) = seen
    assert args == (logits, labels, 0.0, False, group, -100, True) and kwargs == {"overwrite_input": True}
    assert wrapped._primus_overwrite_input  # the marker the overwrite patch checks survives


def test_install_sets_both_modules():
    def original(logits, labels, tp_group, is_cg_capturable=False):
        return is_cg_capturable

    te_ext = SimpleNamespace(te_parallel_cross_entropy=original)
    lm = SimpleNamespace(te_parallel_cross_entropy=original)
    assert mod.install_cg_capturable(te_ext, lm) == "wrapped"
    assert te_ext.te_parallel_cross_entropy is lm.te_parallel_cross_entropy
    assert lm.te_parallel_cross_entropy(None, None, None) is True
    empty = SimpleNamespace(te_parallel_cross_entropy=None)
    assert mod.install_cg_capturable(SimpleNamespace(), empty) == "missing"


def test_leaves_an_installed_deferred_cross_entropy_alone():
    installed = deferred.make_deferred_cross_entropy(lambda *a: "TE")
    te_ext = SimpleNamespace(te_parallel_cross_entropy=installed)
    lm = SimpleNamespace(te_parallel_cross_entropy=installed)
    assert mod.install_cg_capturable(te_ext, lm) == "deferred"
    assert lm.te_parallel_cross_entropy is installed and te_ext.te_parallel_cross_entropy is installed


@pytest.fixture
def megatron_ce(monkeypatch):
    """Megatron's two homes of te_parallel_cross_entropy as plain modules holding a recording TE stand-in, so the
    real patch handlers run on CPU."""
    calls = []

    def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
        calls.append(("TE", is_cg_capturable))
        return "TE"

    te_ext = types.ModuleType("megatron.core.extensions.transformer_engine")
    lm_pkg = types.ModuleType("megatron.core.models.common.language_module")
    lm = types.ModuleType("megatron.core.models.common.language_module.language_module")
    te_ext.te_parallel_cross_entropy = lm.te_parallel_cross_entropy = te_parallel_cross_entropy
    lm_pkg.language_module = lm
    for m in (te_ext, lm_pkg, lm):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    for parent, child in (("megatron.core.extensions", te_ext), ("megatron.core.models.common", lm_pkg)):
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child.__name__.rsplit(".", 1)[1], child, raising=False)
    return SimpleNamespace(te_ext=te_ext, lm=lm, calls=calls)


def test_with_deferred_cross_entropy_the_deferred_path_stays(megatron_ce, monkeypatch):
    """The three cross-entropy patches in the order run_patches applies them: overwrite_input, cg_capturable,
    deferred. Eligible calls take the deferred path; the calls it hands back reach TE with is_cg_capturable=True.
    """
    ids = (
        "megatron.te.cross_entropy_overwrite_input",
        "megatron.te.cross_entropy_cg_capturable",
        "megatron.te.cross_entropy_deferred",
    )
    patches = sorted((PatchRegistry.get(i) for i in ids), key=lambda p: p.priority)
    assert [p.id for p in patches] == list(ids)
    monkeypatch.setattr(deferred, "eligible", lambda logits, labels, tp_group, cg: not cg)
    monkeypatch.setattr(deferred, "DeferredCrossEntropy", SimpleNamespace(apply=lambda *a: "deferred"))
    ctx = _ctx(
        te_ce_overwrite_input=True,
        te_cross_entropy_cg_capturable=True,
        deferred_cross_entropy=True,
        use_turbo_cross_entropy=False,
    )
    for p in patches:
        assert p.condition(ctx)
        p.apply(ctx)
    ce = megatron_ce.lm.te_parallel_cross_entropy
    assert megatron_ce.te_ext.te_parallel_cross_entropy is ce
    assert ce(None, None, None, False) == "deferred" and megatron_ce.calls == []
    assert ce(None, None, None, True) == "TE"  # ineligible (CUDA-graph capture): TE, without its host sync
    assert megatron_ce.calls == [("TE", True)]


def test_te_backward_only_skips_the_scalar_one_check():
    """The assumption behind bitwise: TE's backward uses is_cg_capturable only to skip the scalar-1 shortcut,
    and Megatron's per-token loss gradient (shape [s, b]) is never equal to the scalar 1."""
    tce = pytest.importorskip("transformer_engine.pytorch.triton.cross_entropy")
    src = " ".join(inspect.getsource(tce.cross_entropy_backward).split())
    assert (
        "if not is_cg_capturable and torch.equal( grad_output, torch.tensor(1.0, device=grad_output.device) )"
        in src
    )
    assert src.count("is_cg_capturable") == 2  # the signature and that guard, nothing else
    assert not torch.equal(torch.ones(4096, 2), torch.tensor(1.0))
