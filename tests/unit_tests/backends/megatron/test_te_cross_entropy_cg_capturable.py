###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU coverage of megatron.te.cross_entropy_cg_capturable: Megatron's te_parallel_cross_entropy (or Primus'
overwrite_input wrapper) called with is_cg_capturable=True, and the TE assumption that makes it bitwise."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches as mod


def _ctx(**args):
    defaults = dict(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
    defaults.update(args)
    return SimpleNamespace(extra={"args": SimpleNamespace(**defaults)})


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv("PRIMUS_TE_CE_CG_CAPTURABLE", raising=False)
    assert not mod.cg_capturable_requested()
    monkeypatch.setenv("PRIMUS_TE_CE_CG_CAPTURABLE", "1")
    assert mod.cg_capturable_requested()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, True),
        ({"cross_entropy_loss_fusion": False}, False),
        ({"cross_entropy_fusion_impl": "native"}, False),
    ],
)
def test_only_for_te_fused_cross_entropy(monkeypatch, args, expected):
    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])
    assert mod._te_fused_cross_entropy(_ctx(**args)) is expected


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
    assert mod.install_cg_capturable(te_ext, lm)
    assert te_ext.te_parallel_cross_entropy is lm.te_parallel_cross_entropy
    assert lm.te_parallel_cross_entropy(None, None, None) is True
    assert not mod.install_cg_capturable(SimpleNamespace(), SimpleNamespace(te_parallel_cross_entropy=None))


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
