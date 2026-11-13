# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""MegaMoE's mxfp8 experts as gfx1250 runs them, without a GPU: the test-only EP1 override of the patch, the
activation exactly as configured (gfx1250's MXFP8 stages compute nothing else), and the wgrad accumulation
pattern the stages take. The Turbo ops are stubbed; ``GLUActivation`` is Turbo's own class."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import megatron.training
import pytest
import torch
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.core.patches import PatchContext

_ROOT = Path(__file__).resolve().parents[4]
_CALLS = []


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage1(x, topk_idx, topk_weights, w1, group, fuse_wgrad_accum_pattern=None):
    _CALLS.append(("stage1", fuse_wgrad_accum_pattern))
    return x, topk_weights, (), None


def _stage2(l1, dw, handle, state, topk_idx, topk_weights, w2, group, activation=None, fc1_bias=None, fc2_bias=None,
            fuse_wgrad_accum_pattern=None):
    _CALLS.append(("stage2", fuse_wgrad_accum_pattern, activation))
    return l1


@pytest.fixture(scope="module")
def glu():
    spec = importlib.util.find_spec("primus_turbo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("no Primus-Turbo on the path")
    path = Path(list(spec.submodule_search_locations)[0]) / "flydsl/utils/glu_activation.py"
    return _load(path, "glu_activation_under_test")


@pytest.fixture(scope="module")
def mega_moe(glu):
    """``mega_moe.py`` under a private name: the Turbo op modules stubbed, Turbo's ``GLUActivation`` kept."""
    with pytest.MonkeyPatch.context() as mp:
        for name in ("primus_turbo", "primus_turbo.pytorch", "primus_turbo.pytorch.ops", "primus_turbo.pytorch.ops.moe"):
            package = ModuleType(name)
            package.__path__ = []
            mp.setitem(sys.modules, name, package)
        ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe")
        ops.GLUActivation = glu.GLUActivation
        ops.fused_mega_moe_stage1 = ops.fused_mega_moe_stage2 = None
        fp8_ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8")
        fp8_ops.fused_mega_moe_fp8_stage1, fp8_ops.fused_mega_moe_fp8_stage2 = _stage1, _stage2
        mp.setitem(sys.modules, ops.__name__, ops)
        mp.setitem(sys.modules, fp8_ops.__name__, fp8_ops)
        return _load(_ROOT / "primus/backends/megatron/core/extensions/mega_moe.py", "mega_moe_mxfp8_under_test")


@pytest.fixture
def args(monkeypatch):
    args = SimpleNamespace(turbo_mega_moe_precision="mxfp8", turbo_mega_moe_wgrad_accum_fusion=None)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    return args


class _EPGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


def _config(**kw):
    base = dict(num_layers=1, hidden_size=32, num_attention_heads=2, num_moe_experts=4, moe_ffn_hidden_size=8,
                moe_router_topk=2, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                add_bias_linear=False, params_dtype=torch.bfloat16, bf16=True, use_cpu_initialization=True)
    return TransformerConfig(**{**base, **kw})


def _experts(mega_moe, cls, config):
    return getattr(mega_moe, cls)(config, 4, config.hidden_size, config.moe_ffn_hidden_size, _EPGroup())


@pytest.mark.parametrize(
    "ep, allow, want",
    [(1, None, False), (1, False, False), (1, True, True), (2, False, True), (8, True, True), (16, True, False)],
)
def test_ep1_needs_the_test_override(monkeypatch, ep, allow, want):
    from primus.backends.megatron.patches.turbo import mega_moe_patches as P

    monkeypatch.setattr(P, "is_primus_turbo_can_patch", lambda ctx: True)
    params = dict(use_turbo_mega_moe=True, bf16=True, expert_model_parallel_size=ep)
    if allow is not None:
        params["turbo_mega_moe_allow_ep1"] = allow
    ctx = PatchContext(backend="megatron", phase="before_train",
                       extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))})  # fmt: skip
    assert P._is_turbo_mega_moe_can_patch(ctx) is want


@pytest.mark.parametrize("clamp", [None, 10.0])
def test_mxfp8_activation_exact_on_gfx1250(mega_moe, glu, args, monkeypatch, clamp):
    """DeepSeek-V3's plain SwiGLU (no clamp) and Megatron's clamp form (gate from above, up on both sides) reach the
    gfx1250 MXFP8 stages as such; elsewhere, and for the bf16 experts, the kernels' default (None) stays."""
    config = _config(activation_func_clamp_value=clamp)
    monkeypatch.setattr(mega_moe, "_on_gfx1250", lambda: True)
    want = glu.GLUActivation(1.0, 0.0, None, clamp, clamp)
    assert _experts(mega_moe, "MegaMoEFP8Experts", config).activation == want
    assert _experts(mega_moe, "MegaMoEExperts", config).activation is None
    monkeypatch.setattr(mega_moe, "_on_gfx1250", lambda: False)
    assert _experts(mega_moe, "MegaMoEFP8Experts", config).activation is None
    gelu = _config(activation_func=quick_gelu, activation_func_clamp_value=7.0, glu_linear_offset=1.0)
    assert mega_moe.mega_moe_exact_activation(gelu) == mega_moe.mega_moe_activation(gelu)


@pytest.mark.parametrize("stage2_takes", [True, False])
def test_mxfp8_wgrad_accum_pattern(mega_moe, args, monkeypatch, stage2_takes):
    """With wgrad fusion on, the mxfp8 experts hand ``fuse_wgrad_accum_pattern`` to both stages when this
    Primus-Turbo's stages take it, and nothing otherwise (DDP keeps its add)."""
    args.turbo_mega_moe_wgrad_accum_fusion = True
    if not stage2_takes:
        monkeypatch.setattr(mega_moe, "fused_mega_moe_fp8_stage2",
                            lambda l1, dw, handle, state, topk_idx, topk_weights, w2, group, activation=None,
                            fc1_bias=None, fc2_bias=None: _CALLS.append(("stage2", None, activation)) or l1)  # fmt: skip
    experts = _experts(mega_moe, "MegaMoEFP8Experts", _config())
    pattern = "megatron" if stage2_takes else None
    assert experts.wgrad_accum_pattern == pattern
    _CALLS.clear()
    x = torch.zeros(3, 32, dtype=torch.bfloat16)
    experts(x, torch.zeros(3, 2, dtype=torch.int64), torch.ones(3, 2))
    assert [c[1] for c in _CALLS] == [pattern, pattern]
