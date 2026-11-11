# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MXFP8 with BF16 primary weights keeps the precision-aware optimizer's TE-managed master weights.

Upstream's ``use_precision_aware_optimizer_no_fp8_or_ds_fp8`` turns False for every FP8 recipe but delayed
scaling (MCore then owns FP32 master params, meant for FP8 primary weights). Primus-Turbo quantizes MXFP8 inside its
autograd functions and the parameters stay BF16, so the patch keeps the BF16 run's optimizer path.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig

from primus.backends.megatron.patches import optimizer_patches


@pytest.fixture
def restore_post_init():
    original = OptimizerConfig.__post_init__
    yield
    OptimizerConfig.__post_init__ = original


def _config(**kw):
    base = dict(optimizer="adam", use_distributed_optimizer=True, use_precision_aware_optimizer=True,
                main_params_dtype=torch.float32, exp_avg_dtype=torch.bfloat16, exp_avg_sq_dtype=torch.bfloat16)  # fmt: skip
    base.update(kw)
    return OptimizerConfig(**base)


def test_upstream_switches_master_weights_for_mxfp8(restore_post_init):
    assert _config(fp8_recipe="delayed").use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert _config(fp8_recipe=None).use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not _config(fp8_recipe="mxfp8").use_precision_aware_optimizer_no_fp8_or_ds_fp8


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(fp8_recipe="mxfp8"), True),  # the fix: same optimizer path as without FP8
        (dict(fp8_recipe="delayed"), True),
        (dict(fp8_recipe="blockwise"), False),  # other recipes untouched
        (dict(fp8_recipe="mxfp8", use_precision_aware_optimizer=False, exp_avg_dtype=torch.float32,
              exp_avg_sq_dtype=torch.float32), False),  # fmt: skip
    ],
)
def test_patch_keeps_te_master_weights(restore_post_init, kw, want):
    optimizer_patches.patch_precision_aware_for_turbo_mxfp8(SimpleNamespace())
    assert _config(**kw).use_precision_aware_optimizer_no_fp8_or_ds_fp8 is want


def test_patch_condition_needs_bf16_primary_weights():
    """The registered patch applies only to MXFP8 runs without FP8 primary weights (--fp8-param-gather)."""
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    patch = next(p for p in PatchRegistry._all_patches if p.id == "megatron.optimizer.precision_aware_turbo_mxfp8")

    def cond(**args):
        params = dict(use_precision_aware_optimizer=True, fp8="e4m3", fp8_recipe="mxfp8", fp8_param_gather=False)
        params.update(args)
        ctx = PatchContext(backend="megatron", phase="before_train",
                           extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))})  # fmt: skip
        return bool(patch.condition(ctx))

    assert cond()
    assert not cond(fp8_param_gather=True)
    assert not cond(fp8=None)
    assert not cond(fp8_recipe="tensorwise")
    assert not cond(use_precision_aware_optimizer=False)
