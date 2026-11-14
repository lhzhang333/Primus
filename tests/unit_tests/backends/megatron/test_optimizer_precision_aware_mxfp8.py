###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo MXFP8 on gfx1250 with BF16 primary weights keeps the precision-aware optimizer's TE-managed master
weights.

Upstream's ``use_precision_aware_optimizer_no_fp8_or_ds_fp8`` turns False for every FP8 recipe but delayed
scaling (MCore then owns FP32 master params, meant for FP8 primary weights). Primus-Turbo quantizes MXFP8 inside its
autograd functions and the parameters stay BF16, so the patch keeps the BF16 run's optimizer path; any other run
keeps Megatron's.
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
    base = dict(
        optimizer="adam",
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
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
        (
            dict(
                fp8_recipe="mxfp8",
                use_precision_aware_optimizer=False,
                exp_avg_dtype=torch.float32,
                exp_avg_sq_dtype=torch.float32,
            ),
            False,
        ),
    ],
)
def test_patch_keeps_te_master_weights(restore_post_init, kw, want):
    optimizer_patches.patch_precision_aware_for_turbo_mxfp8(SimpleNamespace())
    assert _config(**kw).use_precision_aware_optimizer_no_fp8_or_ds_fp8 is want


_TURBO_MXFP8 = dict(
    use_precision_aware_optimizer=True,
    fp8="e4m3",
    fp8_recipe="mxfp8",
    fp8_param_gather=False,
    enable_primus_turbo=True,
    use_turbo_gemm=True,
    use_turbo_grouped_gemm=True,
)


@pytest.mark.parametrize(
    "args, gfx1250, want",
    [
        ({}, True, True),
        ({}, False, False),  # gfx950: TE's MXFP8 recipe works there, Megatron's behaviour stays
        (dict(enable_primus_turbo=False), True, False),
        (dict(use_turbo_gemm=False, use_turbo_grouped_gemm=False), True, False),  # nothing Turbo quantizes
        (dict(use_turbo_gemm=False), True, True),  # the experts through Turbo
        (
            dict(
                use_turbo_gemm=False,
                use_turbo_grouped_gemm=False,
                use_turbo_mega_moe=True,
                turbo_mega_moe_precision="mxfp8",
            ),
            True,
            True,
        ),
        (dict(fp8_param_gather=True), True, False),  # FP8 primary weights need MCore's FP32 masters
        (dict(fp8=None), True, False),
        (dict(fp8_recipe="tensorwise"), True, False),
        (dict(use_precision_aware_optimizer=False), True, False),
    ],
)
def test_patch_condition(monkeypatch, args, gfx1250, want):
    """The registered patch applies only to Primus-Turbo MXFP8 on gfx1250 with BF16 primary weights."""
    import primus.core.utils.gpu_arch as gpu_arch
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.optimizer.precision_aware_turbo_mxfp8")
    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: gfx1250)
    params = SimpleNamespace(**{**_TURBO_MXFP8, **args})
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    assert bool(patch.condition(ctx)) is want
