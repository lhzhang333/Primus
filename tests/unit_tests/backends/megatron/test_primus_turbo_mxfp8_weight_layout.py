# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cached MXFP8 weights on gfx1250 are requested in the FlyDSL GEMM layout (E8M0 scales packed, codes 16x16
preshuffled); other architectures and recipes keep Primus-Turbo's default recipe."""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus.backends.megatron.core.extensions.primus_turbo as primus_turbo_ext  # noqa: E402  isort:skip
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboQuantConfig,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    _maybe_create_quantized_weight_buffers,
    _weight_scaling_recipe,
    float8_e4m3,
)

MXFP8 = PrimusTurboQuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)
BLOCKWISE = PrimusTurboQuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)
TENSORWISE = PrimusTurboQuantConfig()
GEMM_LAYOUT = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)


@pytest.mark.parametrize("gfx1250", [True, False])
def test_weight_scaling_recipe(monkeypatch, gfx1250):
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: gfx1250)
    expected_mx = GEMM_LAYOUT if gfx1250 else ScalingRecipe(use_2d_block=True)
    assert _weight_scaling_recipe(MXFP8.data()) == expected_mx
    assert _weight_scaling_recipe(BLOCKWISE.data()) == ScalingRecipe(use_2d_block=True)
    assert _weight_scaling_recipe(TENSORWISE.data()) == ScalingRecipe()


def _recording_create(calls, *, takes_recipe):
    if takes_recipe:

        def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False, scaling_recipe=None):
            calls.append(scaling_recipe)
            return "rowwise", "colwise"

    else:

        def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False):
            calls.append("no scaling_recipe argument")
            return "rowwise", "colwise"

    return create


@pytest.mark.parametrize(
    "gfx1250, config, takes_recipe, expected",
    [
        (True, MXFP8, True, GEMM_LAYOUT),
        (False, MXFP8, True, None),  # other archs: Turbo's own recipe, as before
        (True, BLOCKWISE, True, None),
        (True, TENSORWISE, True, None),
        (True, MXFP8, False, "no scaling_recipe argument"),  # an older Turbo keeps its default layout
    ],
)
def test_create_quantized_weight_gets_the_gemm_layout_on_gfx1250_only(
    monkeypatch, gfx1250, config, takes_recipe, expected
):
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: gfx1250)
    calls = []
    monkeypatch.setattr(
        primus_turbo_ext, "create_quantized_weight", _recording_create(calls, takes_recipe=takes_recipe)
    )
    weight = torch.zeros(64, 64, dtype=torch.bfloat16)
    out = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, config, disable_parameter_transpose_cache=False
    )
    assert out == ("rowwise", "colwise")
    assert calls == [expected]
