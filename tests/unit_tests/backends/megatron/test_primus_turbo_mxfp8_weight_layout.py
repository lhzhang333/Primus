# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cached MXFP8 weights on gfx1250 take the layout Primus-Turbo's ``mx_weight_layout_recipe(grouped)`` reports for
the op kind (dense for a 2-D weight, grouped for the 3-D expert stack); other architectures and recipes, and a
Primus-Turbo without the query, keep Turbo's default recipe."""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as primus_turbo_torch  # noqa: E402  isort:skip

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
RAW = ScalingRecipe(use_2d_block=True)


def test_weight_scaling_recipe_is_turbos_default():
    assert _weight_scaling_recipe(MXFP8.data()) == RAW
    assert _weight_scaling_recipe(BLOCKWISE.data()) == RAW
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


def _turbo_layout_query(monkeypatch, *, present):
    """A Turbo whose dense MX GEMM reads the GEMM layout and whose grouped one does not (yet)."""
    asked = []
    if present:

        def mx_weight_layout_recipe(grouped, shape=None):
            asked.append((grouped, shape))
            return RAW if grouped else GEMM_LAYOUT

        monkeypatch.setattr(
            primus_turbo_torch.ops, "mx_weight_layout_recipe", mx_weight_layout_recipe, raising=False
        )
    else:
        monkeypatch.delattr(primus_turbo_torch.ops, "mx_weight_layout_recipe", raising=False)
    return asked


@pytest.mark.parametrize(
    "gfx1250, config, query, takes_recipe, shape, expected, asked",
    [
        (True, MXFP8, True, True, (64, 64), GEMM_LAYOUT, [(False, (64, 64))]),  # dense linear
        (True, MXFP8, True, True, (2, 64, 64), RAW, [(True, (2, 64, 64))]),  # grouped linear's expert stack
        (True, MXFP8, False, True, (64, 64), None, []),  # a Turbo without the query: its default layout
        (True, MXFP8, True, False, (64, 64), "no scaling_recipe argument", [(False, (64, 64))]),
        (False, MXFP8, True, True, (64, 64), None, []),  # other archs: unchanged
        (True, BLOCKWISE, True, True, (64, 64), None, []),
        (True, TENSORWISE, True, True, (64, 64), None, []),
    ],
)
def test_cached_weight_layout_follows_turbos_query(
    monkeypatch, gfx1250, config, query, takes_recipe, shape, expected, asked
):
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: gfx1250)
    asked_now = _turbo_layout_query(monkeypatch, present=query)
    calls = []
    monkeypatch.setattr(
        primus_turbo_ext, "create_quantized_weight", _recording_create(calls, takes_recipe=takes_recipe)
    )
    weight = torch.zeros(shape, dtype=torch.bfloat16)
    out = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, config, disable_parameter_transpose_cache=False
    )
    assert out == ("rowwise", "colwise")
    assert calls == [expected]
    assert asked_now == asked
