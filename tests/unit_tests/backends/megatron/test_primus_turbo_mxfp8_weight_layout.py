###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cached quantized weights take the layout Primus-Turbo's ``create_quantized_weight`` picks: Primus passes no
scaling recipe (on gfx1250 an MXFP8 weight then gets the GEMM layout its MX GEMMs read there; other archs and
recipes keep Turbo's raw layouts)."""

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
    _maybe_create_quantized_weight_buffers,
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


@pytest.mark.parametrize("config", [MXFP8, BLOCKWISE, TENSORWISE], ids=["mxfp8", "blockwise", "tensorwise"])
@pytest.mark.parametrize("shape", [(64, 64), (2, 64, 64)], ids=["dense", "grouped"])
def test_cached_weight_layout_is_turbos_choice(monkeypatch, config, shape):
    """The cache passes no scaling recipe: Primus-Turbo's create_quantized_weight picks the layout (on gfx1250, for an
    MXFP8 weight, the one its GEMMs read there)."""
    calls = []

    def create(weight, dest_dtype, quant_config, need_weight_transpose_cache=False, **kwargs):
        calls.append((tuple(weight.shape), quant_config, need_weight_transpose_cache, kwargs))
        return "rowwise", "colwise"

    monkeypatch.setattr(primus_turbo_ext, "create_quantized_weight", create)
    weight = torch.zeros(shape, dtype=torch.bfloat16)
    out = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, config, disable_parameter_transpose_cache=False
    )
    assert out == ("rowwise", "colwise")
    assert calls == [(shape, config.data(), True, {})]
