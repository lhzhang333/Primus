# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Guards for Primus-Turbo fused MXFP8 weight-gradient accumulation (wgrad added into a bf16 main_grad, beta=1)."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as primus_turbo_torch  # noqa: E402  isort:skip
from primus_turbo.pytorch.core.low_precision import check_mxfp8_support  # noqa: E402  isort:skip

from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboLowPrecisionGlobalStateManager,
    PrimusTurboQuantConfig,
    PrimusTurboQuantizedTensorPair,
    ScaleDtype,
    ScalingGranularity,
    _bridge_weight_grad,
    _fuse_wgrad_accum_pattern,
    _maybe_create_quantized_weight_buffers,
    float8_e4m3,
)

FUSION = SimpleNamespace(gradient_accumulation_fusion=True)


def _mx_config():
    return PrimusTurboQuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    )


@pytest.fixture(autouse=True)
def _turbo_mxfp8_state(monkeypatch):
    manager = PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", True)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", _mx_config())


def _weight(shape=(8, 8), *, dtype=torch.bfloat16, main_grad_dtype=torch.bfloat16, device="cpu"):
    weight = torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
    weight.main_grad = torch.zeros(shape, dtype=main_grad_dtype, device=device)
    weight.grad_added_to_main_grad = False
    return weight


def _turbo_answers(monkeypatch, answer):
    calls = []

    def supported(grouped, main_grad_dtype):
        calls.append((grouped, main_grad_dtype))
        return answer

    monkeypatch.setattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported", supported, raising=False)
    return calls


@pytest.mark.parametrize("shape", [(8, 8), (2, 8, 8)])
def test_bf16_main_grad_fuses_when_turbo_accumulates(monkeypatch, shape):
    calls = _turbo_answers(monkeypatch, True)
    assert _fuse_wgrad_accum_pattern(FUSION, _weight(shape)) == "megatron"
    assert calls == [(len(shape) == 3, torch.bfloat16)]  # grouped for the 3-D expert stack


def test_turbo_without_beta1_backend_fails_closed(monkeypatch):
    _turbo_answers(monkeypatch, False)
    assert _fuse_wgrad_accum_pattern(FUSION, _weight()) is None


def test_turbo_that_cannot_be_asked_fails_closed(monkeypatch):
    monkeypatch.delattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported", raising=False)
    assert _fuse_wgrad_accum_pattern(FUSION, _weight()) is None


@pytest.mark.parametrize(
    "dtype, main_grad_dtype", [(torch.bfloat16, torch.float32), (torch.float32, torch.float32)]
)
def test_non_bf16_main_grad_fails_closed(monkeypatch, dtype, main_grad_dtype):
    calls = _turbo_answers(monkeypatch, True)
    assert _fuse_wgrad_accum_pattern(FUSION, _weight(dtype=dtype, main_grad_dtype=main_grad_dtype)) is None
    assert calls == []


def test_gradient_accumulation_fusion_flag_still_controls_gate(monkeypatch):
    _turbo_answers(monkeypatch, True)
    assert _fuse_wgrad_accum_pattern(SimpleNamespace(gradient_accumulation_fusion=False), _weight()) is None


def _accumulate_mxfp8_wgrad(weight, inputs, grad_outputs, *, fused, group_lens=None):
    """Two micro-batches through the cached quantized weight and _bridge_weight_grad, as the Turbo linears run them."""
    quant_config = _mx_config()
    quantized_weight, quantized_weight_trans = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, quant_config, disable_parameter_transpose_cache=False
    )
    for x, grad_output in zip(inputs, grad_outputs):
        bridged_x, bridged_weight = _bridge_weight_grad(
            x.detach().clone().requires_grad_(True),
            weight,
            PrimusTurboQuantizedTensorPair(data=quantized_weight, data_t=quantized_weight_trans),
            fuse_wgrad_accum=fused,
        )
        pattern = "megatron" if fused else None
        if group_lens is None:
            output = primus_turbo_torch.ops.gemm_fp8(
                bridged_x,
                bridged_weight,
                trans_b=True,
                config=quant_config.data(),
                fuse_bgrad_accum_pattern=pattern,
            )
        else:
            output = primus_turbo_torch.ops.grouped_gemm_fp8(
                bridged_x,
                bridged_weight,
                group_lens,
                trans_b=True,
                config=quant_config.data(),
                fuse_bgrad_accum_pattern=pattern,
            )
        output.backward(grad_output)
    return weight.main_grad


@pytest.mark.skipif(
    not check_mxfp8_support()[0] or not hasattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported"),
    reason="needs MXFP8 and a Primus-Turbo that reports its MXFP8 wgrad accumulation",
)
@pytest.mark.parametrize("grouped", [False, True])
def test_mxfp8_fused_wgrad_matches_unfused(grouped):
    """The beta=1 GEMM path against the framework's add, on top of an existing gradient, two micro-batches."""
    if not primus_turbo_torch.ops.mx_wgrad_accum_supported(grouped, torch.bfloat16):
        pytest.skip("no MXFP8 beta=1 wgrad into a bf16 main_grad here")
    torch.manual_seed(42)
    if grouped:
        G, m, n, k = 4, 256, 512, 384
        shape, rows = (G, n, k), G * m
        group_lens = torch.full((G,), m, dtype=torch.int64, device="cuda")
    else:
        n, k, rows = 512, 384, 512
        shape, group_lens = (n, k), None
    weight_data = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 16
    inputs = [torch.randn(rows, k, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    grad_outputs = [torch.randn(rows, n, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    initial_main_grad = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 16

    results = {}
    for fused in (True, False):
        weight = torch.nn.Parameter(weight_data.clone())
        weight.main_grad = initial_main_grad.clone()
        weight.grad_added_to_main_grad = False
        results[fused] = _accumulate_mxfp8_wgrad(
            weight, inputs, grad_outputs, fused=fused, group_lens=group_lens
        ).clone()
        assert weight.grad_added_to_main_grad

    assert not torch.equal(results[True], initial_main_grad)
    torch.testing.assert_close(results[True], results[False], rtol=5e-3, atol=5e-3)
