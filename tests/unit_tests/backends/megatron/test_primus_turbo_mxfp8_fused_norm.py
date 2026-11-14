# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""PRIMUS_MXFP8_FUSED_NORM: RMSNorm outputs handed to Primus-Turbo MXFP8 linears already quantized
(Primus-Turbo's ``ops.normalization.rmsnorm_mxfp8``).

Site ``up``: the LN-fused linear (MLA's q_up / kv_up) norms and quantizes in one call. Site ``input``: the
MLA input RMSNorm returns a NaN carrier with the pair attached, which both down projections take. Both are
checked against the unfused path through the real ``forward_internal`` / ``forward`` code with stand-in
modules: outputs, input and norm-weight gradients, and the weight gradients accumulated into ``main_grad``."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus.backends.megatron.core.extensions.primus_turbo as ext  # noqa: E402  isort:skip
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboLayerNormColumnParallelLinear,
    PrimusTurboLinear,
    PrimusTurboLowPrecisionGlobalStateManager,
    PrimusTurboQuantConfig,
    PrimusTurboRMSNorm,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    _FUSED_MXFP8_ACTIVATION_ATTR,
    _mxfp8_gemm_input,
    _take_fused_mxfp8_activation,
    attach_fused_mxfp8_activation,
    link_mxfp8_norm_readers,
)

HAVE_OP = ext._turbo_rmsnorm_mxfp8() is not None


@pytest.fixture(autouse=True)
def _turbo_mxfp8_state(monkeypatch):
    manager = PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", True)
    monkeypatch.setattr(
        manager,
        "PRIMUS_TURBO_QUANT_CONFIG",
        PrimusTurboQuantConfig(
            format=Format.E4M3,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=32,
            scale_dtype=ScaleDtype.E8M0,
        ),
    )
    monkeypatch.setattr(ext, "_use_split_wgrad_op", lambda: False)
    monkeypatch.setattr(ext, "get_num_microbatches", lambda: 2)  # the cached-weight + bridge path


def test_pair_taken_once_per_reader():
    carrier, pair = torch.empty(2), object()
    attach_fused_mxfp8_activation(carrier, pair, 2)
    assert _take_fused_mxfp8_activation(carrier) is pair
    assert hasattr(carrier, _FUSED_MXFP8_ACTIVATION_ATTR)
    assert _take_fused_mxfp8_activation(carrier) is pair
    assert not hasattr(carrier, _FUSED_MXFP8_ACTIVATION_ATTR)
    assert _take_fused_mxfp8_activation(carrier) is None


def test_pair_outside_mxfp8_raises(monkeypatch):
    monkeypatch.setattr(PrimusTurboLowPrecisionGlobalStateManager, "PRIMUS_TURBO_FP8_ENABLED", False)
    carrier = torch.empty(2)
    attach_fused_mxfp8_activation(carrier, object(), 1)
    with pytest.raises(RuntimeError, match="does not run Turbo MXFP8"):
        _take_fused_mxfp8_activation(carrier)


def test_layout_mismatch_raises(monkeypatch):
    monkeypatch.setattr(ext, "_mxfp8_activation_recipes", lambda w: (ScalingRecipe(), ScalingRecipe()))
    packed = ScalingRecipe(shuffle_scale=True)
    pair = SimpleNamespace(
        data=SimpleNamespace(scaling_recipe=packed), data_t=SimpleNamespace(scaling_recipe=packed)
    )
    with pytest.raises(RuntimeError, match="layout"):
        _mxfp8_gemm_input(None, pair, None)
    assert _mxfp8_gemm_input("inp", None, None) == "inp"


class _Stub:
    """Stand-in for a Primus-Turbo module: the attributes forward_internal / forward read (weakref-able)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(n, k, device="cuda", generator=g).bfloat16() / math.sqrt(k))
    w.main_grad = torch.zeros(n, k, dtype=torch.float32, device="cuda")
    w.grad_added_to_main_grad = False
    return w


def _linear(n, k, seed, **kw):
    return _Stub(
        _parameters={"weight": _weight(n, k, seed)},
        use_bias=False,
        offload=False,
        config=SimpleNamespace(gradient_accumulation_fusion=False, normalization="RMSNorm"),
        disable_parameter_transpose_cache=False,
        quantized_weight_buffer=None,
        quantized_weight_t_buffer=None,
        **kw,
    )


def _snr(a, ref):
    a, ref = a.double(), ref.double()
    err = (a - ref).pow(2).sum().item()
    return math.inf if err == 0 else 10 * math.log10(ref.pow(2).sum().item() / err)


def _gamma(h, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.nn.Parameter((1 + 0.1 * torch.randn(h, device="cuda", generator=g)).bfloat16())


def _run_up(monkeypatch, sites, x_full, h, n):
    """kv_up-like: LN-fused linear over a strided row view (the first h of each row of x_full)."""
    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", sites)
    lin = _linear(
        n, h, 1, layer_norm_weight=_gamma(h, 2), zero_centered_gamma=False, eps=1e-5, layer_norm_bias=None
    )
    x_full = x_full.detach().clone().requires_grad_(True)
    x = x_full[..., :h]
    out, _ = PrimusTurboLayerNormColumnParallelLinear.forward_internal(lin, x, True)
    g = torch.Generator(device="cuda").manual_seed(3)
    out.backward(torch.randn(out.shape, device="cuda", generator=g).bfloat16())
    w = lin._parameters["weight"]
    return out, x_full.grad, lin.layer_norm_weight.grad, w.main_grad


@pytest.mark.skipif(not HAVE_OP, reason="needs Primus-Turbo's rmsnorm_mxfp8")
def test_up_site_matches_unfused(monkeypatch):
    s, b, h, n = 256, 2, 512, 1024
    g = torch.Generator(device="cuda").manual_seed(0)
    x_full = torch.randn(s, b, h + 64, device="cuda", generator=g).bfloat16()
    calls = []
    op = ext._turbo_rmsnorm_mxfp8()
    monkeypatch.setattr(ext, "_turbo_rmsnorm_mxfp8", lambda: lambda *a, **k: calls.append(1) or op(*a, **k))
    fused = _run_up(monkeypatch, "up", x_full, h, n)
    assert calls, "the fused producer did not run"
    ref = _run_up(monkeypatch, "0", x_full, h, n)
    for name, a, r in zip(("out", "dx", "dgamma", "dW"), fused, ref):  # Turbo's rstd / y / codes: bit for bit
        assert torch.isfinite(a).all(), name
        assert torch.equal(a, r), (name, _snr(a, r))


def _run_input(monkeypatch, sites, x, h, ns):
    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", sites)
    readers = [_linear(n, h, 10 + i) for i, n in enumerate(ns)]
    norm = _Stub(weight=_gamma(h, 4), zero_centered_gamma=False, eps=1e-5)
    norm._primus_mxfp8_readers = tuple(weakref.ref(r) for r in readers)
    x = x.detach().clone().requires_grad_(True)
    y = PrimusTurboRMSNorm.forward(norm, x)
    outs = [PrimusTurboLinear.forward_internal(r, y, True)[0] for r in readers]
    assert not hasattr(y, _FUSED_MXFP8_ACTIVATION_ATTR)
    g = torch.Generator(device="cuda").manual_seed(5)
    loss = sum((o.float() * torch.randn(o.shape, device="cuda", generator=g)).sum() for o in outs)
    loss.backward()
    return y, outs, x.grad, norm.weight.grad, [r._parameters["weight"].main_grad for r in readers]


@pytest.mark.skipif(not HAVE_OP, reason="needs Primus-Turbo's rmsnorm_mxfp8")
def test_input_site_matches_unfused(monkeypatch):
    s, b, h, ns = 256, 2, 1024, (768, 576)  # q_down / kv_down-like
    # stand-ins, not Turbo linears
    monkeypatch.setattr(ext, "can_consume_fused_mxfp8_activation", lambda m: True)
    g = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn(s, b, h, device="cuda", generator=g).bfloat16()
    y, *fused = _run_input(monkeypatch, "input", x, h, ns)
    assert y.stride() == (0, 0, 0) and torch.isnan(y[0, 0, 0])  # the never-read carrier
    y_ref, *ref = _run_input(monkeypatch, "0", x, h, ns)
    assert torch.isfinite(y_ref).all()
    names = ("out", "dx", "dgamma", "dW")
    for name, a, r in zip(names, fused, ref):
        for ai, ri in zip(a, r) if isinstance(a, list) else ((a, r),):
            assert torch.isfinite(ai).all(), name
            assert torch.equal(ai, ri), (name, _snr(ai, ri))


@pytest.mark.skipif(not HAVE_OP, reason="needs Primus-Turbo's rmsnorm_mxfp8")
def test_input_site_falls_back_when_a_reader_cannot_take_it(monkeypatch):
    monkeypatch.setattr(ext, "can_consume_fused_mxfp8_activation", lambda m: m.ok)
    monkeypatch.setenv("PRIMUS_MXFP8_FUSED_NORM", "1")
    readers = [_linear(64, 128, 20, ok=True), _linear(64, 128, 21, ok=False)]
    norm = _Stub(weight=_gamma(128, 7), zero_centered_gamma=False, eps=1e-5)
    norm._primus_mxfp8_readers = tuple(weakref.ref(r) for r in readers)
    y = PrimusTurboRMSNorm.forward(norm, torch.randn(32, 2, 128, device="cuda").bfloat16())
    assert y.is_contiguous() and torch.isfinite(y).all() and not hasattr(y, _FUSED_MXFP8_ACTIVATION_ATTR)


def _bare(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def test_link_only_mla_down_projections():
    q, kv = _bare(PrimusTurboLinear), _bare(PrimusTurboLinear)
    attn = SimpleNamespace(
        config=SimpleNamespace(q_lora_rank=1536),
        linear_q_down_proj=q,
        linear_kv_down_proj=kv,
        core_attention=None,
    )
    layer = SimpleNamespace(input_layernorm=PrimusTurboRMSNorm(64).cuda(), self_attention=attn)
    assert link_mxfp8_norm_readers(layer)
    assert [r() for r in layer.input_layernorm._primus_mxfp8_readers] == [q, kv]
    dsa = SimpleNamespace(requires_dsa_inputs=True)
    for change in (
        dict(recompute_input_layernorm=True),
        dict(offload_attn_norm=True),
        dict(self_attention=SimpleNamespace(**{**vars(attn), "linear_kv_down_proj": torch.nn.Linear(2, 2)})),
        dict(self_attention=SimpleNamespace(**{**vars(attn), "config": SimpleNamespace(q_lora_rank=None)})),
        dict(self_attention=SimpleNamespace(**{**vars(attn), "offload_qkv_linear": True})),
        dict(self_attention=SimpleNamespace(**{**vars(attn), "core_attention": dsa})),
        dict(input_layernorm=torch.nn.LayerNorm(64)),
    ):  # fmt: skip
        other = SimpleNamespace(**{**vars(layer), "input_layernorm": PrimusTurboRMSNorm(64).cuda(), **change})
        assert not link_mxfp8_norm_readers(other), change
        assert not hasattr(other.input_layernorm, "_primus_mxfp8_readers")
