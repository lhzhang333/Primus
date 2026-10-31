###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests for the fused Triton router: bit for bit against Megatron's routing on TE's fused kernels
(moe_router_fusion) followed by MegaMoE's top-k of the dense probabilities."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_ROOT = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frt = _load(
    "fused_router_triton_under_test", "primus/backends/megatron/core/transformer/moe/fused_router_triton.py"
)
gemm = _load(
    "router_triton_gemm_for_fused_router",
    "primus/backends/megatron/patches/moe_patches/router_triton_gemm_patches.py",
)


def _gfx1250():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250"


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _slots(t, topk, e, mode):
    """PrimusTopKRouter's forced-balancing slots (force_balanced_slots)."""
    if mode == 2:
        tokens = torch.arange(t, device="cuda").unsqueeze(1)
        return (tokens + torch.arange(topk, device="cuda").unsqueeze(0) * (e // topk)) % e
    return torch.arange(t * topk, device="cuda").view(t, topk) % e


def _reference(logits, bias, topk, groups, group_topk, scaling, seq, bsz, mode, coeff, aux):
    """Megatron's routing with moe_router_fusion + PrimusTopKRouter's forced balancing + MegaMoE's top-k."""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    from transformer_engine.pytorch.router import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

    t, e = logits.shape
    probs, rmap = fused_topk_with_score_function(
        logits,
        topk,
        False,
        groups if group_topk else None,
        group_topk if group_topk else None,
        scaling,
        "sigmoid",
        bias,
    )
    tpe = aux_loss = None
    if aux:
        rmap_aux, scores_aux = fused_compute_score_for_moe_aux_loss(logits, topk, "sigmoid")
        tpe = rmap_aux.reshape(seq, -1).sum(dim=0)
        aux_loss = fused_moe_aux_loss(scores_aux.reshape(seq, -1), tpe, seq, e, topk, coeff) / bsz
        probs = MoEAuxLossAutoScaler.apply(probs, aux_loss)
    if mode:
        vals, _ = torch.topk(probs, topk, dim=1)
        new = torch.zeros_like(probs)
        new.scatter_(1, _slots(t, topk, e, mode), vals.to(probs.dtype))
        probs = new
    w, i = probs.topk(topk, dim=-1)
    return w, i, rmap.sum(dim=0).float(), tpe, aux_loss


CASES = [
    # E, groups, group_topk, seq, bsz, mode, logit scale
    (128, 8, 4, 4096, 2, 0, 1.0),
    (128, 8, 4, 4096, 2, 1, 1.0),
    (128, 8, 4, 4096, 2, 2, 1.0),
    (128, 8, 4, 2048, 2, 0, 4.0),
    (32, 8, 4, 4096, 2, 1, 1.0),
    (256, 8, 4, 1024, 4, 0, 1.0),
    (64, 1, 0, 1000, 1, 0, 1.0),
]


@pytest.mark.parametrize("e,groups,group_topk,seq,bsz,mode,scale", CASES)
def test_route_kernels_match_te(e, groups, group_topk, seq, bsz, mode, scale):
    if not _gfx1250():
        pytest.skip("checked against TE's kernels on gfx1250")
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    topk, scaling, coeff, loss_scale = 8, 2.5, 1e-3, 0.125
    t = seq * bsz
    g = torch.Generator(device="cuda").manual_seed(e + mode + seq)
    logits = torch.randn(t, e, device="cuda", generator=g) * scale
    logits[::97, : e // 8] = logits[::97, e // 8 : e // 4]  # equal scores: TE's tie-breaks
    bias = torch.randn(e, device="cuda", generator=g) * 1e-2
    bias[::5] = 0.0
    dw = torch.randn(t, topk, device="cuda", generator=g)

    lg = logits.clone().requires_grad_(True)
    w_ref, i_ref, count_ref, tpe_ref, aux_ref = _reference(
        lg, bias, topk, groups, group_topk, scaling, seq, bsz, mode, coeff, True
    )
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(loss_scale, device="cuda"))
    (w_ref * dw).sum().backward()

    c = frt.te_aux_loss_coeff(e, coeff, topk, seq)
    kwargs = dict(topk=topk, num_groups=groups, group_topk=group_topk, scaling=scaling, bsz=bsz, mode=mode)
    count = torch.zeros(e, device="cuda")
    prob, idx, sel, sig, seq_count, aux = frt.fused_route_forward(
        logits, bias, count=count, aux=True, aux_coeff=c, **kwargs
    )
    assert torch.equal(_bits(prob), _bits(w_ref.detach()))
    assert torch.equal(idx, i_ref)
    assert torch.equal(count, count_ref)
    assert torch.equal(seq_count.reshape(-1).long(), tpe_ref)
    # TE sums the aux loss with atomics: equal up to the summation order
    torch.testing.assert_close(aux, aux_ref.detach().reshape(()), rtol=1e-5, atol=0)
    # run to run: the same bits, the aux loss included
    again = frt.fused_route_forward(
        logits, bias, count=torch.zeros(e, device="cuda"), aux=True, aux_coeff=c, **kwargs
    )
    for a, b in zip(again, (prob, idx, sel, sig, seq_count, aux)):
        assert torch.equal(_bits(a), _bits(b))

    grad_aux = (torch.ones((), device="cuda") * torch.tensor(loss_scale, device="cuda")) / bsz
    hi, lo, dlog = frt.fused_route_backward(
        dw,
        sel,
        sig,
        scaling=scaling,
        bsz=bsz,
        seq_count=seq_count,
        grad_aux=grad_aux,
        aux_coeff=c,
        write_dlogits=True,
    )
    assert torch.equal(_bits(dlog), _bits(lg.grad))
    assert torch.equal(hi, lg.grad.to(torch.bfloat16))
    assert torch.equal(lo, (lg.grad - hi.float()).to(torch.bfloat16))


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("aux", [True, False])
def test_fused_router_function_matches_the_unfused_path(mode, aux):
    """Input to MegaMoE's top-k and back: the Triton gating GEMM, routing, and the GEMM backward."""
    if not _gfx1250():
        pytest.skip("the gfx1250 router path")
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    seq, bsz, hidden, e, topk, scaling, coeff, loss_scale = 2048, 2, 7168, 128, 8, 2.5, 1e-3, 0.25
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(seq, bsz, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(e, hidden, device="cuda", generator=g) * hidden**-0.5).to(torch.bfloat16)
    bias = torch.randn(e, device="cuda", generator=g) * 1e-2
    dw = torch.randn(seq * bsz, topk, device="cuda", generator=g)
    noise = torch.randn(seq * bsz, e, device="cuda", generator=g)

    class _RandomSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits):
            return noise.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad

    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(loss_scale, device="cuda"))
    # unfused: the patched gating GEMM's own math, TE routing, MegaMoE's top-k
    xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    logits = gemm._fwd(xr.reshape(-1, hidden).detach(), wr.detach())
    lg = logits.clone().requires_grad_(True)
    routed = _RandomSTE.apply(lg) if mode else lg
    w_ref, i_ref, count_ref, _, _ = _reference(routed, bias, topk, 8, 4, scaling, seq, bsz, mode, coeff, aux)
    (w_ref * dw).sum().backward()
    gl = lg.grad.float().contiguous()
    dx_ref, dwt_ref = gemm._dx(gl, w), gemm._dw(gl, x.reshape(-1, hidden)).to(w.dtype)

    count = torch.zeros(e, device="cuda")
    cfg = dict(
        topk=topk,
        num_groups=8,
        group_topk=4,
        scaling=scaling,
        bsz=bsz,
        mode=mode,
        aux=aux,
        aux_coeff=frt.te_aux_loss_coeff(e, coeff, topk, seq) if aux else 0.0,
        random_logits=(lambda logits: noise.clone()) if mode else None,
    )
    xf, wf = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    prob, idx, _, _ = frt.FusedRouterFunction.apply(xf, wf, bias, count, cfg)
    assert torch.equal(_bits(prob), _bits(w_ref.detach()))
    assert torch.equal(idx, i_ref)
    assert torch.equal(count, count_ref)
    (prob * dw).sum().backward()
    assert torch.equal(xf.grad, dx_ref.view(seq, bsz, hidden))
    assert torch.equal(wf.grad, dwt_ref)


def test_aux_coeff_folds_like_te():
    # (E * coeff) / topk / T / T in fp32, step by step
    import numpy as np

    c = np.float32(np.float32(128) * np.float32(1e-3))
    for d in (8, 4096, 4096):
        c = np.float32(c / np.float32(d))
    assert frt.te_aux_loss_coeff(128, 1e-3, 8, 4096) == float(c)
