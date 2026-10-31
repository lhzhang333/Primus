###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MoE routing on Triton for MegaMoE (opt-in, ``moe_router_fused_triton: true``).

With ``moe_router_fusion`` and ``moe_router_triton_gemm``, a MegaMoE layer's router runs per micro-batch
the gating GEMM, TE's fused top-k (sigmoid scores, expert bias, group-limited top-k), TE's aux-loss scores
and seq aux loss, the token counts of the expert bias and of the aux loss, and MegaMoE's top-k over the
dense probabilities: about 30 kernels forward and backward. Here the forward is the gating GEMM and one
routing kernel that writes MegaMoE's top-k indices and weights directly; the backward is one kernel from
the top-k weights' gradient to the logits' gradient (as the hi + lo pair the GEMMs read) and the two
backward GEMMs of ``router_triton_gemm_patches``.

The results are bit-identical to that path: the routing kernels replay the arithmetic of TE's
one-warp-per-token kernels (the expf-based sigmoid, IEEE division, the warp butterfly sums and the
naive top-k's tie-breaks) and MegaMoE's ordering of the top-k by probability. They compile without fp
contraction, as TE's kernels are: contraction would also change ocml's exp. Only the logged aux-loss
value is summed in another order (TE sums it with atomics too); its gradient does not depend on it.
"""

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

# tokens per program: 8 for the forward (fewer live [tokens, experts] tiles), 32 for the backward
_BLOCK_T, _BLOCK_T_BWD = 8, 32


@triton.jit
def _bitrev5(x):
    return ((x & 1) << 4) | ((x & 2) << 2) | (x & 4) | ((x & 8) >> 2) | ((x & 16) >> 4)


@triton.jit
def _tie_key(i, L: tl.constexpr):
    """Among equal maxima, TE's naive warp top-k over ``L`` values keeps the one of the lowest key: the
    value's lane (i % 32) ranks by its bit-reversed number -- each butterfly step keeps its own value on
    a tie -- and a lane keeps the first of its values."""
    if L > 32:
        return _bitrev5(i % 32) * (L // 32) + i // 32
    else:
        return _bitrev5(i)


_REMOVED = tl.constexpr(-(1 << 62))


@triton.jit
def _packed(v, key):
    """int64 whose max is TE's pick: v's bits in signed-sortable form high, the inverted tie key in the
    low 16 bits (the highest value, among equal values the lowest key). ``_REMOVED`` is below all."""
    b = v.to(tl.int32, bitcast=True)
    return (tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64) << 16) | (0xFFFF - key)


@triton.jit
def _unpacked(m):
    """(value, key) of a ``_packed`` maximum."""
    s = (m >> 16).to(tl.int32)
    return tl.where(s >= 0, s, s ^ 0x7FFFFFFF).to(tl.float32, bitcast=True), (0xFFFF - (m & 0xFFFF)).to(
        tl.int32
    )


@triton.jit
def _key_index(k, L: tl.constexpr):
    """The value index of tie key ``k`` (``_tie_key``'s inverse)."""
    if L > 32:
        return (k % (L // 32)) * 32 + _bitrev5(k // (L // 32))
    else:
        return _bitrev5(k)


@triton.jit
def _column(x, cols, k):
    """Column ``k`` of a 2-D tensor, bits included (-0.0 stays -0.0)."""
    if x.dtype == tl.float32:
        bits = tl.sum(tl.where(cols[None, :] == k, x.to(tl.int32, bitcast=True), 0), axis=1)
        return bits.to(tl.float32, bitcast=True)
    else:
        return tl.sum(tl.where(cols[None, :] == k, x, 0), axis=1)


@triton.jit
def _halve(v, BLOCK_T: tl.constexpr, W: tl.constexpr):
    """One butterfly step over W lanes: lane l < W / 2 gets v[l] + v[l + W / 2]."""
    return tl.sum(tl.reshape(v, (BLOCK_T, 2, W // 2)), axis=1)


@triton.jit
def _butterfly(lane, W: tl.constexpr, BLOCK_T: tl.constexpr):
    """The xor-16 ... xor-1 butterfly over W lanes (lanes past W hold zeros, so the wider steps change
    nothing)."""
    if W >= 32:
        lane = _halve(lane, BLOCK_T, 32)
    if W >= 16:
        lane = _halve(lane, BLOCK_T, 16)
    if W >= 8:
        lane = _halve(lane, BLOCK_T, 8)
    if W >= 4:
        lane = _halve(lane, BLOCK_T, 4)
    if W >= 2:
        lane = _halve(lane, BLOCK_T, 2)
    return tl.reshape(lane, (BLOCK_T,))


@triton.jit
def _lane_slice(v3, j, jj):
    """Slice ``jj`` of [BLOCK_T, L // 32, 32]: lane l's (jj + 1)-th value, bits included."""
    return tl.sum(tl.where(j[None, :, None] == jj, v3.to(tl.int32, bitcast=True), 0), axis=1).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _warp_sum(v, L: tl.constexpr, BLOCK_T: tl.constexpr):
    """TE's warp sum of each row's ``L`` values: lane l adds v[l], v[l + 32], ... in order, then the
    butterfly."""
    if L > 32:
        v3 = tl.reshape(v, (BLOCK_T, L // 32, 32))
        j = tl.arange(0, L // 32)
        lane = _lane_slice(v3, j, 0)
        for jj in tl.static_range(1, L // 32):
            lane = lane + _lane_slice(v3, j, jj)
        return _butterfly(lane, 32, BLOCK_T)
    else:
        return _butterfly(v, L, BLOCK_T)


@triton.jit
def _warp_dot(g, a, mask, L: tl.constexpr, BLOCK_T: tl.constexpr):
    """TE's warp sum of g * a over each row's values where ``mask``: lane l runs s += g * a from 0 over
    v[l], v[l + 32], ... in order, product and sum each rounded, then the butterfly."""
    return _warp_sum(tl.where(mask, g * a, 0.0), L, BLOCK_T)


@triton.jit
def _fused_route_fwd_kernel(
    LOGITS,
    BIAS,
    SIG,
    PROBS,
    PART,
    T,
    scaling,
    E: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_TOPK: tl.constexpr,
    TOPK: tl.constexpr,
    BSZ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    COUNT_TOKENS: tl.constexpr,
    AUX: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Per token, TE's fused top-k with sigmoid scores: PROBS the dense probabilities, SIG the sigmoid
    (saved for backward). This program's row of PART [programs, E * (1 + 2 * BSZ)]: the tokens it routed to
    each expert, then with AUX, per sequence of the micro-batch (token t is sequence t % BSZ), the top-k
    counts and the column sums of TE's aux-loss scores. ``_route_partials_kernel`` sums the rows."""
    pid = tl.program_id(0)
    rt = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = rt < T
    re = tl.arange(0, E)
    offs = rt[:, None] * E + re[None, :]
    x = tl.load(LOGITS + offs, mask=tmask[:, None], other=0.0)
    sig = tl.math.div_rn(tl.full((BLOCK_T, E), 1.0, tl.float32), 1.0 + libdevice.exp(-x))
    tl.store(SIG + offs, sig, mask=tmask[:, None])
    if HAS_BIAS:
        scores = sig + tl.load(BIAS + re)[None, :]
    else:
        scores = sig
    key = _tie_key(re, E)

    # group-limited routing: the groups of the highest sums of their top (TOPK / GROUP_TOPK) scores
    if GROUPS > 1:
        GS: tl.constexpr = E // GROUPS
        s3 = tl.reshape(scores, (BLOCK_T, GROUPS, GS))
        gkey = _tie_key(tl.arange(0, GS), GS)
        pk3 = _packed(s3, gkey[None, None, :])
        gscore = tl.zeros((BLOCK_T, GROUPS), dtype=tl.float32)
        for _k in tl.static_range(TOPK // GROUP_TOPK):
            m, kmin = _unpacked(tl.max(pk3, axis=2))
            pk3 = tl.where(gkey[None, None, :] == kmin[:, :, None], _REMOVED, pk3)
            gscore = gscore + m
        grkey = _tie_key(tl.arange(0, GROUPS), GROUPS)
        gpk = _packed(gscore, grkey[None, :])
        gsel = tl.zeros((BLOCK_T, GROUPS), dtype=tl.int32)
        for _k in tl.static_range(GROUP_TOPK):
            kmin = _unpacked(tl.max(gpk, axis=1))[1]
            hit = grkey[None, :] == kmin[:, None]
            gsel = tl.where(hit, 1, gsel)
            gpk = tl.where(hit, _REMOVED, gpk)
        masked = tl.reshape(tl.where(gsel[:, :, None] != 0, s3, float("-inf")), (BLOCK_T, E))
    else:
        masked = scores

    # top-k in TE's selection order
    kk = tl.arange(0, TOPK)
    sel = tl.zeros((BLOCK_T, TOPK), dtype=tl.int32)
    val = tl.zeros((BLOCK_T, TOPK), dtype=tl.float32)
    routed = tl.zeros((BLOCK_T, E), dtype=tl.int32)
    pk = _packed(masked, key[None, :])
    for k in tl.static_range(TOPK):
        m, kmin = _unpacked(tl.max(pk, axis=1))
        hit = key[None, :] == kmin[:, None]
        pk = tl.where(hit, _REMOVED, pk)
        routed = tl.where(hit, 1, routed)
        sel = tl.where(kk[None, :] == k, _key_index(kmin, E)[:, None], sel)
        val = tl.where(kk[None, :] == k, m[:, None], val)
    if HAS_BIAS:
        val = val - tl.load(BIAS + sel)
    if TOPK > 1:
        denom = _warp_sum(val, TOPK, BLOCK_T) + 1e-20
        val = tl.math.div_rn(val, tl.broadcast_to(denom[:, None], (BLOCK_T, TOPK)))
    prob = scaling * val
    dense = tl.zeros((BLOCK_T, E), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        dense = tl.where(re[None, :] == _column(sel, kk, k)[:, None], _column(prob, kk, k)[:, None], dense)
    tl.store(PROBS + offs, dense, mask=tmask[:, None])

    if COUNT_TOKENS:
        tl.store(
            PART + pid * (E * (1 + 2 * BSZ)) + re,
            tl.sum(tl.where(tmask[:, None], routed, 0), axis=0).to(tl.float32),
        )

    if AUX:
        # TE's aux-loss scores: the sigmoid normalized over all experts, and their plain top-k
        ascores = tl.math.div_rn(
            sig, tl.broadcast_to((_warp_sum(sig, E, BLOCK_T) + 1e-20)[:, None], (BLOCK_T, E))
        )
        apk = _packed(ascores, key[None, :])
        arouted = tl.zeros((BLOCK_T, E), dtype=tl.int32)
        for _k in tl.static_range(TOPK):
            hit = key[None, :] == _unpacked(tl.max(apk, axis=1))[1][:, None]
            apk = tl.where(hit, _REMOVED, apk)
            arouted = tl.where(hit, 1, arouted)
        seq = rt % BSZ
        for b in tl.static_range(BSZ):
            rows = tmask & (seq == b)
            row = PART + pid * (E * (1 + 2 * BSZ)) + E
            tl.store(row + b * E + re, tl.sum(tl.where(rows[:, None], arouted, 0), axis=0).to(tl.float32))
            tl.store(row + (BSZ + b) * E + re, tl.sum(tl.where(rows[:, None], ascores, 0.0), axis=0))


@triton.jit
def _route_partials_kernel(
    PART,
    OUT,
    COUNT,
    P,
    E: tl.constexpr,
    BSZ: tl.constexpr,
    COUNT_TOKENS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """OUT[c] = the sum of PART's P rows in column c, ROWS rows at a time in order (deterministic); the
    token counts (columns below E) added into COUNT instead."""
    N: tl.constexpr = E * (1 + 2 * BSZ)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rr = tl.arange(0, ROWS)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r0 in range(0, P, ROWS):
        rows = r0 + rr
        acc += tl.sum(
            tl.load(PART + rows[:, None] * N + cols[None, :], mask=(rows < P)[:, None], other=0.0), axis=0
        )
    if COUNT_TOKENS:
        tl.store(COUNT + cols, tl.load(COUNT + cols, mask=cols < E) + acc, mask=cols < E)
    tl.store(OUT + cols, acc, mask=cols >= E)


@triton.jit
def _route_aux_loss_kernel(OUT, AUX, coeff, E: tl.constexpr, BSZ: tl.constexpr):
    """AUX = coeff * sum(aux-loss token counts * score column sums) -- TE's aux loss, divided by BSZ."""
    c = tl.arange(0, BSZ * E)
    tl.store(AUX, tl.sum(tl.load(OUT + E + c) * tl.load(OUT + E + BSZ * E + c)) * coeff)


@triton.jit
def _fused_route_bwd_kernel(
    DPROB,
    SEL,
    SIG,
    SEQ_COUNT,
    GRAD_AUX,
    HI,
    LO,
    DLOGITS,
    T,
    scaling,
    aux_coeff,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BSZ: tl.constexpr,
    AUX: tl.constexpr,
    WRITE_DLOGITS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """The logits' gradient from the gradient of the top-k probabilities (DPROB, in SEL's order): TE's
    fused top-k backward, plus with AUX the backward of TE's aux loss (GRAD_AUX, aux_coeff, SEQ_COUNT)
    and aux-loss scores, summed; written as its bf16 hi + lo pair (and with WRITE_DLOGITS in fp32)."""
    pid = tl.program_id(0)
    rt = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = rt < T
    re = tl.arange(0, E)
    offs = rt[:, None] * E + re[None, :]
    sig = tl.load(SIG + offs, mask=tmask[:, None], other=0.5)
    kk = tl.arange(0, TOPK)
    toffs = rt[:, None] * TOPK + kk[None, :]
    dp = tl.load(DPROB + toffs, mask=tmask[:, None], other=0.0)
    sel = tl.load(SEL + toffs, mask=tmask[:, None], other=0)
    gprob = tl.zeros((BLOCK_T, E), dtype=tl.float32)
    routed = tl.zeros((BLOCK_T, E), dtype=tl.int32)
    for k in tl.static_range(TOPK):
        hit = re[None, :] == _column(sel, kk, k)[:, None]
        gprob = tl.where(hit, _column(dp, kk, k)[:, None], gprob)
        routed = tl.where(hit, 1, routed)
    isr = routed != 0
    g = gprob * scaling
    if TOPK > 1:
        sum_act = _warp_sum(tl.where(isr, sig, 0.0), E, BLOCK_T)
        sum_gact = _warp_dot(g, sig, isr, E, BLOCK_T)
        denom = sum_act + 1e-20
        g = (
            tl.math.div_rn(g, tl.broadcast_to(denom[:, None], (BLOCK_T, E)))
            - tl.math.div_rn(sum_gact, denom * denom)[:, None]
        )
    g = tl.where(isr, g, 0.0)
    d = (g * sig) * (1.0 - sig)
    if AUX:
        tpe = tl.load(SEQ_COUNT + (rt % BSZ)[:, None] * E + re[None, :], mask=tmask[:, None], other=0.0)
        ga = (aux_coeff * tpe) * tl.load(GRAD_AUX)
        sum_act = _warp_sum(sig, E, BLOCK_T)
        sum_gact = _warp_dot(ga, sig, tl.broadcast_to(re[None, :] >= 0, (BLOCK_T, E)), E, BLOCK_T)
        denom = sum_act + 1e-20
        ga = (
            tl.math.div_rn(ga, tl.broadcast_to(denom[:, None], (BLOCK_T, E)))
            - tl.math.div_rn(sum_gact, denom * denom)[:, None]
        )
        d = d + (ga * sig) * (1.0 - sig)
    hi = d.to(tl.bfloat16)
    tl.store(HI + offs, hi, mask=tmask[:, None])
    tl.store(LO + offs, (d - hi.to(tl.float32)).to(tl.bfloat16), mask=tmask[:, None])
    if WRITE_DLOGITS:
        tl.store(DLOGITS + offs, d, mask=tmask[:, None])


def te_aux_loss_coeff(num_experts: int, coeff: float, topk: int, total_num_tokens: int) -> float:
    """TE's fused aux loss coefficient, folded on the host in fp32 as TE folds it:
    ``(num_experts * coeff) / topk / total_num_tokens / total_num_tokens``."""
    c = np.float32(num_experts) * np.float32(coeff)
    for d in (topk, total_num_tokens, total_num_tokens):
        c = np.float32(c / np.float32(d))
    return float(c)


def fused_route_forward(
    logits: torch.Tensor,
    expert_bias,
    *,
    topk: int,
    num_groups,
    group_topk,
    scaling: float,
    bsz: int,
    mode: int,
    count=None,
    aux: bool = False,
    aux_coeff: float = 0.0,
):
    """Routing outputs of logits [T, E] fp32: (weights [T, topk] fp32, idx [T, topk] int64, sel [T, topk]
    int32, sig [T, E], seq_count [bsz, E] fp32 or None, aux_loss 0-dim or None -- with ``aux``, Megatron's
    seq aux loss, TE's ``aux_coeff`` times the scores' column sums times the counts over ``bsz``). (weights, idx) are
    MegaMoE's top-k of TE's dense probabilities -- with ``mode`` 1 / 2 after PrimusTopKRouter's ``even`` /
    ``spread`` forced balancing -- taken by torch.topk itself, whose order of equal values is its own;
    sel[:, r] is the expert whose probability weights[:, r] is. ``count`` (fp32 [E]) += the tokens routed
    to each expert."""
    t, e = logits.shape
    dev = logits.device
    probs = torch.empty(t, e, dtype=torch.float32, device=dev)
    sig = torch.empty(t, e, dtype=torch.float32, device=dev)
    programs = triton.cdiv(t, _BLOCK_T)
    width = e * (1 + 2 * bsz)
    part = torch.empty(programs, width, dtype=torch.float32, device=dev)
    grouped = bool(group_topk) and num_groups is not None and num_groups > 1
    _fused_route_fwd_kernel[(programs,)](
        logits,
        expert_bias if expert_bias is not None else sig,
        sig,
        probs,
        part,
        t,
        scaling,
        E=e,
        GROUPS=num_groups if grouped else 1,
        GROUP_TOPK=group_topk if grouped else 1,
        TOPK=topk,
        BSZ=bsz,
        HAS_BIAS=expert_bias is not None,
        COUNT_TOKENS=count is not None,
        AUX=aux,
        BLOCK_T=_BLOCK_T,
        num_warps=4,
        enable_fp_fusion=False,
    )
    seq_count = aux_loss = None
    if count is not None or aux:
        out = torch.empty(width, dtype=torch.float32, device=dev)
        _route_partials_kernel[(triton.cdiv(width if aux else e, 16),)](
            part,
            out,
            count if count is not None else out,
            programs,
            E=e,
            BSZ=bsz,
            COUNT_TOKENS=count is not None,
            ROWS=256,
            BLOCK=16,
            num_warps=4,
        )
        if aux:
            seq_count = out[e : e * (1 + bsz)].view(bsz, e)
            aux_loss = torch.empty((), dtype=torch.float32, device=dev)
            _route_aux_loss_kernel[(1,)](out, aux_loss, aux_coeff / bsz, E=e, BSZ=bsz)
    if not mode:
        weights, idx = probs.topk(topk, dim=-1)
        return weights, idx, idx.to(torch.int32), sig, seq_count, aux_loss
    # PrimusTopKRouter._force_even_routing, then MegaMoE's top-k of the balanced probabilities
    from primus.backends.megatron.core.transformer.moe.router import (
        force_balanced_slots,
    )

    vals, experts = torch.topk(probs, topk, dim=1)
    slot = force_balanced_slots(t, topk, e, mode == 2, dev)
    balanced = torch.zeros_like(probs)
    balanced.scatter_(1, slot, vals)
    weights, idx = balanced.topk(topk, dim=-1)
    # MegaMoE's r-th pick is slot[:, k] for one k: its probability is expert experts[:, k]'s
    k_of = (idx.unsqueeze(-1) == slot.unsqueeze(1)).to(torch.int32).argmax(dim=-1)
    return weights, idx, experts.gather(1, k_of).to(torch.int32), sig, seq_count, aux_loss


class FusedRouterFunction(torch.autograd.Function):
    """The router of a MegaMoE layer from its input to MegaMoE's top-k: gating GEMM (Triton, FP32
    logits), optional random logits of forced load balancing, routing; backward: routing, then dX and dW
    on the pre-split hi + lo pair. Returns (top-k weights, top-k indices, aux-loss token counts or None,
    seq aux loss or None)."""

    @staticmethod
    def forward(ctx, inp, weight, expert_bias, count, cfg):
        from primus.backends.megatron.patches.moe_patches import (
            router_triton_gemm_patches as gemm,
        )

        x = inp.reshape(-1, inp.shape[-1]).contiguous()
        logits = gemm._fwd(x, weight)
        if cfg["random_logits"] is not None:
            logits = cfg["random_logits"](logits)
        prob, idx, sel, sig, seq_count, aux_loss = fused_route_forward(
            logits,
            expert_bias,
            topk=cfg["topk"],
            num_groups=cfg["num_groups"],
            group_topk=cfg["group_topk"],
            scaling=cfg["scaling"],
            bsz=cfg["bsz"],
            mode=cfg["mode"],
            count=count,
            aux=cfg["aux"],
            aux_coeff=cfg["aux_coeff"],
        )
        ctx.save_for_backward(x, weight, sel, sig, seq_count)
        ctx.cfg = cfg
        ctx.in_shape = inp.shape
        outputs = (prob, idx) + ((seq_count, aux_loss) if cfg["aux"] else (None, None))
        ctx.mark_non_differentiable(*[o for o in outputs[1:] if o is not None])
        return outputs

    @staticmethod
    def backward(ctx, dprob, *_):
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        from primus.backends.megatron.patches.moe_patches import (
            router_triton_gemm_patches as gemm,
        )

        x, weight, sel, sig, seq_count = ctx.saved_tensors
        cfg = ctx.cfg
        grad_aux = None
        if cfg["aux"]:
            # the aux loss gradient MoEAuxLossAutoScaler hands back, through the seq aux loss's / bsz
            if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=sig.device)
            scale = MoEAuxLossAutoScaler.main_loss_backward_scale
            grad_aux = (torch.ones((), dtype=torch.float32, device=sig.device) * scale) / cfg["bsz"]
        hi, lo, _ = fused_route_backward(
            dprob,
            sel,
            sig,
            scaling=cfg["scaling"],
            bsz=cfg["bsz"],
            seq_count=seq_count,
            grad_aux=grad_aux,
            aux_coeff=cfg["aux_coeff"],
        )
        grad_input = gemm._dx_split(hi, lo, weight).view(ctx.in_shape)
        grad_weight = gemm._dw_split(hi, lo, x).to(weight.dtype)
        return grad_input, grad_weight, None, None, None


def fused_route_backward(
    dprob: torch.Tensor,
    sel: torch.Tensor,
    sig: torch.Tensor,
    *,
    scaling: float,
    bsz: int,
    seq_count=None,
    grad_aux=None,
    aux_coeff: float = 0.0,
    write_dlogits: bool = False,
):
    """The logits' gradient as (hi, lo, dlogits or None); the aux-loss part with ``seq_count``."""
    t, e = sig.shape
    hi = torch.empty(t, e, dtype=torch.bfloat16, device=sig.device)
    lo = torch.empty_like(hi)
    dlog = torch.empty(t, e, dtype=torch.float32, device=sig.device) if write_dlogits else None
    aux = seq_count is not None
    _fused_route_bwd_kernel[(triton.cdiv(t, _BLOCK_T_BWD),)](
        dprob.contiguous(),
        sel,
        sig,
        seq_count if aux else sel,
        grad_aux if aux else sig,
        hi,
        lo,
        dlog if write_dlogits else sig,
        t,
        scaling,
        aux_coeff,
        E=e,
        TOPK=sel.shape[1],
        BSZ=bsz,
        AUX=aux,
        WRITE_DLOGITS=write_dlogits,
        BLOCK_T=_BLOCK_T_BWD,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return hi, lo, dlog
