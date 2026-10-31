###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton GEMMs for an FP32 MoE router (opt-in, ``moe_router_triton_gemm: true``).

The router GEMMs are [tokens x hidden] x [hidden x experts] with experts = 128: the output is too
narrow for hipBLASLt on gfx1250 (the forward BF16->FP32 GEMM gets ~32 workgroups, the backward runs
FP32 GEMMs with 32x32 tiles on FP32 copies of the input and weight). These kernels give each
workgroup a block of tokens and the whole expert dimension:

- forward: FP32 logits = BF16 x @ BF16 W^T, FP32 accumulation (same math as TE's BF16->FP32 GEMM);
- dX: BF16 = FP32 dlogits @ BF16 W, and dW: FP32 partial sums over 4 token splits, summed and cast to
  the weight dtype. dlogits enter the MMAs as a BF16 hi + lo pair, so the backward keeps FP32
  accuracy (measured equal to Megatron's FP32 backward against an FP64 reference) without FP32
  copies of the input and weight.

Router dtype FP32 with BF16 input and weight, no bias, and shapes the kernels tile exactly; anything
else keeps Megatron's path. Measured on MI455X for [8192 x 7168] x [7168 x 128]: fwd 346 -> 56 us,
dX 282 -> 84 us, dW 324 -> 100 us per call; with the gfx1250 launch configs below, fwd 58 -> 25 us and
dX 76 -> 69 us, bit-identical.

On gfx1250 the backward forms the hi + lo pair once, in a small kernel both GEMMs read, instead of in
every workgroup of each: at 128 experts dW 100 -> 57 us (a three-stage k-loop now pays off) and dX
72 -> 69 us per call, the split included, bit-identical. ``router_dx_into`` adds dX
into an existing gradient the way autograd would sum the two.
"""

import functools

import torch
import triton
import triton.language as tl

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_FWD_BLOCK_M, _FWD_BLOCK_K = 32, 64
# Measured on MI455X at 8192 tokens, hidden 7168, 16-256 experts. Four warps for dX: eight cost
# 2.8x at 256 experts (338 -> 120 us). dW (BLOCK_M, BLOCK_H, splits, warps) depends on the expert
# count: the narrow block wins from 128 experts up (256: 217 -> 130 us), the shipped one below.
_DX_BLOCK_M, _DX_BLOCK_H, _DX_WARPS = 128, 128, 4


@functools.lru_cache(maxsize=None)
def _is_gfx1250(device_index: int) -> bool:
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0] == "gfx1250"


def _gfx1250(t: torch.Tensor) -> bool:
    return _is_gfx1250(t.device.index if t.device.index is not None else torch.cuda.current_device())


# gfx1250 launch configs, from a sweep at 8192 x 7168 x 128 in which every config gave the shipped
# outputs bit for bit (each output sums the same 32-deep k-steps in the same order; the tile shape
# only decides who computes it). The forward's deep, three-stage k-loop keeps W's tile in flight
# (58 -> 25 us at k-steps of 256); its stages must fit the LDS, so the k-step shrinks with the expert
# count (128 at 256 experts), and gfx950 keeps the shipped config.
_FWD_STAGES, _FWD_LDS_BUDGET = 3, 256 * 1024


def _fwd_launch(k: int, n: int, gfx1250: bool):
    """(BLOCK_K, launch kwargs) of the forward for K = ``k`` and ``n`` experts."""
    if gfx1250:
        for block_k in (256, 128):
            if k % block_k == 0 and _FWD_STAGES * (_FWD_BLOCK_M + n) * block_k * 2 <= _FWD_LDS_BUDGET:
                return block_k, {"num_warps": 4, "num_stages": _FWD_STAGES}
    return _FWD_BLOCK_K, {"num_warps": 4}


def _dx_launch(n: int, gfx1250: bool):
    """(BLOCK_M, BLOCK_H, launch kwargs) of dX for ``n`` experts: on gfx1250 76 -> 69 us at 128 experts,
    even at 32, and slower at 256, which keeps the shipped config."""
    if gfx1250 and n <= 128:
        return 256, 64, {"num_warps": 4, "num_stages": 2}
    return _DX_BLOCK_M, _DX_BLOCK_H, {"num_warps": _DX_WARPS}


def _dw_config(num_experts: int):
    return (32, 64, 4, 4) if num_experts >= 128 else (64, 64, 4, 8)


# gfx1250 backward on a pre-split hi + lo pair (``_split``). The row block and the split count of dW stay
# those of ``_dw_config``: they fix each output's summation order. Measured at 8192 x 7168, the split
# (10 us) included in both: dW (BLOCK_H 64, 3 stages) 100 -> 57 us at 128 experts, 31 -> 29 us at 32,
# 128 -> 88 us at 256 (4 warps there, 8 below); dX (256-wide, 8 warps) 72 -> 69 us at 128, flat at 32,
# 122 -> 99 us at 256.
_SPLIT_MAX_EXPERTS = 256
_SPLIT_BLOCK = 4096
_DX_SPLIT_LAUNCH = (128, 256, {"num_warps": 8, "num_stages": 1})
_DW_SPLIT_BLOCK_H, _DW_SPLIT_STAGES = 64, 3


def _dw_split_warps(num_experts: int) -> int:
    return 4 if num_experts >= 256 else 8


def _split_backward(g: torch.Tensor) -> bool:
    """Whether the backward for dlogits ``g`` runs the pre-split kernels."""
    return g.shape[1] <= _SPLIT_MAX_EXPERTS and _gfx1250(g)


@triton.jit
def _router_fwd_kernel(
    X, W, O, M, sxm, swn, som, K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    """O[M, N] (fp32) = X[M, K] @ W[N, K]^T; N is the whole expert dimension."""
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mmask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(X + rm[:, None] * sxm + (k0 + rk)[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(W + rn[None, :] * swn + (k0 + rk)[:, None])
        acc += tl.dot(x, w)
    tl.store(O + rm[:, None] * som + rn[None, :], acc, mask=mmask[:, None])


@triton.jit
def _router_dx_kernel(
    G, W, DX, M, sgm, swn, sdm, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    """DX[M, H] (bf16) = G[M, N] (fp32, as bf16 hi + lo) @ W[N, H] (bf16)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rn = tl.arange(0, N)
    mmask = rm < M
    g = tl.load(G + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    w = tl.load(W + rn[:, None] * swn + rh[None, :])
    g_hi = g.to(tl.bfloat16)
    acc = tl.dot(g_hi, w)
    acc += tl.dot((g - g_hi.to(tl.float32)).to(tl.bfloat16), w)
    tl.store(DX + rm[:, None] * sdm + rh[None, :], acc.to(tl.bfloat16), mask=mmask[:, None])


@triton.jit
def _router_dw_kernel(
    G, X, P, M, sgm, sxm, sps, spn, N: tl.constexpr, M_PER_SPLIT, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    """P[s, N, H] (fp32) = G[rows of split s, N]^T @ X[rows of split s, H] (G as bf16 hi + lo)."""
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = tl.arange(0, N)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((N, BLOCK_H), dtype=tl.float32)
    m_start = pid_s * M_PER_SPLIT
    for m0 in range(m_start, m_start + M_PER_SPLIT, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M)
        mmask = rm < M
        g = tl.load(G + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        x = tl.load(X + rm[:, None] * sxm + rh[None, :], mask=mmask[:, None], other=0.0)
        g_hi = g.to(tl.bfloat16)
        acc += tl.dot(g_hi, x)
        acc += tl.dot((g - g_hi.to(tl.float32)).to(tl.bfloat16), x)
    tl.store(P + pid_s * sps + rn[:, None] * spn + rh[None, :], acc)


@triton.jit
def _router_split_kernel(G, HI, LO, numel, BLOCK: tl.constexpr):
    """HI, LO (bf16) = the hi + lo pair of G (fp32), formed as the kernels above form it."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(G + offs, mask=mask)
    hi = g.to(tl.bfloat16)
    tl.store(HI + offs, hi, mask=mask)
    tl.store(LO + offs, (g - hi.to(tl.float32)).to(tl.bfloat16), mask=mask)


@triton.jit
def _router_dx_split_kernel(
    GH,
    GL,
    W,
    DX,
    M,
    sgm,
    swn,
    sdm,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """DX[M, H] (bf16) = (GH + GL)[M, N] @ W[N, H]; ACCUMULATE: DX += that, rounded as autograd's sum
    of two bf16 gradients (each rounded, added in fp32, rounded again)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rn = tl.arange(0, N)
    mmask = rm < M
    gh = tl.load(GH + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    gl = tl.load(GL + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    w = tl.load(W + rn[:, None] * swn + rh[None, :])
    acc = tl.dot(gh, w)
    acc += tl.dot(gl, w)
    out = acc.to(tl.bfloat16)
    ptrs = DX + rm[:, None] * sdm + rh[None, :]
    if ACCUMULATE:
        out = (out.to(tl.float32) + tl.load(ptrs, mask=mmask[:, None]).to(tl.float32)).to(tl.bfloat16)
    tl.store(ptrs, out, mask=mmask[:, None])


@triton.jit
def _router_dw_split_kernel(
    GH,
    GL,
    X,
    P,
    M,
    sgm,
    sxm,
    sps,
    spn,
    N: tl.constexpr,
    M_PER_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """P[s, N, H] (fp32) = (GH + GL)[rows of split s, N]^T @ X[rows of split s, H], the products summed
    in the order of ``_router_dw_kernel``."""
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = tl.arange(0, N)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((N, BLOCK_H), dtype=tl.float32)
    m_start = pid_s * M_PER_SPLIT
    for m0 in range(m_start, m_start + M_PER_SPLIT, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M)
        mmask = rm < M
        gh = tl.load(GH + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        gl = tl.load(GL + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        x = tl.load(X + rm[:, None] * sxm + rh[None, :], mask=mmask[:, None], other=0.0)
        acc += tl.dot(gh, x)
        acc += tl.dot(gl, x)
    tl.store(P + pid_s * sps + rn[:, None] * spn + rh[None, :], acc)


def _supported(inp: torch.Tensor, weight: torch.Tensor, bias, router_dtype) -> bool:
    n, k = weight.shape
    return (
        router_dtype == torch.float32
        and inp.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and bias is None
        and inp.is_cuda
        and weight.is_contiguous()
        and inp.shape[-1] == k
        and 16 <= n <= 256
        and n & (n - 1) == 0
        and k % _FWD_BLOCK_K == 0
        and k % _DX_BLOCK_H == 0
        and k % _dw_config(n)[1] == 0
        # the kernels index with int32 offsets
        and inp.numel() < 2**31
    )


def router_gemm_on_triton(inp: torch.Tensor, weight: torch.Tensor, bias, router_dtype) -> bool:
    """Whether the router gating GEMM for these operands runs on these Triton kernels: the patch is
    installed and the shapes are ones it takes (so the GEMM uses no TE workspace)."""
    from megatron.core.transformer.moe import moe_utils

    return bool(
        getattr(moe_utils.RouterGatingLinearFunction, "_primus_router_triton_gemm_patched", False)
    ) and (_supported(inp.reshape(-1, inp.shape[-1]), weight, bias, router_dtype))


def _fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    block_k, launch = _fwd_launch(x.shape[1], w.shape[0], _gfx1250(x))
    _router_fwd_kernel[(triton.cdiv(x.shape[0], _FWD_BLOCK_M),)](
        x,
        w,
        out,
        x.shape[0],
        x.stride(0),
        w.stride(0),
        out.stride(0),
        K=x.shape[1],
        N=w.shape[0],
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_K=block_k,
        **launch,
    )
    return out


def _dx(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    dx = torch.empty(g.shape[0], w.shape[1], dtype=torch.bfloat16, device=g.device)
    block_m, block_h, launch = _dx_launch(w.shape[0], _gfx1250(g))
    _router_dx_kernel[(triton.cdiv(g.shape[0], block_m), w.shape[1] // block_h)](
        g,
        w,
        dx,
        g.shape[0],
        g.stride(0),
        w.stride(0),
        dx.stride(0),
        N=w.shape[0],
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        **launch,
    )
    return dx


def _dw(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = g.shape[0]
    block_m, block_h, splits, warps = _dw_config(g.shape[1])
    per = triton.cdiv(triton.cdiv(m, splits), block_m) * block_m
    p = torch.empty(splits, g.shape[1], x.shape[1], dtype=torch.float32, device=g.device)
    _router_dw_kernel[(x.shape[1] // block_h, splits)](
        g,
        x,
        p,
        m,
        g.stride(0),
        x.stride(0),
        p.stride(0),
        p.stride(1),
        N=g.shape[1],
        M_PER_SPLIT=per,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=warps,
    )
    return p.sum(0)


def _split(g: torch.Tensor):
    hi = torch.empty(g.shape, dtype=torch.bfloat16, device=g.device)
    lo = torch.empty_like(hi)
    _router_split_kernel[(triton.cdiv(g.numel(), _SPLIT_BLOCK),)](g, hi, lo, g.numel(), BLOCK=_SPLIT_BLOCK)
    return hi, lo


def _dx_split(hi: torch.Tensor, lo: torch.Tensor, w: torch.Tensor, out=None) -> torch.Tensor:
    accumulate = out is not None
    if out is None:
        out = torch.empty(hi.shape[0], w.shape[1], dtype=torch.bfloat16, device=hi.device)
    block_m, block_h, launch = _DX_SPLIT_LAUNCH
    _router_dx_split_kernel[(triton.cdiv(hi.shape[0], block_m), w.shape[1] // block_h)](
        hi,
        lo,
        w,
        out,
        hi.shape[0],
        hi.stride(0),
        w.stride(0),
        out.stride(0),
        N=w.shape[0],
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        ACCUMULATE=accumulate,
        **launch,
    )
    return out


def _dw_split(hi: torch.Tensor, lo: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = hi.shape[0]
    block_m, _, splits, _ = _dw_config(hi.shape[1])
    per = triton.cdiv(triton.cdiv(m, splits), block_m) * block_m
    p = torch.empty(splits, hi.shape[1], x.shape[1], dtype=torch.float32, device=hi.device)
    _router_dw_split_kernel[(x.shape[1] // _DW_SPLIT_BLOCK_H, splits)](
        hi,
        lo,
        x,
        p,
        m,
        hi.stride(0),
        x.stride(0),
        p.stride(0),
        p.stride(1),
        N=hi.shape[1],
        M_PER_SPLIT=per,
        BLOCK_M=block_m,
        BLOCK_H=_DW_SPLIT_BLOCK_H,
        num_warps=_dw_split_warps(hi.shape[1]),
        num_stages=_DW_SPLIT_STAGES,
    )
    return p.sum(0)


def router_dx_into(grad_logits: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out += grad_logits @ weight`` for the router's input gradient, in place: FP32-accurate dX
    rounded to bf16, then added to ``out`` exactly as autograd sums two bf16 gradients, so a caller
    that owns the input gradient can take the router's share without a separate add kernel.

    ``grad_logits`` [tokens, experts] fp32, ``weight`` [experts, hidden] bf16 contiguous, ``out``
    [tokens, hidden] bf16 contiguous; the shapes must be ones ``router_gemm_on_triton`` accepts.
    """
    g = grad_logits.reshape(-1, grad_logits.shape[-1]).float().contiguous()
    o = out.view(-1, out.shape[-1])
    assert out.dtype == torch.bfloat16 and out.is_contiguous() and o.shape == (g.shape[0], weight.shape[1])
    if _split_backward(g):
        hi, lo = _split(g)
        _dx_split(hi, lo, weight, out=o)
    else:
        o.add_(_dx(g, weight))
    return out


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_triton_gemm", False))


@register_patch(
    "megatron.moe.router_triton_gemm",
    backend="megatron",
    phase="before_train",
    description="Triton forward/backward GEMMs for the FP32 MoE router (FP32-accurate backward)",
    condition=_enabled,
)
def patch_router_triton_gemm(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    if getattr(fn, "_primus_router_triton_gemm_patched", False):
        return
    if getattr(get_args(ctx), "moe_router_bwd_bf16", False):
        log_rank_0(
            "[Patch:megatron.moe.router_triton_gemm] not installed: moe_router_bwd_bf16 is set "
            "(the two options are mutually exclusive)"
        )
        return
    orig_forward, orig_backward = fn.forward, fn.backward

    def forward(ctx_, inp, weight, bias, router_dtype):
        ctx_.primus_triton = _supported(inp, weight, bias, router_dtype)
        if not ctx_.primus_triton:
            return orig_forward(ctx_, inp, weight, bias, router_dtype)
        ctx_.save_for_backward(inp, weight, bias)
        ctx_.router_dtype = router_dtype
        ctx_.input_dtype = inp.dtype
        ctx_.weight_dtype = weight.dtype
        shape = inp.shape
        out = _fwd(inp.reshape(-1, shape[-1]).contiguous(), weight)
        return out.view(*shape[:-1], -1)

    def backward(ctx_, grad_output):
        if not getattr(ctx_, "primus_triton", False):
            return orig_backward(ctx_, grad_output)
        inp, weight, _ = ctx_.saved_tensors
        shape = inp.shape
        x = inp.reshape(-1, shape[-1]).contiguous()
        g = grad_output.reshape(-1, grad_output.shape[-1]).float().contiguous()
        if _split_backward(g):
            hi, lo = _split(g)
            grad_input = _dx_split(hi, lo, weight).view(*shape)
            grad_weight = _dw_split(hi, lo, x).to(ctx_.weight_dtype)
        else:
            grad_input = _dx(g, weight).view(*shape)
            grad_weight = _dw(g, x).to(ctx_.weight_dtype)
        return grad_input, grad_weight, None, None

    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    fn._primus_router_triton_gemm_patched = True
    log_rank_0("[Patch:megatron.moe.router_triton_gemm] FP32 router GEMMs on Triton kernels")
