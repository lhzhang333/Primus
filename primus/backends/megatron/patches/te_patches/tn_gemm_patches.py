###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer Engine's BF16 GEMMs in the TN form: dgrad on a per-iteration transposed weight, and
optionally the TN GEMMs on Primus-Turbo's HipKittens gfx1250 kernels.

TE's ``Linear`` / ``LayerNormLinear`` backward compute ``dx = dy @ w`` with
``general_gemm(w, dy, layout="NN")``. On gfx1250 the NN kernels run well below the TN ones at the
DeepSeek-V3 shapes; ``general_gemm(w.t(), dy, layout="TN")`` is the same product with both operands
K-contiguous. ``te_dgrad_tn: true`` keeps one transposed copy of each weight per training iteration
(the weights change only in the optimizer step; a copy is keyed by ``args.curr_iteration`` and the
weight's storage, and rewritten in place each iteration), so the transpose costs one pass over the
dense weights per step instead of one per micro-batch, and runs those dgrad calls as TN.

``te_gemm_hipkittens: true`` (gfx1250, Primus-Turbo with ``primus_turbo.hipkittens.gemm_gfx1250``,
needs ``te_dgrad_tn``) runs those transposed dgrad GEMMs on the HipKittens kernels when they take the
shape; the dgrad shapes HipKittens does not take keep TE's NN call (and its tuned hipBLASLt rows),
untransposed. The forward GEMMs stay on hipBLASLt: with the example's tuned hipBLASLt rows they are
as fast there or faster, while the dgrad shapes without tuned rows are where HipKittens is ahead.

``te_wgrad_hipkittens: true`` (gfx1250, Primus-Turbo with ``hk_gemm_nt``) runs TE's weight-gradient
GEMMs (``dw = dy^T @ x``, layout NT) on HipKittens' NT kernel where it takes the shape: it reads ``x``
[tokens, in] and ``dy`` [tokens, out] as they are (K-major LDS panels, transposing LDS reads), where the
hipBLASLt NT kernels are well behind its TN ones on gfx1250. The other wgrad shapes keep TE's NT call.

Only calls with nothing else attached qualify: BF16 2-D weight, no quantization, no Userbuffers
overlap, no extra output, no accumulation, no GELU, no bias, default alpha/beta. Every other call is
passed through unchanged. The products are the same in exact arithmetic; the kernel and its summation
order change.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.te.tn_gemm"
_MODULES = ("transformer_engine.pytorch.module.linear", "transformer_engine.pytorch.module.layernorm_linear")


def _is_gfx1250() -> bool:
    import torch

    return torch.cuda.is_available() and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "te_dgrad_tn", False))
        or bool(getattr(args, "te_gemm_hipkittens", False))
        or bool(getattr(args, "te_wgrad_hipkittens", False))
    ) and _is_gfx1250()


def _plain(kwargs) -> bool:
    return (
        kwargs.get("quantization_params") is None
        and kwargs.get("ub") is None
        and kwargs.get("ub_type") is None
        and kwargs.get("extra_output") is None
        and not kwargs.get("bulk_overlap", False)
        and not kwargs.get("accumulate", False)
        and not kwargs.get("gelu", False)
        and kwargs.get("gelu_in") is None
        and kwargs.get("bias") is None
        and kwargs.get("out") is None
        and kwargs.get("alpha", 1.0) == 1.0
        and kwargs.get("beta") is None
        and kwargs.get("out_dtype") in (None, __import__("torch").bfloat16)
    )


def _make_general_gemm(original, dgrad_tn, hipkittens, transposed, wgrad_hk=False):
    import torch

    hk = None
    if hipkittens:
        from primus_turbo.hipkittens.gemm_gfx1250 import (
            hk_gemm_tn,
            hk_gemm_tn_plan,
            hk_gemm_tn_supported,
        )

        hk = (hk_gemm_tn, hk_gemm_tn_supported, hk_gemm_tn_plan)
    nt = None
    if wgrad_hk:
        from primus_turbo.hipkittens.gemm_gfx1250 import (
            hk_gemm_nt,
            hk_gemm_nt_plan,
            hk_gemm_nt_supported,
        )

        nt = (hk_gemm_nt, hk_gemm_nt_supported, hk_gemm_nt_plan)

    hk_dgrad = hk

    def tn(A, B, kwargs):
        """D = B @ A^T for a [M, K] A and a [..., K] B, on HipKittens (dgrad) when it takes the shape."""
        if hk_dgrad is not None and B.is_contiguous():
            b2 = B.reshape(-1, B.shape[-1])
            if hk[1](A, b2):
                return hk[0](A, b2).view(*B.shape[:-1], A.shape[0]), None, None, None
        kw = dict(kwargs)
        kw["layout"] = "TN"
        return original(A, B, **kw)

    def general_gemm(A, B, *args, **kwargs):
        if (
            not args
            and type(A) in (torch.Tensor, torch.nn.Parameter)
            and type(B) is torch.Tensor
            and A.dim() == 2
            and A.dtype == torch.bfloat16
            and B.dtype == torch.bfloat16
            and A.is_cuda
            and A.is_contiguous()
            and _plain(kwargs)
        ):
            layout = kwargs.get("layout", "TN")
            if (
                layout == "NT"
                and nt is not None
                and kwargs.get("grad", False)
                and B.dim() == 2
                and nt[2](A.shape[1], B.shape[1], A.shape[0]) is not None
                and nt[1](A, B)
            ):
                # dw [out, in] = dy^T @ x: HipKittens NT on x [tokens, in] and dy [tokens, out] as they are
                return nt[0](A, B), None, None, None
            if layout == "NN" and dgrad_tn and kwargs.get("grad", False):
                # With HipKittens, only the shapes it takes go TN; the others keep TE's NN call (and the
                # example's tuned hipBLASLt rows for it), without a transpose.
                if hk_dgrad is not None and hk[2](A.shape[1], B.numel() // B.shape[-1], A.shape[0]) is None:
                    return original(A, B, *args, **kwargs)
                return tn(transposed(A), B, kwargs)
        return original(A, B, *args, **kwargs)

    general_gemm._primus_tn_gemm = True
    return general_gemm


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="TE dgrad as TN on a per-iteration transposed weight; optional HipKittens TN GEMMs (gfx1250)",
    condition=_enabled,
)
def patch_te_tn_gemm(ctx: PatchContext) -> None:
    import importlib

    from primus.backends.megatron.core.extensions._triton.transpose import (
        iteration_transposer,
    )

    args = get_args(ctx)
    dgrad_tn = bool(getattr(args, "te_dgrad_tn", False))
    hipkittens = bool(getattr(args, "te_gemm_hipkittens", False))
    wgrad_hk = bool(getattr(args, "te_wgrad_hipkittens", False))
    if hipkittens or wgrad_hk:
        try:
            from primus_turbo.hipkittens import gemm_gfx1250

            available = gemm_gfx1250.hk_gemm_available()
            has_nt = hasattr(gemm_gfx1250, "hk_gemm_nt")
        except ImportError:
            available = has_nt = False
        if not available:
            log_rank_0(f"[Patch:{_PATCH_KEY}] HipKittens gfx1250 GEMM unavailable, hipBLASLt only")
        elif wgrad_hk and not has_nt:
            log_rank_0(f"[Patch:{_PATCH_KEY}] te_wgrad_hipkittens: this Primus-Turbo has no hk_gemm_nt, hipBLASLt wgrad")
        hipkittens = hipkittens and available
        wgrad_hk = wgrad_hk and available and has_nt

    transposed = iteration_transposer()

    patched = []
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        original = getattr(module, "general_gemm", None)
        if original is None or getattr(original, "_primus_tn_gemm", False):
            continue
        module.general_gemm = _make_general_gemm(original, dgrad_tn, hipkittens, transposed, wgrad_hk)
        patched.append(name.rsplit(".", 1)[-1])
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] dgrad TN={dgrad_tn} HipKittens dgrad={hipkittens} HipKittens wgrad={wgrad_hk} "
        f"in: {', '.join(patched) or 'none'}"
    )
