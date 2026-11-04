###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run Transformer Engine's BF16 dgrad GEMMs as TN on a transposed copy of the weight (gfx1250).

TE's ``Linear`` and ``LayerNormLinear`` backward compute ``dx = dy @ w`` with
``general_gemm(w, dy, layout="NN")``. On gfx1250, hipBLASLt runs the same product as a TN GEMM
(``general_gemm(w.t(), dy, layout="TN")``, both operands K-contiguous) faster than the NN one at the
dense Llama shapes, by more than the cost of transposing the weight with a tiled copy right before.

``te_dgrad_transposed_weight: true`` wraps the ``general_gemm`` the two TE modules call: a dgrad call
(``layout="NN"``, ``grad=True``) on a plain contiguous 2-D BF16 / FP16 weight with nothing else
attached (no quantization, Userbuffers overlap, extra output or accumulation) transposes the weight
(``transpose_2d``, bit for bit) and runs TN; every other call is passed through unchanged. The product
is the same in exact arithmetic; the kernel and its summation order change. Off by default; gfx1250
only (where it was measured). The TN shapes want their own hipBLASLt tuning rows.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.te.dgrad_transposed_weight"
_MODULES = ("transformer_engine.pytorch.module.linear", "transformer_engine.pytorch.module.layernorm_linear")


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "te_dgrad_transposed_weight", False)) and is_gfx1250()


def _make_general_gemm(original, transpose_2d):
    import torch

    def general_gemm(A, B, *args, **kwargs):
        if (
            not args
            and kwargs.get("layout") == "NN"
            and kwargs.get("grad", False)
            and type(A) in (torch.Tensor, torch.nn.Parameter)
            and A.dim() == 2
            and A.dtype in (torch.bfloat16, torch.float16)
            and A.is_cuda
            and A.is_contiguous()
            and kwargs.get("quantization_params") is None
            and kwargs.get("ub") is None
            and kwargs.get("extra_output") is None
            and not kwargs.get("bulk_overlap", False)
            and not kwargs.get("accumulate", False)
            and not kwargs.get("gelu", False)
            and kwargs.get("bias") is None
        ):
            kwargs["layout"] = "TN"
            return original(transpose_2d(A.detach()), B, **kwargs)
        return original(A, B, *args, **kwargs)

    general_gemm._primus_dgrad_transposed_weight = True
    return general_gemm


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="TE Linear / LayerNormLinear dgrad GEMMs as TN on a transposed weight (gfx1250)",
    condition=_enabled,
)
def patch_te_dgrad_transposed_weight(ctx: PatchContext) -> None:
    import importlib

    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d

    patched = []
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        original = getattr(module, "general_gemm", None)
        if original is None or getattr(original, "_primus_dgrad_transposed_weight", False):
            continue
        module.general_gemm = _make_general_gemm(original, transpose_2d)
        patched.append(name.rsplit(".", 1)[-1])
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] dgrad GEMMs run TN on a transposed weight in: {', '.join(patched) or 'none'}"
    )
