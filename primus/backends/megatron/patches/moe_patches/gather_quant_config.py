# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Argument validation and scoped handoff for the Turbo MXFP4 / MXFP8 MoE consumer."""

import importlib
from contextvars import ContextVar
from functools import wraps

_GATHER_CONSUMER = ContextVar("primus_turbo_gather_consumer", default=False)


def validate_gather_quant_config(args):
    forward = bool(getattr(args, "moe_permute_quant_fusion", False))
    backward = bool(getattr(args, "moe_backward_permute_quant_fusion", False))
    if backward and not forward:
        raise ValueError("moe_backward_permute_quant_fusion requires moe_permute_quant_fusion")
    if not forward:
        return forward, backward
    required = {
        "tensor_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "moe_token_dispatcher_type": "alltoall",
        "moe_permute_fusion": True,
        "enable_primus_turbo": True,
        "use_turbo_grouped_gemm": True,
        "turbo_fused_grouped_gemm": True,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"moe_permute_quant_fusion requires {name}={expected!r}")
    # the fused grouped MLP's quantizer gathers: Turbo MXFP4 (grouped_mlp_fp4) or MXFP8 (grouped_mlp_fp8)
    mxfp4 = (getattr(args, "fp4", None), getattr(args, "fp4_recipe", None)) == ("e2m1", "mxfp4")
    mxfp8 = (getattr(args, "fp8", None), getattr(args, "fp8_recipe", None)) == ("e4m3", "mxfp8")
    if not (mxfp4 or mxfp8):
        raise ValueError(
            "moe_permute_quant_fusion requires fp4='e2m1' with fp4_recipe='mxfp4', "
            "or fp8='e4m3' with fp8_recipe='mxfp8'"
        )
    if mxfp8 and not turbo_consumes_permute_seam_mxfp8():
        raise ValueError(
            "moe_permute_quant_fusion with fp8_recipe='mxfp8' requires a Primus-Turbo whose fused grouped MXFP8 MLP "
            "takes the permute's handoff (primus_turbo.pytorch.ops.grouped_mlp_fp8.CONSUMES_PERMUTE_SEAM_MXFP8); "
            "the installed one would quantize the permute's unwritten placeholder"
        )
    if not getattr(args, "moe_skip_identity_sort", True):
        raise ValueError("moe_permute_quant_fusion requires moe_skip_identity_sort=true")
    if getattr(args, "expert_tensor_parallel_size", None) not in (None, 1):
        raise ValueError("moe_permute_quant_fusion requires expert_tensor_parallel_size=1")
    for name in (
        "moe_pad_expert_input_to_capacity",
        "moe_router_padding_for_quantization",
        "moe_apply_probs_on_input",
        "use_turbo_mega_moe",
        "use_turbo_deepep",
    ):
        if getattr(args, name, False):
            raise ValueError(f"moe_permute_quant_fusion requires {name}=false")
    return forward, backward


def turbo_consumes_permute_seam_mxfp8() -> bool:
    """Whether the installed Primus-Turbo's ``grouped_mlp_fp8`` gathers the permuted rows in its MXFP8 quantizer
    (it exports ``CONSUMES_PERMUTE_SEAM_MXFP8``). A build without the flag quantizes the permute's placeholder as
    if it were the activation, so the fusion must not be accepted. Imports Primus-Turbo; called only once the
    fusion is requested with MXFP8."""
    try:
        module = importlib.import_module("primus_turbo.pytorch.ops.grouped_mlp_fp8")
    except Exception:  # noqa: BLE001 - no Primus-Turbo, or one that does not import on this host
        return False
    return bool(getattr(module, "CONSUMES_PERMUTE_SEAM_MXFP8", False))


def _uses_mx():
    """Whether the experts' fused grouped MLP quantizes MX here: MXFP8 (it picks Turbo FP8 first) or MXFP4."""
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager as state,
    )

    if state.is_turbo_fp8_enabled():
        return state.get_turbo_quant_config().mxfp8_scaling()
    return state.is_turbo_fp4_enabled() and state.get_turbo_quant_config().mxfp4_scaling()


def configure_gather_quant_fusion(args, permute):
    forward, backward = validate_gather_quant_config(args)
    if not forward:
        return permute
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    original = MoEAlltoAllTokenDispatcher.dispatch_preprocess
    if not getattr(original, "_primus_gather_consumer", False):

        @wraps(original)
        def dispatch_preprocess(self, *inputs, **kwargs):
            # Only the configured all-to-all MX expert path may hand off a
            # placeholder. Generic TE/Megatron permutation calls remain eager,
            # including BF16 override layers outside the FP4 / FP8 autocast region.
            token = _GATHER_CONSUMER.set(_uses_mx())
            try:
                return original(self, *inputs, **kwargs)
            finally:
                _GATHER_CONSUMER.reset(token)

        dispatch_preprocess._primus_gather_consumer = True
        MoEAlltoAllTokenDispatcher.dispatch_preprocess = dispatch_preprocess

    @wraps(permute)
    def scoped_permute(*inputs, **kwargs):
        enabled = _GATHER_CONSUMER.get()
        return permute(
            *inputs,
            **kwargs,
            fuse_permute_quant=enabled,
            fuse_backward_permute_quant=enabled and backward,
        )

    return scoped_permute
