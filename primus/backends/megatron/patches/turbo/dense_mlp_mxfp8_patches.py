###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Dense MXFP8 SwiGLU MLP patch (gfx1250)
======================================

Routes Megatron ``MLP.forward`` (fc1 -> SwiGLU -> fc2) of the non-expert SwiGLU MLPs on Primus-Turbo linears --
the DeepSeek-V3 shared experts (``*.mlp.shared_experts``), the full model's dense FFN layers -- through
Primus-Turbo ``mlp_fp8``: one autograd Function whose GLU and dGLU write their MXFP8 dual casts directly, so the
activation and its gradient are neither kept in bf16 nor quantized a second time. The weights are cached on the
first micro-batch the way the Turbo MX linears cache them (``_maybe_create_quantized_weight_buffers``: the layout
Primus-Turbo's ``mx_weight_layout_recipe`` asks for) and the wgrads reach ``main_grad`` the same way
(``_fuse_wgrad_accum_pattern``, ``_bridge_weight_grad``). Opt-in, with the MXFP4 dense MLP's switch:

    YAML  turbo_fused_gemm: true
    env   PRIMUS_TURBO_FUSED_DENSE_MLP=1   (overrides YAML)

under ``--fp8 e4m3 --fp8_recipe mxfp8`` and ``use_turbo_gemm``, on gfx1250 only (Primus-Turbo's ``mlp_fp8`` raises
on other archs; there the switch keeps meaning the MXFP4 dense MLP alone). Per forward, a module takes the op only
where both of its linears run the same Turbo MXFP8 recipe (a BF16 matcher of ``te_precision_config_file`` or a BF16
autocast keeps Megatron's path); routed experts never do.
"""

from __future__ import annotations

import torch.nn.functional as F

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches.turbo.dense_mlp_fp4_patches import (
    _apply_fc1_fused_norm,
    _fused_dense_mlp_requested,
    _run_module_forward_pre_hooks,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.turbo_dense_mlp_mxfp8"


def _ext():
    import primus.backends.megatron.core.extensions.primus_turbo as ext

    return ext


def _turbo_linear_types():
    """(fc1 types, fc2 types): the Primus-Turbo linears whose MXFP8 path ``mlp_fp8`` stands in for."""
    ext = _ext()
    fc1 = (ext.PrimusTurboColumnParallelLinear, ext.PrimusTurboLayerNormColumnParallelLinear)
    return fc1, (ext.PrimusTurboRowParallelLinear,)


def _mlp_fp8_op():
    try:
        from primus_turbo.pytorch.ops import mlp_fp8
    except ImportError:
        return None
    return mlp_fp8


def _mxfp8_recipe(args) -> bool:
    return bool(getattr(args, "fp8", None)) and getattr(args, "fp8_recipe", None) == "mxfp8"


def _requested_here(args) -> bool:
    """The switch, under the MXFP8 recipe, on gfx1250 (the only arch ``mlp_fp8`` runs on); the arch is probed
    last."""
    return _fused_dense_mlp_requested(args) and _mxfp8_recipe(args) and is_gfx1250()


def _should_enable(mlp, args, is_expert: bool) -> bool:
    """Whether a constructed Megatron MLP may take ``mlp_fp8``: switch on under the MXFP8 recipe on gfx1250 with
    Turbo linears, not an expert, SwiGLU without bias, clamp or GLU offset (the op's math)."""
    if is_expert or not _requested_here(args):
        return False
    if not getattr(args, "use_turbo_gemm", False) or _mlp_fp8_op() is None:
        return False
    cfg = mlp.config
    if not getattr(cfg, "gated_linear_unit", False) or getattr(cfg, "activation_func", None) is not F.silu:
        return False
    if getattr(cfg, "add_bias_linear", False) or getattr(cfg, "use_te_activation_func", False):
        return False
    if (
        getattr(cfg, "glu_linear_offset", 0.0)
        or getattr(cfg, "activation_func_clamp_value", None) is not None
    ):
        return False
    fc1_types, fc2_types = _turbo_linear_types()
    return isinstance(mlp.linear_fc1, fc1_types) and isinstance(mlp.linear_fc2, fc2_types)


def _routes_here(mlp) -> bool:
    """Whether this forward takes ``mlp_fp8`` (the extension decides: both linears under the same Turbo MXFP8
    recipe here)."""
    return _ext()._dense_mlp_routes_to_mlp_fp8(mlp)


def _forward_turbo_dense_mlp_mxfp8(mlp, hidden_states):
    """``MLP.forward`` through ``mlp_fp8``: the linears' pre-hooks and fused norm, their weight caches and wgrad
    accumulation, one op for fc1 -> SwiGLU -> fc2, inside fc1's quant context (``_routes_here`` checked fc2's).
    """
    ext = _ext()
    fc1, fc2 = mlp.linear_fc1, mlp.linear_fc2
    w1, w2 = fc1._parameters["weight"], fc2._parameters["weight"]
    # DistOpt param-gather overlap in registration order: wait fc1, dispatch fc2, wait fc2, dispatch the next layer
    _run_module_forward_pre_hooks(fc1, (hidden_states,))
    _run_module_forward_pre_hooks(fc2, (hidden_states,))
    carrier = hidden_states  # a pre-MLP RMSNorm's output may carry its MXFP8 pair (PRIMUS_MXFP8_FUSED_NORM=mlp)
    hidden_states = _apply_fc1_fused_norm(fc1, hidden_states, mlp.config)
    leading = hidden_states.shape[:-1]
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    with ext._get_fp8_autocast_for_quant_params(fc1.te_quant_params, fc1.training):
        fused_pair = ext._take_fused_mxfp8_activation(carrier)
        quant = ext.PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        pattern = ext._fuse_wgrad_accum_pattern(fc1.config, w1)
        if ext._fuse_wgrad_accum_pattern(fc2.config, w2) != pattern:
            pattern = None  # the op takes one pattern for both weights
        if ext.get_num_microbatches() == 1:
            weights = [w1, w2]
        else:
            weights = []
            for fc, w in ((fc1, w1), (fc2, w2)):
                if fc.is_first_microbatch:
                    fc.quantized_weight_buffer, fc.quantized_weight_t_buffer = (
                        ext._maybe_create_quantized_weight_buffers(
                            w,
                            ext.float8_e4m3,
                            quant,
                            disable_parameter_transpose_cache=fc.disable_parameter_transpose_cache,
                        )
                    )
                pair = ext.PrimusTurboQuantizedTensorPair(
                    data=fc.quantized_weight_buffer, data_t=fc.quantized_weight_t_buffer
                )
                x, pair = ext._bridge_weight_grad(x, w, pair, fuse_wgrad_accum=pattern is not None)
                weights.append(pair)
        y = _mlp_fp8_op()(
            ext._mxfp8_gemm_input(x, fused_pair, w1),
            weights[0],
            weights[1],
            trans_w1=True,
            trans_w2=True,
            out_dtype=hidden_states.dtype,
            config=quant.data(),
            fuse_wgrad_accum_pattern=pattern,
            activation="silu",
        )
    fc1.is_first_microbatch = False
    fc2.is_first_microbatch = False
    return y.view(*leading, y.shape[-1]), None


def _install_dense_mlp_mxfp8_patch() -> None:
    from megatron.core.transformer.mlp import MLP
    from megatron.training import get_args as _get_args

    if is_patched(MLP, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP already patched; skipping.")
        return

    orig_init = MLP.__init__
    orig_forward = MLP.forward
    orig_backward_dw = MLP.backward_dw

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        is_expert = kwargs.get("is_expert", args[2] if len(args) > 2 else False)
        self._turbo_dense_mlp_mxfp8 = _should_enable(self, _get_args(), bool(is_expert))
        self._turbo_dense_mlp_mxfp8_routed = False

    def _forward(self, hidden_states, per_token_scale=None, **kwargs):
        routed = (
            per_token_scale is None and getattr(self, "_turbo_dense_mlp_mxfp8", False) and _routes_here(self)
        )
        self._turbo_dense_mlp_mxfp8_routed = routed
        if routed:
            return _forward_turbo_dense_mlp_mxfp8(self, hidden_states)
        return orig_forward(self, hidden_states, per_token_scale, **kwargs)

    def _backward_dw(self):
        if getattr(self, "_turbo_dense_mlp_mxfp8_routed", False):
            return  # mlp_fp8 computed (or accumulated) both wgrads in its backward
        return orig_backward_dw(self)

    MLP.__init__ = _init
    MLP.forward = _forward
    MLP.backward_dw = _backward_dw
    mark_patched(MLP, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] MLP.forward routes the non-expert SwiGLU MLPs on Turbo MXFP8 linears through "
        "primus_turbo.ops.mlp_fp8 when turbo_fused_gemm is set."
    )


def _patch_condition(ctx: PatchContext) -> bool:
    return _requested_here(get_args(ctx))


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Route the non-expert SwiGLU MLPs (DeepSeek-V3 shared experts) through Primus-Turbo's MXFP8 mlp_fp8 "
        "(fused GLU / dGLU dual casts, cached weights, wgrad into main_grad) when turbo_fused_gemm is set (gfx1250)."
    ),
    priority=60,
    condition=_patch_condition,
)
def patch_turbo_dense_mlp_mxfp8(ctx: PatchContext) -> None:
    _install_dense_mlp_mxfp8_patch()
