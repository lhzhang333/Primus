###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MLA self-attention fast path: Megatron's fused MLA RoPE folded into Primus-Turbo's MXFP8 attention (hk_mx).

Megatron's apply_rope_fusion path rotates q in place, writes bf16 key / value copies (fused_apply_mla_rope_for_kv) and
hands query / key / value to the core attention, which quantizes them to MXFP8 again. Here the up-projections' outputs
(q, kv) and k_pos_emb go straight to ``primus_turbo.hipkittens.gfx1250_mla_mx.mx.mla_rope_attention``: the rotation runs
inside the quantization, the backward writes the kv gradient in the [k nope | v] layout and the k_pos_emb / q-rope
gradients itself, so no bf16 query / key / value are materialized and Megatron's RoPE kernels do not run.

Taken only for: training-style calls without inference context, packed sequences, CP / TP / SP, recompute / offload
of the attention pieces; YaRN rope with apply_rope_fusion; causal; head dims 128 + 64 / 128; seq len % 256 == 0;
anything else runs Megatron's forward unchanged.
"""

import os

import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.typed_torch import apply_module

_ORIG_FORWARD = None
_MX = None
_STATS = {"fused": 0, "fallback": 0}


def _mx():
    global _MX
    if _MX is None:
        from primus_turbo.hipkittens.gfx1250_mla_mx import mx

        _MX = mx
    return _MX


def _eligible(self, hidden_states, kw):
    cfg = self.config
    if any(
        kw.get(n) is not None
        for n in (
            "key_value_states",
            "inference_context",
            "inference_params",
            "packed_seq_params",
            "rotary_pos_emb",
            "rotary_pos_cos",
            "rotary_pos_sin",
            "attention_bias",
        )
    ):
        return False
    if kw.get("rotary_pos_cos_sin"):
        return False
    if not getattr(cfg, "apply_rope_fusion", False) or getattr(cfg, "rope_type", "") == "rope":
        return False
    if getattr(cfg, "cache_mla_latents", False) or getattr(self, "recompute_up_proj", False):
        return False
    if getattr(self, "checkpoint_core_attention", False):
        return False
    if any(
        getattr(self, n, False) for n in ("offload_qkv_linear", "offload_core_attention", "offload_attn_proj")
    ):
        return False
    if (
        getattr(cfg, "context_parallel_size", 1) != 1
        or getattr(cfg, "tensor_model_parallel_size", 1) != 1
        or getattr(cfg, "sequence_parallel", False)
    ):
        return False
    if self.attn_mask_type != AttnMaskType.causal:
        return False
    if (cfg.qk_head_dim, cfg.qk_pos_emb_head_dim, cfg.v_head_dim) != (128, 64, 128):
        return False
    if hidden_states.ndim != 3 or hidden_states.size(0) % 256 != 0:
        return False
    return _mx().unavailable_reason() is None


def _forward_fused(self, hidden_states):
    cfg = self.config
    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(None, None, hidden_states, cfg, None)
    cos, sin = self.rotary_pos_emb.get_cached_cos_sin(
        rotary_seq_len, dtype=hidden_states.dtype, packed_seq=False
    )
    q_compressed, kv_combined = self._qkv_down_projection(hidden_states)
    kv_compressed, k_pos_emb = torch.split(kv_combined, [cfg.kv_lora_rank, cfg.qk_pos_emb_head_dim], dim=-1)
    if cfg.q_lora_rank is not None:
        q_compressed = apply_module(self.q_layernorm)(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
    else:
        q, _ = self.linear_q_proj(q_compressed)
    kv_compressed = apply_module(self.kv_layernorm)(kv_compressed)
    kv, _ = self.linear_kv_up_proj(kv_compressed)
    q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
    kv = kv.view(*kv.size()[:-1], self.num_attention_heads_per_partition, cfg.qk_head_dim + cfg.v_head_dim)
    k_pos_emb = torch.unsqueeze(k_pos_emb, -2)  # [s, b, 1, 64]
    mx = _mx()
    reason = mx.rope_unsupported_reason(
        q.transpose(0, 1),
        kv.transpose(0, 1),
        k_pos_emb.transpose(0, 1),
        cos.reshape(-1, 64),
        sin.reshape(-1, 64),
        True,
    )
    if reason is not None:
        raise RuntimeError(f"MLA fused-RoPE MX path: {reason}")
    out = mx.mla_rope_attention(q, kv, k_pos_emb, cos, sin, self.softmax_scale)  # [s, b, h, 128]
    core_attn_out = out.view(out.size(0), out.size(1), -1)
    output, bias = apply_module(self.linear_proj)(core_attn_out)
    return output, bias


def _forward(self, hidden_states, attention_mask, *args, **kwargs):
    named = dict(kwargs)
    names = (
        "key_value_states",
        "inference_context",
        "rotary_pos_emb",
        "rotary_pos_cos",
        "rotary_pos_sin",
        "rotary_pos_cos_sin",
        "attention_bias",
        "packed_seq_params",
        "position_ids",
        "sequence_len_offset",
    )
    for n, a in zip(names, args):
        named[n] = a
    if _eligible(self, hidden_states, named):
        _STATS["fused"] += 1
        if _STATS["fused"] == 1:
            print(
                f"[mla_rope_fused_mx] fused-RoPE MXFP8 path active (layer {getattr(self, 'layer_number', '?')})",
                flush=True,
            )
        return _forward_fused(self, hidden_states)
    _STATS["fallback"] += 1
    if _STATS["fallback"] == 1:
        print(
            f"[mla_rope_fused_mx] fallback to Megatron's MLA forward (layer {getattr(self, 'layer_number', '?')})",
            flush=True,
        )
    return _ORIG_FORWARD(self, hidden_states, attention_mask, *args, **kwargs)


def install(mla_module):
    """Wrap MultiLatentAttention.forward (inherited by MLASelfAttention and Primus' subclass)."""
    global _ORIG_FORWARD
    cls = mla_module.MultiLatentAttention
    if getattr(cls.forward, "_primus_mla_rope_fused_mx", False):
        return
    _ORIG_FORWARD = cls.forward
    _forward._primus_mla_rope_fused_mx = True
    cls.forward = _forward


def enabled():
    return os.environ.get("PRIMUS_TURBO_MLA_ROPE_FUSION", "0") == "1"
