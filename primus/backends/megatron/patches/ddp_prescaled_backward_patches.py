###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Scale the backward by 1/DP at its seed instead of every gradient bucket before its reduce.

Without ``average_in_collective`` Megatron's DDP multiplies each bucket by
``gradient_scaling_factor`` (1/DP, for expert and dense buckets alike) right before the
reduce-scatter: a read and a write of every gradient, every step. With the NaN check on, Primus'
``fused_grad_check_scaling`` already folds that multiply into the pass that takes the bucket norm;
on the 4-layer DeepSeek-V3 proxy (EP4) a rank still reads and writes ~17 GB of bf16 gradients
there, ~4.1 ms of a ~935 ms step.

Every gradient is linear in the loss the backward starts from. So the patch moves the 1/DP into
that seed, ``optimizer.scale_loss``, which Megatron applies to the last stage's loss and to the
MoE / MTP auxiliary-loss scales (``config.grad_scale_func``), and sets every bucket's factor to 1:
the norm pass becomes read-only and the scaling loop skips the bucket.

- 1/DP must be a power of two: scaling by it is exact in floating point, and rounding commutes
  with it, so every gradient comes out bit for bit the old one times 1/DP (outside the subnormal
  range) and the reduced gradients, the grad norm, the clipping and the update are unchanged.
- The NaN / Inf check still runs on every bucket before its reduce, with the same calls and
  messages; the norms it takes are the old ones times 1/DP, which does not change finiteness.
- Only when both factors are the same 1/DP (not ``average_in_collective``, not per-token loss), no
  fp16 training (whose grad scaler unscales) and no separate MoE / MTP loss-scale function is set;
  otherwise nothing changes. ``get_loss_scale`` (logged as "loss scale") is left alone.
- On gfx1250 only, where it was validated; ``prescale_backward_by_dp: false`` turns it off there too.
"""

import math

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.ddp.prescaled_backward"
# 1/DP of the DDP modules built with their bucket factors folded into the backward seed
_STATE = {"prescale": None}


def _power_of_two(x: float) -> bool:
    if not x > 0.0:
        return False
    mantissa, _ = math.frexp(x)
    return mantissa == 0.5


def _bucket_factors(ddp):
    for buffer in ddp.buffers + ddp.expert_parallel_buffers:
        yield buffer.gradient_scaling_factor
        for bucket in buffer.buckets:
            yield bucket.gradient_scaling_factor


def fold_scaling_into_seed(ddp) -> bool:
    """Set ``ddp``'s bucket factors to 1 and record their common 1/DP for ``scale_loss``."""
    factors = set(_bucket_factors(ddp))
    if len(factors) != 1:
        return False
    (factor,) = factors
    if factor == 1.0 or not _power_of_two(factor):
        return False
    prescale = _STATE["prescale"]
    if prescale is not None and prescale != factor:
        return False
    for buffer in ddp.buffers + ddp.expert_parallel_buffers:
        buffer.gradient_scaling_factor = 1.0
        for bucket in buffer.buckets:
            bucket.gradient_scaling_factor = 1.0
    _STATE["prescale"] = factor
    return True


def _foldable(ddp, config, ddp_config) -> bool:
    return (
        not getattr(config, "fp16", False)
        and not getattr(config, "calculate_per_token_loss", False)
        and not getattr(ddp_config, "average_in_collective", False)
        and getattr(config, "moe_grad_scale_func", None) is None
        and getattr(config, "mtp_grad_scale_func", None) is None
        and hasattr(ddp, "buffers")
        and hasattr(ddp, "expert_parallel_buffers")
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Scale the backward by 1/DP at its seed, not each gradient bucket before its reduce",
    # validated on gfx1250 only; other architectures keep the per-bucket scaling
    condition=lambda ctx: is_gfx1250() and getattr(get_args(ctx), "prescale_backward_by_dp", True),
)
def patch_prescaled_backward(ctx: PatchContext) -> None:
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    if getattr(DistributedDataParallel.__init__, "_primus_prescaled", False):
        return
    original_init = DistributedDataParallel.__init__
    original_scale_loss = MegatronOptimizer.scale_loss

    def __init__(self, config, ddp_config, module, *args, **kwargs):
        original_init(self, config, ddp_config, module, *args, **kwargs)
        if _foldable(self, config, ddp_config) and fold_scaling_into_seed(self):
            log_rank_0(
                f"[Patch:{_PATCH_KEY}] backward seeds scaled by {_STATE['prescale']}; "
                "gradient buckets are no longer scaled before their reduce"
            )

    def scale_loss(self, loss):
        prescale = _STATE["prescale"]
        if prescale is None:
            return original_scale_loss(self, loss)
        if getattr(self, "grad_scaler", None) is not None:
            raise RuntimeError(
                f"[Patch:{_PATCH_KEY}] the gradient buckets were built unscaled, but this optimizer "
                "has a grad scaler; set prescale_backward_by_dp: false"
            )
        return (self.get_loss_scale() * prescale) * loss

    __init__._primus_prescaled = True
    DistributedDataParallel.__init__ = __init__
    MegatronOptimizer.scale_loss = scale_loss
    log_rank_0(f"[Patch:{_PATCH_KEY}] installed (DDP bucket scaling folds into the backward seed)")
