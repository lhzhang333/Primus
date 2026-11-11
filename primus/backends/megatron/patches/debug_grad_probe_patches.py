###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Debug only (PRIMUS_DEBUG_GRAD_PROBE=<jsonl path>): per-parameter checksums of main_grad and of the parameters at
fixed points of the first PRIMUS_DEBUG_GRAD_PROBE_ITERS (default 3) iterations, rank 0, one JSON line per
(probe, parameter):

  Z  right after DDP.zero_grad_buffer()            main_grad (must be all zero)
  B  right before the iteration's first backward   main_grad (still zero) and the parameters the iteration uses
  G  right after the iteration's last backward     main_grad (before finalize_model_grads)
  F  right after finalize_model_grads              main_grad as the optimizer sees it

Each checksum: float64 sum, float64 sum of |x|, and a position-weighted integer hash of the raw bits (bitwise
equality). Inert without the environment variable.
"""

import json
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_DEBUG_GRAD_PROBE"


def _checksums(t):
    import torch

    flat = t.detach().reshape(-1)
    n = flat.numel()
    s = a = 0.0
    h = 0
    chunk = 1 << 26
    ibits = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(flat.dtype)
    for i in range(0, n, chunk):
        c = flat[i : i + chunk]
        cf = c.float()
        s += float(cf.sum(dtype=torch.float64))
        a += float(cf.abs().sum(dtype=torch.float64))
        if ibits is not None:
            iv = c.contiguous().view(ibits).to(torch.int64)
            w = (torch.arange(i, i + c.numel(), device=c.device, dtype=torch.int64) % 65521) + 1
            h = (h + int((iv * w).sum())) % (1 << 61)
    return s, a, h


def install(path: str, iters: int) -> None:
    import torch
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel import schedules
    from megatron.training import training
    from megatron.training.global_vars import get_args

    state = {"model": None, "bw": 0}
    rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    out = open(path, "a") if rank0 else None

    def it():
        return int(getattr(get_args(), "curr_iteration", -1)) + 1  # 1-based iteration being run

    def active():
        return out is not None and state["model"] is not None and 1 <= it() <= iters

    def dump(probe, grads=True, params=False):
        torch.cuda.synchronize()
        mod = state["model"].module
        for name, p in mod.named_parameters():
            rec = {"iter": it(), "probe": probe, "name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
            mg = getattr(p, "main_grad", None)
            if grads:
                if mg is None:
                    rec["main_grad"] = None
                else:
                    rec["main_grad"] = _checksums(mg)
                    rec["mg_dtype"] = str(mg.dtype)
                rec["added"] = bool(getattr(p, "grad_added_to_main_grad", False))
            if params:
                rec["param"] = _checksums(p.data)
            out.write(json.dumps(rec) + "\n")
        out.flush()

    orig_zero = DistributedDataParallel.zero_grad_buffer

    def zero_grad_buffer(self, *args, **kwargs):
        r = orig_zero(self, *args, **kwargs)
        state["model"] = self
        state["bw"] = 0
        if active():
            dump("Z")
        return r

    DistributedDataParallel.zero_grad_buffer = zero_grad_buffer

    orig_bw = schedules.backward_step

    def backward_step(*args, **kwargs):
        state["bw"] += 1
        if state["bw"] == 1 and active():
            dump("B", params=True)
        r = orig_bw(*args, **kwargs)
        if state["bw"] == get_num_microbatches() and active():
            dump("G")
        return r

    schedules.backward_step = backward_step

    orig_fin = training.finalize_model_grads

    def finalize_model_grads(*args, **kwargs):
        r = orig_fin(*args, **kwargs)
        if active():
            dump("F")
        return r

    training.finalize_model_grads = finalize_model_grads
    log_rank_0(f"[Patch:megatron.debug.grad_probe] main_grad / param checksums of iterations 1..{iters} -> {path}")


@register_patch(
    "megatron.debug.grad_probe",
    backend="megatron",
    phase="before_train",
    description="Debug: per-parameter main_grad / param checksums of the first iterations (PRIMUS_DEBUG_GRAD_PROBE)",
    condition=lambda ctx: bool(os.environ.get(_ENV)),
)
def patch_grad_probe(ctx: PatchContext):
    install(os.environ[_ENV], int(os.environ.get(_ENV + "_ITERS", "3")))
