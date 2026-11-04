###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A micro-batch's host-to-device copies on a side stream.

``pretrain_gpt.get_batch`` copies the batch the data loader pinned to the GPU with ``.cuda(non_blocking=True)``
on the compute stream, behind every kernel the previous micro-batch queued. The copies (a few small tensors,
each mostly transfer latency) therefore start only once the previous backward has finished, and the forward
that needs them waits too: a gap on the GPU at every micro-batch boundary, although the host issued the
copies long before. With ``batch_h2d_side_stream: true`` the same copies are issued at the same moment on a
side stream, which is idle and runs them right away; the compute stream waits on that stream before the
forward, by which time they are done.

Only the iterator ``get_batch`` reads from is wrapped, for its ``next()``: what the data iterator yields, and
when, is unchanged (no batch is read ahead), and ``get_batch``'s own ``.cuda()`` finds the tensors already on
the GPU. Batches that are not dicts pass through untouched.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.data.batch_h2d_side_stream"


class SideStreamBatchIterator:
    """``next()`` of ``it``, with the batch's CPU tensors copied to the GPU on ``stream`` and the current stream
    made to wait for the copies."""

    __slots__ = ("_it", "_stream")

    def __init__(self, it, stream):
        self._it = it
        self._stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        import torch

        batch = next(self._it)
        if not isinstance(batch, dict):
            return batch
        current = torch.cuda.current_stream()
        with torch.cuda.stream(self._stream):
            out = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) and not v.is_cuda else v
                for k, v in batch.items()
            }
        current.wait_stream(self._stream)
        for v in out.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                # allocated on the side stream, used on the current one
                v.record_stream(current)
        return out


def make_get_batch(original):
    """``original`` (``pretrain_gpt.get_batch``) reading its batches through a :class:`SideStreamBatchIterator`."""
    import torch

    streams = {}

    def get_batch(data_iterator, *args, **kwargs):
        if data_iterator is not None:
            dev = torch.cuda.current_device()
            stream = streams.get(dev)
            if stream is None:
                stream = streams[dev] = torch.cuda.Stream(device=dev)
            data_iterator = SideStreamBatchIterator(data_iterator, stream)
        return original(data_iterator, *args, **kwargs)

    get_batch._primus_batch_h2d_side_stream = True
    return get_batch


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Copy each micro-batch to the GPU on a side stream instead of behind the previous micro-batch",
    condition=lambda ctx: bool(getattr(get_args(ctx), "batch_h2d_side_stream", False)),
)
def patch_batch_h2d_side_stream(ctx: PatchContext) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    try:
        import pretrain_gpt
    except ImportError:
        log_rank_0(f"[Patch:{_PATCH_KEY}] pretrain_gpt is not importable; skipped")
        return
    original = pretrain_gpt.get_batch
    if getattr(original, "_primus_batch_h2d_side_stream", False):
        return
    pretrain_gpt.get_batch = make_get_batch(original)
    log_rank_0(f"[Patch:{_PATCH_KEY}] pretrain_gpt.get_batch copies each batch to the GPU on a side stream")
