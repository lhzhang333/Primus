###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A micro-batch's host-to-device copies on a side stream (``batch_h2d_side_stream``)."""

import pytest
import torch

from primus.backends.megatron.patches.batch_h2d_side_stream_patches import (
    SideStreamBatchIterator,
    make_get_batch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _batches(n, seq=4096):
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        yield {
            "tokens": torch.randint(0, 129280, (2, seq), generator=g).pin_memory(),
            "labels": torch.randint(0, 129280, (2, seq), generator=g).pin_memory(),
            "loss_mask": torch.rand(2, seq, generator=g).pin_memory(),
            "position_ids": torch.arange(seq).repeat(2, 1).pin_memory(),
            "note": f"batch {i}",
        }


def test_yields_the_same_batches_on_the_gpu_one_for_one():
    ref = list(_batches(4))
    calls = []

    def inner():
        for b in _batches(4):
            calls.append(1)
            yield b

    it = SideStreamBatchIterator(inner(), torch.cuda.Stream())
    for k in range(4):
        out = next(it)
        assert len(calls) == k + 1  # nothing read ahead
        for key, v in ref[k].items():
            if isinstance(v, torch.Tensor):
                assert out[key].is_cuda and torch.equal(out[key].cpu(), v)
            else:
                assert out[key] == v
    with pytest.raises(StopIteration):
        next(it)


def test_current_stream_sees_the_copy_behind_queued_work():
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(20):  # queue work on the current stream
        a = a @ a / 64
    it = SideStreamBatchIterator(_batches(1, seq=1 << 16), stream)
    out = next(it)
    total = out["position_ids"].sum()  # on the current stream, ordered after the copies
    assert total.item() == 2 * sum(range(1 << 16))


def test_non_dict_batches_pass_through():
    it = SideStreamBatchIterator(iter([(1, 2), None]), torch.cuda.Stream())
    assert next(it) == (1, 2)
    assert next(it) is None


def test_get_batch_wrapper():
    seen = []

    def get_batch(data_iterator, vp_stage=None):
        if data_iterator is None:
            return [None] * 4
        b = next(data_iterator)
        seen.append(vp_stage)
        return {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}

    wrapped = make_get_batch(get_batch)
    assert wrapped._primus_batch_h2d_side_stream
    assert wrapped(None) == [None] * 4
    ref = next(_batches(1))
    out = wrapped(_batches(1), vp_stage=3)
    assert seen == [3] and torch.equal(out["tokens"].cpu(), ref["tokens"])
