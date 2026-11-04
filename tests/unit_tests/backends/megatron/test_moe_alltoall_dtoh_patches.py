###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The all-to-all dispatcher's D2H point under ``use_turbo_grouped_gemm``: tokens_per_expert stays on the
device, the splits are copied as before, and the host waits only when a copy was made."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def dtoh(monkeypatch):
    from megatron.core.transformer.moe import token_dispatcher

    from primus.backends.megatron.patches.moe_patches.moe_alltoall_dtoh_patches import (
        patch_moe_alltoall_dtoh,
    )

    cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    monkeypatch.setattr(cls, "_maybe_dtoh_and_synchronize", cls._maybe_dtoh_and_synchronize)
    patch_moe_alltoall_dtoh(None)
    return cls._maybe_dtoh_and_synchronize


class _Event:
    def __init__(self):
        self.synced = 0

    def synchronize(self):
        self.synced += 1


def _dispatcher(**splits):
    stream = torch.cuda.Stream()
    d = SimpleNamespace(
        drop_and_pad=False,
        cuda_dtoh_point="before_permutation_1",
        cuda_sync_point="before_finish",
        cuda_dtoh_stream=stream,
        num_local_experts=32,
        config=SimpleNamespace(moe_permute_fusion=True),
        input_splits=None,
        output_splits=None,
        output_splits_tp=None,
        num_out_tokens=8192 * 8,
        num_global_tokens_per_local_expert=None,
    )
    d.__dict__.update(splits)
    return d


def test_single_gpu_dropless_does_not_wait(dtoh):
    d = _dispatcher()
    tpe = torch.ones(32, dtype=torch.long, device="cuda")
    assert dtoh(d, "before_permutation_1", tpe) is tpe
    assert dtoh(d, "before_finish", tpe) is tpe
    # no copy was made, so no event was recorded and none was waited for
    assert not hasattr(d, "d2h_event") and tpe.is_cuda and d.num_out_tokens == 8192 * 8


def test_device_splits_are_copied_and_waited_for(dtoh):
    splits = torch.tensor([3, 5], device="cuda")
    d = _dispatcher(input_splits=splits, output_splits=splits.clone(), cuda_sync_point="before_ep_alltoall")
    tpe = torch.ones(32, dtype=torch.long, device="cuda")
    assert dtoh(d, "before_permutation_1", tpe) is tpe  # tokens_per_expert stays on the device
    assert not torch.is_tensor(d.input_splits) and list(d.input_splits) == [3, 5]  # numpy on the host
    real = d.d2h_event
    event = _Event()
    d.d2h_event = event
    dtoh(d, "before_ep_alltoall", tpe)
    assert event.synced == 1
    real.synchronize()


def test_unfused_permute_copies_per_expert_counts(dtoh):
    counts = torch.arange(32, device="cuda").view(1, 32)
    d = _dispatcher(config=SimpleNamespace(moe_permute_fusion=False), num_global_tokens_per_local_expert=counts)
    tpe = torch.ones(32, dtype=torch.long, device="cuda")
    dtoh(d, "before_permutation_1", tpe)
    assert not d.num_global_tokens_per_local_expert.is_cuda
    d.d2h_event.synchronize()
