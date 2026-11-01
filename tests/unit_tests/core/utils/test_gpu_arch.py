###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import types

import pytest
import torch

from primus.core.utils import gpu_arch


@pytest.fixture(autouse=True)
def _fresh_cache():
    gpu_arch.is_gfx1250.cache_clear()
    yield
    gpu_arch.is_gfx1250.cache_clear()


@pytest.mark.parametrize(
    "arch, expected",
    [
        ("gfx1250", True),
        ("gfx1250:sramecc+:xnack-", True),
        ("gfx950:sramecc+:xnack-", False),
        ("gfx942", False),
    ],
)
def test_reads_the_device_arch(monkeypatch, arch, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(gcnArchName=arch)
    )
    assert gpu_arch.is_gfx1250() is expected


def test_no_gpu_is_not_gfx1250(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_arch.is_gfx1250() is False


def test_a_failing_probe_is_not_gfx1250(monkeypatch):
    def boom(i):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert gpu_arch.is_gfx1250() is False
