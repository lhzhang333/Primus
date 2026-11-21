###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import sys

import pytest

from primus.core.utils import turbo_capability
from primus.core.utils.turbo_capability import turbo_has

MHA, MEGA, WGRAD = "attention.flydsl.mha", "mega_moe.bf16", "grouped_gemm.flydsl.bf16.wgrad_accum"


def _write(root, relpath, text=""):
    path = root.joinpath(*relpath.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _package(root, *relpaths):
    """A fake ``primus_turbo`` under ``root``: every directory of ``relpaths`` is a package."""
    _write(root, "primus_turbo/__init__.py")
    for relpath in relpaths:
        parts = relpath.split("/")
        for depth in range(1, len(parts)):
            init = root.joinpath("primus_turbo", *parts[:depth], "__init__.py")
            init.parent.mkdir(parents=True, exist_ok=True)
            init.touch()
        if relpath.endswith(".py"):
            _write(root, "primus_turbo/" + relpath)
        else:
            _write(root, "primus_turbo/" + relpath + "/__init__.py")


# The layout Primus probed before primus_turbo.capabilities existed (Primus-Turbo up to 2026-11).
OLD_LAYOUT = ("flydsl/attention/gfx1250", "flydsl/mega/gfx1250")
OLD_GROUPED_GEMM = "primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py"


@pytest.fixture
def turbo(monkeypatch, tmp_path):
    """Put a fake Primus-Turbo first on sys.path, in place of an installed one."""
    for name in [m for m in sys.modules if m == "primus_turbo" or m.startswith("primus_turbo.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    turbo_has.cache_clear()
    yield tmp_path
    for name in [m for m in sys.modules if m == "primus_turbo" or m.startswith("primus_turbo.")]:
        del sys.modules[name]
    importlib.invalidate_caches()
    turbo_has.cache_clear()


def test_a_turbo_with_capabilities_answers_itself(turbo):
    # the new layout: none of the old directories or modules exist
    _package(turbo, "flydsl/attention/gfx1250/mha")
    _write(
        turbo,
        "primus_turbo/capabilities.py",
        "def has(feature, arch=None):\n"
        f"    return (feature, arch) in {{({MHA!r}, 'gfx1250'), ({WGRAD!r}, 'gfx1250'), ({MEGA!r}, 'gfx950')}}\n",
    )
    assert turbo_has(MHA) is True
    assert turbo_has(WGRAD) is True
    assert turbo_has(MEGA) is False
    assert turbo_has(MEGA, "gfx950") is True
    assert turbo_has("no.such.feature") is False
    assert "primus_turbo.flydsl" not in sys.modules  # asked, not probed


def test_an_older_turbo_is_probed_as_before(turbo):
    _package(turbo, *OLD_LAYOUT, "flydsl/grouped_gemm")
    _write(
        turbo, OLD_GROUPED_GEMM, "def grouped_gemm_bf16_variable_k_accum_supported(*a):\n    return True\n"
    )
    assert turbo_has(MHA) is True
    assert turbo_has(MEGA) is True
    assert turbo_has(WGRAD) is True
    # the old probes were gfx1250 ones, and the other features had none
    assert turbo_has(MEGA, "gfx950") is False
    assert turbo_has("attention.flydsl.mla") is False
    assert turbo_capability.turbo_package_dir() == str(turbo / "primus_turbo")


def test_an_older_turbo_without_the_gfx1250_kernels(turbo):
    _package(turbo, "flydsl/attention", "flydsl/grouped_gemm")
    assert turbo_has(MHA) is False
    assert turbo_has(MEGA) is False
    assert turbo_has(WGRAD) is False


def test_an_older_grouped_gemm_module_without_the_accumulate_gate(turbo):
    _package(turbo, *OLD_LAYOUT, "flydsl/grouped_gemm")
    _write(turbo, OLD_GROUPED_GEMM, "def grouped_gemm_bf16_variable_k_supported(*a):\n    return True\n")
    assert turbo_has(MEGA) is True
    assert turbo_has(WGRAD) is False


class _NoTurbo:
    """A meta path finder for an environment without Primus-Turbo."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name.split(".")[0] == "primus_turbo":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def test_no_turbo(turbo, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [_NoTurbo, *sys.meta_path])
    assert turbo_capability.turbo_package_dir() == ""
    assert turbo_has(MHA) is False
    assert turbo_has(MEGA) is False
    assert turbo_has(WGRAD) is False


def test_cli(turbo, capsys):
    _package(turbo, *OLD_LAYOUT)
    assert turbo_capability.main([MEGA]) == 0
    assert turbo_capability.main(["attention.flydsl.mla"]) == 0
    assert turbo_capability.main(["--where"]) == 0
    assert capsys.readouterr().out.splitlines() == ["True", "False", str(turbo / "primus_turbo")]
    with pytest.raises(SystemExit):
        turbo_capability.main([])
