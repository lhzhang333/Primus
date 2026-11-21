###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "runner" / "helpers" / "patches" / "13_triton_gfx1250_expert_scheduling.sh"

GATE = "def is_expert_scheduling_enabled(arch):\n"
STOCK = GATE + '    return arch in ["gfx1250"]\n'
PATCHED = GATE + "    return False\n" + '    return arch in ["gfx1250"]\n'


@pytest.fixture
def compiler(tmp_path):
    """backends/amd/compiler.py of a stand-in Triton that the patch's python3 imports."""
    amd = tmp_path / "triton" / "backends" / "amd"
    amd.mkdir(parents=True)
    (tmp_path / "triton" / "__init__.py").touch()
    path = amd / "compiler.py"
    path.write_text(STOCK)
    return path


def run_patch(compiler, **extra_env):
    env = {k: v for k, v in os.environ.items() if "PATCH_TRITON_EXPERT_SCHEDULING" not in k}
    env.update(PYTHONPATH=str(compiler.parents[3]), PYTHONDONTWRITEBYTECODE="1", **extra_env)
    return subprocess.run(["/bin/bash", str(PATCH)], env=env, text=True, capture_output=True)


def test_patches_once(compiler):
    first = run_patch(compiler)
    assert first.returncode == 0, first.stderr
    assert first.stdout == f"patched {compiler}: is_expert_scheduling_enabled() -> False\n"
    assert compiler.read_text() == PATCHED
    # 2 = nothing to do, which runner/helpers/execute_patches.sh takes as a skip
    second = run_patch(compiler)
    assert (second.returncode, second.stdout, second.stderr) == (2, "", "")
    assert compiler.read_text() == PATCHED


@pytest.mark.parametrize("name", ["PATCH_TRITON_EXPERT_SCHEDULING", "PRIMUS_PATCH_TRITON_EXPERT_SCHEDULING"])
def test_switched_off(compiler, name):
    result = run_patch(compiler, **{name: "0"})
    assert (result.returncode, result.stdout, result.stderr) == (2, "", "")
    assert compiler.read_text() == STOCK


def test_a_triton_without_the_gate_is_left_alone(compiler):
    compiler.write_text("def is_fpsan_supported(arch):\n    return True\n")
    result = run_patch(compiler)
    assert result.returncode == 2
    assert result.stderr == f"no is_expert_scheduling_enabled(arch) in {compiler}; Triton not patched\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_cannot_patch_without_write_access(compiler):
    compiler.parent.chmod(0o555)
    try:
        result = run_patch(compiler)
    finally:
        compiler.parent.chmod(0o755)
    assert result.returncode == 1
    assert result.stderr.startswith(f"cannot patch {compiler} as uid {os.geteuid()}; run as root or set ")
    assert compiler.read_text() == STOCK
