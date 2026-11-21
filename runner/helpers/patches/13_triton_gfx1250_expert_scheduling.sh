#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script, also run by the MI455X launch scripts under
# examples/megatron/models/: make the installed Triton keep AMDGPU expert
# scheduling off.
#
# Why: Triton 3.8 turns expert scheduling on for gfx1250 (and for no other
# architecture), which makes Triton kernels fault or return NaN at random.
# TRITON_HIP_USE_EXPERT_SCHEDULING=0 (the default in runner/helpers/envs/MI455X.sh)
# turns it off whenever Triton compiles a kernel, but Inductor loads kernels from
# its own cache without compiling them. Making is_expert_scheduling_enabled()
# return False changes Triton's default, which also invalidates that cache.
#
# The edit stays in the container. It needs write access to Triton's files (root
# in the stock images).
#
# Controlled by PATCH_TRITON_EXPERT_SCHEDULING, or PRIMUS_PATCH_TRITON_EXPERT_SCHEDULING
# (the PRIMUS_ prefix auto-forwards into the container, see primus-cli-container.sh
# env passthrough): 1, the default, patches; anything else skips.
#
# Exit codes, per runner/helpers/execute_patches.sh: 0 patched; 2 nothing to do
# (switched off, already patched, or this Triton has no such function); 1 cannot
# patch.
###############################################################################
set -euo pipefail

if [[ "${PRIMUS_PATCH_TRITON_EXPERT_SCHEDULING:-${PATCH_TRITON_EXPERT_SCHEDULING:-1}}" != 1 ]]; then
    exit 2
fi

TRITON_AMD_COMPILER=$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py"))')
if ! grep -q '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}"; then
    echo "no is_expert_scheduling_enabled(arch) in ${TRITON_AMD_COMPILER}; Triton not patched" >&2
    exit 2
fi
if grep -A1 '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}" | grep -q '^    return False$'; then
    exit 2
fi
if [[ ! -w "$(dirname "${TRITON_AMD_COMPILER}")" ]]; then
    echo "cannot patch ${TRITON_AMD_COMPILER} as uid $(id -u); run as root or set PATCH_TRITON_EXPERT_SCHEDULING=0" >&2
    exit 1
fi
sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${TRITON_AMD_COMPILER}"
echo "patched ${TRITON_AMD_COMPILER}: is_expert_scheduling_enabled() -> False"
