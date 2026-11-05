#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Kimi K3.1 Pro 4-layer proxy (KDA x 3 + NoPE Gated MLA x 1, official widths) pretraining on MI455X.
#
#   bash examples/megatron/models/kimi-k3/run_kimi_k3.1_pro_4L_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/kimi-k3/run_kimi_k3.1_pro_4L_mi455x.sh  # in a container
#
# One node of 4 GPUs at EP 4 (896 experts, 224 per GPU) by default. One GPU with the same per-GPU shape:
#   GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_NUM_EXPERTS=224 GBS=16
#
# MLA core attention: USE_TURBO_ATTENTION=False runs TE's DotProductAttention (the yaml's validated setting;
# on gfx1250 TE has no fused backend for qk 192 / v 128 and runs its unfused path), True runs Primus-Turbo's
# flash attention: the gfx1250 FlyDSL MLA kernels, or with PRIMUS_TURBO_MLA_FWD_IMPL / PRIMUS_TURBO_MLA_BWD_IMPL
# = hk the HipKittens ones (Primus-Turbo checkouts that have primus_turbo/hipkittens/gfx1250_mla).
#
# Settings are the environment variables below. Extra arguments are appended to the Primus command line.
###############################################################################
set -euo pipefail

SCRIPT=$(realpath "$0")
# relative to the caller's directory
if [[ -n "${TURBO_DIR:-}" ]]; then
    TURBO_DIR=$(realpath -e "${TURBO_DIR}")
fi
cd "$(dirname "${SCRIPT}")/../../../.."

# container: re-run this script in a new container of DOCKER_IMAGE; direct: run in this container
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261019}
export EXP=${EXP:-examples/megatron/configs/MI455X/kimi_k3.1_pro_4L-BF16-pretrain.yaml}
# Primus-Turbo checkout built in place for DOCKER_IMAGE (deepseek-v3/README.md); empty: the image's primus_turbo
export TURBO_DIR=${TURBO_DIR:-}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-2}
export GBS=${GBS:-64}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_EP=${PRIMUS_EP:-4}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-896}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
# Primus-Turbo flash-attention backend (the MLA kernels run under FLYDSL)
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
# gfx1250 MLA forward / backward kernels: flydsl or hk
export PRIMUS_TURBO_MLA_FWD_IMPL=${PRIMUS_TURBO_MLA_FWD_IMPL:-flydsl}
export PRIMUS_TURBO_MLA_BWD_IMPL=${PRIMUS_TURBO_MLA_BWD_IMPL:-flydsl}
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
export ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}
# 1: switch Triton's gfx1250 expert scheduling off in the container's Triton (needs root): fla's KDA kernels
# are Triton, see deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
export PATCH_TRITON_EXPERT_SCHEDULING=${PATCH_TRITON_EXPERT_SCHEDULING:-1}

export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(id -un)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-kimi_k3.1_pro_4L_${GPUS_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)}

case "${PRIMUS_LAUNCHER}" in
    container)
        ENV_ARGS=(-e PRIMUS_LAUNCHER=direct)
        while IFS= read -r v; do
            case "${v}" in
                PRIMUS_LAUNCHER) ;;
                EXP | TURBO_DIR | GPUS_PER_NODE | HIP_VISIBLE_DEVICES | TRAIN_ITERS | MBS | GBS | USE_TURBO_* | \
                    MOCK_DATA | PROFILE | PATCH_TRITON_EXPERT_SCHEDULING | MASTER_ADDR | MASTER_PORT | \
                    ENABLE_NUMA_BINDING | GPU_MAX_HW_QUEUES | HSA_* | NVTE_* | PRIMUS_* | NCCL_* | RCCL_* | GLOO_*)
                    ENV_ARGS+=(-e "${v}")
                    ;;
            esac
        done < <(compgen -e)
        DOCKER_ARGS=(--device /dev/kfd --device /dev/dri -v "${PWD}:${PWD}")
        if [[ -n "${TURBO_DIR}" ]]; then
            DOCKER_ARGS+=(-v "${TURBO_DIR}:${TURBO_DIR}")
        fi
        exec docker run --rm --name "kimi-k3-mi455x-$$" \
            --ipc host --network host --privileged --group-add video \
            --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN --security-opt seccomp=unconfined \
            --ulimit nofile=1048576:1048576 "${DOCKER_ARGS[@]}" -w "${PWD}" "${ENV_ARGS[@]}" \
            "${DOCKER_IMAGE}" bash "${SCRIPT}" "$@"
        ;;
    direct) ;;
    *)
        echo "PRIMUS_LAUNCHER must be container or direct, not '${PRIMUS_LAUNCHER}'" >&2
        exit 2
        ;;
esac

if [[ "${PATCH_TRITON_EXPERT_SCHEDULING}" == 1 ]]; then
    TRITON_AMD_COMPILER=$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py"))')
    if ! grep -q '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}"; then
        echo "no is_expert_scheduling_enabled(arch) in ${TRITON_AMD_COMPILER}; Triton not patched" >&2
    elif ! grep -A1 '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}" | grep -q '^    return False$'; then
        if [[ ! -w "$(dirname "${TRITON_AMD_COMPILER}")" ]]; then
            echo "cannot patch ${TRITON_AMD_COMPILER} as uid $(id -u); run as root or set PATCH_TRITON_EXPERT_SCHEDULING=0" >&2
            exit 2
        fi
        sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${TRITON_AMD_COMPILER}"
        echo "patched ${TRITON_AMD_COMPILER}: is_expert_scheduling_enabled() -> False"
    fi
fi

ENV_ARGS=(
    --env PRIMUS_GPU_MODEL=MI455X
    --env NNODES=1
    --env GPUS_PER_NODE="${GPUS_PER_NODE}"
    --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0
    --env PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND}"
    --env PRIMUS_TURBO_MLA_FWD_IMPL="${PRIMUS_TURBO_MLA_FWD_IMPL}"
    --env PRIMUS_TURBO_MLA_BWD_IMPL="${PRIMUS_TURBO_MLA_BWD_IMPL}"
    --env HSA_COREDUMP_PATTERN=/dev/null
)
if [[ -n "${TURBO_DIR}" ]]; then
    # --env is applied after Primus builds PYTHONPATH, so this checkout comes first.
    ENV_ARGS+=(
        --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM"
        --env PYTHONDONTWRITEBYTECODE=1
    )
fi

RUNNER_ARGS=()
if [[ "${ENABLE_NUMA_BINDING}" == 1 ]]; then
    RUNNER_ARGS+=(--numa)
fi

LOG_DIR="${PRIMUS_WORKSPACE:-output}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOG_DIR}"
./primus-cli direct "${RUNNER_ARGS[@]}" "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --num_experts "${PRIMUS_NUM_EXPERTS}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
