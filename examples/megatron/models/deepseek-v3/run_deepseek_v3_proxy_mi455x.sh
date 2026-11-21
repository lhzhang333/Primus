#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# DeepSeek-V3 proxy pretraining on one MI455X node.
#
#   bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh  # in a container
#
# Settings are the environment variables below. Extra arguments are appended
# to the Primus command line.
###############################################################################
set -euo pipefail

# What the MI455X launch scripts share is in ../mi455x_launcher.sh: PRIMUS_LAUNCHER (container: re-run
# this script in a new container of DOCKER_IMAGE; direct: run in this container), TURBO_DIR (a
# Primus-Turbo checkout built in place for DOCKER_IMAGE, see README.md; empty: the image's
# primus_turbo), HIPBLASLT_TUNING_OVERRIDE_FILE and PATCH_TRITON_EXPERT_SCHEDULING.
# shellcheck source=examples/megatron/models/mi455x_launcher.sh
source "$(dirname "$(realpath "$0")")/../mi455x_launcher.sh"
mi455x_init

export EXP=${EXP:-examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-2}
export GBS=${GBS:-64}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-4}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-128}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-8}
# layers to recompute in full, e.g. 0,3; empty: no recompute
export PRIMUS_RECOMPUTE_LAYER_IDS=${PRIMUS_RECOMPUTE_LAYER_IDS:-}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
export USE_TURBO_GROUPED_GEMM=${USE_TURBO_GROUPED_GEMM:-True}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}
# MegaMoE (--use_turbo_mega_moe): True, False, or auto, on when the Primus-Turbo in use has its gfx1250 kernels
export USE_TURBO_MEGA_MOE=${USE_TURBO_MEGA_MOE:-auto}
# Primus-Turbo backends. Attention: flydsl or triton; grouped GEMM: bf16:flydsl or default
# (Primus-Turbo's choice: Triton, or auto-tuning with PRIMUS_TURBO_AUTO_TUNE=1)
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
# lower case: Primus compares the value with "default", Primus-Turbo ignores case
export TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND:-bf16:flydsl}
TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND,,}
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
# Primus overwrites PRIMUS_TURBO_GROUPED_GEMM_BACKEND from TURBO_GROUPED_GEMM_BACKEND unless that is default
if [[ -n "${PRIMUS_TURBO_GROUPED_GEMM_BACKEND:-}" ]]; then
    echo "set TURBO_GROUPED_GEMM_BACKEND instead of PRIMUS_TURBO_GROUPED_GEMM_BACKEND" >&2
    exit 2
fi
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
# 1: bind each rank's CPU threads and host memory to the NUMA node of its GPU (a little faster)
export ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-1}
# 4 hardware queues per process, not base_env.sh's 2 (a little faster)
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}
# hipBLASLt kernels tuned for this example's GEMMs (README.md), unless HIPBLASLT_TUNING_OVERRIDE_FILE
# names another file or is empty (hipBLASLt's own choice)
mi455x_hipblaslt_tuning "$(dirname "${SCRIPT}")/hipblaslt_tuning/deepseek_v3_proxy-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt"

mi455x_experiment "deepseek_v3_proxy_${GPUS_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)"

# PRIMUS_LAUNCHER=container: re-run in a container with the settings above, plus the single-node
# variables `primus-cli container` passes through
mi455x_launch deepseek-v3-mi455x "${MI455X_FORWARD[@]}" TURBO_GROUPED_GEMM_BACKEND -- "$@"

# Triton's gfx1250 expert scheduling off in the container's Triton, unless PATCH_TRITON_EXPERT_SCHEDULING=0
mi455x_patch_triton

# tas-20261019's primus_turbo predates the gfx1250 MegaMoE kernels
if [[ "${USE_TURBO_MEGA_MOE}" == auto ]]; then
    USE_TURBO_MEGA_MOE=$(mi455x_turbo_has mega_moe.bf16)
    echo "USE_TURBO_MEGA_MOE=auto: ${USE_TURBO_MEGA_MOE} (Primus-Turbo at $(mi455x_turbo_has --where))"
fi

# ENV_ARGS: the MI455X options of `primus-cli direct`, with TURBO_DIR first on PYTHONPATH when set
mi455x_env_args PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND}"

# base_env.sh only exports ENABLE_NUMA_BINDING; primus-cli direct binds with --numa
RUNNER_ARGS=()
if [[ "${ENABLE_NUMA_BINDING}" == 1 ]]; then
    RUNNER_ARGS+=(--numa)
fi

if [[ -n "${PRIMUS_RECOMPUTE_LAYER_IDS}" ]]; then
    # as a list: Primus reads a single id as an int
    RECOMPUTE_ARGS=(--recompute_granularity full --recompute_layer_ids "[${PRIMUS_RECOMPUTE_LAYER_IDS//[][]/}]")
else
    RECOMPUTE_ARGS=(--recompute_granularity null --recompute_layer_ids null)
fi

mi455x_log_dir
./primus-cli direct "${RUNNER_ARGS[@]}" "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    --expert_model_parallel_size "${PRIMUS_EP}" \
    --num_layers "${PRIMUS_TOTAL_LAYERS}" \
    --num_experts "${PRIMUS_NUM_EXPERTS}" \
    --moe_router_topk "${PRIMUS_MOE_TOPK}" \
    "${RECOMPUTE_ARGS[@]}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_grouped_gemm "${USE_TURBO_GROUPED_GEMM}" \
    --turbo_grouped_gemm_backend "${TURBO_GROUPED_GEMM_BACKEND}" \
    --use_turbo_deepep "${USE_TURBO_DEEPEP}" \
    --use_turbo_mega_moe "${USE_TURBO_MEGA_MOE}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
