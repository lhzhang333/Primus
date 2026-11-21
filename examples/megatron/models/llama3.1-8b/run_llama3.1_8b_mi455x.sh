#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Llama-3.1-8B pretraining on one MI455X node.
#
#   bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh  # in a container
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

export EXP=${EXP:-examples/megatron/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-4}
export GBS=${GBS:-$((MBS * GPUS_PER_NODE))}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
# max_position_embeddings follows the sequence length unless set: Megatron needs it >= seq_length
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-8192}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
# layers to recompute in full, e.g. 0,3; empty: no recompute
export PRIMUS_RECOMPUTE_LAYER_IDS=${PRIMUS_RECOMPUTE_LAYER_IDS:-}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
export USE_TURBO_RMS_NORM=${USE_TURBO_RMS_NORM:-True}
# Primus-Turbo attention backend: flydsl (needs the gfx1250 head-dim-128 kernels, see README.md)
# or triton
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
# hipBLASLt kernels tuned for this example's GEMMs (README.md), unless HIPBLASLT_TUNING_OVERRIDE_FILE
# names another file or is empty (hipBLASLt's own choice)
mi455x_hipblaslt_tuning "$(dirname "${SCRIPT}")/hipblaslt_tuning/llama3.1_8b-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt"

mi455x_experiment "llama3.1_8b_${GPUS_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)"

# PRIMUS_LAUNCHER=container: re-run in a container with the settings above, plus the single-node
# variables `primus-cli container` passes through
mi455x_launch llama3.1-8b-mi455x "${MI455X_FORWARD[@]}" -- "$@"

# The flydsl backend needs Primus-Turbo's gfx1250 head-dim-128 kernels (README.md). Without them the
# first attention call raises ValueError, after the model and optimizer are built; stop here instead.
if [[ "${USE_TURBO_ATTENTION,,}" =~ ^(true|1|yes)$ && "${PRIMUS_TURBO_ATTN_BACKEND,,}" == flydsl ]]; then
    HAS_FLYDSL_ATTENTION=$(mi455x_turbo_has attention.flydsl.mha)
    if [[ "${HAS_FLYDSL_ATTENTION}" != True ]]; then
        TURBO_PKG=$(mi455x_turbo_has --where)
        echo "PRIMUS_TURBO_ATTN_BACKEND=flydsl needs a Primus-Turbo with the gfx1250 FlyDSL attention kernels," \
            "which ${TURBO_PKG:-this Python environment} lacks. Set TURBO_DIR to such a checkout (README.md)" \
            "or PRIMUS_TURBO_ATTN_BACKEND=triton." >&2
        exit 2
    fi
fi

# Triton's gfx1250 expert scheduling off in the container's Triton, unless PATCH_TRITON_EXPERT_SCHEDULING=0
mi455x_patch_triton

# ENV_ARGS: the MI455X options of `primus-cli direct`, with TURBO_DIR first on PYTHONPATH when set
mi455x_env_args PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND}"

if [[ -n "${PRIMUS_RECOMPUTE_LAYER_IDS}" ]]; then
    # as a list: Primus reads a single id as an int
    RECOMPUTE_ARGS=(--recompute_granularity full --recompute_layer_ids "[${PRIMUS_RECOMPUTE_LAYER_IDS//[][]/}]")
else
    RECOMPUTE_ARGS=(--recompute_granularity null --recompute_layer_ids null)
fi

mi455x_log_dir
./primus-cli direct "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    "${RECOMPUTE_ARGS[@]}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_rms_norm "${USE_TURBO_RMS_NORM}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
