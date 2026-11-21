#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# MiniMax-M3 proxy pretraining on one MI455X node.
#
#   bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh  # in a container
#
# One GPU unless GPUS_PER_NODE is set; extra GPUs are data and expert parallel.
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

export EXP=${EXP:-examples/megatron/configs/MI455X/minimax_m3-BF16-pretrain.yaml}

# One GPU until the caller asks for more. Do not default to every device on a
# shared node.
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-1}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
# flydsl or eager (the PyTorch reference)
export PRIMUS_MSA_BACKEND=${PRIMUS_MSA_BACKEND:-flydsl}
# 0: no recomputation. Every proxy shape the launcher allows fits without it (one GPU: 234 of 432 GB),
# and on M3.1-Flash the recomputed first layer is an MSA layer, so recomputing it repeats its
# whole forward (projections, indexer, MSA kernels) inside the backward.
export PRIMUS_RECOMPUTE_NUM_LAYERS=${PRIMUS_RECOMPUTE_NUM_LAYERS:-0}
# Primus-Turbo attention for the three dense layers. Empty: on with the flydsl backend when the
# Primus-Turbo in use has the gfx1250 head-dim-128 kernels (see README.md), off otherwise; decided in
# the container, below. The default backend fails in fmha_bwd here.
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-}
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-}
# Primus-Turbo grouped GEMM for the experts; False: TE's grouped GEMM (one hipBLASLt GEMM per expert)
export USE_TURBO_GROUPED_GEMM=${USE_TURBO_GROUPED_GEMM:-True}
# bf16:flydsl (Primus-Turbo's FlyDSL kernels; Primus raises their 64-expert weight-gradient cap to
# the 128 local experts of one GPU on gfx1250) or default (Primus-Turbo's choice: Triton)
export TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND:-bf16:flydsl}
TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND,,}
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
# hipBLASLt kernels tuned for this example's GEMMs (README.md), unless HIPBLASLT_TUNING_OVERRIDE_FILE
# names another file or is empty (hipBLASLt's own choice)
mi455x_hipblaslt_tuning "$(dirname "${SCRIPT}")/hipblaslt_tuning/minimax_m3-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt"
# 1: Primus's HIP Adam applies the clip coefficient in-kernel and writes the BF16 parameters, so
# the step has no host sync, no separate grad scale and no parameter copy. Megatron then logs no
# grad norm. The other two need the first. The BF16 writeback needs separate FP32 master
# parameters, which the config's precision-aware optimizer (BF16 grads and moments) does not keep,
# so the default is 0.
export PRIMUS_FUSED_ADAM_CLIP=${PRIMUS_FUSED_ADAM_CLIP:-0}
export PRIMUS_FUSED_ADAM_BF16_WRITEBACK=${PRIMUS_FUSED_ADAM_BF16_WRITEBACK:-${PRIMUS_FUSED_ADAM_CLIP}}
export PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=${PRIMUS_STATIC_BF16_LOSS_SCALE_CPU:-${PRIMUS_FUSED_ADAM_CLIP}}

case "${PRIMUS_MSA_BACKEND}" in
    flydsl | eager) ;;
    *)
        echo "PRIMUS_MSA_BACKEND must be flydsl or eager, not '${PRIMUS_MSA_BACKEND}'" >&2
        exit 2
        ;;
esac
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    "" | flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
# FlyDSL's expert weight-gradient GEMM has no beta=1 epilogue, which gradient_accumulation_fusion
# needs; Primus-Turbo would raise at the first backward.
GAF_RE=' --gradient_accumulation_fusion[ =](true|1) '
if [[ "${TURBO_GROUPED_GEMM_BACKEND}" == *flydsl* && " ${*,,} " =~ ${GAF_RE} ]]; then
    echo "--gradient_accumulation_fusion True needs TURBO_GROUPED_GEMM_BACKEND=default," \
        "not '${TURBO_GROUPED_GEMM_BACKEND}'" >&2
    exit 2
fi
if [[ "${PRIMUS_TP}" != 1 ]]; then
    echo "MSA has no tensor parallel; PRIMUS_TP must be 1, not '${PRIMUS_TP}'" >&2
    exit 2
fi
if (( PRIMUS_TOTAL_LAYERS < 4 )); then
    echo "PRIMUS_TOTAL_LAYERS must be >= 4 so the first three layers stay dense (got ${PRIMUS_TOTAL_LAYERS})" >&2
    exit 2
fi
if (( GPUS_PER_NODE % PRIMUS_PP != 0 )); then
    echo "GPUS_PER_NODE=${GPUS_PER_NODE} is not divisible by PRIMUS_PP=${PRIMUS_PP}" >&2
    exit 2
fi
DP=$((GPUS_PER_NODE / PRIMUS_PP))
# expert parallel over the whole data-parallel group unless set
export PRIMUS_EP=${PRIMUS_EP:-${DP}}
if (( DP % PRIMUS_EP != 0 || 128 % PRIMUS_EP != 0 )); then
    echo "PRIMUS_EP=${PRIMUS_EP} must divide the data-parallel size ${DP} and the 128 experts" >&2
    exit 2
fi
export GBS=${GBS:-$((MBS * DP))}
if (( GBS % (MBS * DP) != 0 )); then
    echo "GBS=${GBS} must be divisible by MBS*DP=$((MBS * DP))" >&2
    exit 2
fi
# the yaml's moe_layer_freq, sparse_attention_freq and sparse_disable_index_value
export PRIMUS_LAYER_FREQ="([0]*3+[1]*$((PRIMUS_TOTAL_LAYERS - 3)))"

mi455x_experiment "minimax_m3_${GPUS_PER_NODE}gpu_ep${PRIMUS_EP}_$(date +%Y%m%d_%H%M%S)"

# PRIMUS_LAUNCHER=container: re-run in a container with the settings above, plus the single-node
# variables `primus-cli container` passes through
mi455x_launch minimax-m3-mi455x "${MI455X_FORWARD[@]}" TURBO_GROUPED_GEMM_BACKEND -- "$@"

# The flydsl backend needs Primus-Turbo's gfx1250 head-dim-128 kernels (README.md). Without them the
# first attention call raises ValueError, after the model and optimizer are built; stop here instead.
HAS_FLYDSL_ATTENTION=$(mi455x_turbo_has attention.flydsl.mha)
if [[ "${HAS_FLYDSL_ATTENTION}" == True ]]; then
    USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
    PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
else
    USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
    PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}
    if [[ "${USE_TURBO_ATTENTION,,}" =~ ^(true|1|yes)$ && "${PRIMUS_TURBO_ATTN_BACKEND,,}" == flydsl ]]; then
        TURBO_PKG=$(mi455x_turbo_has --where)
        echo "PRIMUS_TURBO_ATTN_BACKEND=flydsl needs a Primus-Turbo with the gfx1250 FlyDSL attention kernels," \
            "which ${TURBO_PKG:-this Python environment} lacks. Set TURBO_DIR to such a checkout (README.md)" \
            "or PRIMUS_TURBO_ATTN_BACKEND=triton." >&2
        exit 2
    fi
fi
export USE_TURBO_ATTENTION PRIMUS_TURBO_ATTN_BACKEND

# Triton's gfx1250 expert scheduling off in the container's Triton, unless PATCH_TRITON_EXPERT_SCHEDULING=0
mi455x_patch_triton

# ENV_ARGS: the MI455X options of `primus-cli direct`, with TURBO_DIR first on PYTHONPATH when set
# shellcheck disable=SC2119  # no further variables
mi455x_env_args

mi455x_log_dir
./primus-cli direct "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    --expert_model_parallel_size "${PRIMUS_EP}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_grouped_gemm "${USE_TURBO_GROUPED_GEMM}" \
    --turbo_grouped_gemm_backend "${TURBO_GROUPED_GEMM_BACKEND}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
