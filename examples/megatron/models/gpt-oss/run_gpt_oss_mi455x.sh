#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# GPT-OSS BF16 pretrain on one MI455X (gfx1250) node, 4 GPUs, mock data.
# Results and known issues are in README.md.
#
#   From the host (starts DOCKER_IMAGE and re-runs this script inside it):
#     MODEL=20B  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#     MODEL=120B bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#   Already inside the container:
#     PRIMUS_LAUNCHER=direct MODEL=20B bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#
# MODEL=20B  is the full 24-layer model, EP4, MBS8 / GBS64.
# MODEL=120B is an 8-of-36-layer proxy, EP4, MBS1 / GBS8, full recompute.
# Extra Megatron overrides go in EXTRA_ARGS, e.g. EXTRA_ARGS="--sink_sliding_window 128".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO}"
# the container re-run and the Triton patch are common to the MI455X launch scripts
# shellcheck source=examples/megatron/models/mi455x_launcher.sh
source "${HERE}/../mi455x_launcher.sh"

export MODEL=${MODEL:-20B}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261019}
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(whoami)}

if [ "${PRIMUS_LAUNCHER}" = "container" ]; then
  echo "[gpt-oss] launching ${MODEL} in ${DOCKER_IMAGE}"
  SCRIPT="${HERE}/$(basename "${BASH_SOURCE[0]}")"
  # re-runs this script in the container with the variables named here; does not return
  mi455x_launch gpt-oss-mi455x MODEL PRIMUS_TEAM PRIMUS_USER PRIMUS_EXP_NAME PRIMUS_EP HIP_VISIBLE_DEVICES \
    GPUS_PER_NODE TRAIN_ITERS MBS GBS NUM_LAYERS CE_IMPL EXTRA_ARGS PATCH_TRITON_EXPERT_SCHEDULING \
    TRITON_HIP_USE_EXPERT_SCHEDULING PRIMUS_TURBO_ATTN_BACKEND HSA_COREDUMP_PATTERN HF_TOKEN \
    HIPBLASLT_TUNING_OVERRIDE_FILE --
fi

########################## gfx1250 runtime settings ##########################
# Selects runner/helpers/envs/MI455X.sh; Primus also maps gfx1250 to MI455X on its own. That file
# defaults TRITON_HIP_USE_EXPERT_SCHEDULING=0 (Triton 3.8 enables amdgpu-expert-scheduling-mode for
# gfx1250, which causes random memory faults / NaN in Turbo Triton kernels) and
# HSA_COREDUMP_PATTERN=/dev/null. The env var alone misses Inductor kernels loaded from an earlier
# run's cache, so Triton's default is also patched below (PATCH_TRITON_EXPERT_SCHEDULING=0 to skip).
export PRIMUS_GPU_MODEL=${PRIMUS_GPU_MODEL:-MI455X}
# Turbo Triton attention, which the baseline in README.md uses.
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}
# hipBLASLt GEMM tuning for the hipBLASLt build in DOCKER_IMAGE (README.md, "GEMM tuning").
# Set HIPBLASLT_TUNING_OVERRIDE_FILE= (empty) for hipBLASLt's default kernel selection.
export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE-${HERE}/hipblaslt_tuning/gpt_oss-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt}
[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ] || unset HIPBLASLT_TUNING_OVERRIDE_FILE

mi455x_patch_triton

################################ Parallelism #################################
export NNODES=1
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=${PRIMUS_EP:-4}

case "${MODEL}" in
  20B)
    EXP=examples/megatron/configs/MI455X/gpt_oss_20B-BF16-pretrain.yaml
    MBS=${MBS:-8}
    GBS=${GBS:-64}
    # native fused CE runs out of pinned host memory at step 1, see the yaml
    CE_IMPL=${CE_IMPL:-te}
    MODEL_ARGS=()
    ;;
  120B)
    EXP=examples/megatron/configs/MI455X/gpt_oss_120B-BF16-pretrain.yaml
    MBS=${MBS:-1}
    GBS=${GBS:-8}
    CE_IMPL=${CE_IMPL:-native}
    MODEL_ARGS=(--num_layers "${NUM_LAYERS:-8}")
    ;;
  *)
    echo "[gpt-oss] unknown MODEL=${MODEL}, expected 20B or 120B" >&2
    exit 1
    ;;
esac

export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-gpt_oss_${MODEL}_mi455x_ep${PRIMUS_EP}_mbs${MBS}_gbs${GBS}}
LOGDIR="${REPO}/output/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"
echo "[gpt-oss] model=${MODEL} gpus=${GPUS_PER_NODE} ep=${PRIMUS_EP} mbs=${MBS} gbs=${GBS} ce=${CE_IMPL} log=${LOGDIR}"
echo "[gpt-oss] hipBLASLt tuning: ${HIPBLASLT_TUNING_OVERRIDE_FILE:-off}"

# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "${EXP}" \
  --train_iters "${TRAIN_ITERS:-10}" --mock_data True \
  --micro_batch_size "${MBS}" --global_batch_size "${GBS}" \
  --cross_entropy_fusion_impl "${CE_IMPL}" \
  --disable_wandb True --disable_tensorboard True \
  "${MODEL_ARGS[@]}" ${EXTRA_ARGS:-} \
  2>&1 | tee "${LOGDIR}/log_node0.txt"
