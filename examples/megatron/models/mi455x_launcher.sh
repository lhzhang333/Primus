#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# What the MI455X launch scripts of the models in this directory share: path
# handling, the re-run in a container, the Primus-Turbo checkout, the hipBLASLt
# tuning file, the Triton patch and the log directory. Sourced, not run:
#
#   source "$(dirname "$(realpath "$0")")/../mi455x_launcher.sh"
#
# A launch script calls the functions in the order they are defined here and
# keeps its model's settings and its `primus-cli direct` command line.
# Platform settings that every MI455X run needs are not here but in
# runner/helpers/envs/MI455X.sh, which primus-cli loads.
###############################################################################

# Variables every launch script forwards into the container (shell patterns), besides the single-node
# ones `primus-cli container` passes through. PRIMUS_LAUNCHER is never forwarded.
# shellcheck disable=SC2034  # read by the launch scripts
MI455X_FORWARD=(
    EXP TURBO_DIR GPUS_PER_NODE HIP_VISIBLE_DEVICES TRAIN_ITERS MBS GBS 'USE_TURBO_*' MOCK_DATA PROFILE
    PATCH_TRITON_EXPERT_SCHEDULING MASTER_ADDR MASTER_PORT HF_TOKEN WANDB_API_KEY ENABLE_NUMA_BINDING
    GPU_MAX_HW_QUEUES 'HSA_*' 'NVTE_*' 'PRIMUS_*' 'NCCL_*' 'RCCL_*' 'GLOO_*' 'IONIC_*' 'HIPBLASLT_*'
)

# mi455x_init: resolve the caller's paths, go to the repository root, set the launcher defaults.
#   SCRIPT                          the launch script
#   PRIMUS_LAUNCHER                 container: re-run the script in a new container of DOCKER_IMAGE;
#                                   direct: run in this container
#   TURBO_DIR                       Primus-Turbo checkout built in place for DOCKER_IMAGE (see the
#                                   model's README.md); empty: the image's primus_turbo
#   PATCH_TRITON_EXPERT_SCHEDULING  1: switch Triton's gfx1250 expert scheduling off in the
#                                   container's Triton (needs root), see mi455x_patch_triton
mi455x_init() {
    SCRIPT=$(realpath "$0")
    # relative to the caller's directory
    if [[ -n "${TURBO_DIR:-}" ]]; then
        TURBO_DIR=$(realpath -e "${TURBO_DIR}")
    fi
    if [[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" ]]; then
        HIPBLASLT_TUNING_OVERRIDE_FILE=$(realpath -e "${HIPBLASLT_TUNING_OVERRIDE_FILE}")
    fi
    cd "$(dirname "${SCRIPT}")/../../../.." || exit

    export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
    export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261109}
    export TURBO_DIR=${TURBO_DIR:-}
    export PATCH_TRITON_EXPERT_SCHEDULING=${PATCH_TRITON_EXPERT_SCHEDULING:-1}
}

# mi455x_hipblaslt_tuning <file>: HIPBLASLT_TUNING_OVERRIDE_FILE defaults to <file>, the hipBLASLt
# kernels tuned for the example's GEMMs (README.md); empty: hipBLASLt's own choice. hipBLASLt uses a
# row only while its solution index still names the kernel it was tuned with. No default with
# PRIMUS_HIPBLASLT_TUNING=1: its stage 3 would take this file instead of the stage-2 result.
mi455x_hipblaslt_tuning() {
    if [[ "${PRIMUS_HIPBLASLT_TUNING:-0}" == 1 ]]; then
        export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE-}
    else
        export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE-$1}
    fi
    if [[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE}" && ! -r "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
        echo "cannot read HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE}" >&2
        exit 2
    fi
}

# mi455x_experiment <name>: where Primus writes this run; <name> is the default PRIMUS_EXP_NAME.
mi455x_experiment() {
    export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
    export PRIMUS_USER=${PRIMUS_USER:-$(id -un)}
    export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-$1}
}

# _mi455x_matches <name> <patterns ...>: whether the name matches one of the shell patterns
_mi455x_matches() {
    local name=$1 pattern
    shift
    for pattern in "$@"; do
        # shellcheck disable=SC2053  # the right-hand side is a pattern
        if [[ "${name}" == ${pattern} ]]; then
            return 0
        fi
    done
    return 1
}

# mi455x_launch <container name> <variable patterns to forward ...> -- <the script's arguments>
#   PRIMUS_LAUNCHER=direct     returns: the script goes on in this container
#   PRIMUS_LAUNCHER=container  re-runs SCRIPT, with the forwarded variables that are exported and
#                              PRIMUS_LAUNCHER=direct, in a new container of DOCKER_IMAGE; no return.
#                              The repository is mounted, and what a forwarded TURBO_DIR and
#                              HIPBLASLT_TUNING_OVERRIDE_FILE name (absolute paths only).
mi455x_launch() {
    local name=$1 forward=() env_args docker_args v
    shift
    while [[ $# -gt 0 && "$1" != -- ]]; do
        forward+=("$1")
        shift
    done
    shift
    case "${PRIMUS_LAUNCHER}" in
        container) ;;
        direct) return 0 ;;
        *)
            echo "PRIMUS_LAUNCHER must be container or direct, not '${PRIMUS_LAUNCHER}'" >&2
            exit 2
            ;;
    esac
    env_args=(-e PRIMUS_LAUNCHER=direct)
    while IFS= read -r v; do
        if [[ "${v}" != PRIMUS_LAUNCHER ]] && _mi455x_matches "${v}" "${forward[@]}"; then
            env_args+=(-e "${v}")
        fi
    done < <(compgen -e)
    docker_args=(--device /dev/kfd --device /dev/dri -v "${PWD}:${PWD}")
    if [[ -e /dev/infiniband ]]; then
        docker_args+=(--device /dev/infiniband)
    fi
    if _mi455x_matches TURBO_DIR "${forward[@]}" && [[ "${TURBO_DIR:-}" == /* && -e "${TURBO_DIR}" ]]; then
        docker_args+=(-v "${TURBO_DIR}:${TURBO_DIR}")
    fi
    if _mi455x_matches HIPBLASLT_TUNING_OVERRIDE_FILE "${forward[@]}" &&
        [[ "${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" == /* && -e "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
        docker_args+=(-v "${HIPBLASLT_TUNING_OVERRIDE_FILE}:${HIPBLASLT_TUNING_OVERRIDE_FILE}:ro")
    fi
    # Same runtime options as `primus-cli container` (runner/.primus.yaml)
    exec docker run --rm --name "${name}-$$" \
        --ipc host --network host --privileged --group-add video \
        --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN --security-opt seccomp=unconfined \
        --ulimit nofile=1048576:1048576 "${docker_args[@]}" -w "${PWD}" "${env_args[@]}" \
        "${DOCKER_IMAGE}" bash "${SCRIPT}" "$@"
}

# mi455x_turbo_has <feature>: True / False, whether the Primus-Turbo this run imports (TURBO_DIR when
# set, else the image's) has the feature, e.g. mega_moe.bf16; --where: its package directory.
mi455x_turbo_has() {
    PYTHONPATH="${TURBO_DIR:+${TURBO_DIR}:}${PYTHONPATH:-}" python3 -m primus.core.utils.turbo_capability "$@"
}

# mi455x_patch_triton: Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes Triton
# kernels fault or return NaN at random. TRITON_HIP_USE_EXPERT_SCHEDULING=0
# (runner/helpers/envs/MI455X.sh, unless the caller set it) turns it off when a kernel is compiled,
# but Inductor loads kernels from its own cache without compiling them; changing Triton's default
# also invalidates that cache. PATCH_TRITON_EXPERT_SCHEDULING=0 skips the patch. The patch script
# exits 2 when there is nothing to do and 1 when it cannot patch (not root).
mi455x_patch_triton() {
    bash runner/helpers/patches/13_triton_gfx1250_expert_scheduling.sh || [[ $? == 2 ]] || exit 2
}

# mi455x_env_args [NAME=value ...]: ENV_ARGS, the `primus-cli direct` options of an MI455X run, with
# the given variables and, with TURBO_DIR, that checkout first on PYTHONPATH.
mi455x_env_args() {
    local kv
    # hipBLASLt takes a set but empty HIPBLASLT_TUNING_OVERRIDE_FILE as an override file named ""
    if [[ -z "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
        unset HIPBLASLT_TUNING_OVERRIDE_FILE
    fi
    ENV_ARGS=(
        --env PRIMUS_GPU_MODEL=MI455X
        --env NNODES=1
        --env GPUS_PER_NODE="${GPUS_PER_NODE}"
        --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    )
    for kv in "$@"; do
        ENV_ARGS+=(--env "${kv}")
    done
    if [[ -n "${TURBO_DIR}" ]]; then
        # --env is applied after Primus builds PYTHONPATH, so this checkout comes first.
        ENV_ARGS+=(
            --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM"
            --env PYTHONDONTWRITEBYTECODE=1
        )
    fi
}

# mi455x_log_dir: LOG_DIR, the experiment's directory, created.
mi455x_log_dir() {
    LOG_DIR="${PRIMUS_WORKSPACE:-output}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
    mkdir -p "${LOG_DIR}"
}
