#!/usr/bin/env bash
set -euo pipefail

# JiT multinode launcher.
#
# Single-node example:
#   NUM_GPUS=8 IMAGENET_ROOT=/path/to/imagenet ./run_jit_multinode.sh
#
# SLURM multinode example (inside an allocation):
#   AUTO_SRUN=1 NUM_GPUS=8 IMAGENET_ROOT=/path/to/imagenet ./run_jit_multinode.sh
#
# Manual multinode example:
#   # node 0
#   WORLD_SIZE=2 NODE_RANK=0 MASTER_ADDR=node0 MASTER_PORT=29500 NUM_GPUS=8 IMAGENET_ROOT=/path/to/imagenet ./run_jit_multinode.sh
#   # node 1
#   WORLD_SIZE=2 NODE_RANK=1 MASTER_ADDR=node0 MASTER_PORT=29500 NUM_GPUS=8 IMAGENET_ROOT=/path/to/imagenet ./run_jit_multinode.sh
#
# Spectral Forcing is controlled by extra flags forwarded to main_jit.py:
#   ./run_jit_multinode.sh --disable_dct_patchify        # baseline (no SF)
#   ./run_jit_multinode.sh --window_size 32              # linear-SF, DCT window = patch size

usage() {
  cat <<'USAGE'
run_jit_multinode.sh - JiT multinode launcher

Required:
  IMAGENET_ROOT      ImageNet root path (expects train/ under it)

Common env vars:
  DATA_PATH           Dataset path override (supports ImageNet root or train dir)
  DATA_ROOT           Fallback dataset path (supports ImageNet root or train dir)
  IMAGENET_ROOT       ImageNet root path (expects train/ under it)
  MODEL              JiT model name (default: JiT-B/16)
  IMG_SIZE           Image size (default: 256)
  NOISE_SCALE        Training/sample noise scale (default: 1.0)
  PRED_PARAM         x|v|hybrid (default: x)
  HYBRID_X_WEIGHT_MODE  one_minus_t|snr_inverse|snr (default: one_minus_t)
  HYBRID_X_WEIGHT_POWER Weight schedule exponent (default: 1.0)
  HYBRID_X_WEIGHT_MIN   Hybrid x-branch min clamp (default: 0.0)
  HYBRID_X_WEIGHT_MAX   Hybrid x-branch max clamp (default: 1.0)
  NUM_GPUS           GPUs per node (default: auto-detect)
  WORLD_SIZE         Number of nodes (default: SLURM_NNODES or 1)
  NODE_RANK          Node rank (default: RANK/SLURM_NODEID/0)
  MASTER_ADDR        Master node host/IP (default: SLURM first host or local host)
  MASTER_PORT        Master node port (default: 29500)

  WANDB_PROJECT      W&B project (default: jit)
  WANDB_ENTITY       W&B entity/team (optional)
  ENABLE_WANDB       1|0 (default: 1). If 0, forces WANDB_MODE=disabled.
  WANDB_MODE         online|offline|disabled (default: offline)
  WANDB_RUN_NAME     W&B run name (default: jit_b16_256_cfg2.9)
  WANDB_NUM_VIS      Number of sample images to log during eval (default: 16)
  WANDB_API_KEY      Optional; if set, script runs `wandb login` on node rank 0

  OUTPUT_DIR         Output dir (default: output/<WANDB_RUN_NAME>)
  RESUME             Resume dir (default: OUTPUT_DIR)
  ONLINE_EVAL        1|0 run FID/IS eval during training (default: 1)
  EVALUATE_GEN       1|0 eval-only on the resumed checkpoint (default: 0)
  AUTO_SRUN          Auto fan-out with srun under SLURM (default: 1)
  DRY_RUN            Print launch command and exit (default: 0)

Extra flags after the script name are forwarded to main_jit.py, e.g.
  --disable_dct_patchify | --window_size N | --disable_dct_scale_schedule --dct_fixed_scale 1.0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Optionally activate a conda env by exporting CONDA_ENV_PATH=/path/to/env.
# If unset, the script assumes the right Python/torch is already on PATH.
setup_conda() {
  if [[ -z "${CONDA_ENV_PATH:-}" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_PATH}"
    return
  fi
  echo "[WARN] CONDA_ENV_PATH is set but conda is unavailable; continuing without activation." >&2
}
setup_conda

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[ERROR] torchrun not found in PATH. Set CONDA_ENV_PATH or activate an env with PyTorch." >&2
  exit 1
fi

SCRIPT_PATH="${ROOT}/$(basename "${BASH_SOURCE[0]}")"
AUTO_SRUN="${AUTO_SRUN:-1}"
if [[ "${AUTO_SRUN}" == "1" && -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-1}" -gt 1 && -z "${SLURM_PROCID:-}" && -z "${LAUNCHED_WITH_SRUN:-}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    echo "[INFO] Launching with srun across ${SLURM_NNODES} nodes"
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --export=ALL,LAUNCHED_WITH_SRUN=1 "${SCRIPT_PATH}" "$@"
    exit 0
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

MODEL="${MODEL:-JiT-B/16}"
IMG_SIZE="${IMG_SIZE:-256}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
PRED_PARAM="${PRED_PARAM:-x}"
HYBRID_X_WEIGHT_MODE="${HYBRID_X_WEIGHT_MODE:-one_minus_t}"
HYBRID_X_WEIGHT_POWER="${HYBRID_X_WEIGHT_POWER:-1.0}"
HYBRID_X_WEIGHT_MIN="${HYBRID_X_WEIGHT_MIN:-0.0}"
HYBRID_X_WEIGHT_MAX="${HYBRID_X_WEIGHT_MAX:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BLR="${BLR:-5e-5}"
EPOCHS="${EPOCHS:-600}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
GEN_BSZ="${GEN_BSZ:-128}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
CFG="${CFG:-2.9}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"
ONLINE_EVAL="${ONLINE_EVAL:-1}"
EVALUATE_GEN="${EVALUATE_GEN:-0}"

# DATA_PATH -> DATA_ROOT -> IMAGENET_ROOT.
# Accept either ImageNet root (contains train/) or the train directory itself.
DATA_PATH="${DATA_PATH:-${DATA_ROOT:-${IMAGENET_ROOT:-}}}"
if [[ -z "${DATA_PATH}" ]]; then
  echo "[ERROR] Set IMAGENET_ROOT (or DATA_PATH) to your ImageNet directory." >&2
  exit 1
fi
if [[ -d "${DATA_PATH}/train" ]]; then
  IMAGENET_ROOT="${DATA_PATH}"
elif [[ -d "${DATA_PATH}" && "$(basename "${DATA_PATH}")" == "train" ]]; then
  IMAGENET_ROOT="$(dirname "${DATA_PATH}")"
else
  echo "[ERROR] DATA_PATH/IMAGENET_ROOT invalid: ${DATA_PATH}" >&2
  echo "[ERROR] Expected ImageNet root with train/ or a train directory path." >&2
  exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-jit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_NAME_BASE="${RUN_NAME_BASE:-jit_b16_256_cfg2.9}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME_BASE}}"
WANDB_NUM_VIS="${WANDB_NUM_VIS:-16}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

if [[ "${ENABLE_WANDB}" != "1" ]]; then
  WANDB_MODE="disabled"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${WANDB_RUN_NAME}}"
RESUME="${RESUME:-${OUTPUT_DIR}}"

detect_num_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "${visible}" ]]; then
      echo 0
      return
    fi
    IFS=',' read -r -a devs <<< "${visible}"
    echo "${#devs[@]}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
    if [[ -z "${count}" || "${count}" -lt 1 ]]; then
      count=1
    fi
    echo "${count}"
    return
  fi
  echo 1
}

DETECTED_GPUS="$(detect_num_gpus)"
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [[ "${DETECTED_GPUS}" -gt 0 && "${NUM_GPUS}" -gt "${DETECTED_GPUS}" ]]; then
  echo "[WARN] NUM_GPUS=${NUM_GPUS} > visible GPUs=${DETECTED_GPUS}; clamping." >&2
  NUM_GPUS="${DETECTED_GPUS}"
fi

WORLD_SIZE="${WORLD_SIZE:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"

MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "${MASTER_ADDR}" ]]; then
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  else
    MASTER_ADDR="$(hostname -f 2>/dev/null || hostname)"
  fi
fi
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${WANDB_MODE}" != "disabled" && "${NODE_RANK}" == "0" ]]; then
  if [[ -n "${WANDB_API_KEY}" ]]; then
    if command -v wandb >/dev/null 2>&1; then
      wandb login "${WANDB_API_KEY}" >/dev/null 2>&1 || true
    else
      echo "[WARN] wandb CLI not found; skipping wandb login." >&2
    fi
  elif [[ "${WANDB_MODE}" == "online" ]]; then
    echo "[WARN] WANDB_MODE=online but WANDB_API_KEY is empty. Set WANDB_API_KEY or switch to WANDB_MODE=offline." >&2
  fi
fi

echo "[INFO] root=${ROOT}"
echo "[INFO] nnodes=${WORLD_SIZE}, node_rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}, gpus_per_node=${NUM_GPUS}"
echo "[INFO] model=${MODEL}, img_size=${IMG_SIZE}, batch_size=${BATCH_SIZE}, epochs=${EPOCHS}"
echo "[INFO] pred_param=${PRED_PARAM}, hybrid_weight_mode=${HYBRID_X_WEIGHT_MODE}, hybrid_weight_power=${HYBRID_X_WEIGHT_POWER}"
echo "[INFO] imagenet_root=${IMAGENET_ROOT} (from DATA_PATH=${DATA_PATH})"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] wandb: project=${WANDB_PROJECT}, entity=${WANDB_ENTITY:-<none>}, mode=${WANDB_MODE}, run=${WANDB_RUN_NAME}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/jit_node${NODE_RANK}_${STAMP}.log"

CMD=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --nnodes="${WORLD_SIZE}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  main_jit.py
  --model "${MODEL}"
  --proj_dropout "0.0"
  --P_mean "-0.8"
  --P_std "0.8"
  --img_size "${IMG_SIZE}"
  --noise_scale "${NOISE_SCALE}"
  --pred_param "${PRED_PARAM}"
  --hybrid_x_weight_mode "${HYBRID_X_WEIGHT_MODE}"
  --hybrid_x_weight_power "${HYBRID_X_WEIGHT_POWER}"
  --hybrid_x_weight_min "${HYBRID_X_WEIGHT_MIN}"
  --hybrid_x_weight_max "${HYBRID_X_WEIGHT_MAX}"
  --batch_size "${BATCH_SIZE}"
  --blr "${BLR}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --gen_bsz "${GEN_BSZ}"
  --num_images "${NUM_IMAGES}"
  --cfg "${CFG}"
  --interval_min "${INTERVAL_MIN}"
  --interval_max "${INTERVAL_MAX}"
  --output_dir "${OUTPUT_DIR}"
  --resume "${RESUME}"
  --data_path "${IMAGENET_ROOT}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_mode "${WANDB_MODE}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_num_vis "${WANDB_NUM_VIS}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ "${ONLINE_EVAL}" == "1" ]]; then
  CMD+=(--online_eval)
fi

if [[ "${EVALUATE_GEN}" == "1" ]]; then
  CMD+=(--evaluate_gen)
fi

# Forward any extra CLI args to main_jit.py (e.g. --disable_dct_patchify, --window_size N)
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1. Launch command:"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
