#!/usr/bin/env bash

set -euo pipefail

# Example:
# DATASET_REPO_ID=YSanYi/songling_kcvla_medicine_boxes \
# DATASET_ROOT=/path/to/local/dataset \
# POLICY_REPO_ID=YSanYi/kcvla_songling_medicine_boxes_v1 \
# OUTPUT_DIR=/root/autodl-tmp/outputs/train/kcvla_songling_medicine_boxes_v1 \
# bash examples/songling_aloha/train_kcvla.sh

DATASET_REPO_ID="${DATASET_REPO_ID:-YSanYi/songling_kcvla_medicine_boxes}"
DATASET_ROOT="${DATASET_ROOT:-}"

PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
POLICY_REPO_ID="${POLICY_REPO_ID:-YSanYi/kcvla_songling_medicine_boxes_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/train/kcvla_songling_medicine_boxes_v1}"
JOB_NAME="${JOB_NAME:-kcvla_songling_medicine_boxes_v1}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
NORMALIZATION_MAPPING="${NORMALIZATION_MAPPING:-{\"ACTION\":\"QUANTILES\",\"STATE\":\"QUANTILES\",\"VISUAL\":\"IDENTITY\"}}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-kcvla_songling_medicine_boxes}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

KEYWORD_MAX_COUNT="${KEYWORD_MAX_COUNT:-8}"
KEYWORD_TEXT_MAX_TOKENS="${KEYWORD_TEXT_MAX_TOKENS:-16}"
COUNTERFACTUAL_ENABLED="${COUNTERFACTUAL_ENABLED:-true}"
COUNTERFACTUAL_AUTO_GENERATE="${COUNTERFACTUAL_AUTO_GENERATE:-true}"
COUNTERFACTUAL_ACTION_MARGIN="${COUNTERFACTUAL_ACTION_MARGIN:-0.05}"
LOSS_CONTRAST_W="${LOSS_CONTRAST_W:-0.1}"
LOSS_COUNTERFACTUAL_W="${LOSS_COUNTERFACTUAL_W:-0.1}"
LOSS_SPARSE_W="${LOSS_SPARSE_W:-0.01}"

cmd=(
  lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--policy.type=kcvla"
  "--policy.pretrained_path=${PRETRAINED_PATH}"
  "--policy.normalization_mapping=${NORMALIZATION_MAPPING}"
  "--policy.device=${DEVICE}"
  "--policy.dtype=${DTYPE}"
  "--policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  "--policy.freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
  "--policy.train_expert_only=${TRAIN_EXPERT_ONLY}"
  "--policy.keyword_max_count=${KEYWORD_MAX_COUNT}"
  "--policy.keyword_text_max_tokens=${KEYWORD_TEXT_MAX_TOKENS}"
  "--policy.counterfactual_enabled=${COUNTERFACTUAL_ENABLED}"
  "--policy.counterfactual_auto_generate=${COUNTERFACTUAL_AUTO_GENERATE}"
  "--policy.counterfactual_action_margin=${COUNTERFACTUAL_ACTION_MARGIN}"
  "--policy.loss_contrast_w=${LOSS_CONTRAST_W}"
  "--policy.loss_counterfactual_w=${LOSS_COUNTERFACTUAL_W}"
  "--policy.loss_sparse_w=${LOSS_SPARSE_W}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--num_workers=${NUM_WORKERS}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.project=${WANDB_PROJECT}"
  "--policy.repo_id=${POLICY_REPO_ID}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=("--dataset.root=${DATASET_ROOT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=("--wandb.entity=${WANDB_ENTITY}")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
