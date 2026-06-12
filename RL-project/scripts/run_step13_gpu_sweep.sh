#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"${ROOT_DIR}/.venv/bin/python"}"

OPPONENT="${OPPONENT:-project_rf_rule}"
SEEDS="${SEEDS:-0 1 2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
TIMESTEPS_LABEL="${TIMESTEPS_LABEL:-3m}"
N_ENVS="${N_ENVS:-16}"
DEVICE="${DEVICE:-cuda}"
TRAIN_EVAL_EPISODES="${TRAIN_EVAL_EPISODES:-100}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-1000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0}"
RUNS_ROOT="${RUNS_ROOT:-runs/ppo_step13_gpu_sweep_full}"
LOGS_ROOT="${LOGS_ROOT:-logs/ppo_step13_gpu_sweep_full}"

read -r -a SEED_ARGS <<< "${SEEDS}"

run_combo() {
  local observation_mode="$1"
  local reward_mode="$2"
  shift 2

  printf '\n[%s] Step 13 sweep: observation=%s reward=%s\n' \
    "$(date -Is)" "${observation_mode}" "${reward_mode}"

  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_ppo_long_sweep.py" \
    --opponents "${OPPONENT}" \
    --seeds "${SEED_ARGS[@]}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --timesteps-label "${TIMESTEPS_LABEL}" \
    --n-envs "${N_ENVS}" \
    --device "${DEVICE}" \
    --train-eval-episodes "${TRAIN_EVAL_EPISODES}" \
    --final-eval-episodes "${FINAL_EVAL_EPISODES}" \
    --checkpoint-freq "${CHECKPOINT_FREQ}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOGS_ROOT}" \
    --observation-mode "${observation_mode}" \
    --reward-mode "${reward_mode}" \
    "$@"
}

cd "${ROOT_DIR}"

run_combo base terminal "$@"
run_combo base rf_shaped "$@"
run_combo tactical terminal "$@"
run_combo tactical rf_shaped "$@"

printf '\n[%s] Step 13 sweep finished\n' "$(date -Is)"
