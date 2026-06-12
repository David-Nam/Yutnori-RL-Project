#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"${ROOT_DIR}/.venv/bin/python"}"

PROFILE="${PROFILE:-primary}"
OPPONENT="${OPPONENT:-project_rf_rule}"
SEEDS="${SEEDS:-0 1 2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
TIMESTEPS_LABEL="${TIMESTEPS_LABEL:-10m}"
N_ENVS="${N_ENVS:-16}"
VEC_ENV="${VEC_ENV:-dummy}"
DEVICE="${DEVICE:-cuda}"
TRAIN_EVAL_EPISODES="${TRAIN_EVAL_EPISODES:-100}"
OFFICIAL_EVAL_EPISODES="${OFFICIAL_EVAL_EPISODES:-5000}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.60}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000000}"
RUNS_ROOT="${RUNS_ROOT:-runs/ppo_step14_long_training_v2}"
LOGS_ROOT="${LOGS_ROOT:-logs/ppo_step14_long_training_v2}"

read -r -a SEED_ARGS <<< "${SEEDS}"
EXTRA_ARGS=("$@")

has_flag() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" ]]; then
      return 0
    fi
  done
  return 1
}

DRY_RUN=0
if has_flag "--dry-run" "${EXTRA_ARGS[@]}"; then
  DRY_RUN=1
fi

OVERWRITE_EVAL_REQUESTED="${OVERWRITE_EVAL:-0}"
if has_flag "--overwrite" "${EXTRA_ARGS[@]}"; then
  OVERWRITE_EVAL_REQUESTED=1
fi

run_combo() {
  local observation_mode="$1"
  local reward_mode="$2"
  shift 2

  printf '\n[%s] Step 14 training: observation=%s reward=%s\n' \
    "$(date -Is)" "${observation_mode}" "${reward_mode}"

  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_ppo_long_sweep.py" \
    --opponents "${OPPONENT}" \
    --seeds "${SEED_ARGS[@]}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --timesteps-label "${TIMESTEPS_LABEL}" \
    --n-envs "${N_ENVS}" \
    --vec-env "${VEC_ENV}" \
    --device "${DEVICE}" \
    --train-eval-episodes "${TRAIN_EVAL_EPISODES}" \
    --final-eval-episodes 0 \
    --skip-final-eval \
    --checkpoint-freq "${CHECKPOINT_FREQ}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOGS_ROOT}" \
    --observation-mode "${observation_mode}" \
    --reward-mode "${reward_mode}" \
    "$@"

  run_official_eval "${observation_mode}" "${reward_mode}"
}

run_official_eval() {
  local observation_mode="$1"
  local reward_mode="$2"
  local seed

  for seed in "${SEED_ARGS[@]}"; do
    local run_name
    run_name="$(step14_run_name "${observation_mode}" "${reward_mode}" "${seed}")"
    local run_dir="${RUNS_ROOT}/${run_name}"
    local model_path="${run_dir}/model.zip"
    local output_path="${run_dir}/eval_project_rf_rule_official_${OFFICIAL_EVAL_EPISODES}.json"
    local eval_seed=$((seed + 100000))

    if [[ "${DRY_RUN}" == "0" && ! -f "${model_path}" ]]; then
      echo "cannot run official eval without model: ${model_path}" >&2
      exit 1
    fi
    if [[ "${DRY_RUN}" == "0" && -f "${output_path}" && "${OVERWRITE_EVAL_REQUESTED}" != "1" ]]; then
      echo "skip official eval: existing output ${output_path}"
      continue
    fi

    local command=(
      "${PYTHON_BIN}"
      "${ROOT_DIR}/scripts/evaluate_rf_target.py"
      --model-path "${model_path}"
      --episodes "${OFFICIAL_EVAL_EPISODES}"
      --seed "${eval_seed}"
      --device "${DEVICE}"
      --observation-mode "${observation_mode}"
      --reward-mode "${reward_mode}"
      --pass-threshold "${PASS_THRESHOLD}"
      --output "${output_path}"
    )

    printf '\n[%s] Step 14 official eval: %s\n' "$(date -Is)" "${run_name}"
    print_command "${command[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      "${command[@]}"
    fi
  done
}

step14_run_name() {
  local observation_mode="$1"
  local reward_mode="$2"
  local seed="$3"
  local suffix=""

  if [[ "${observation_mode}" != "base" ]]; then
    suffix="${suffix}_${observation_mode}"
  fi
  if [[ "${reward_mode}" != "terminal" ]]; then
    suffix="${suffix}_${reward_mode}"
  fi

  printf '%s_seed_%s_%s_nenv%s%s' \
    "${OPPONENT}" "${seed}" "${TIMESTEPS_LABEL}" "${N_ENVS}" "${suffix}"
}

print_command() {
  printf '$'
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

cd "${ROOT_DIR}"

case "${PROFILE}" in
  primary)
    run_combo tactical terminal "${EXTRA_ARGS[@]}"
    ;;
  comparison)
    run_combo tactical rf_shaped "${EXTRA_ARGS[@]}"
    ;;
  both)
    run_combo tactical terminal "${EXTRA_ARGS[@]}"
    run_combo tactical rf_shaped "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "PROFILE must be one of: primary, comparison, both" >&2
    exit 2
    ;;
esac

printf '\n[%s] Step 14 long training finished\n' "$(date -Is)"
