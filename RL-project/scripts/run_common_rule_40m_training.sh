#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"${ROOT_DIR}/.venv/bin/python"}"

SEEDS="${SEEDS:-0 1 2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-40000000}"
TIMESTEPS_LABEL="${TIMESTEPS_LABEL:-40m}"
N_ENVS="${N_ENVS:-12}"
VEC_ENV="${VEC_ENV:-subproc}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-4000000}"
RUNS_ROOT="${RUNS_ROOT:-runs/ppo_common_rule_40m_subproc}"
LOGS_ROOT="${LOGS_ROOT:-logs/ppo_common_rule_40m_subproc}"
EVAL_SEED_START="${EVAL_SEED_START:-100000}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-2500}"
EVAL_SEED_FILE="${EVAL_SEED_FILE:-}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.60}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

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

print_command() {
  printf '$'
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

DRY_RUN=0
if has_flag "--dry-run" "${EXTRA_ARGS[@]}"; then
  DRY_RUN=1
fi

OVERWRITE_EVAL=0
if has_flag "--overwrite" "${EXTRA_ARGS[@]}"; then
  OVERWRITE_EVAL=1
fi

cd "${ROOT_DIR}"

printf '\n[%s] Common-rule PPO training started\n' "$(date -Is)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_ppo_long_sweep.py" \
  --opponents common_rule_based \
  --seeds "${SEED_ARGS[@]}" \
  --observation-mode tactical \
  --reward-mode terminal \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --timesteps-label "${TIMESTEPS_LABEL}" \
  --n-envs "${N_ENVS}" \
  --vec-env "${VEC_ENV}" \
  --device "${DEVICE}" \
  --checkpoint-freq "${CHECKPOINT_FREQ}" \
  --train-eval-episodes 0 \
  --final-eval-episodes 0 \
  --skip-final-eval \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  "${EXTRA_ARGS[@]}"

for seed in "${SEED_ARGS[@]}"; do
  run_name="common_rule_based_seed_${seed}_${TIMESTEPS_LABEL}_nenv${N_ENVS}_tactical"
  run_dir="${RUNS_ROOT}/${run_name}"
  model_path="${run_dir}/model.zip"
  if [[ -n "${EVAL_SEED_FILE}" ]]; then
    output_path="${run_dir}/eval_common_rule_paired.json"
    eval_seed_args=(--seed-file "${EVAL_SEED_FILE}")
  else
    output_path="${run_dir}/eval_common_rule_paired_$((EVAL_SEED_COUNT * 2)).json"
    eval_seed_args=(
      --seed-start "${EVAL_SEED_START}"
      --seed-count "${EVAL_SEED_COUNT}"
    )
  fi

  if [[ "${DRY_RUN}" == "0" && ! -f "${model_path}" ]]; then
    echo "cannot evaluate missing model: ${model_path}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" == "0" && -f "${output_path}" && "${OVERWRITE_EVAL}" != "1" ]]; then
    echo "skip common evaluation: existing output ${output_path}"
    continue
  fi

  command=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/scripts/evaluate_common_rule.py"
    --model-path "${model_path}"
    --training-seed "${seed}"
    --device "${DEVICE}"
    "${eval_seed_args[@]}"
    --pass-threshold "${PASS_THRESHOLD}"
    --output "${output_path}"
  )

  printf '\n[%s] Common paired evaluation: %s\n' "$(date -Is)" "${run_name}"
  print_command "${command[@]}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${command[@]}"
  fi
done

printf '\n[%s] Common-rule PPO training and evaluation finished\n' "$(date -Is)"
