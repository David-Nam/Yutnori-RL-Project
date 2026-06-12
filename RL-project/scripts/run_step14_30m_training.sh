#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PROFILE="${PROFILE:-primary}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}"
export TIMESTEPS_LABEL="${TIMESTEPS_LABEL:-30m}"
export N_ENVS="${N_ENVS:-12}"
export VEC_ENV="${VEC_ENV:-subproc}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-3000000}"
export RUNS_ROOT="${RUNS_ROOT:-runs/ppo_step14_30m_subproc}"
export LOGS_ROOT="${LOGS_ROOT:-logs/ppo_step14_30m_subproc}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "${ROOT_DIR}/scripts/run_step14_long_training.sh" "$@"
