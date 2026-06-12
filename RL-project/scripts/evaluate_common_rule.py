"""Evaluate a saved MaskablePPO model with the common paired-seed protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402

from yutnori.core import ACTION_SIZE  # noqa: E402
from yutnori.env import (  # noqa: E402
    OBSERVATION_MODES,
    REWARD_MODES,
    RULESET,
    observation_size,
)
from yutnori.training import (  # noqa: E402
    CommonPolicyEvaluationResult,
    evaluate_common_rule_policy,
    resolve_model_observation_mode,
    resolve_model_reward_mode,
    resolve_model_ruleset,
)

DEFAULT_BASE_SEED_START = 100_000
DEFAULT_BASE_SEED_COUNT = 2_500
DEFAULT_PASS_THRESHOLD = 0.60


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_BASE_SEED_START)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_BASE_SEED_COUNT)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--model-type", default="Pure RL")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=None)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=None)
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    base_seeds = load_base_seeds(
        seed_file=args.seed_file,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    )
    ruleset = resolve_model_ruleset(args.model_path)
    observation_mode = resolve_model_observation_mode(
        args.model_path,
        args.observation_mode,
    )
    reward_mode = resolve_model_reward_mode(
        args.model_path,
        args.reward_mode,
    )
    model = MaskablePPO.load(args.model_path, device=args.device)
    result = evaluate_common_rule_policy(
        model,
        base_seeds=base_seeds,
        observation_mode=observation_mode,
        deterministic=not args.stochastic,
        max_decisions=args.max_decisions,
        show_progress=not args.no_progress_bar,
    )
    payload = build_payload(
        result,
        model_path=args.model_path,
        deterministic=not args.stochastic,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
        pass_threshold=args.pass_threshold,
        training_seed=args.training_seed,
        model_type=args.model_type,
        seed_source=(
            str(args.seed_file)
            if args.seed_file is not None
            else f"range:{args.seed_start}:{args.seed_count}"
        ),
        ruleset=ruleset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result.evaluation_error_count:
        raise SystemExit(2)


def load_base_seeds(
    *,
    seed_file: Path | None,
    seed_start: int,
    seed_count: int,
) -> list[int]:
    if seed_file is None:
        return list(range(seed_start, seed_start + seed_count))
    payload = json.loads(seed_file.read_text())
    if not isinstance(payload, list) or not all(
        isinstance(seed, int) for seed in payload
    ):
        raise ValueError("seed file must contain a JSON array of integers")
    return payload


def build_payload(
    result: CommonPolicyEvaluationResult,
    *,
    model_path: Path,
    deterministic: bool,
    observation_mode: str,
    reward_mode: str,
    pass_threshold: float,
    training_seed: int | None,
    model_type: str,
    seed_source: str,
    ruleset: str = RULESET,
) -> dict[str, Any]:
    valid_official_result = (
        result.completed_games == result.scheduled_games
        and result.evaluation_error_count == 0
    )
    return {
        "model_path": str(model_path),
        "model_type": model_type,
        "training_seed": training_seed,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deterministic": deterministic,
        "ruleset": ruleset,
        "action_size": ACTION_SIZE,
        "observation_size": observation_size(observation_mode),
        "observation_mode": observation_mode,
        "training_reward_mode": reward_mode,
        "seed_source": seed_source,
        "pass_threshold": pass_threshold,
        "valid_official_result": valid_official_result,
        "passed": valid_official_result and result.win_rate >= pass_threshold,
        **result.to_dict(),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if args.seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if args.max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    if not 0.0 <= args.pass_threshold <= 1.0:
        raise ValueError("pass_threshold must be in [0, 1]")


if __name__ == "__main__":
    main()
