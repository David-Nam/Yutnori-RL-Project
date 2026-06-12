"""Evaluate a saved MaskablePPO policy against the project-RF target."""

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
    PolicyEvaluationResult,
    evaluate_maskable_policy,
    resolve_model_observation_mode,
    resolve_model_reward_mode,
    resolve_model_ruleset,
)

TARGET_OPPONENT = "project_rf_rule"
DEFAULT_EPISODES = 5_000
DEFAULT_PASS_THRESHOLD = 0.60
DEFAULT_SEED = 100_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=None)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

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
    result = evaluate_maskable_policy(
        model,
        opponent=TARGET_OPPONENT,
        episodes=args.episodes,
        seed=args.seed,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
        deterministic=not args.stochastic,
        max_decisions=args.max_decisions,
        show_progress=not args.no_progress_bar,
        progress_desc=f"Evaluate {TARGET_OPPONENT}",
    )
    payload = build_payload(
        result,
        model_path=args.model_path,
        deterministic=not args.stochastic,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
        pass_threshold=args.pass_threshold,
        ruleset=ruleset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_payload(
    result: PolicyEvaluationResult,
    *,
    model_path: Path,
    deterministic: bool,
    observation_mode: str,
    reward_mode: str,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ruleset: str = RULESET,
) -> dict[str, Any]:
    return {
        "model_path": str(model_path),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deterministic": deterministic,
        "ruleset": ruleset,
        "action_size": ACTION_SIZE,
        "observation_size": observation_size(observation_mode),
        "observation_mode": observation_mode,
        "reward_mode": reward_mode,
        "target_opponent": TARGET_OPPONENT,
        "official_episodes": result.episodes,
        "pass_threshold": pass_threshold,
        "passed": result.win_rate >= pass_threshold,
        **result.to_dict(),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    if not 0.0 <= args.pass_threshold <= 1.0:
        raise ValueError("pass_threshold must be in [0, 1]")


if __name__ == "__main__":
    main()
