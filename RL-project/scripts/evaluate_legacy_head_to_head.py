"""Run a paired match between the best 40M PPO and a project-RF checkpoint."""

from __future__ import annotations

import argparse
import csv
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

from yutnori.agents.project_rf_checkpoint import (  # noqa: E402
    PROJECT_RF_TACTICAL_WEIGHT,
    ProjectRFCheckpointAgent,
)
from yutnori.eval.legacy_head_to_head import (  # noqa: E402
    LegacyMaskablePPOAgent,
    evaluate_legacy_head_to_head,
    file_sha256,
)

DEFAULT_SEED_START = 200_000
DEFAULT_SEED_COUNT = 2_500


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model-path", type=Path, required=True)
    parser.add_argument("--project-rf-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument(
        "--project-rf-tactical-weight",
        type=float,
        default=PROJECT_RF_TACTICAL_WEIGHT,
    )
    parser.add_argument("--project-rf-network-only", action="store_true")
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

    model_a = LegacyMaskablePPOAgent.load(
        args.rl_model_path,
        device=args.device,
        observation_mode="tactical",
        deterministic=True,
        name="rl_40m_seed1",
    )
    model_b = ProjectRFCheckpointAgent(
        args.project_rf_model_path,
        device=args.device,
        use_tactical_prior=not args.project_rf_network_only,
        tactical_weight=args.project_rf_tactical_weight,
    )
    model_b.name = "project_rf_ppo_capture_imitation"

    result = evaluate_legacy_head_to_head(
        model_a,
        model_b,
        base_seeds=base_seeds,
        max_decisions=args.max_decisions,
        show_progress=not args.no_progress_bar,
    )
    payload = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "seed_source": (
            str(args.seed_file)
            if args.seed_file is not None
            else f"range:{args.seed_start}:{args.seed_count}"
        ),
        "model_a_checkpoint": {
            **model_a.metadata,
            "sha256": file_sha256(args.rl_model_path),
        },
        "model_b_checkpoint": {
            "name": model_b.name,
            "model_path": str(args.project_rf_model_path),
            "sha256": file_sha256(args.project_rf_model_path),
            "model_type": (
                "Pure RL"
                if args.project_rf_network_only
                else "RL + Rule Hybrid"
            ),
            "projection": model_b.projection_metadata,
        },
        **result.to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_games_csv(args.output_dir / "games.csv", result.games)
    (args.output_dir / "report.md").write_text(_report_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid_official_result"]:
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


def _write_games_csv(path: Path, games: Sequence[Any]) -> None:
    rows = [game.to_dict() for game in games]
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(payload: dict[str, Any]) -> str:
    model_a = payload["model_a"]
    model_b = payload["model_b"]
    pair = payload["pair_outcomes"]
    return f"""# Legacy 20-action Head-to-Head Evaluation

## 조건

- ruleset: `{payload['ruleset']}`
- protocol: `{payload['protocol']}`
- paired seeds: {payload['base_seed_count']}
- total games: {payload['scheduled_games']}
- deterministic policy: true
- project-RF tactical prior: {payload['model_b_checkpoint']['projection']['use_tactical_prior']}

## 결과

| Model | Wins | Win rate | First | Second | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| {model_a['name']} | {model_a['wins']} | {model_a['win_rate']:.2%} | {model_a['as_first']['win_rate']:.2%} | {model_a['as_second']['win_rate']:.2%} | {model_a['confidence_interval_95']['lower']:.2%}~{model_a['confidence_interval_95']['upper']:.2%} |
| {model_b['name']} | {model_b['wins']} | {model_b['win_rate']:.2%} | {model_b['as_first']['win_rate']:.2%} | {model_b['as_second']['win_rate']:.2%} | {model_b['confidence_interval_95']['lower']:.2%}~{model_b['confidence_interval_95']['upper']:.2%} |

## Paired seed 결과

- Model A 2승: {pair['model_a_sweeps']}
- 1승 1패: {pair['split_pairs']}
- Model B 2승: {pair['model_b_sweeps']}
- 오류 pair: {pair['error_pairs']}

## 무결성

- illegal Model A: {model_a['illegal_action_count']}
- illegal Model B: {model_b['illegal_action_count']}
- evaluation errors: {payload['evaluation_error_count']}
- valid official result: {payload['valid_official_result']}
"""


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if args.seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if args.max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    if args.project_rf_tactical_weight < 0:
        raise ValueError("project_rf_tactical_weight must be non-negative")


if __name__ == "__main__":
    main()
