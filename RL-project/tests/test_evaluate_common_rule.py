import json
from pathlib import Path

from scripts.evaluate_common_rule import (
    DEFAULT_BASE_SEED_COUNT,
    DEFAULT_BASE_SEED_START,
    load_base_seeds,
    parse_args,
)


def test_common_eval_defaults_to_2500_paired_base_seeds():
    args = parse_args(["--model-path", "model.zip", "--output", "eval.json"])

    assert args.seed_start == DEFAULT_BASE_SEED_START
    assert args.seed_count == DEFAULT_BASE_SEED_COUNT
    assert args.stochastic is False
    assert args.model_type == "Pure RL"


def test_common_eval_builds_default_consecutive_seed_list():
    assert load_base_seeds(seed_file=None, seed_start=10, seed_count=3) == [
        10,
        11,
        12,
    ]


def test_common_eval_loads_frozen_seed_file(tmp_path: Path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps([3, 8, 13]))

    assert load_base_seeds(
        seed_file=seed_file,
        seed_start=0,
        seed_count=1,
    ) == [3, 8, 13]
