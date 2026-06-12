import json
from pathlib import Path

from scripts.evaluate_project_rf_common import (
    DEFAULT_BASE_SEED_COUNT,
    DEFAULT_BASE_SEED_START,
    load_base_seeds,
    parse_args,
)


def test_project_rf_eval_defaults_to_common_paired_seed_protocol():
    args = parse_args(["--model-path", "model.pt", "--output", "eval.json"])

    assert args.seed_start == DEFAULT_BASE_SEED_START
    assert args.seed_count == DEFAULT_BASE_SEED_COUNT
    assert args.device == "cpu"
    assert args.network_only is False


def test_project_rf_eval_loads_frozen_seed_file(tmp_path: Path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps([3, 8, 13]))

    assert load_base_seeds(
        seed_file=seed_file,
        seed_start=0,
        seed_count=1,
    ) == [3, 8, 13]
