import json
from pathlib import Path

from scripts.evaluate_legacy_head_to_head import (
    DEFAULT_SEED_COUNT,
    DEFAULT_SEED_START,
    load_base_seeds,
    parse_args,
)


def test_head_to_head_defaults_to_holdout_paired_seeds():
    args = parse_args(
        [
            "--rl-model-path",
            "rl.zip",
            "--project-rf-model-path",
            "rf.pt",
            "--output-dir",
            "out",
        ]
    )

    assert args.seed_start == DEFAULT_SEED_START
    assert args.seed_count == DEFAULT_SEED_COUNT
    assert args.project_rf_network_only is False


def test_head_to_head_loads_seed_file(tmp_path: Path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps([3, 8, 13]))

    assert load_base_seeds(
        seed_file=seed_file,
        seed_start=0,
        seed_count=1,
    ) == [3, 8, 13]
