import json

import pytest

from yutnori.env import RULESET
from yutnori.training import (
    resolve_model_observation_mode,
    resolve_model_reward_mode,
    resolve_model_ruleset,
)


def test_resolve_model_observation_mode_prefers_explicit_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "base"}),
    )

    assert resolve_model_observation_mode(model_path, "tactical") == "tactical"


def test_resolve_model_observation_mode_reads_training_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "tactical"}),
    )

    assert resolve_model_observation_mode(model_path) == "tactical"


def test_resolve_model_observation_mode_defaults_to_base_without_config(tmp_path):
    assert resolve_model_observation_mode(tmp_path / "model.zip") == "base"


def test_resolve_model_observation_mode_rejects_invalid_config_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "unknown"}),
    )

    with pytest.raises(ValueError, match="observation_mode"):
        resolve_model_observation_mode(model_path)


def test_resolve_model_reward_mode_prefers_explicit_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"reward_mode": "terminal"}),
    )

    assert resolve_model_reward_mode(model_path, "rf_shaped") == "rf_shaped"


def test_resolve_model_reward_mode_reads_training_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"reward_mode": "rf_shaped"}),
    )

    assert resolve_model_reward_mode(model_path) == "rf_shaped"


def test_resolve_model_reward_mode_defaults_to_terminal_without_config(tmp_path):
    assert resolve_model_reward_mode(tmp_path / "model.zip") == "terminal"


def test_resolve_model_reward_mode_defaults_to_terminal_for_old_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "tactical"}),
    )

    assert resolve_model_reward_mode(model_path) == "terminal"


def test_resolve_model_reward_mode_rejects_invalid_config_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"reward_mode": "unknown"}),
    )

    with pytest.raises(ValueError, match="reward_mode"):
        resolve_model_reward_mode(model_path)


def test_resolve_model_ruleset_accepts_full_backdo_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"ruleset": RULESET}),
    )

    assert resolve_model_ruleset(model_path) == RULESET


def test_resolve_model_ruleset_rejects_legacy_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "tactical"}),
    )

    with pytest.raises(ValueError, match="incompatible"):
        resolve_model_ruleset(model_path)


def test_resolve_model_ruleset_requires_config(tmp_path):
    with pytest.raises(ValueError, match="missing config"):
        resolve_model_ruleset(tmp_path / "model.zip")
