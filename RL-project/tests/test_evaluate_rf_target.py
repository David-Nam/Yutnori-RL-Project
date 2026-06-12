from pathlib import Path

import pytest

from scripts.evaluate_rf_target import (
    DEFAULT_EPISODES,
    DEFAULT_PASS_THRESHOLD,
    TARGET_OPPONENT,
    _validate_args,
    build_payload,
    parse_args,
)
from yutnori.training import PolicyEvaluationResult


def test_rf_target_eval_defaults_to_official_episode_count():
    args = parse_args(
        [
            "--model-path",
            "model.zip",
            "--output",
            "eval.json",
        ]
    )

    assert args.episodes == DEFAULT_EPISODES
    assert args.pass_threshold == DEFAULT_PASS_THRESHOLD
    assert args.seed == 100_000
    assert args.observation_mode is None
    assert args.reward_mode is None


def test_rf_target_eval_allows_custom_episode_count():
    args = parse_args(
        [
            "--model-path",
            "model.zip",
            "--episodes",
            "17",
            "--output",
            "eval.json",
        ]
    )

    assert args.episodes == 17


def test_rf_target_eval_allows_explicit_observation_mode():
    args = parse_args(
        [
            "--model-path",
            "model.zip",
            "--observation-mode",
            "tactical",
            "--output",
            "eval.json",
        ]
    )

    assert args.observation_mode == "tactical"


def test_rf_target_eval_allows_explicit_reward_mode():
    args = parse_args(
        [
            "--model-path",
            "model.zip",
            "--reward-mode",
            "rf_shaped",
            "--output",
            "eval.json",
        ]
    )

    assert args.reward_mode == "rf_shaped"


def test_rf_target_payload_marks_passing_result():
    result = PolicyEvaluationResult(
        opponent=TARGET_OPPONENT,
        episodes=17,
        learner_player=0,
        wins=11,
        losses=6,
        win_rate=11 / 17,
        average_turns=20.0,
        average_decisions=25.0,
        illegal_action_count=0,
        starting_player_counts={0: 8, 1: 9},
    )

    payload = build_payload(
        result,
        model_path=Path("runs/model.zip"),
        deterministic=True,
        observation_mode="tactical",
        reward_mode="rf_shaped",
    )

    assert payload["target_opponent"] == TARGET_OPPONENT
    assert payload["official_episodes"] == 17
    assert payload["pass_threshold"] == DEFAULT_PASS_THRESHOLD
    assert payload["observation_mode"] == "tactical"
    assert payload["reward_mode"] == "rf_shaped"
    assert payload["passed"] is True
    assert payload["episodes"] == 17
    assert payload["win_rate"] == pytest.approx(11 / 17)


def test_rf_target_payload_marks_failing_result():
    result = PolicyEvaluationResult(
        opponent=TARGET_OPPONENT,
        episodes=10,
        learner_player=0,
        wins=5,
        losses=5,
        win_rate=0.5,
        average_turns=20.0,
        average_decisions=25.0,
        illegal_action_count=0,
        starting_player_counts={0: 5, 1: 5},
    )

    payload = build_payload(
        result,
        model_path=Path("runs/model.zip"),
        deterministic=False,
        observation_mode="base",
        reward_mode="terminal",
    )

    assert payload["deterministic"] is False
    assert payload["observation_mode"] == "base"
    assert payload["reward_mode"] == "terminal"
    assert payload["passed"] is False


def test_rf_target_eval_rejects_invalid_threshold():
    args = parse_args(
        [
            "--model-path",
            "model.zip",
            "--pass-threshold",
            "1.5",
            "--output",
            "eval.json",
        ]
    )

    with pytest.raises(ValueError, match="pass_threshold"):
        _validate_args(args)
