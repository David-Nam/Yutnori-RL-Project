"""Helpers for reading training metadata saved beside PPO models."""

from __future__ import annotations

import json
from pathlib import Path

from yutnori.env import (
    OBSERVATION_MODE_BASE,
    OBSERVATION_MODES,
    REWARD_MODE_TERMINAL,
    REWARD_MODES,
    RULESET,
)

LEGACY_RULESET = "legacy_no_backdo_v1"


def resolve_model_observation_mode(
    model_path: Path,
    requested_observation_mode: str | None = None,
) -> str:
    """Resolve the observation mode to use when evaluating a saved model."""

    if requested_observation_mode is not None:
        return _validate_observation_mode(
            requested_observation_mode,
            source="--observation-mode",
        )

    config = _read_model_config(model_path)
    if config is None:
        return OBSERVATION_MODE_BASE

    observation_mode = config.get("observation_mode", OBSERVATION_MODE_BASE)
    return _validate_observation_mode(
        observation_mode,
        source=f"{model_path.parent / 'config.json'} observation_mode",
    )


def resolve_model_reward_mode(
    model_path: Path,
    requested_reward_mode: str | None = None,
) -> str:
    """Resolve the reward mode to use when evaluating a saved model."""

    if requested_reward_mode is not None:
        return _validate_reward_mode(
            requested_reward_mode,
            source="--reward-mode",
        )

    config = _read_model_config(model_path)
    if config is None:
        return REWARD_MODE_TERMINAL

    reward_mode = config.get("reward_mode", REWARD_MODE_TERMINAL)
    return _validate_reward_mode(
        reward_mode,
        source=f"{model_path.parent / 'config.json'} reward_mode",
    )


def resolve_model_ruleset(model_path: Path) -> str:
    """Require a saved model to match the active environment ruleset."""

    config = _read_model_config(model_path)
    if config is None:
        raise ValueError(
            f"missing config.json beside {model_path}; cannot verify ruleset "
            f"compatibility with {RULESET}"
        )

    ruleset = config.get("ruleset", LEGACY_RULESET)
    if ruleset != RULESET:
        raise ValueError(
            f"{model_path.parent / 'config.json'} ruleset={ruleset!r} is "
            f"incompatible with active ruleset={RULESET!r}"
        )
    return RULESET


def _validate_observation_mode(observation_mode: object, *, source: str) -> str:
    if isinstance(observation_mode, str) and observation_mode in OBSERVATION_MODES:
        return observation_mode
    expected = ", ".join(OBSERVATION_MODES)
    raise ValueError(f"{source} must be one of {expected}")


def _validate_reward_mode(reward_mode: object, *, source: str) -> str:
    if isinstance(reward_mode, str) and reward_mode in REWARD_MODES:
        return reward_mode
    expected = ", ".join(REWARD_MODES)
    raise ValueError(f"{source} must be one of {expected}")


def _read_model_config(model_path: Path) -> dict[str, object] | None:
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return config
