"""Reward shaping helpers for PPO training experiments."""

from __future__ import annotations

from collections.abc import Iterable

from yutnori.core import PLAYER_COUNT, GameEvent

RF_SHAPING_CAPTURE_WEIGHT = 0.08
RF_SHAPING_FINISH_WEIGHT = 0.15
RF_SHAPING_SHORTCUT_BONUS = 0.02


def project_rf_event_shaping_reward(
    event: GameEvent,
    *,
    learner_player: int,
) -> float:
    """Return non-terminal RF-style shaping reward for one game event."""

    _validate_learner_player(learner_player)
    _validate_event_actor(event.actor)
    if event.actor == learner_player:
        return (
            RF_SHAPING_CAPTURE_WEIGHT * event.captured_count
            + RF_SHAPING_FINISH_WEIGHT * event.finished_count
            + (RF_SHAPING_SHORTCUT_BONUS if event.entered_shortcut else 0.0)
        )
    return -(
        RF_SHAPING_CAPTURE_WEIGHT * event.captured_count
        + RF_SHAPING_FINISH_WEIGHT * event.finished_count
    )


def project_rf_events_shaping_reward(
    learner_event: GameEvent,
    opponent_events: Iterable[GameEvent],
    *,
    learner_player: int,
) -> float:
    """Return summed RF-style shaping reward for one learner env step."""

    total = project_rf_event_shaping_reward(
        learner_event,
        learner_player=learner_player,
    )
    for opponent_event in opponent_events:
        total += project_rf_event_shaping_reward(
            opponent_event,
            learner_player=learner_player,
        )
    return total


def _validate_learner_player(learner_player: int) -> None:
    if learner_player < 0 or learner_player >= PLAYER_COUNT:
        raise ValueError(f"learner_player must be in [0, {PLAYER_COUNT})")


def _validate_event_actor(actor: int) -> None:
    if actor < 0 or actor >= PLAYER_COUNT:
        raise ValueError(f"event actor must be in [0, {PLAYER_COUNT})")
