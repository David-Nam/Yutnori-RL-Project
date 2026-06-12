import pytest

from yutnori.core import GameEvent, YutResult, encode_action
from yutnori.training import (
    RF_SHAPING_CAPTURE_WEIGHT,
    RF_SHAPING_FINISH_WEIGHT,
    RF_SHAPING_SHORTCUT_BONUS,
    project_rf_event_shaping_reward,
    project_rf_events_shaping_reward,
)


def _event(
    *,
    actor: int,
    captured_count: int = 0,
    finished_count: int = 0,
    entered_shortcut: bool = False,
) -> GameEvent:
    return GameEvent(
        actor=actor,
        action=encode_action(0, YutResult.DO),
        piece_id=0,
        yut_result=YutResult.DO,
        moved_piece_ids=[0],
        captured=captured_count > 0,
        captured_count=captured_count,
        captured_piece_ids=list(range(captured_count)),
        finished_count=finished_count,
        entered_shortcut=entered_shortcut,
    )


def test_project_rf_event_shaping_rewards_learner_capture_count():
    reward = project_rf_event_shaping_reward(
        _event(actor=0, captured_count=2),
        learner_player=0,
    )

    assert reward == pytest.approx(2 * RF_SHAPING_CAPTURE_WEIGHT)


def test_project_rf_event_shaping_rewards_learner_finished_count():
    reward = project_rf_event_shaping_reward(
        _event(actor=0, finished_count=3),
        learner_player=0,
    )

    assert reward == pytest.approx(3 * RF_SHAPING_FINISH_WEIGHT)


def test_project_rf_event_shaping_rewards_learner_shortcut():
    reward = project_rf_event_shaping_reward(
        _event(actor=0, entered_shortcut=True),
        learner_player=0,
    )

    assert reward == pytest.approx(RF_SHAPING_SHORTCUT_BONUS)


def test_project_rf_event_shaping_penalizes_opponent_capture_and_finish():
    reward = project_rf_event_shaping_reward(
        _event(actor=1, captured_count=2, finished_count=1),
        learner_player=0,
    )

    expected = -(2 * RF_SHAPING_CAPTURE_WEIGHT + RF_SHAPING_FINISH_WEIGHT)
    assert reward == pytest.approx(expected)


def test_project_rf_event_shaping_does_not_penalize_opponent_shortcut():
    reward = project_rf_event_shaping_reward(
        _event(actor=1, entered_shortcut=True),
        learner_player=0,
    )

    assert reward == 0.0


def test_project_rf_events_shaping_sums_learner_and_opponent_events():
    reward = project_rf_events_shaping_reward(
        _event(actor=0, captured_count=1, entered_shortcut=True),
        [
            _event(actor=1, finished_count=1),
            _event(actor=1, captured_count=2),
        ],
        learner_player=0,
    )

    expected = (
        RF_SHAPING_CAPTURE_WEIGHT
        + RF_SHAPING_SHORTCUT_BONUS
        - RF_SHAPING_FINISH_WEIGHT
        - 2 * RF_SHAPING_CAPTURE_WEIGHT
    )
    assert reward == pytest.approx(expected)


def test_project_rf_event_shaping_rejects_invalid_learner_player():
    with pytest.raises(ValueError, match="learner_player"):
        project_rf_event_shaping_reward(_event(actor=0), learner_player=2)


def test_project_rf_event_shaping_rejects_invalid_event_actor():
    with pytest.raises(ValueError, match="event actor"):
        project_rf_event_shaping_reward(_event(actor=2), learner_player=0)
