"""Tactical action features for policy observations and analysis."""

from __future__ import annotations

import numpy as np

from yutnori.agents.baseline import (
    evaluate_action,
    project_rf_action_score,
    project_rf_distance_to_finish,
)
from yutnori.core import ACTION_SIZE, GameState, PieceStatus, decode_action
from yutnori.core.yut import steps_for

TACTICAL_ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "legal",
    "capture",
    "captured_count",
    "finish",
    "finished_count",
    "moved_count",
    "waiting_move",
    "stack_size",
    "distance_after",
    "rf_score",
)
TACTICAL_ACTION_FEATURE_SIZE = len(TACTICAL_ACTION_FEATURE_NAMES)


def tactical_action_features(state: GameState) -> np.ndarray:
    """Return raw tactical features for every action in action-id order."""

    features = np.zeros(
        (ACTION_SIZE, TACTICAL_ACTION_FEATURE_SIZE),
        dtype=np.float32,
    )
    for action in range(ACTION_SIZE):
        if state.is_legal_action(action):
            features[action] = tactical_action_feature_row(state, action)
    return features


def tactical_action_feature_row(state: GameState, action: int) -> np.ndarray:
    """Return raw tactical features for one legal action."""

    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be featurized: {action}")

    actor = state.current_player
    piece_id, yut_result = decode_action(action)
    old_position = state.pieces[actor][piece_id]
    move_result = state.board.move(old_position, steps_for(yut_result))
    evaluation = evaluate_action(state, action)

    return np.array(
        [
            1.0,
            float(evaluation.captured_count > 0),
            float(evaluation.captured_count),
            float(evaluation.finished_count > 0),
            float(evaluation.finished_count),
            float(evaluation.moved_count),
            float(old_position.status == PieceStatus.WAITING),
            float(evaluation.moved_count),
            float(project_rf_distance_to_finish(move_result.position)),
            float(project_rf_action_score(state, action)),
        ],
        dtype=np.float32,
    )
