import numpy as np
import pytest

from yutnori.agents.tactical_features import (
    TACTICAL_ACTION_FEATURE_NAMES,
    TACTICAL_ACTION_FEATURE_SIZE,
    tactical_action_feature_row,
    tactical_action_features,
)
from yutnori.core import ACTION_SIZE, GameState, Position, Route, YutResult, encode_action

FEATURE_INDEX = {
    name: index for index, name in enumerate(TACTICAL_ACTION_FEATURE_NAMES)
}


def test_tactical_action_features_returns_one_row_per_action():
    state = GameState()
    state.set_pool(YutResult.DO)

    features = tactical_action_features(state)

    assert features.dtype == np.float32
    assert features.shape == (ACTION_SIZE, TACTICAL_ACTION_FEATURE_SIZE)


def test_tactical_action_features_zeroes_illegal_actions():
    state = GameState()
    state.set_pool(YutResult.DO)

    features = tactical_action_features(state)
    illegal_action = encode_action(0, YutResult.GAE)

    np.testing.assert_array_equal(
        features[illegal_action],
        np.zeros(TACTICAL_ACTION_FEATURE_SIZE, dtype=np.float32),
    )


def test_tactical_action_feature_row_rejects_illegal_action():
    state = GameState()
    state.set_pool(YutResult.DO)

    with pytest.raises(ValueError, match="illegal action"):
        tactical_action_feature_row(state, encode_action(0, YutResult.GAE))


def test_tactical_action_features_mark_capture_counts_and_rf_score():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.pieces[1][1] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    action = encode_action(0, YutResult.GAE)
    row = tactical_action_features(state)[action]

    assert row[FEATURE_INDEX["legal"]] == 1.0
    assert row[FEATURE_INDEX["capture"]] == 1.0
    assert row[FEATURE_INDEX["captured_count"]] == 2.0
    assert row[FEATURE_INDEX["finish"]] == 0.0
    assert row[FEATURE_INDEX["distance_after"]] == 18.0
    assert row[FEATURE_INDEX["rf_score"]] == pytest.approx(41.0)


def test_tactical_action_features_mark_finish_counts_and_distance():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 19)
    state.set_pool(YutResult.GAE)

    action = encode_action(0, YutResult.GAE)
    row = tactical_action_features(state)[action]

    assert row[FEATURE_INDEX["legal"]] == 1.0
    assert row[FEATURE_INDEX["finish"]] == 1.0
    assert row[FEATURE_INDEX["finished_count"]] == 1.0
    assert row[FEATURE_INDEX["distance_after"]] == 0.0
    assert row[FEATURE_INDEX["rf_score"]] == pytest.approx(100.0)


def test_tactical_action_features_mark_stack_movement():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[0][2] = Position.at(Route.OUTER, 2)
    state.pieces[0][3] = Position.finished()
    state.set_pool(YutResult.DO)

    action = encode_action(0, YutResult.DO)
    row = tactical_action_features(state)[action]

    assert row[FEATURE_INDEX["moved_count"]] == 2.0
    assert row[FEATURE_INDEX["stack_size"]] == 2.0
    assert row[FEATURE_INDEX["waiting_move"]] == 0.0


def test_tactical_action_features_describe_back_do_capture():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 3)
    state.pieces[1][0] = Position.at(Route.OUTER, 2)
    state.pieces[1][1] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.BACK_DO)

    row = tactical_action_feature_row(
        state,
        encode_action(0, YutResult.BACK_DO),
    )

    assert row[FEATURE_INDEX["legal"]] == 1.0
    assert row[FEATURE_INDEX["capture"]] == 1.0
    assert row[FEATURE_INDEX["captured_count"]] == 2.0
    assert row[FEATURE_INDEX["waiting_move"]] == 0.0


def test_tactical_action_features_mark_waiting_piece_move():
    state = GameState()
    state.set_pool(YutResult.DO)

    action = encode_action(0, YutResult.DO)
    row = tactical_action_features(state)[action]

    assert row[FEATURE_INDEX["waiting_move"]] == 1.0
    assert row[FEATURE_INDEX["moved_count"]] == 1.0
    assert row[FEATURE_INDEX["stack_size"]] == 1.0
    assert row[FEATURE_INDEX["distance_after"]] == 20.0
