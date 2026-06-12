import pytest

from yutnori.core import (
    ACTION_SIZE,
    Cell,
    GameState,
    PieceStatus,
    Position,
    Route,
    YutResult,
    decode_action,
    encode_action,
)


class SequenceSampler:
    def __init__(self, results):
        self.results = list(results)
        self.index = 0

    def sample(self):
        if self.index >= len(self.results):
            raise AssertionError("SequenceSampler exhausted")
        result = self.results[self.index]
        self.index += 1
        return result


def test_action_encoding_round_trips_all_supported_actions():
    for piece_id in range(4):
        for yut_result in YutResult:
            action = encode_action(piece_id, yut_result)
            assert decode_action(action) == (piece_id, yut_result)

    with pytest.raises(ValueError):
        encode_action(4, YutResult.DO)
    with pytest.raises(ValueError):
        decode_action(ACTION_SIZE)


def test_start_turn_rolls_until_non_bonus_result():
    sampler = SequenceSampler([YutResult.YUT, YutResult.MO, YutResult.GAE])
    state = GameState(yut_sampler=sampler)

    rolled = state.start_turn()

    assert rolled == [YutResult.YUT, YutResult.MO, YutResult.GAE]
    assert state.pool_counts[YutResult.YUT] == 1
    assert state.pool_counts[YutResult.MO] == 1
    assert state.pool_counts[YutResult.GAE] == 1
    assert state.turn_count == 1


def test_legal_actions_require_pool_and_stack_representative():
    state = GameState()
    state.set_pool(YutResult.DO)

    assert state.get_legal_actions() == [
        encode_action(0, YutResult.DO),
        encode_action(1, YutResult.DO),
        encode_action(2, YutResult.DO),
        encode_action(3, YutResult.DO),
    ]

    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)

    assert state.is_legal_action(encode_action(0, YutResult.DO))
    assert not state.is_legal_action(encode_action(1, YutResult.DO))


def test_stack_moves_together_and_unifies_destination_stack():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[0][2] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE, YutResult.DO)

    event = state.apply_action(encode_action(0, YutResult.GAE))

    assert event.moved_piece_ids == [0, 1]
    assert event.stacked
    assert event.stack_size == 3
    assert state.pieces[0][0].physical_cell == Cell.O3
    assert state.pieces[0][1].physical_cell == Cell.O3
    assert state.pieces[0][2].physical_cell == Cell.O3
    assert state.pool_counts[YutResult.DO] == 1
    assert state.current_player == 0


def test_capture_resets_opponent_stack_and_adds_bonus_for_do_gae_geol():
    sampler = SequenceSampler([YutResult.DO])
    state = GameState(yut_sampler=sampler)
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.pieces[1][1] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    event = state.apply_action(encode_action(0, YutResult.GAE))

    assert event.captured
    assert event.captured_count == 2
    assert event.captured_piece_ids == [0, 1]
    assert event.bonus_rolls == [YutResult.DO]
    assert state.pieces[1][0].status == PieceStatus.WAITING
    assert state.pieces[1][1].status == PieceStatus.WAITING
    assert state.pool_counts[YutResult.DO] == 1
    assert state.current_player == 0


def test_passing_opponent_piece_does_not_capture_or_add_bonus():
    sampler = SequenceSampler([YutResult.DO])
    state = GameState(yut_sampler=sampler)
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.GAE)

    event = state.apply_action(encode_action(0, YutResult.GAE))

    assert not event.captured
    assert event.bonus_rolls == []
    assert state.pieces[1][0].physical_cell == Cell.O2
    assert event.turn_changed
    assert state.current_player == 1
    assert state.pool_counts[YutResult.DO] == 1


def test_capture_opportunity_without_capture_does_not_add_bonus():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.DO, YutResult.GAE)

    event = state.apply_action(encode_action(1, YutResult.DO))

    assert not event.captured
    assert event.bonus_rolls == []
    assert state.pieces[1][0].physical_cell == Cell.O2
    assert state.pool_counts[YutResult.GAE] == 1
    assert state.current_player == 0


def test_yut_or_mo_capture_does_not_add_duplicate_capture_bonus():
    sampler = SequenceSampler([YutResult.DO])
    state = GameState(yut_sampler=sampler)
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.C1_DIAGONAL, 0)
    state.set_pool(YutResult.YUT)

    event = state.apply_action(encode_action(0, YutResult.YUT))

    assert event.captured
    assert event.captured_count == 1
    assert event.bonus_rolls == []
    assert event.turn_changed
    assert state.current_player == 1
    assert state.pool_counts[YutResult.DO] == 1


def test_pool_depletion_changes_turn_and_starts_next_turn():
    sampler = SequenceSampler([YutResult.GEOL])
    state = GameState(yut_sampler=sampler)
    state.set_pool(YutResult.DO)

    event = state.apply_action(encode_action(0, YutResult.DO))

    assert event.turn_changed
    assert state.current_player == 1
    assert state.turn_count == 1
    assert state.pool_counts[YutResult.GEOL] == 1


def test_finishing_all_pieces_sets_winner_without_turn_change():
    state = GameState()
    for piece_id in range(3):
        state.pieces[0][piece_id] = Position.finished()
    state.pieces[0][3] = Position.at(Route.OUTER, 19)
    state.set_pool(YutResult.GAE)

    event = state.apply_action(encode_action(3, YutResult.GAE))

    assert event.finished_count == 1
    assert event.winner == 0
    assert state.winner == 0
    assert not event.turn_changed


def test_back_do_is_legal_for_on_board_pieces_but_not_waiting_pieces():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.BACK_DO)

    assert state.get_legal_actions() == [
        encode_action(0, YutResult.BACK_DO),
    ]
    assert not state.is_legal_action(encode_action(1, YutResult.BACK_DO))


def test_back_do_moves_stack_and_captures_opponent_stack_with_bonus_roll():
    sampler = SequenceSampler([YutResult.DO])
    state = GameState(yut_sampler=sampler)
    state.pieces[0][0] = Position.at(Route.OUTER, 3)
    state.pieces[0][1] = Position.at(Route.OUTER, 3)
    state.pieces[1][0] = Position.at(Route.OUTER, 2)
    state.pieces[1][1] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.BACK_DO)

    event = state.apply_action(encode_action(0, YutResult.BACK_DO))

    assert event.moved_piece_ids == [0, 1]
    assert event.captured_piece_ids == [0, 1]
    assert event.captured_count == 2
    assert event.bonus_rolls == [YutResult.DO]
    assert state.pieces[0][0].physical_cell == Cell.O2
    assert state.pieces[0][1].physical_cell == Cell.O2
    assert state.pieces[1][0].status == PieceStatus.WAITING
    assert state.pieces[1][1].status == PieceStatus.WAITING
    assert state.pool_counts[YutResult.DO] == 1
    assert state.current_player == 0


def test_back_do_from_o1_can_capture_opponent_on_home():
    sampler = SequenceSampler([YutResult.DO])
    state = GameState(yut_sampler=sampler)
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.home(Route.OUTER)
    state.set_pool(YutResult.BACK_DO)

    event = state.apply_action(encode_action(0, YutResult.BACK_DO))

    assert event.captured_count == 1
    assert state.pieces[0][0].physical_cell == Cell.HOME
    assert state.pieces[1][0].status == PieceStatus.WAITING


def test_start_turn_auto_passes_back_do_when_all_pieces_are_waiting():
    sampler = SequenceSampler([YutResult.BACK_DO, YutResult.DO])
    state = GameState(starting_player=0, yut_sampler=sampler)

    rolls = state.start_turn()

    assert rolls == [YutResult.DO]
    assert state.current_player == 1
    assert state.turn_count == 2
    assert state.pool_counts[YutResult.DO] == 1
    assert len(state.last_auto_passes) == 1
    assert state.last_auto_passes[0].player == 0
    assert state.last_auto_passes[0].rolls == [YutResult.BACK_DO]
