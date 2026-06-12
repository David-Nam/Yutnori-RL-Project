import pytest

from yutnori.agents import (
    CaptureFirstAgent,
    CommonRuleBasedAgent,
    GreedyFinishAgent,
    RandomAgent,
)
from yutnori.agents.baseline import (
    ProjectRFRuleBasedAgent,
    project_rf_action_score,
    project_rf_distance_to_finish,
)
from yutnori.core import Cell, GameState, Position, Route, YutResult, encode_action


def test_random_agent_selects_one_of_the_legal_actions():
    state = GameState()
    state.set_pool(YutResult.DO)
    legal_actions = state.get_legal_actions()

    action = RandomAgent(seed=1).select_action(state, legal_actions)

    assert action in legal_actions


def test_capture_first_agent_prefers_actual_capture():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.DO, YutResult.GAE)

    action = CaptureFirstAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_capture_first_agent_counts_stacked_opponents():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 2)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.pieces[1][1] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.DO, YutResult.GAE)

    action = CaptureFirstAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_capture_first_agent_ignores_passed_opponent_piece():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    for piece_id in range(1, 4):
        state.pieces[0][piece_id] = Position.finished()
    state.pieces[1][0] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.GAE)

    action = CaptureFirstAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_greedy_finish_agent_prefers_finishing_action():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 19)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.set_pool(YutResult.DO, YutResult.GAE)

    action = GreedyFinishAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_greedy_finish_agent_prefers_moving_a_stack_when_no_finish_exists():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[0][2] = Position.at(Route.OUTER, 2)
    state.set_pool(YutResult.DO)

    action = GreedyFinishAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.DO)


def test_baseline_agent_selected_actions_remain_legal_after_mutation():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[0][2] = Position.at(Route.OUTER, 4)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE, YutResult.MO)

    for agent in (
        RandomAgent(seed=3),
        CaptureFirstAgent(),
        GreedyFinishAgent(),
        ProjectRFRuleBasedAgent(),
    ):
        legal_actions = state.get_legal_actions()
        action = agent.select_action(state, legal_actions)
        assert action in legal_actions


def test_capture_first_agent_can_be_used_as_env_opponent_policy():
    state = GameState(starting_player=1)
    state.pieces[1][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    action = CaptureFirstAgent().select_action(state, state.get_legal_actions())
    event = state.apply_action(action)

    assert event.captured
    assert state.pieces[0][0].physical_cell is None
    assert state.pieces[0][0].status.value == "WAITING"
    assert event.captured_piece_ids == [0]
    assert event.captured_count == 1
    assert event.moved_piece_ids == [0]
    assert state.pieces[1][0].physical_cell == Cell.O3


def test_project_rf_distance_to_finish_counts_exact_home_as_one_step():
    assert project_rf_distance_to_finish(Position.finished()) == 0
    assert project_rf_distance_to_finish(Position.home()) == 1
    assert project_rf_distance_to_finish(Position.at(Route.OUTER, 19)) == 2


def test_project_rf_action_score_rewards_finishing_destination():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 19)
    state.set_pool(YutResult.GAE)

    score = project_rf_action_score(state, encode_action(0, YutResult.GAE))

    assert score == pytest.approx(100.0)


def test_project_rf_action_score_rewards_capture_once():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.pieces[1][1] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    score = project_rf_action_score(state, encode_action(0, YutResult.GAE))

    assert score == pytest.approx(41.0)


def test_project_rf_action_score_rewards_waiting_piece_start():
    state = GameState()
    state.set_pool(YutResult.DO)

    score = project_rf_action_score(state, encode_action(0, YutResult.DO))

    assert score == pytest.approx(-5.0)


def test_project_rf_action_score_rewards_moving_stack():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.set_pool(YutResult.GAE)

    score = project_rf_action_score(state, encode_action(0, YutResult.GAE))

    assert score == pytest.approx(-5.0)


def test_project_rf_action_score_prefers_shorter_distance_to_finish():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.set_pool(YutResult.DO, YutResult.GAE)

    do_score = project_rf_action_score(state, encode_action(0, YutResult.DO))
    gae_score = project_rf_action_score(state, encode_action(0, YutResult.GAE))

    assert gae_score > do_score


def test_project_rf_rule_agent_rejects_empty_legal_actions():
    with pytest.raises(ValueError, match="legal_actions must not be empty"):
        ProjectRFRuleBasedAgent().select_action(GameState(), [])


def test_project_rf_rule_agent_prefers_finish_over_capture():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 19)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    action = ProjectRFRuleBasedAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_project_rf_rule_agent_prefers_capture_over_simple_advance():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 4)
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)

    action = ProjectRFRuleBasedAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_project_rf_rule_agent_prefers_moving_stack():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 1)
    state.pieces[0][2] = Position.at(Route.OUTER, 2)
    state.pieces[0][3] = Position.finished()
    state.set_pool(YutResult.DO)

    action = ProjectRFRuleBasedAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.DO)


def test_project_rf_rule_agent_prefers_shorter_distance_to_finish():
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    for piece_id in range(1, 4):
        state.pieces[0][piece_id] = Position.finished()
    state.set_pool(YutResult.DO, YutResult.GAE)

    action = ProjectRFRuleBasedAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_project_rf_rule_agent_breaks_score_ties_by_current_action_id():
    state = GameState()
    state.set_pool(YutResult.DO)

    action = ProjectRFRuleBasedAgent().select_action(state, state.get_legal_actions())

    assert action == encode_action(3, YutResult.DO)


def test_common_rule_agent_breaks_score_ties_by_smallest_action_id():
    state = GameState()
    state.set_pool(YutResult.DO)

    action = CommonRuleBasedAgent().select_action(
        state,
        state.get_legal_actions(),
    )

    assert action == encode_action(0, YutResult.DO)
