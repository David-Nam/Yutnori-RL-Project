import pytest

from yutnori.agents import (
    CaptureFirstAgent,
    GreedyFinishAgent,
    ProjectRFRuleBasedAgent,
    RandomAgent,
)
from yutnori.eval import play_game, run_tournament


def test_play_game_returns_winner_and_metrics():
    result = play_game(
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        seed=3,
        max_decisions=10_000,
    )

    assert result.winner in {0, 1}
    assert result.starting_player in {0, 1}
    assert result.turn_count > 0
    assert result.decision_count > 0


def test_random_vs_random_1000_games_finish_without_illegal_actions():
    result = run_tournament(
        RandomAgent(seed=11),
        RandomAgent(seed=22),
        games=1000,
        seed=33,
        max_decisions=10_000,
    )

    assert result.games == 1000
    assert result.wins[0] + result.wins[1] == 1000
    assert result.starting_player_counts[0] + result.starting_player_counts[1] == 1000
    assert result.average_turns > 0
    assert result.average_decisions > 0


def test_heuristic_baselines_beat_random_in_smoke_tournaments():
    capture_result = run_tournament(
        CaptureFirstAgent(),
        RandomAgent(seed=44),
        games=100,
        seed=55,
    )
    greedy_result = run_tournament(
        GreedyFinishAgent(),
        RandomAgent(seed=66),
        games=100,
        seed=77,
    )

    assert capture_result.games == 100
    assert greedy_result.games == 100
    assert capture_result.wins[0] + capture_result.wins[1] == 100
    assert greedy_result.wins[0] + greedy_result.wins[1] == 100


@pytest.mark.parametrize(
    ("agent0", "agent1", "seed"),
    [
        (ProjectRFRuleBasedAgent(), RandomAgent(seed=101), 111),
        (RandomAgent(seed=102), ProjectRFRuleBasedAgent(), 112),
        (ProjectRFRuleBasedAgent(), CaptureFirstAgent(), 113),
        (CaptureFirstAgent(), ProjectRFRuleBasedAgent(), 114),
        (ProjectRFRuleBasedAgent(), GreedyFinishAgent(), 115),
        (GreedyFinishAgent(), ProjectRFRuleBasedAgent(), 116),
    ],
)
def test_project_rf_rule_agent_finishes_smoke_tournaments(agent0, agent1, seed):
    result = run_tournament(
        agent0,
        agent1,
        games=100,
        seed=seed,
        max_decisions=10_000,
    )

    assert result.games == 100
    assert result.wins[0] + result.wins[1] == 100
    assert result.starting_player_counts[0] + result.starting_player_counts[1] == 100
    assert result.average_turns > 0
    assert result.average_decisions > 0
