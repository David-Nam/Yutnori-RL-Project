import numpy as np
import pytest

from yutnori.training import (
    evaluate_common_rule_agent,
    evaluate_common_rule_policy,
    wilson_interval,
)


class FirstLegalModel:
    def predict(self, _observation, *, deterministic, action_masks, **_kwargs):
        assert deterministic is True
        return np.array(np.flatnonzero(action_masks)[0]), None


class FirstLegalAgent:
    def select_action(self, _state, legal_actions):
        return legal_actions[0]


def test_common_evaluation_runs_each_seed_with_both_starting_positions():
    result = evaluate_common_rule_policy(
        FirstLegalModel(),
        base_seeds=[7, 11],
        deterministic=True,
    )

    assert result.scheduled_games == 4
    assert result.completed_games == 4
    assert result.model_first.games == 2
    assert result.model_second.games == 2
    assert result.wins + result.losses == 4
    assert result.illegal_action_count == 0
    assert result.evaluation_error_count == 0
    assert result.seed_sha256


def test_common_evaluation_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        evaluate_common_rule_policy(FirstLegalModel(), base_seeds=[7, 7])


def test_common_agent_evaluation_uses_same_paired_seed_protocol():
    result = evaluate_common_rule_agent(FirstLegalAgent(), base_seeds=[7, 11])

    assert result.scheduled_games == 4
    assert result.completed_games == 4
    assert result.model_first.games == 2
    assert result.model_second.games == 2
    assert result.illegal_action_count == 0
    assert result.evaluation_error_count == 0


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(3000, 5000)

    assert lower == pytest.approx(0.586349, abs=1e-6)
    assert upper == pytest.approx(0.613497, abs=1e-6)
