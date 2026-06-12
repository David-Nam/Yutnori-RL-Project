from types import SimpleNamespace

import numpy as np
import pytest

from yutnori.core import GameState, YutResult, encode_action
from yutnori.eval.legacy_head_to_head import (
    LEGACY_ACTION_SIZE,
    LEGACY_TACTICAL_OBSERVATION_SIZE,
    LegacyMaskablePPOAgent,
    LegacyNoBackdoSampler,
    encode_legacy_observation,
    evaluate_legacy_head_to_head,
    legacy_action_to_local,
    local_action_to_legacy,
)


class FirstLegalAgent:
    def __init__(self, name):
        self.name = name

    def select_action(self, _state, legal_actions):
        return legal_actions[0]


class FirstLegalModel:
    action_space = SimpleNamespace(n=20)
    observation_space = SimpleNamespace(shape=(253,))

    def predict(self, observation, *, deterministic, action_masks, **_kwargs):
        assert observation.shape == (253,)
        assert deterministic is True
        return np.array(np.flatnonzero(action_masks)[0]), None


def test_legacy_action_mapping_round_trips_all_forward_actions():
    for legacy_action in range(LEGACY_ACTION_SIZE):
        assert local_action_to_legacy(
            legacy_action_to_local(legacy_action)
        ) == legacy_action


def test_legacy_action_mapping_rejects_back_do():
    with pytest.raises(ValueError, match="BACK_DO"):
        local_action_to_legacy(encode_action(0, YutResult.BACK_DO))


def test_legacy_sampler_never_emits_back_do():
    sampler = LegacyNoBackdoSampler(seed=17)

    assert {
        sampler.sample() for _ in range(10_000)
    }.issubset(
        {
            YutResult.DO,
            YutResult.GAE,
            YutResult.GEOL,
            YutResult.YUT,
            YutResult.MO,
        }
    )


def test_legacy_tactical_observation_matches_40m_shape():
    state = GameState()
    state.set_pool(YutResult.DO, YutResult.YUT)

    observation = encode_legacy_observation(
        state,
        0,
        observation_mode="tactical",
    )

    assert observation.shape == (LEGACY_TACTICAL_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32


def test_legacy_ppo_adapter_maps_masked_action_to_current_action():
    state = GameState()
    state.set_pool(YutResult.DO)
    agent = LegacyMaskablePPOAgent(FirstLegalModel())

    action = agent.select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.DO)


def test_paired_evaluation_runs_both_starting_positions_and_is_reproducible():
    kwargs = {
        "base_seeds": [7, 11],
        "show_progress": False,
    }
    first = evaluate_legacy_head_to_head(
        FirstLegalAgent("model_a"),
        FirstLegalAgent("model_b"),
        **kwargs,
    )
    second = evaluate_legacy_head_to_head(
        FirstLegalAgent("model_a"),
        FirstLegalAgent("model_b"),
        **kwargs,
    )

    assert first.scheduled_games == 4
    assert first.completed_games == 4
    assert first.error_count == 0
    assert [game.to_dict() for game in first.games] == [
        game.to_dict() for game in second.games
    ]
    assert sum(game.model_a_starts for game in first.games) == 2
