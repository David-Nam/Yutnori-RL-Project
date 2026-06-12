import numpy as np
import pytest

from yutnori.agents import ProjectRFRuleBasedAgent
from yutnori.agents.tactical_features import (
    TACTICAL_ACTION_FEATURE_NAMES,
    TACTICAL_ACTION_FEATURE_SIZE,
)
from yutnori.core import (
    ACTION_SIZE,
    Cell,
    GameState,
    Position,
    Route,
    YutResult,
    encode_action,
)
from yutnori.env import (
    OBSERVATION_SIZE,
    POSITION_WAITING,
    REWARD_MODE_RF_SHAPED,
    REWARD_MODE_TERMINAL,
    TACTICAL_OBSERVATION_SIZE,
    YutnoriEnv,
    encode_observation,
    observation_size,
)
from yutnori.training import (
    RF_SHAPING_CAPTURE_WEIGHT,
    RF_SHAPING_FINISH_WEIGHT,
    RF_SHAPING_SHORTCUT_BONUS,
)

FEATURE_INDEX = {
    name: index for index, name in enumerate(TACTICAL_ACTION_FEATURE_NAMES)
}


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


def first_legal_action(_state: GameState, legal_actions: list[int]) -> int:
    return legal_actions[0]


def sequence_factory(results):
    def factory(_rng):
        return SequenceSampler(results)

    return factory


def sequence_attempt_factory(attempts):
    attempts = [list(results) for results in attempts]

    def factory(_rng):
        if not attempts:
            raise AssertionError("sequence_attempt_factory exhausted")
        return SequenceSampler(attempts.pop(0))

    return factory


def test_reset_returns_vector_observation_and_mask_for_learner_turn():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
    )

    obs, info = env.reset(seed=123)

    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["starting_player"] == 0
    assert info["initial_rolls"] == ["GAE"]
    mask = env.action_masks()
    assert mask.dtype == np.bool_
    assert mask.shape == (ACTION_SIZE,)
    assert np.flatnonzero(mask).tolist() == [
        encode_action(0, YutResult.GAE),
        encode_action(1, YutResult.GAE),
        encode_action(2, YutResult.GAE),
        encode_action(3, YutResult.GAE),
    ]


def test_observation_size_rejects_unknown_mode():
    with pytest.raises(ValueError, match="observation_mode"):
        observation_size("unknown")


def test_full_backdo_observation_and_action_sizes():
    assert ACTION_SIZE == 24
    assert OBSERVATION_SIZE == 62
    assert TACTICAL_OBSERVATION_SIZE == 302


def test_tactical_observation_mode_appends_action_features():
    base_env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
    )
    tactical_env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
        observation_mode="tactical",
    )

    base_obs, _base_info = base_env.reset(seed=123)
    tactical_obs, _tactical_info = tactical_env.reset(seed=123)

    assert observation_size("base") == OBSERVATION_SIZE
    assert observation_size("tactical") == TACTICAL_OBSERVATION_SIZE
    assert tactical_obs.shape == (TACTICAL_OBSERVATION_SIZE,)
    assert tactical_obs.dtype == np.float32
    assert tactical_env.observation_space.contains(tactical_obs)
    np.testing.assert_array_equal(tactical_obs[:OBSERVATION_SIZE], base_obs)


def test_tactical_observation_legal_feature_matches_action_mask():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
        observation_mode="tactical",
    )

    obs, _info = env.reset(seed=123)
    tactical_rows = obs[OBSERVATION_SIZE:].reshape(
        ACTION_SIZE,
        TACTICAL_ACTION_FEATURE_SIZE,
    )

    np.testing.assert_array_equal(
        tactical_rows[:, FEATURE_INDEX["legal"]].astype(np.bool_),
        env.action_masks(),
    )
    assert tactical_rows[
        encode_action(0, YutResult.GAE),
        FEATURE_INDEX["waiting_move"],
    ] == 1.0
    assert tactical_rows[
        encode_action(0, YutResult.DO),
        FEATURE_INDEX["legal"],
    ] == 0.0


def test_yutnori_env_rejects_unknown_observation_mode():
    with pytest.raises(ValueError, match="observation_mode"):
        YutnoriEnv(observation_mode="unknown")


def test_yutnori_env_rejects_unknown_reward_mode():
    with pytest.raises(ValueError, match="reward_mode"):
        YutnoriEnv(reward_mode="unknown")


def test_reset_seed_reproducibly_returns_same_initial_observation_and_mask():
    first_env = YutnoriEnv(starting_player=0)
    second_env = YutnoriEnv(starting_player=0)

    first_obs, first_info = first_env.reset(seed=77)
    second_obs, second_info = second_env.reset(seed=77)

    np.testing.assert_array_equal(first_obs, second_obs)
    np.testing.assert_array_equal(first_env.action_masks(), second_env.action_masks())
    assert first_info["initial_rolls"] == second_info["initial_rolls"]


def test_observation_is_from_learner_perspective():
    env = YutnoriEnv(
        learner_player=1,
        starting_player=1,
        yut_sampler_factory=sequence_factory([YutResult.DO]),
    )
    obs, _ = env.reset(seed=3)

    assert obs[0:4].tolist() == [POSITION_WAITING] * 4
    assert obs[4:8].tolist() == [0.0] * 4


def test_observation_encodes_back_do_pool_count():
    state = GameState()
    state.set_pool(YutResult.BACK_DO)

    obs = encode_observation(state, 0)

    assert obs[-1] == 1.0


def test_observation_distinguishes_center_entry_routes():
    state_from_c1 = GameState()
    state_from_c2 = GameState()
    state_from_c1.pieces[0][0] = state_from_c1.board.move(
        Position.at(Route.C1_DIAGONAL, 2),
        1,
    ).position
    state_from_c2.pieces[0][0] = state_from_c2.board.move(
        Position.at(Route.C2_DIAGONAL, 2),
        1,
    ).position

    obs_from_c1 = encode_observation(state_from_c1, 0)
    obs_from_c2 = encode_observation(state_from_c2, 0)

    assert obs_from_c1[0] == obs_from_c2[0] == float(Cell.CENTER)
    assert obs_from_c1[8] != obs_from_c2[8]


def test_action_masks_respect_stack_representative():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.DO]),
    )
    env.reset(seed=1)
    assert env.state is not None
    env.state.pieces[0][0] = Position.at(Route.OUTER, 1)
    env.state.pieces[0][1] = Position.at(Route.OUTER, 1)
    env.state.set_pool(YutResult.DO)

    legal_actions = np.flatnonzero(env.action_masks()).tolist()

    assert encode_action(0, YutResult.DO) in legal_actions
    assert encode_action(1, YutResult.DO) not in legal_actions


def test_step_applies_learner_action_and_returns_next_decision_state():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory(
            [YutResult.GAE, YutResult.DO, YutResult.GEOL]
        ),
    )
    env.reset(seed=5)

    obs, reward, terminated, truncated, info = env.step(
        encode_action(0, YutResult.GAE)
    )

    assert obs.shape == (OBSERVATION_SIZE,)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["learner_event"]["yut_result"] == "GAE"
    assert info["learner_event"]["turn_changed"]
    assert len(info["opponent_events"]) == 1
    assert info["current_player"] == 0
    assert np.flatnonzero(env.action_masks()).tolist() == [
        encode_action(0, YutResult.GEOL),
        encode_action(1, YutResult.GEOL),
        encode_action(2, YutResult.GEOL),
        encode_action(3, YutResult.GEOL),
    ]


def test_reset_auto_advances_opponent_until_learner_turn():
    env = YutnoriEnv(
        starting_player=1,
        opponent_policy=first_legal_action,
        yut_sampler_factory=sequence_factory([YutResult.DO, YutResult.GEOL]),
    )

    _obs, info = env.reset(seed=9)

    assert len(info["opponent_events"]) == 1
    assert info["opponent_events"][0]["actor"] == 1
    assert info["current_player"] == 0
    assert np.flatnonzero(env.action_masks()).tolist() == [
        encode_action(0, YutResult.GEOL),
        encode_action(1, YutResult.GEOL),
        encode_action(2, YutResult.GEOL),
        encode_action(3, YutResult.GEOL),
    ]


def test_reset_records_auto_pass_when_back_do_has_no_on_board_piece():
    env = YutnoriEnv(
        starting_player=0,
        opponent_policy=first_legal_action,
        yut_sampler_factory=sequence_factory(
            [YutResult.BACK_DO, YutResult.DO, YutResult.GAE]
        ),
    )

    _obs, info = env.reset(seed=9)

    assert info["initial_auto_passes"] == [
        {
            "player": 0,
            "rolls": ["BACK_DO"],
            "pool_counts": {
                "DO": 0,
                "GAE": 0,
                "GEOL": 0,
                "YUT": 0,
                "MO": 0,
                "BACK_DO": 1,
            },
            "reason": "NO_LEGAL_ACTION",
        }
    ]
    assert info["current_player"] == 0
    assert np.flatnonzero(env.action_masks()).tolist() == [
        encode_action(0, YutResult.GAE),
        encode_action(1, YutResult.GAE),
        encode_action(2, YutResult.GAE),
        encode_action(3, YutResult.GAE),
    ]


def test_reset_resamples_terminal_opponent_opening_before_learner_turn():
    env = YutnoriEnv(
        starting_player=1,
        opponent_policy=ProjectRFRuleBasedAgent().select_action,
        yut_sampler_factory=sequence_attempt_factory(
            [
                [YutResult.MO] * 10 + [YutResult.DO],
                [YutResult.DO, YutResult.GEOL],
            ]
        ),
    )

    _obs, info = env.reset(seed=9)

    assert env.state is not None
    assert env.state.winner is None
    assert info["skipped_terminal_resets"] == 1
    assert info["current_player"] == 0
    assert np.flatnonzero(env.action_masks()).tolist() == [
        encode_action(0, YutResult.GEOL),
        encode_action(1, YutResult.GEOL),
        encode_action(2, YutResult.GEOL),
        encode_action(3, YutResult.GEOL),
    ]


def test_illegal_action_raises_value_error():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.DO]),
    )
    env.reset(seed=1)

    with pytest.raises(ValueError):
        env.step(encode_action(0, YutResult.GAE))


def test_terminal_reward_when_learner_wins():
    env = YutnoriEnv(
        starting_player=0,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
    )
    env.reset(seed=1)
    assert env.state is not None
    for piece_id in range(3):
        env.state.pieces[0][piece_id] = Position.finished()
    env.state.pieces[0][3] = Position.at(Route.OUTER, 19)
    env.state.set_pool(YutResult.GAE)

    _obs, reward, terminated, truncated, info = env.step(
        encode_action(3, YutResult.GAE)
    )

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["winner"] == 0
    assert not env.action_masks().any()


def test_terminal_reward_mode_ignores_non_terminal_shaping_events():
    env = YutnoriEnv(
        starting_player=0,
        reward_mode=REWARD_MODE_TERMINAL,
        yut_sampler_factory=sequence_factory([YutResult.DO, YutResult.DO]),
    )
    env.reset(seed=1)
    assert env.state is not None
    env.state.pieces[0][0] = Position.at(Route.OUTER, 1)
    env.state.pieces[1][0] = Position.at(Route.OUTER, 3)
    env.state.pieces[1][1] = Position.at(Route.OUTER, 3)
    env.state.set_pool(YutResult.GAE)

    _obs, reward, terminated, _truncated, info = env.step(
        encode_action(0, YutResult.GAE)
    )

    assert reward == 0.0
    assert not terminated
    assert info["reward_mode"] == REWARD_MODE_TERMINAL
    assert info["terminal_reward"] == 0.0
    assert info["shaping_reward"] == 0.0
    assert info["learner_event"]["captured_count"] == 2


def test_rf_shaped_reward_adds_learner_capture_reward():
    env = YutnoriEnv(
        starting_player=0,
        reward_mode=REWARD_MODE_RF_SHAPED,
        yut_sampler_factory=sequence_factory([YutResult.DO, YutResult.DO]),
    )
    env.reset(seed=1)
    assert env.state is not None
    env.state.pieces[0][0] = Position.at(Route.OUTER, 1)
    env.state.pieces[1][0] = Position.at(Route.OUTER, 3)
    env.state.pieces[1][1] = Position.at(Route.OUTER, 3)
    env.state.set_pool(YutResult.GAE)

    _obs, reward, terminated, _truncated, info = env.step(
        encode_action(0, YutResult.GAE)
    )

    expected = 2 * RF_SHAPING_CAPTURE_WEIGHT
    assert reward == pytest.approx(expected)
    assert not terminated
    assert info["reward_mode"] == REWARD_MODE_RF_SHAPED
    assert info["terminal_reward"] == 0.0
    assert info["shaping_reward"] == pytest.approx(expected)


def test_rf_shaped_reward_adds_learner_shortcut_reward():
    env = YutnoriEnv(
        starting_player=0,
        reward_mode=REWARD_MODE_RF_SHAPED,
        yut_sampler_factory=sequence_factory([YutResult.DO]),
    )
    env.reset(seed=1)
    assert env.state is not None
    env.state.pieces[0][0] = Position.at(Route.OUTER, 1)
    env.state.set_pool(YutResult.YUT, YutResult.DO)

    _obs, reward, terminated, _truncated, info = env.step(
        encode_action(0, YutResult.YUT)
    )

    assert reward == pytest.approx(RF_SHAPING_SHORTCUT_BONUS)
    assert not terminated
    assert info["terminal_reward"] == 0.0
    assert info["shaping_reward"] == pytest.approx(RF_SHAPING_SHORTCUT_BONUS)
    assert info["learner_event"]["entered_shortcut"] is True


def test_rf_shaped_reward_penalizes_opponent_capture_events():
    env = YutnoriEnv(
        starting_player=0,
        reward_mode=REWARD_MODE_RF_SHAPED,
        opponent_policy=first_legal_action,
        yut_sampler_factory=sequence_factory(
            [YutResult.DO, YutResult.GAE, YutResult.DO, YutResult.DO]
        ),
    )
    env.reset(seed=1)
    assert env.state is not None
    env.state.pieces[0][0] = Position.at(Route.OUTER, 18)
    env.state.pieces[0][1] = Position.at(Route.OUTER, 3)
    env.state.pieces[1][0] = Position.at(Route.OUTER, 1)
    env.state.set_pool(YutResult.DO)

    _obs, reward, terminated, _truncated, info = env.step(
        encode_action(0, YutResult.DO)
    )

    assert not terminated
    assert len(info["opponent_events"]) == 2
    assert info["opponent_events"][0]["captured_count"] == 1
    assert reward == pytest.approx(-RF_SHAPING_CAPTURE_WEIGHT)
    assert info["terminal_reward"] == 0.0
    assert info["shaping_reward"] == pytest.approx(-RF_SHAPING_CAPTURE_WEIGHT)


def test_rf_shaped_reward_combines_terminal_and_shaping_rewards():
    env = YutnoriEnv(
        starting_player=0,
        reward_mode=REWARD_MODE_RF_SHAPED,
        yut_sampler_factory=sequence_factory([YutResult.GAE]),
    )
    env.reset(seed=1)
    assert env.state is not None
    for piece_id in range(3):
        env.state.pieces[0][piece_id] = Position.finished()
    env.state.pieces[0][3] = Position.at(Route.OUTER, 19)
    env.state.set_pool(YutResult.GAE)

    _obs, reward, terminated, _truncated, info = env.step(
        encode_action(3, YutResult.GAE)
    )

    assert terminated
    assert reward == pytest.approx(1.0 + RF_SHAPING_FINISH_WEIGHT)
    assert info["terminal_reward"] == 1.0
    assert info["shaping_reward"] == pytest.approx(RF_SHAPING_FINISH_WEIGHT)


def test_gymnasium_spaces_accept_reset_outputs():
    env = YutnoriEnv(starting_player=0)

    obs, info = env.reset(seed=10)

    assert env.action_space.n == ACTION_SIZE
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_mask_aware_random_rollouts_finish_without_illegal_actions():
    completed = 0
    for seed in range(100):
        env = YutnoriEnv(starting_player=0)
        obs, _info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        terminated = False
        for _step in range(5000):
            mask = env.action_masks()
            if terminated:
                break
            if not mask.any():
                raise AssertionError("non-terminal learner state has no legal actions")
            action = int(np.flatnonzero(mask)[0])
            obs, _reward, terminated, truncated, _info = env.step(action)
            assert not truncated
            assert env.observation_space.contains(obs)
            if terminated:
                completed += 1
                break
        else:
            raise AssertionError("rollout did not finish within safety bound")

    assert completed == 100
