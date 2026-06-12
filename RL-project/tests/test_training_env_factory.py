import numpy as np
import pytest
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv

from yutnori.agents import ProjectRFRuleBasedAgent
from yutnori.core import ACTION_SIZE
from yutnori.env import REWARD_MODE_RF_SHAPED, TACTICAL_OBSERVATION_SIZE
from yutnori.training import (
    OPPONENT_NAMES,
    VEC_ENV_SUBPROC,
    make_yutnori_env,
    make_yutnori_vec_env,
)
from yutnori.training.env_factory import make_opponent


def test_project_rf_rule_opponent_is_registered():
    assert "project_rf_rule" in OPPONENT_NAMES
    assert isinstance(make_opponent("project_rf_rule"), ProjectRFRuleBasedAgent)


def test_make_opponent_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_opponent("unknown")


@pytest.mark.parametrize("opponent", OPPONENT_NAMES)
def test_make_yutnori_env_supports_all_baseline_opponents(opponent):
    env = make_yutnori_env(opponent=opponent, seed=11)

    try:
        obs, _info = env.reset(seed=12)
        mask = env.action_masks()

        assert env.observation_space.contains(obs)
        assert mask.dtype == np.bool_
        assert mask.shape == (ACTION_SIZE,)
        assert mask.any()
    finally:
        env.close()


def test_make_yutnori_env_is_reproducible_for_same_seed():
    first_env = make_yutnori_env(
        opponent="random",
        seed=21,
        starting_player=1,
    )
    second_env = make_yutnori_env(
        opponent="random",
        seed=21,
        starting_player=1,
    )

    try:
        first_obs, first_info = first_env.reset(seed=22)
        second_obs, second_info = second_env.reset(seed=22)

        np.testing.assert_array_equal(first_obs, second_obs)
        np.testing.assert_array_equal(
            first_env.action_masks(),
            second_env.action_masks(),
        )
        assert first_info["initial_rolls"] == second_info["initial_rolls"]
        assert first_info["opponent_events"] == second_info["opponent_events"]
    finally:
        first_env.close()
        second_env.close()


def test_make_yutnori_vec_env_exposes_action_masks():
    vec_env = make_yutnori_vec_env(opponent="random", n_envs=2, seed=31)

    try:
        obs = vec_env.reset()
        masks = get_action_masks(vec_env)

        assert obs.shape[0] == 2
        assert masks.dtype == np.bool_
        assert masks.shape == (2, ACTION_SIZE)
        assert masks.any(axis=1).all()
    finally:
        vec_env.close()


def test_make_yutnori_vec_env_supports_subprocess_workers():
    vec_env = make_yutnori_vec_env(
        opponent="project_rf_rule",
        n_envs=2,
        seed=35,
        observation_mode="tactical",
        vec_env_type=VEC_ENV_SUBPROC,
    )

    try:
        obs = vec_env.reset()
        masks = get_action_masks(vec_env)

        assert isinstance(vec_env, SubprocVecEnv)
        assert obs.shape == (2, TACTICAL_OBSERVATION_SIZE)
        assert masks.dtype == np.bool_
        assert masks.shape == (2, ACTION_SIZE)
        assert masks.any(axis=1).all()
    finally:
        vec_env.close()


def test_make_yutnori_vec_env_rejects_unknown_type():
    with pytest.raises(ValueError, match="vec_env_type"):
        make_yutnori_vec_env(vec_env_type="unknown")


def test_make_yutnori_env_supports_tactical_observation_mode():
    env = make_yutnori_env(
        opponent="project_rf_rule",
        seed=41,
        observation_mode="tactical",
    )

    try:
        obs, _info = env.reset(seed=42)
        mask = env.action_masks()

        assert obs.shape == (TACTICAL_OBSERVATION_SIZE,)
        assert env.observation_space.contains(obs)
        assert mask.dtype == np.bool_
        assert mask.shape == (ACTION_SIZE,)
        assert mask.any()
    finally:
        env.close()


def test_make_yutnori_env_supports_reward_mode():
    env = make_yutnori_env(
        opponent="project_rf_rule",
        seed=45,
        reward_mode=REWARD_MODE_RF_SHAPED,
    )

    try:
        _obs, info = env.reset(seed=46)

        assert env.reward_mode == REWARD_MODE_RF_SHAPED
        assert info["reward_mode"] == REWARD_MODE_RF_SHAPED
    finally:
        env.close()


def test_make_yutnori_vec_env_supports_tactical_observation_mode():
    vec_env = make_yutnori_vec_env(
        opponent="project_rf_rule",
        n_envs=2,
        seed=51,
        observation_mode="tactical",
    )

    try:
        obs = vec_env.reset()
        masks = get_action_masks(vec_env)

        assert obs.shape == (2, TACTICAL_OBSERVATION_SIZE)
        assert masks.dtype == np.bool_
        assert masks.shape == (2, ACTION_SIZE)
        assert masks.any(axis=1).all()
    finally:
        vec_env.close()


def test_make_yutnori_vec_env_supports_reward_mode():
    vec_env = make_yutnori_vec_env(
        opponent="project_rf_rule",
        n_envs=2,
        seed=61,
        reward_mode=REWARD_MODE_RF_SHAPED,
    )

    try:
        vec_env.reset()

        assert vec_env.get_attr("reward_mode") == [
            REWARD_MODE_RF_SHAPED,
            REWARD_MODE_RF_SHAPED,
        ]
    finally:
        vec_env.close()
