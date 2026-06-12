from argparse import Namespace
from io import BytesIO
from pathlib import Path

from scripts.run_ppo_long_sweep import (
    _CarriageReturnLogFilter,
    _eval_command,
    _run_name,
    _train_command,
)


def _filtered_log(*chunks: bytes) -> bytes:
    output = BytesIO()
    log_filter = _CarriageReturnLogFilter(output)
    for chunk in chunks:
        log_filter.write(chunk)
    log_filter.close()
    return output.getvalue()


def test_progress_log_filter_collapses_carriage_return_redraws():
    output = _filtered_log(
        b"started\r\n",
        b"\rPPO training: 1%",
        b"\rPPO training: 2%",
        b"\rPPO training: 100%\r\n",
        b"finished\r\n",
    )

    assert output == b"started\nPPO training: 100%\nfinished\n"


def test_progress_log_filter_handles_split_crlf():
    output = _filtered_log(b"summary line\r", b"\n")

    assert output == b"summary line\n"


def test_progress_log_filter_flushes_final_unterminated_line():
    output = _filtered_log(b"\rEvaluate random: 100%")

    assert output == b"Evaluate random: 100%\n"


def test_run_name_keeps_base_names_backward_compatible():
    assert _run_name("random", 1, "10m", 16, "base", "terminal") == (
        "random_seed_1_10m_nenv16"
    )


def test_run_name_adds_suffix_for_tactical_observations():
    assert _run_name("random", 1, "10m", 16, "tactical", "terminal") == (
        "random_seed_1_10m_nenv16_tactical"
    )


def test_run_name_adds_suffix_for_rf_shaped_reward():
    assert _run_name("random", 1, "10m", 16, "base", "rf_shaped") == (
        "random_seed_1_10m_nenv16_rf_shaped"
    )


def test_run_name_combines_observation_and_reward_suffixes():
    assert _run_name("random", 1, "10m", 16, "tactical", "rf_shaped") == (
        "random_seed_1_10m_nenv16_tactical_rf_shaped"
    )


def test_train_command_passes_observation_mode():
    command = _train_command(
        _args(observation_mode="tactical"),
        "project_rf_rule",
        7,
        Path("runs/ppo/project_rf_rule_seed_7"),
    )

    mode_index = command.index("--observation-mode")
    assert command[mode_index + 1] == "tactical"


def test_train_command_passes_reward_mode():
    command = _train_command(
        _args(reward_mode="rf_shaped"),
        "project_rf_rule",
        7,
        Path("runs/ppo/project_rf_rule_seed_7"),
    )

    mode_index = command.index("--reward-mode")
    assert command[mode_index + 1] == "rf_shaped"


def test_train_command_passes_vec_env_type():
    command = _train_command(
        _args(vec_env="subproc"),
        "project_rf_rule",
        7,
        Path("runs/ppo/project_rf_rule_seed_7"),
    )

    mode_index = command.index("--vec-env")
    assert command[mode_index + 1] == "subproc"


def test_eval_command_passes_observation_mode():
    command = _eval_command(
        _args(observation_mode="tactical"),
        Path("runs/ppo/model.zip"),
        "project_rf_rule",
        7,
        Path("runs/ppo/eval.json"),
    )

    mode_index = command.index("--observation-mode")
    assert command[mode_index + 1] == "tactical"


def test_eval_command_passes_reward_mode():
    command = _eval_command(
        _args(reward_mode="rf_shaped"),
        Path("runs/ppo/model.zip"),
        "project_rf_rule",
        7,
        Path("runs/ppo/eval.json"),
    )

    mode_index = command.index("--reward-mode")
    assert command[mode_index + 1] == "rf_shaped"


def _args(**overrides):
    defaults = {
        "total_timesteps": 10_000,
        "n_envs": 2,
        "vec_env": "dummy",
        "device": "cpu",
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "checkpoint_freq": 0,
        "train_eval_episodes": 0,
        "final_eval_episodes": 10,
        "overwrite": False,
        "no_progress_bar": True,
        "observation_mode": "base",
        "reward_mode": "terminal",
    }
    defaults.update(overrides)
    return Namespace(**defaults)
