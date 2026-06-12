"""Train a MaskablePPO policy on the Yutnori environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium  # noqa: E402
import sb3_contrib  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from tqdm.auto import tqdm  # noqa: E402

from yutnori.core import ACTION_SIZE  # noqa: E402
from yutnori.env import (  # noqa: E402
    OBSERVATION_MODES,
    REWARD_MODES,
    RULESET,
    observation_size,
)
from yutnori.training import (  # noqa: E402
    OPPONENT_NAMES,
    VEC_ENV_TYPES,
    evaluate_maskable_policy,
    make_yutnori_vec_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default="base")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="terminal")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=VEC_ENV_TYPES, default="dummy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--early-stop-eval-freq", type=int, default=0)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=100)
    parser.add_argument("--early-stop-opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--early-stop-win-rate", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-min-timesteps", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = _prepare_run_dir(args.run_dir, overwrite=args.overwrite)

    config = _config_dict(args, run_dir)
    _write_json(run_dir / "config.json", config)
    callback = _callbacks(args, run_dir)

    vec_env = make_yutnori_vec_env(
        opponent=args.opponent,
        n_envs=args.n_envs,
        seed=args.seed,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        vec_env_type=args.vec_env,
    )
    try:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard") if args.tensorboard else None,
            verbose=args.verbose,
        )

        eval_summary: dict[str, Any] = {}
        if args.eval_episodes > 0:
            before = evaluate_maskable_policy(
                model,
                opponent="random",
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                observation_mode=args.observation_mode,
                reward_mode=args.reward_mode,
                show_progress=not args.no_progress_bar,
                progress_desc="Eval before random",
            )
            eval_summary["before_random"] = before.to_dict()
            _write_json(run_dir / "eval_before_random.json", before.to_dict())

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            use_masking=True,
        )
        model_path = run_dir / "model.zip"
        model.save(model_path)

        if args.eval_episodes > 0:
            after = evaluate_maskable_policy(
                model,
                opponent="random",
                episodes=args.eval_episodes,
                seed=args.seed + 20_000,
                observation_mode=args.observation_mode,
                reward_mode=args.reward_mode,
                show_progress=not args.no_progress_bar,
                progress_desc="Eval after random",
            )
            eval_summary["after_random"] = after.to_dict()
            _write_json(run_dir / "eval_after_random.json", after.to_dict())

        episode_stats = _episode_stats_summary(
            run_dir,
            trained_timesteps=model.num_timesteps,
        )
        summary = {
            "model_path": str(model_path),
            "started_at": config["started_at"],
            "finished_at": datetime.now(UTC).isoformat(),
            "checkpoint_dir": config["checkpoint_dir"],
            "ruleset": RULESET,
            "action_size": ACTION_SIZE,
            "observation_size": observation_size(args.observation_mode),
            "observation_mode": args.observation_mode,
            "reward_mode": args.reward_mode,
            "target_total_timesteps": args.total_timesteps,
            "trained_timesteps": model.num_timesteps,
            "episode_stats": episode_stats,
            "evaluation": eval_summary,
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        vec_env.close()


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.n_steps <= 1:
        raise ValueError("n_steps must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.eval_episodes < 0:
        raise ValueError("eval_episodes must be non-negative")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint_freq must be non-negative")
    if args.early_stop_eval_freq < 0:
        raise ValueError("early_stop_eval_freq must be non-negative")
    if args.early_stop_eval_episodes <= 0:
        raise ValueError("early_stop_eval_episodes must be positive")
    if args.early_stop_win_rate is not None and not (
        0.0 <= args.early_stop_win_rate <= 1.0
    ):
        raise ValueError("early_stop_win_rate must be in [0, 1]")
    if args.early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if args.early_stop_min_delta < 0.0:
        raise ValueError("early_stop_min_delta must be non-negative")
    if args.early_stop_min_timesteps < 0:
        raise ValueError("early_stop_min_timesteps must be non-negative")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError("batch_size must be <= n_steps * n_envs")


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}. "
            "Use --overwrite or choose a new --run-dir."
        )
    return run_dir


class TqdmProgressCallback(BaseCallback):
    """Show env-timestep progress and ETA during long PPO runs."""

    def __init__(
        self,
        total_timesteps: int,
        *,
        episode_stats: "EpisodeStatsCallback | None" = None,
    ) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.episode_stats = episode_stats
        self._progress_bar: tqdm | None = None
        self._last_num_timesteps = 0
        self._last_postfix_timestep = 0
        self._last_postfix_episodes = 0

    def _on_training_start(self) -> None:
        self._last_num_timesteps = 0
        self._progress_bar = tqdm(
            total=self.total_timesteps,
            desc="PPO training",
            unit="ts",
            dynamic_ncols=True,
            leave=True,
        )

    def _on_step(self) -> bool:
        if self._progress_bar is None:
            return True

        delta = self.num_timesteps - self._last_num_timesteps
        if delta > 0:
            self._progress_bar.update(delta)
            self._last_num_timesteps = self.num_timesteps
            self._refresh_episode_postfix()
        return True

    def _on_training_end(self) -> None:
        if self._progress_bar is None:
            return

        self._refresh_episode_postfix(force=True)
        self._progress_bar.close()
        self._progress_bar = None

    def _refresh_episode_postfix(self, *, force: bool = False) -> None:
        if self._progress_bar is None or self.episode_stats is None:
            return

        stats = self.episode_stats.summary(trained_timesteps=self.num_timesteps)
        completed_episodes = int(stats["completed_episodes"])
        should_refresh = (
            force
            or completed_episodes != self._last_postfix_episodes
            or self.num_timesteps - self._last_postfix_timestep >= 1_000
        )
        if not should_refresh:
            return

        self._progress_bar.set_postfix(
            {
                "eps": completed_episodes,
                "ep_ts": f"{stats['average_learner_timesteps']:.1f}",
                "ep/100k": f"{stats['episodes_per_100k_timesteps']:.1f}",
                "ep_wr": f"{stats['learner_win_rate']:.3f}",
            },
            refresh=False,
        )
        self._last_postfix_timestep = self.num_timesteps
        self._last_postfix_episodes = completed_episodes


class MaskableEarlyStoppingCallback(BaseCallback):
    """Periodically evaluate with action masks and stop on configured criteria."""

    def __init__(
        self,
        *,
        eval_freq: int,
        eval_episodes: int,
        opponent: str,
        seed: int,
        observation_mode: str,
        reward_mode: str,
        min_timesteps: int,
        win_rate_threshold: float | None,
        patience: int,
        min_delta: float,
        output_path: Path,
        episode_stats: "EpisodeStatsCallback | None" = None,
        show_progress: bool = True,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponent = opponent
        self.seed = seed
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.min_timesteps = min_timesteps
        self.win_rate_threshold = win_rate_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.output_path = output_path
        self.episode_stats = episode_stats
        self.show_progress = show_progress
        self._next_eval_timestep = eval_freq
        self._eval_index = 0
        self._best_win_rate: float | None = None
        self._no_improvement_count = 0

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_timestep:
            return True

        result = evaluate_maskable_policy(
            self.model,
            opponent=self.opponent,
            episodes=self.eval_episodes,
            seed=self.seed + self._eval_index,
            observation_mode=self.observation_mode,
            reward_mode=self.reward_mode,
            show_progress=self.show_progress,
            progress_desc=f"Early eval {self._eval_index + 1}",
        )
        self._eval_index += 1

        improved = self._is_improved(result.win_rate)
        if improved:
            self._best_win_rate = result.win_rate
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        can_stop = self.num_timesteps >= self.min_timesteps
        stop_reason = self._stop_reason(result.win_rate) if can_stop else None
        episode_summary = self._episode_summary()
        payload = {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "timesteps": self.num_timesteps,
            "eval_index": self._eval_index,
            "observation_mode": self.observation_mode,
            "reward_mode": self.reward_mode,
            "best_win_rate": self._best_win_rate,
            "improved": improved,
            "no_improvement_count": self._no_improvement_count,
            "stop_reason": stop_reason,
            "training_episode_stats": episode_summary,
            "result": result.to_dict(),
        }
        with self.output_path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

        message = (
            f"eval {self._eval_index}: timesteps={self.num_timesteps}, "
            f"opponent={self.opponent}, win_rate={result.win_rate:.4f}, "
            f"illegal={result.illegal_action_count}, "
            f"best={self._best_win_rate:.4f}"
        )
        if episode_summary is not None:
            message = (
                f"{message}, train_eps={episode_summary['completed_episodes']}, "
                f"avg_ep_ts={episode_summary['average_learner_timesteps']:.1f}, "
                f"avg_decisions={episode_summary['average_decisions']:.1f}, "
                f"ep/100k={episode_summary['episodes_per_100k_timesteps']:.1f}, "
                f"train_ep_wr={episode_summary['learner_win_rate']:.3f}"
            )
        if stop_reason is not None:
            message = f"{message}, stop_reason={stop_reason}"
        tqdm.write(message)

        self._next_eval_timestep += self.eval_freq
        return stop_reason is None

    def _is_improved(self, win_rate: float) -> bool:
        if self._best_win_rate is None:
            return True
        return win_rate > self._best_win_rate + self.min_delta

    def _stop_reason(self, win_rate: float) -> str | None:
        if (
            self.win_rate_threshold is not None
            and win_rate >= self.win_rate_threshold
        ):
            return f"win_rate>={self.win_rate_threshold}"
        if self.patience > 0 and self._no_improvement_count >= self.patience:
            return f"no_improvement_patience={self.patience}"
        return None

    def _episode_summary(self) -> dict[str, Any] | None:
        if self.episode_stats is None:
            return None
        return self.episode_stats.summary(trained_timesteps=self.num_timesteps)


class EpisodeStatsCallback(BaseCallback):
    """Record completed episode statistics during training."""

    def __init__(self, output_path: Path) -> None:
        super().__init__(verbose=0)
        self.output_path = output_path
        self._episode_lengths: list[int] = []
        self._completed_episodes = 0
        self._learner_wins = 0
        self._total_turns = 0
        self._total_decisions = 0
        self._total_learner_timesteps = 0
        self._max_decisions = 0
        self._max_turns = 0
        self._max_learner_timesteps = 0
        self._last_completed_episode_timestep = 0
        self._back_do_stats: dict[str, int] = {}

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")
        env_count = int(getattr(self.training_env, "num_envs", 1))
        self._episode_lengths = [0 for _ in range(env_count)]

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for env_index, done in enumerate(dones):
            self._episode_lengths[env_index] += 1
            if bool(done):
                info = infos[env_index]
                self._record_episode(env_index, info)
        return True

    def _record_episode(self, env_index: int, info: dict[str, Any]) -> None:
        self._completed_episodes += 1
        self._last_completed_episode_timestep = self.num_timesteps
        learner_player = int(info["learner_player"])
        winner = info["winner"]
        learner_win = winner == learner_player
        if learner_win:
            self._learner_wins += 1

        turn_count = int(info["turn_count"])
        decision_count = int(info["decision_count"])
        learner_decisions = self._episode_lengths[env_index]
        self._total_turns += turn_count
        self._total_decisions += decision_count
        self._total_learner_timesteps += learner_decisions
        self._max_turns = max(self._max_turns, turn_count)
        self._max_decisions = max(self._max_decisions, decision_count)
        self._max_learner_timesteps = max(
            self._max_learner_timesteps,
            learner_decisions,
        )
        for name, count in info["back_do_stats"].items():
            self._back_do_stats[name] = self._back_do_stats.get(name, 0) + int(count)

        payload = {
            "completed_episodes": self._completed_episodes,
            "timesteps": self.num_timesteps,
            "env_index": env_index,
            "learner_player": learner_player,
            "winner": winner,
            "learner_win": learner_win,
            "learner_decisions": learner_decisions,
            "turn_count": turn_count,
            "decision_count": decision_count,
            "back_do_stats": info["back_do_stats"],
        }
        with self.output_path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

        self._episode_lengths[env_index] = 0

    def summary(self, *, trained_timesteps: int | None = None) -> dict[str, Any]:
        if self._completed_episodes == 0:
            return {
                "completed_episodes": 0,
                "learner_wins": 0,
                "learner_win_rate": 0.0,
                "average_learner_timesteps": 0.0,
                "average_turns": 0.0,
                "average_decisions": 0.0,
                "max_learner_timesteps": 0,
                "max_turns": 0,
                "max_decisions": 0,
                "episodes_per_100k_timesteps": 0.0,
                "last_completed_episode_timestep": 0,
                "back_do_stats": {},
            }
        return {
            "completed_episodes": self._completed_episodes,
            "learner_wins": self._learner_wins,
            "learner_win_rate": self._learner_wins / self._completed_episodes,
            "average_learner_timesteps": (
                self._total_learner_timesteps / self._completed_episodes
            ),
            "average_turns": self._total_turns / self._completed_episodes,
            "average_decisions": self._total_decisions / self._completed_episodes,
            "max_learner_timesteps": self._max_learner_timesteps,
            "max_turns": self._max_turns,
            "max_decisions": self._max_decisions,
            "episodes_per_100k_timesteps": (
                0.0
                if not trained_timesteps
                else self._completed_episodes / trained_timesteps * 100_000
            ),
            "last_completed_episode_timestep": self._last_completed_episode_timestep,
            "back_do_stats": self._back_do_stats.copy(),
        }


def _callbacks(
    args: argparse.Namespace,
    run_dir: Path,
) -> CallbackList | BaseCallback | None:
    callbacks: list[BaseCallback] = []
    episode_stats_callback = EpisodeStatsCallback(run_dir / "episodes.jsonl")
    callbacks.append(episode_stats_callback)
    if not args.no_progress_bar:
        callbacks.append(
            TqdmProgressCallback(
                args.total_timesteps,
                episode_stats=episode_stats_callback,
            )
        )

    checkpoint_callback = _checkpoint_callback(args, run_dir)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    early_stopping_callback = _early_stopping_callback(
        args,
        run_dir,
        episode_stats=episode_stats_callback,
    )
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)

    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)


def _checkpoint_callback(
    args: argparse.Namespace,
    run_dir: Path,
) -> CheckpointCallback | None:
    if args.checkpoint_freq == 0:
        return None

    checkpoint_dir = _resolve_checkpoint_dir(args, run_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_freq_calls = _checkpoint_save_freq_calls(args)
    return CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_yutnori",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def _checkpoint_save_freq_calls(args: argparse.Namespace) -> int | None:
    if args.checkpoint_freq == 0:
        return None
    return max((args.checkpoint_freq + args.n_envs - 1) // args.n_envs, 1)


def _early_stopping_callback(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    episode_stats: EpisodeStatsCallback | None = None,
) -> MaskableEarlyStoppingCallback | None:
    if args.early_stop_eval_freq == 0:
        return None

    return MaskableEarlyStoppingCallback(
        eval_freq=args.early_stop_eval_freq,
        eval_episodes=args.early_stop_eval_episodes,
        opponent=args.early_stop_opponent,
        seed=args.seed + 30_000,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        min_timesteps=args.early_stop_min_timesteps,
        win_rate_threshold=args.early_stop_win_rate,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        output_path=run_dir / "eval_during_training.jsonl",
        episode_stats=episode_stats,
        show_progress=not args.no_progress_bar,
    )


def _resolve_checkpoint_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.checkpoint_freq == 0:
        return None
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return run_dir / "checkpoints"


def _config_dict(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    checkpoint_dir = _resolve_checkpoint_dir(args, run_dir)
    return {
        "command": sys.argv,
        "git_commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "ruleset": RULESET,
        "action_size": ACTION_SIZE,
        "observation_size": observation_size(args.observation_mode),
        "opponent": args.opponent,
        "observation_mode": args.observation_mode,
        "reward_mode": args.reward_mode,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "checkpoint_save_freq_calls": _checkpoint_save_freq_calls(args),
        "early_stop_eval_freq": args.early_stop_eval_freq,
        "early_stop_eval_episodes": args.early_stop_eval_episodes,
        "early_stop_opponent": args.early_stop_opponent,
        "early_stop_win_rate": args.early_stop_win_rate,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_min_timesteps": args.early_stop_min_timesteps,
        "early_stop_eval_log": (
            None
            if args.early_stop_eval_freq == 0
            else str(run_dir / "eval_during_training.jsonl")
        ),
        "episode_stats_log": str(run_dir / "episodes.jsonl"),
        "tensorboard": args.tensorboard,
        "progress_bar": not args.no_progress_bar,
        "system": _system_info(),
    }


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _episode_stats_summary(
    run_dir: Path,
    *,
    trained_timesteps: int,
) -> dict[str, Any]:
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        return {
            "completed_episodes": 0,
            "learner_wins": 0,
            "learner_win_rate": 0.0,
            "average_learner_timesteps": 0.0,
            "average_turns": 0.0,
            "average_decisions": 0.0,
            "max_learner_timesteps": 0,
            "max_turns": 0,
            "max_decisions": 0,
            "episodes_per_100k_timesteps": 0.0,
            "last_completed_episode_timestep": 0,
            "back_do_stats": {},
        }

    completed = 0
    learner_wins = 0
    total_turns = 0
    total_decisions = 0
    total_learner_timesteps = 0
    max_turns = 0
    max_decisions = 0
    max_learner_timesteps = 0
    final_timesteps = 0
    back_do_stats: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        item = json.loads(line)
        completed += 1
        learner_wins += int(bool(item["learner_win"]))
        turn_count = int(item["turn_count"])
        decision_count = int(item["decision_count"])
        learner_timesteps = int(item["learner_decisions"])
        total_turns += turn_count
        total_decisions += decision_count
        total_learner_timesteps += learner_timesteps
        max_turns = max(max_turns, turn_count)
        max_decisions = max(max_decisions, decision_count)
        max_learner_timesteps = max(max_learner_timesteps, learner_timesteps)
        final_timesteps = max(final_timesteps, int(item["timesteps"]))
        for name, count in item.get("back_do_stats", {}).items():
            back_do_stats[name] = back_do_stats.get(name, 0) + int(count)

    if completed == 0:
        return {
            "completed_episodes": 0,
            "learner_wins": 0,
            "learner_win_rate": 0.0,
            "average_learner_timesteps": 0.0,
            "average_turns": 0.0,
            "average_decisions": 0.0,
            "max_learner_timesteps": 0,
            "max_turns": 0,
            "max_decisions": 0,
            "episodes_per_100k_timesteps": 0.0,
            "last_completed_episode_timestep": 0,
            "back_do_stats": {},
        }

    return {
        "completed_episodes": completed,
        "learner_wins": learner_wins,
        "learner_win_rate": learner_wins / completed,
        "average_learner_timesteps": total_learner_timesteps / completed,
        "average_turns": total_turns / completed,
        "average_decisions": total_decisions / completed,
        "max_learner_timesteps": max_learner_timesteps,
        "max_turns": max_turns,
        "max_decisions": max_decisions,
        "episodes_per_100k_timesteps": (
            0.0
            if trained_timesteps == 0
            else completed / trained_timesteps * 100_000
        ),
        "last_completed_episode_timestep": final_timesteps,
        "back_do_stats": back_do_stats,
    }


if __name__ == "__main__":
    main()
