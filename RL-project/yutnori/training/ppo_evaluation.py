"""Mask-aware evaluation helpers for PPO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from tqdm.auto import tqdm

from yutnori.env import OBSERVATION_MODE_BASE, REWARD_MODE_TERMINAL
from yutnori.training.env_factory import make_yutnori_env


class MaskablePredictor(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        ...


@dataclass(frozen=True)
class PolicyEvaluationResult:
    opponent: str
    episodes: int
    learner_player: int
    wins: int
    losses: int
    win_rate: float
    average_turns: float
    average_decisions: float
    illegal_action_count: int
    starting_player_counts: dict[int, int]
    back_do_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "opponent": self.opponent,
            "episodes": self.episodes,
            "learner_player": self.learner_player,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "average_turns": self.average_turns,
            "average_decisions": self.average_decisions,
            "illegal_action_count": self.illegal_action_count,
            "starting_player_counts": {
                str(player): count
                for player, count in sorted(self.starting_player_counts.items())
            },
            "back_do_stats": self.back_do_stats,
        }


def evaluate_maskable_policy(
    model: MaskablePredictor,
    *,
    opponent: str,
    episodes: int,
    seed: int,
    learner_player: int = 0,
    observation_mode: str = OBSERVATION_MODE_BASE,
    reward_mode: str = REWARD_MODE_TERMINAL,
    deterministic: bool = True,
    max_decisions: int = 10_000,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> PolicyEvaluationResult:
    """Evaluate a MaskablePPO-style model while passing masks to predict()."""

    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    env = make_yutnori_env(
        opponent=opponent,
        seed=seed,
        learner_player=learner_player,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    wins = 0
    total_turns = 0
    total_decisions = 0
    illegal_action_count = 0
    starting_player_counts = {0: 0, 1: 0}
    back_do_stats: dict[str, int] = {}

    try:
        episode_iter = range(episodes)
        if show_progress:
            episode_iter = tqdm(
                episode_iter,
                total=episodes,
                desc=progress_desc or f"Evaluating vs {opponent}",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )

        for episode in episode_iter:
            obs, info = env.reset(seed=seed + episode)
            starting_player = int(info["starting_player"])
            starting_player_counts[starting_player] += 1
            terminated = False
            truncated = False

            while not (terminated or truncated):
                mask = env.action_masks()
                if not mask.any():
                    raise RuntimeError("non-terminal learner state has no legal actions")
                action, _state = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=mask,
                )
                action_int = int(np.asarray(action).item())
                if action_int < 0 or action_int >= mask.shape[0] or not mask[action_int]:
                    illegal_action_count += 1
                    raise ValueError(
                        f"model selected illegal action {action_int}; "
                        f"legal_actions={np.flatnonzero(mask).tolist()}"
                    )

                obs, _reward, terminated, truncated, info = env.step(action_int)
                if int(info["decision_count"]) > max_decisions:
                    raise RuntimeError(
                        f"evaluation game exceeded max_decisions={max_decisions}"
                    )

            winner = info["winner"]
            if winner == learner_player:
                wins += 1
            total_turns += int(info["turn_count"])
            total_decisions += int(info["decision_count"])
            for name, count in info["back_do_stats"].items():
                back_do_stats[name] = back_do_stats.get(name, 0) + int(count)

            if show_progress and hasattr(episode_iter, "set_postfix"):
                completed = episode + 1
                episode_iter.set_postfix(
                    {
                        "wr": f"{wins / completed:.3f}",
                        "avg_dec": f"{total_decisions / completed:.1f}",
                    },
                    refresh=False,
                )
    finally:
        env.close()

    losses = episodes - wins
    return PolicyEvaluationResult(
        opponent=opponent,
        episodes=episodes,
        learner_player=learner_player,
        wins=wins,
        losses=losses,
        win_rate=0.0 if episodes == 0 else wins / episodes,
        average_turns=0.0 if episodes == 0 else total_turns / episodes,
        average_decisions=0.0 if episodes == 0 else total_decisions / episodes,
        illegal_action_count=illegal_action_count,
        starting_player_counts=starting_player_counts,
        back_do_stats=back_do_stats,
    )
