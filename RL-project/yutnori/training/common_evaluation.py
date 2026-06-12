"""Paired-seed evaluation against the frozen common rule-based agent."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from math import sqrt
from collections.abc import Callable
from typing import Protocol, Sequence

import numpy as np
from tqdm.auto import tqdm

from yutnori.agents import CommonRuleBasedAgent
from yutnori.core import ACTION_SIZE, GameState, YutSampler
from yutnori.env import OBSERVATION_MODE_BASE, RULESET, encode_observation

COMMON_EVALUATION_PROTOCOL = "common_rule_based_paired_full_backdo_v1"
COMMON_RULE_OPPONENT = "common_rule_based"


class CommonMaskablePredictor(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        ...


class CommonStateAgent(Protocol):
    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        ...


CommonActionSelector = Callable[[GameState, list[int]], int]


@dataclass(frozen=True)
class CommonEvaluationSplit:
    games: int
    wins: int
    losses: int
    win_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True)
class CommonEvaluationError:
    base_seed: int
    model_starts: bool
    error_type: str
    message: str

    def to_dict(self) -> dict[str, int | bool | str]:
        return {
            "base_seed": self.base_seed,
            "model_starts": self.model_starts,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class CommonPolicyEvaluationResult:
    base_seeds: tuple[int, ...]
    scheduled_games: int
    completed_games: int
    wins: int
    losses: int
    win_rate: float
    model_first: CommonEvaluationSplit
    model_second: CommonEvaluationSplit
    confidence_interval_95: tuple[float, float]
    average_turns: float
    average_decisions: float
    illegal_action_count: int
    evaluation_errors: tuple[CommonEvaluationError, ...]
    elapsed_seconds: float
    back_do_stats: dict[str, int]

    @property
    def evaluation_error_count(self) -> int:
        return len(self.evaluation_errors)

    @property
    def seed_sha256(self) -> str:
        return seed_list_sha256(self.base_seeds)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": COMMON_EVALUATION_PROTOCOL,
            "ruleset": RULESET,
            "opponent": COMMON_RULE_OPPONENT,
            "base_seed_count": len(self.base_seeds),
            "base_seed_sha256": self.seed_sha256,
            "scheduled_games": self.scheduled_games,
            "completed_games": self.completed_games,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "model_first": self.model_first.to_dict(),
            "model_second": self.model_second.to_dict(),
            "confidence_interval_95": {
                "method": "wilson",
                "lower": self.confidence_interval_95[0],
                "upper": self.confidence_interval_95[1],
            },
            "average_turns": self.average_turns,
            "average_decisions": self.average_decisions,
            "illegal_action_count": self.illegal_action_count,
            "evaluation_error_count": self.evaluation_error_count,
            "evaluation_errors": [
                evaluation_error.to_dict()
                for evaluation_error in self.evaluation_errors
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "back_do_stats": self.back_do_stats,
            "average_game_seconds": (
                0.0
                if self.scheduled_games == 0
                else self.elapsed_seconds / self.scheduled_games
            ),
        }


@dataclass(frozen=True)
class _GameResult:
    model_won: bool
    turns: int
    decisions: int
    illegal_action: bool = False
    back_do_stats: dict[str, int] | None = None


def evaluate_common_rule_policy(
    model: CommonMaskablePredictor,
    *,
    base_seeds: Sequence[int],
    observation_mode: str = OBSERVATION_MODE_BASE,
    deterministic: bool = True,
    max_decisions: int = 10_000,
    show_progress: bool = False,
    progress_desc: str = "Common rule evaluation",
) -> CommonPolicyEvaluationResult:
    """Evaluate both starting positions for every independent base seed."""

    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)

    def select_model_action(state: GameState, legal_actions: list[int]) -> int:
        observation = encode_observation(
            state,
            state.current_player,
            observation_mode=observation_mode,
        )
        action_mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        action_mask[legal_actions] = True
        action, _model_state = model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_mask,
        )
        return int(np.asarray(action).item())

    return _evaluate_common_rule_selector(
        select_model_action,
        base_seeds=base_seeds,
        max_decisions=max_decisions,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def evaluate_common_rule_agent(
    agent: CommonStateAgent,
    *,
    base_seeds: Sequence[int],
    max_decisions: int = 10_000,
    show_progress: bool = False,
    progress_desc: str = "Common rule agent evaluation",
) -> CommonPolicyEvaluationResult:
    """Evaluate a state-based agent with the same paired-seed protocol."""

    return _evaluate_common_rule_selector(
        agent.select_action,
        base_seeds=base_seeds,
        max_decisions=max_decisions,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def _evaluate_common_rule_selector(
    select_model_action: CommonActionSelector,
    *,
    base_seeds: Sequence[int],
    max_decisions: int,
    show_progress: bool,
    progress_desc: str,
) -> CommonPolicyEvaluationResult:
    seeds = _validate_base_seeds(base_seeds)
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    first_wins = 0
    first_losses = 0
    second_wins = 0
    second_losses = 0
    total_turns = 0
    total_decisions = 0
    illegal_action_count = 0
    errors: list[CommonEvaluationError] = []
    completed_games = 0
    back_do_stats: dict[str, int] = {}
    started_at = time.perf_counter()

    game_specs = (
        (base_seed, model_starts)
        for base_seed in seeds
        for model_starts in (True, False)
    )
    if show_progress:
        game_specs = tqdm(
            game_specs,
            total=len(seeds) * 2,
            desc=progress_desc,
            unit="game",
            dynamic_ncols=True,
            leave=True,
        )

    for base_seed, model_starts in game_specs:
        try:
            game = _play_common_game(
                select_model_action,
                base_seed=base_seed,
                model_starts=model_starts,
                max_decisions=max_decisions,
            )
        except Exception as exc:  # Evaluation errors are reported, not scored.
            errors.append(
                CommonEvaluationError(
                    base_seed=base_seed,
                    model_starts=model_starts,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        completed_games += 1
        total_turns += game.turns
        total_decisions += game.decisions
        illegal_action_count += int(game.illegal_action)
        for name, count in (game.back_do_stats or {}).items():
            back_do_stats[name] = back_do_stats.get(name, 0) + int(count)
        if model_starts:
            first_wins += int(game.model_won)
            first_losses += int(not game.model_won)
        else:
            second_wins += int(game.model_won)
            second_losses += int(not game.model_won)

        if show_progress and hasattr(game_specs, "set_postfix"):
            wins = first_wins + second_wins
            game_specs.set_postfix(
                {
                    "wr": f"{wins / completed_games:.3f}",
                    "first": f"{first_wins / max(1, first_wins + first_losses):.3f}",
                    "second": (
                        f"{second_wins / max(1, second_wins + second_losses):.3f}"
                    ),
                },
                refresh=False,
            )

    elapsed_seconds = time.perf_counter() - started_at
    wins = first_wins + second_wins
    losses = first_losses + second_losses
    win_rate = 0.0 if completed_games == 0 else wins / completed_games
    return CommonPolicyEvaluationResult(
        base_seeds=seeds,
        scheduled_games=len(seeds) * 2,
        completed_games=completed_games,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        model_first=_split(first_wins, first_losses),
        model_second=_split(second_wins, second_losses),
        confidence_interval_95=wilson_interval(wins, completed_games),
        average_turns=(
            0.0 if completed_games == 0 else total_turns / completed_games
        ),
        average_decisions=(
            0.0 if completed_games == 0 else total_decisions / completed_games
        ),
        illegal_action_count=illegal_action_count,
        evaluation_errors=tuple(errors),
        elapsed_seconds=elapsed_seconds,
        back_do_stats=back_do_stats,
    )


def wilson_interval(
    wins: int,
    games: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if games < 0:
        raise ValueError("games must be non-negative")
    if wins < 0 or wins > games:
        raise ValueError("wins must be in [0, games]")
    if games == 0:
        return 0.0, 0.0
    proportion = wins / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    half_width = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / games
            + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return center - half_width, center + half_width


def seed_list_sha256(base_seeds: Sequence[int]) -> str:
    payload = json.dumps(list(base_seeds), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _play_common_game(
    select_model_action: CommonActionSelector,
    *,
    base_seed: int,
    model_starts: bool,
    max_decisions: int,
) -> _GameResult:
    model_player = 0
    starting_player = model_player if model_starts else 1
    state = GameState(
        starting_player=starting_player,
        yut_sampler=YutSampler(seed=base_seed),
    )
    rule_agent = CommonRuleBasedAgent()
    state.start_turn()

    while state.winner is None:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("non-terminal state has no legal actions")

        if state.current_player == model_player:
            action_int = int(select_model_action(state, legal_actions))
            if (
                action_int < 0
                or action_int >= ACTION_SIZE
                or action_int not in legal_actions
            ):
                return _GameResult(
                    model_won=False,
                    turns=state.turn_count,
                    decisions=state.decision_count,
                    illegal_action=True,
                    back_do_stats=state.back_do_stats(),
                )
        else:
            action_int = rule_agent.select_action(state, legal_actions)

        state.apply_action(action_int)
        if state.decision_count > max_decisions:
            raise RuntimeError(
                f"evaluation game exceeded max_decisions={max_decisions}"
            )

    return _GameResult(
        model_won=state.winner == model_player,
        turns=state.turn_count,
        decisions=state.decision_count,
        back_do_stats=state.back_do_stats(),
    )


def _validate_base_seeds(base_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("base_seeds must not be empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("base_seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("base_seeds must be unique")
    return seeds


def _split(wins: int, losses: int) -> CommonEvaluationSplit:
    games = wins + losses
    return CommonEvaluationSplit(
        games=games,
        wins=wins,
        losses=losses,
        win_rate=0.0 if games == 0 else wins / games,
    )
