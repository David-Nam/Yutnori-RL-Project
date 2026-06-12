"""Paired head-to-head evaluation for legacy no-backdo checkpoints."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from yutnori.agents.tactical_features import (
    TACTICAL_ACTION_FEATURE_SIZE,
    tactical_action_feature_row,
)
from yutnori.core import (
    PIECES_PER_PLAYER,
    GameState,
    PieceStatus,
    Position,
    YutResult,
    decode_action,
    encode_action,
)
from yutnori.training import seed_list_sha256, wilson_interval

LEGACY_RULESET = "legacy_no_backdo_v1"
LEGACY_HEAD_TO_HEAD_PROTOCOL = "legacy_no_backdo_head_to_head_paired_v1"
LEGACY_YUT_ORDER = (
    YutResult.DO,
    YutResult.GAE,
    YutResult.GEOL,
    YutResult.YUT,
    YutResult.MO,
)
LEGACY_YUT_PROBABILITIES = {
    YutResult.DO: 0.1536,
    YutResult.GAE: 0.3456,
    YutResult.GEOL: 0.3456,
    YutResult.YUT: 0.1296,
    YutResult.MO: 0.0256,
}
LEGACY_ACTION_SIZE = PIECES_PER_PLAYER * len(LEGACY_YUT_ORDER)
LEGACY_BASE_OBSERVATION_SIZE = (4 + 4 + 16) * 2 + len(LEGACY_YUT_ORDER)
LEGACY_TACTICAL_OBSERVATION_SIZE = (
    LEGACY_BASE_OBSERVATION_SIZE
    + LEGACY_ACTION_SIZE * TACTICAL_ACTION_FEATURE_SIZE
)
LEGACY_OBSERVATION_MODES = ("base", "tactical")
POSITION_WAITING = 29
POSITION_FINISHED = 30


class StateAgent(Protocol):
    name: str

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        ...


class LegacyNoBackdoSampler:
    """Sample the exact five-outcome distribution used by the 40M models."""

    def __init__(
        self,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._rng = rng if rng is not None else random.Random(seed)

    def sample(self) -> YutResult:
        value = self._rng.random()
        cumulative = 0.0
        for result in LEGACY_YUT_ORDER:
            cumulative += LEGACY_YUT_PROBABILITIES[result]
            if value < cumulative:
                return result
        return LEGACY_YUT_ORDER[-1]


class LegacyMaskablePPOAgent:
    """Expose a frozen 20-action MaskablePPO model as a current state agent."""

    def __init__(
        self,
        model: Any,
        *,
        observation_mode: str = "tactical",
        deterministic: bool = True,
        name: str = "legacy_maskable_ppo",
        model_path: str | Path | None = None,
    ) -> None:
        if observation_mode not in LEGACY_OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be one of "
                f"{', '.join(LEGACY_OBSERVATION_MODES)}"
            )
        self.model = model
        self.observation_mode = observation_mode
        self.deterministic = deterministic
        self.name = name
        self.model_path = None if model_path is None else Path(model_path)
        self._validate_model_spaces()

        policy = getattr(self.model, "policy", None)
        if policy is not None and hasattr(policy, "set_training_mode"):
            policy.set_training_mode(False)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        device: str = "cpu",
        observation_mode: str = "tactical",
        deterministic: bool = True,
        name: str = "legacy_maskable_ppo",
    ) -> "LegacyMaskablePPOAgent":
        from sb3_contrib import MaskablePPO

        path = Path(model_path)
        model = MaskablePPO.load(path, device=device)
        return cls(
            model,
            observation_mode=observation_mode,
            deterministic=deterministic,
            name=name,
            model_path=path,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_path": (
                None if self.model_path is None else str(self.model_path)
            ),
            "model_type": "Pure RL",
            "source_action_size": LEGACY_ACTION_SIZE,
            "source_observation_size": legacy_observation_size(
                self.observation_mode
            ),
            "observation_mode": self.observation_mode,
            "deterministic": self.deterministic,
        }

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        legacy_legal = [local_action_to_legacy(action) for action in legal_actions]
        action_mask = np.zeros(LEGACY_ACTION_SIZE, dtype=np.bool_)
        action_mask[legacy_legal] = True
        observation = encode_legacy_observation(
            state,
            state.current_player,
            observation_mode=self.observation_mode,
        )
        action, _model_state = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=action_mask,
        )
        legacy_action = int(np.asarray(action).item())
        return legacy_action_to_local(legacy_action)

    def _validate_model_spaces(self) -> None:
        action_space = getattr(self.model, "action_space", None)
        action_count = getattr(action_space, "n", None)
        if action_count is not None and int(action_count) != LEGACY_ACTION_SIZE:
            raise ValueError(
                f"legacy PPO action space must be {LEGACY_ACTION_SIZE}; "
                f"got {action_count}"
            )

        observation_space = getattr(self.model, "observation_space", None)
        shape = getattr(observation_space, "shape", None)
        expected = (legacy_observation_size(self.observation_mode),)
        if shape is not None and tuple(shape) != expected:
            raise ValueError(
                f"legacy PPO observation shape must be {expected}; got {shape}"
            )


@dataclass(frozen=True)
class HeadToHeadGame:
    base_seed: int
    pair_game: str
    model_a_starts: bool
    winner_model: str | None
    first_player_won: bool | None
    turns: int
    decisions: int
    captures_a: int
    captures_b: int
    finished_a: int
    finished_b: int
    illegal_model: str | None = None
    evaluation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "pair_game": self.pair_game,
            "model_a_starts": self.model_a_starts,
            "winner_model": self.winner_model or "",
            "first_player_won": (
                "" if self.first_player_won is None else self.first_player_won
            ),
            "turns": self.turns,
            "decisions": self.decisions,
            "captures_a": self.captures_a,
            "captures_b": self.captures_b,
            "finished_a": self.finished_a,
            "finished_b": self.finished_b,
            "illegal_model": self.illegal_model or "",
            "evaluation_error": self.evaluation_error or "",
        }


@dataclass(frozen=True)
class HeadToHeadSplit:
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
class HeadToHeadResult:
    base_seeds: tuple[int, ...]
    games: tuple[HeadToHeadGame, ...]
    model_a_name: str
    model_b_name: str
    elapsed_seconds: float

    @property
    def scheduled_games(self) -> int:
        return len(self.base_seeds) * 2

    @property
    def completed_games(self) -> int:
        return sum(game.winner_model is not None for game in self.games)

    @property
    def error_count(self) -> int:
        return sum(game.evaluation_error is not None for game in self.games)

    def to_dict(self) -> dict[str, Any]:
        completed = [
            game for game in self.games if game.winner_model is not None
        ]
        wins_a = sum(
            game.winner_model == self.model_a_name for game in completed
        )
        wins_b = sum(
            game.winner_model == self.model_b_name for game in completed
        )
        first_games = [
            game for game in completed if game.model_a_starts
        ]
        second_games = [
            game for game in completed if not game.model_a_starts
        ]
        pair_counts = _pair_outcomes(
            self.games,
            self.base_seeds,
            self.model_a_name,
        )
        valid = (
            self.completed_games == self.scheduled_games
            and self.error_count == 0
        )
        return {
            "protocol": LEGACY_HEAD_TO_HEAD_PROTOCOL,
            "ruleset": LEGACY_RULESET,
            "legacy_action_size": LEGACY_ACTION_SIZE,
            "legacy_yut_probabilities": {
                result.value: probability
                for result, probability in LEGACY_YUT_PROBABILITIES.items()
            },
            "base_seed_count": len(self.base_seeds),
            "base_seed_sha256": seed_list_sha256(self.base_seeds),
            "scheduled_games": self.scheduled_games,
            "completed_games": self.completed_games,
            "valid_official_result": valid,
            "model_a": {
                "name": self.model_a_name,
                "wins": wins_a,
                "losses": wins_b,
                "win_rate": _rate(wins_a, self.completed_games),
                "confidence_interval_95": _ci_dict(
                    wins_a,
                    self.completed_games,
                ),
                "as_first": _split(first_games, self.model_a_name),
                "as_second": _split(second_games, self.model_a_name),
                "average_captures": _mean(
                    game.captures_a for game in completed
                ),
                "average_finished_pieces": _mean(
                    game.finished_a for game in completed
                ),
                "illegal_action_count": sum(
                    game.illegal_model == self.model_a_name
                    for game in self.games
                ),
            },
            "model_b": {
                "name": self.model_b_name,
                "wins": wins_b,
                "losses": wins_a,
                "win_rate": _rate(wins_b, self.completed_games),
                "confidence_interval_95": _ci_dict(
                    wins_b,
                    self.completed_games,
                ),
                "as_first": _split(second_games, self.model_b_name),
                "as_second": _split(first_games, self.model_b_name),
                "average_captures": _mean(
                    game.captures_b for game in completed
                ),
                "average_finished_pieces": _mean(
                    game.finished_b for game in completed
                ),
                "illegal_action_count": sum(
                    game.illegal_model == self.model_b_name
                    for game in self.games
                ),
            },
            "first_player_win_rate": _mean(
                float(game.first_player_won)
                for game in completed
                if game.first_player_won is not None
            ),
            "pair_outcomes": pair_counts,
            "average_turns": _mean(game.turns for game in completed),
            "average_decisions": _mean(game.decisions for game in completed),
            "evaluation_error_count": self.error_count,
            "evaluation_errors": [
                {
                    "base_seed": game.base_seed,
                    "pair_game": game.pair_game,
                    "error": game.evaluation_error,
                }
                for game in self.games
                if game.evaluation_error is not None
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "average_game_seconds": (
                0.0
                if self.scheduled_games == 0
                else self.elapsed_seconds / self.scheduled_games
            ),
        }


def evaluate_legacy_head_to_head(
    model_a: StateAgent,
    model_b: StateAgent,
    *,
    base_seeds: Sequence[int],
    max_decisions: int = 10_000,
    show_progress: bool = False,
) -> HeadToHeadResult:
    seeds = _validate_seeds(base_seeds)
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    if model_a.name == model_b.name:
        raise ValueError("model names must be distinct")

    specs: Any = (
        (seed, model_a_starts)
        for seed in seeds
        for model_a_starts in (True, False)
    )
    if show_progress:
        from tqdm.auto import tqdm

        specs = tqdm(
            specs,
            total=len(seeds) * 2,
            desc="Legacy head-to-head",
            unit="game",
            dynamic_ncols=True,
        )

    started_at = time.perf_counter()
    games = tuple(
        _play_game(
            model_a,
            model_b,
            base_seed=base_seed,
            model_a_starts=model_a_starts,
            max_decisions=max_decisions,
        )
        for base_seed, model_a_starts in specs
    )
    return HeadToHeadResult(
        base_seeds=seeds,
        games=games,
        model_a_name=model_a.name,
        model_b_name=model_b.name,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def legacy_action_to_local(action: int) -> int:
    if action < 0 or action >= LEGACY_ACTION_SIZE:
        raise ValueError(
            f"legacy action must be in [0, {LEGACY_ACTION_SIZE})"
        )
    piece_id = action // len(LEGACY_YUT_ORDER)
    yut_result = LEGACY_YUT_ORDER[action % len(LEGACY_YUT_ORDER)]
    return encode_action(piece_id, yut_result)


def local_action_to_legacy(action: int) -> int:
    piece_id, yut_result = decode_action(action)
    if yut_result == YutResult.BACK_DO:
        raise ValueError("legacy checkpoints do not support BACK_DO")
    return piece_id * len(LEGACY_YUT_ORDER) + LEGACY_YUT_ORDER.index(yut_result)


def legacy_observation_size(observation_mode: str) -> int:
    if observation_mode == "base":
        return LEGACY_BASE_OBSERVATION_SIZE
    if observation_mode == "tactical":
        return LEGACY_TACTICAL_OBSERVATION_SIZE
    raise ValueError(
        f"observation_mode must be one of {', '.join(LEGACY_OBSERVATION_MODES)}"
    )


def encode_legacy_observation(
    state: GameState,
    player: int,
    *,
    observation_mode: str = "tactical",
) -> np.ndarray:
    opponent = 1 - player
    values: list[float] = []
    values.extend(_position_values(state, player))
    values.extend(_status_values(state, player))
    values.extend(_stack_matrix_values(state, player))
    values.extend(_position_values(state, opponent))
    values.extend(_status_values(state, opponent))
    values.extend(_stack_matrix_values(state, opponent))
    values.extend(float(state.pool_counts[result]) for result in LEGACY_YUT_ORDER)
    base = np.asarray(values, dtype=np.float32)
    if observation_mode == "base":
        return base
    if observation_mode != "tactical":
        raise ValueError(
            "observation_mode must be one of "
            f"{', '.join(LEGACY_OBSERVATION_MODES)}"
        )

    tactical = np.zeros(
        (LEGACY_ACTION_SIZE, TACTICAL_ACTION_FEATURE_SIZE),
        dtype=np.float32,
    )
    for legacy_action in range(LEGACY_ACTION_SIZE):
        local_action = legacy_action_to_local(legacy_action)
        if state.is_legal_action(local_action):
            tactical[legacy_action] = tactical_action_feature_row(
                state,
                local_action,
            )
    return np.concatenate([base, tactical.reshape(-1)]).astype(
        np.float32,
        copy=False,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _play_game(
    model_a: StateAgent,
    model_b: StateAgent,
    *,
    base_seed: int,
    model_a_starts: bool,
    max_decisions: int,
) -> HeadToHeadGame:
    model_a_player = 0
    model_b_player = 1
    starting_player = model_a_player if model_a_starts else model_b_player
    state = GameState(
        starting_player=starting_player,
        yut_sampler=LegacyNoBackdoSampler(seed=base_seed),
    )
    agents = {model_a_player: model_a, model_b_player: model_b}
    names = {
        model_a_player: model_a.name,
        model_b_player: model_b.name,
    }
    captures = {model_a_player: 0, model_b_player: 0}
    state.start_turn()

    while state.winner is None:
        player = state.current_player
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return _error_game(
                state,
                base_seed=base_seed,
                model_a_starts=model_a_starts,
                captures=captures,
                error="RuntimeError: non-terminal state has no legal actions",
            )
        try:
            action = int(agents[player].select_action(state, legal_actions))
        except Exception as exc:
            return _error_game(
                state,
                base_seed=base_seed,
                model_a_starts=model_a_starts,
                captures=captures,
                error=f"{type(exc).__name__}: {exc}",
            )

        if action not in legal_actions:
            winner = 1 - player
            return _completed_game(
                state,
                base_seed=base_seed,
                model_a_starts=model_a_starts,
                captures=captures,
                winner_model=names[winner],
                winner_player=winner,
                illegal_model=names[player],
                starting_player=starting_player,
            )

        event = state.apply_action(action)
        captures[player] += event.captured_count
        if state.decision_count > max_decisions:
            return _error_game(
                state,
                base_seed=base_seed,
                model_a_starts=model_a_starts,
                captures=captures,
                error=(
                    "RuntimeError: game exceeded "
                    f"max_decisions={max_decisions}"
                ),
            )

    return _completed_game(
        state,
        base_seed=base_seed,
        model_a_starts=model_a_starts,
        captures=captures,
        winner_model=names[state.winner],
        winner_player=state.winner,
        starting_player=starting_player,
    )


def _completed_game(
    state: GameState,
    *,
    base_seed: int,
    model_a_starts: bool,
    captures: dict[int, int],
    winner_model: str,
    winner_player: int,
    starting_player: int,
    illegal_model: str | None = None,
) -> HeadToHeadGame:
    return HeadToHeadGame(
        base_seed=base_seed,
        pair_game="A" if model_a_starts else "B",
        model_a_starts=model_a_starts,
        winner_model=winner_model,
        first_player_won=winner_player == starting_player,
        turns=state.turn_count,
        decisions=state.decision_count,
        captures_a=captures[0],
        captures_b=captures[1],
        finished_a=_finished_count(state, 0),
        finished_b=_finished_count(state, 1),
        illegal_model=illegal_model,
    )


def _error_game(
    state: GameState,
    *,
    base_seed: int,
    model_a_starts: bool,
    captures: dict[int, int],
    error: str,
) -> HeadToHeadGame:
    return HeadToHeadGame(
        base_seed=base_seed,
        pair_game="A" if model_a_starts else "B",
        model_a_starts=model_a_starts,
        winner_model=None,
        first_player_won=None,
        turns=state.turn_count,
        decisions=state.decision_count,
        captures_a=captures[0],
        captures_b=captures[1],
        finished_a=_finished_count(state, 0),
        finished_b=_finished_count(state, 1),
        evaluation_error=error,
    )


def _split(
    games: Sequence[HeadToHeadGame],
    winner_name: str,
) -> dict[str, int | float]:
    wins = sum(game.winner_model == winner_name for game in games)
    losses = len(games) - wins
    return HeadToHeadSplit(
        games=len(games),
        wins=wins,
        losses=losses,
        win_rate=_rate(wins, len(games)),
    ).to_dict()


def _pair_outcomes(
    games: Sequence[HeadToHeadGame],
    base_seeds: Sequence[int],
    model_a_name: str,
) -> dict[str, int]:
    by_seed = {
        seed: [game for game in games if game.base_seed == seed]
        for seed in base_seeds
    }
    counts = {
        "model_a_sweeps": 0,
        "split_pairs": 0,
        "model_b_sweeps": 0,
        "error_pairs": 0,
    }
    for pair in by_seed.values():
        if len(pair) != 2 or any(
            game.winner_model is None for game in pair
        ):
            counts["error_pairs"] += 1
            continue
        wins_a = sum(game.winner_model == model_a_name for game in pair)
        if wins_a == 2:
            counts["model_a_sweeps"] += 1
        elif wins_a == 1:
            counts["split_pairs"] += 1
        else:
            counts["model_b_sweeps"] += 1
    return counts


def _position_values(state: GameState, player: int) -> list[float]:
    return [float(_position_value(position)) for position in state.pieces[player]]


def _position_value(position: Position) -> int:
    if position.status == PieceStatus.WAITING:
        return POSITION_WAITING
    if position.status == PieceStatus.FINISHED:
        return POSITION_FINISHED
    if position.physical_cell is None:
        raise ValueError("on-board position requires physical_cell")
    return int(position.physical_cell)


def _status_values(state: GameState, player: int) -> list[float]:
    return [
        float(_status_value(position.status))
        for position in state.pieces[player]
    ]


def _status_value(status: PieceStatus) -> int:
    if status == PieceStatus.WAITING:
        return 0
    if status == PieceStatus.ON_BOARD:
        return 1
    if status == PieceStatus.FINISHED:
        return 2
    raise ValueError(f"unknown piece status: {status}")


def _stack_matrix_values(state: GameState, player: int) -> list[float]:
    return [
        float(_same_stack(left, right))
        for left in state.pieces[player]
        for right in state.pieces[player]
    ]


def _same_stack(left: Position, right: Position) -> bool:
    return (
        left.status == PieceStatus.ON_BOARD
        and right.status == PieceStatus.ON_BOARD
        and left.physical_cell is not None
        and left.physical_cell == right.physical_cell
    )


def _finished_count(state: GameState, player: int) -> int:
    return sum(
        position.status == PieceStatus.FINISHED
        for position in state.pieces[player]
    )


def _validate_seeds(base_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("base_seeds must not be empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("base_seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("base_seeds must be unique")
    return seeds


def _ci_dict(wins: int, games: int) -> dict[str, float | str]:
    lower, upper = wilson_interval(wins, games)
    return {"method": "wilson", "lower": lower, "upper": upper}


def _rate(wins: int, games: int) -> float:
    return 0.0 if games == 0 else wins / games


def _mean(values: Any) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)
