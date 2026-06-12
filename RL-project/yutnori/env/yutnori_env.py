"""Gymnasium-compatible wrapper for the Yutnori game state."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from yutnori.agents.tactical_features import (
    TACTICAL_ACTION_FEATURE_SIZE,
    tactical_action_features,
)
from yutnori.core import (
    ACTION_SIZE,
    PIECES_PER_PLAYER,
    PLAYER_COUNT,
    GameEvent,
    GameState,
    PieceStatus,
    Position,
    Route,
    YUT_ORDER,
    YutSampler,
)
from yutnori.core.game import Sampler

POSITION_WAITING = 29
POSITION_FINISHED = 30
TRACK_NONE = 0
TRACK_OUTER = 1
TRACK_C1_VIA_C3 = 2
TRACK_C1_VIA_CENTER_HOME = 3
TRACK_C2_VIA_CENTER_HOME = 4
OBSERVATION_SIZE = (4 + 4 + 4 + 16) * 2 + len(YUT_ORDER)
TACTICAL_OBSERVATION_SIZE = (
    OBSERVATION_SIZE + ACTION_SIZE * TACTICAL_ACTION_FEATURE_SIZE
)
RULESET_FULL_BACKDO = "full_backdo_v1"
RULESET = RULESET_FULL_BACKDO
OBSERVATION_MODE_BASE = "base"
OBSERVATION_MODE_TACTICAL = "tactical"
OBSERVATION_MODES = (OBSERVATION_MODE_BASE, OBSERVATION_MODE_TACTICAL)
REWARD_MODE_TERMINAL = "terminal"
REWARD_MODE_RF_SHAPED = "rf_shaped"
REWARD_MODES = (REWARD_MODE_TERMINAL, REWARD_MODE_RF_SHAPED)
MAX_RESET_ATTEMPTS = 1_000

OpponentPolicy = Callable[[GameState, list[int]], int]
YutSamplerFactory = Callable[[random.Random], Sampler]


class YutnoriEnv(gym.Env[np.ndarray, int]):
    """Single-learner Gymnasium view over a two-player Yutnori game.

    The environment only returns decision states for ``learner_player``.
    Opponent turns are advanced internally with ``opponent_policy``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        learner_player: int = 0,
        starting_player: int | None = None,
        opponent_policy: OpponentPolicy | None = None,
        yut_sampler_factory: YutSamplerFactory | None = None,
        observation_mode: str = OBSERVATION_MODE_BASE,
        reward_mode: str = REWARD_MODE_TERMINAL,
    ) -> None:
        if learner_player < 0 or learner_player >= PLAYER_COUNT:
            raise ValueError(f"learner_player must be in [0, {PLAYER_COUNT})")
        if starting_player is not None and (
            starting_player < 0 or starting_player >= PLAYER_COUNT
        ):
            raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {', '.join(OBSERVATION_MODES)}"
            )
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {', '.join(REWARD_MODES)}")

        self.learner_player = learner_player
        self._fixed_starting_player = starting_player
        self._opponent_policy = opponent_policy
        self._yut_sampler_factory = yut_sampler_factory
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self._rng = random.Random()
        self.state: GameState | None = None

        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=_observation_space_low(observation_mode),
            high=1_000_000.0,
            shape=(observation_size(observation_mode),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = random.Random(seed)

        skipped_terminal_resets = 0
        for _attempt in range(MAX_RESET_ATTEMPTS):
            starting_player = self._resolve_starting_player(options)
            sampler = self._create_yut_sampler()
            self.state = GameState(
                starting_player=starting_player,
                yut_sampler=sampler,
            )
            initial_rolls = self.state.start_turn()
            initial_auto_passes = self.state.last_auto_passes.copy()
            opponent_events = self._advance_opponent_turns()
            if self.state.winner is None:
                break
            skipped_terminal_resets += 1
        else:
            raise RuntimeError(
                "reset could not produce a learner decision state before terminal"
            )

        info = self._base_info()
        info.update(
            {
                "starting_player": starting_player,
                "initial_rolls": [result.value for result in initial_rolls],
                "initial_auto_passes": [
                    self._auto_pass_to_dict(auto_pass)
                    for auto_pass in initial_auto_passes
                ],
                "opponent_events": [self._event_to_dict(event) for event in opponent_events],
                "skipped_terminal_resets": skipped_terminal_resets,
            }
        )
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.current_player != self.learner_player:
            raise RuntimeError("step called while learner is not the current player")
        if not state.is_legal_action(int(action), self.learner_player):
            raise ValueError(f"illegal learner action: {action}")

        learner_event = state.apply_action(int(action))
        opponent_events = self._advance_opponent_turns()
        terminated = state.winner is not None
        terminal_reward, shaping_reward = self._reward_components(
            learner_event,
            opponent_events,
        )
        reward = terminal_reward + shaping_reward
        info = self._base_info()
        info.update(
            {
                "learner_event": self._event_to_dict(learner_event),
                "opponent_events": [
                    self._event_to_dict(event) for event in opponent_events
                ],
                "terminal_reward": terminal_reward,
                "shaping_reward": shaping_reward,
            }
        )
        return self._get_obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        state = self.state
        if state is None or state.winner is not None:
            return np.zeros(ACTION_SIZE, dtype=np.bool_)
        if state.current_player != self.learner_player:
            return np.zeros(ACTION_SIZE, dtype=np.bool_)
        return np.array(
            [state.is_legal_action(action, self.learner_player) for action in range(ACTION_SIZE)],
            dtype=np.bool_,
        )

    def render(self) -> None:
        return None

    def _resolve_starting_player(self, options: dict[str, Any] | None) -> int:
        if options is not None and "starting_player" in options:
            starting_player = int(options["starting_player"])
            if starting_player < 0 or starting_player >= PLAYER_COUNT:
                raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")
            return starting_player
        if self._fixed_starting_player is not None:
            return self._fixed_starting_player
        return self._rng.randrange(PLAYER_COUNT)

    def _create_yut_sampler(self) -> Sampler:
        if self._yut_sampler_factory is not None:
            return self._yut_sampler_factory(self._rng)
        return YutSampler(rng=self._rng)

    def _advance_opponent_turns(self) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        while state.winner is None and state.current_player != self.learner_player:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                raise RuntimeError("opponent has no legal actions during its turn")
            action = self._select_opponent_action(state, legal_actions)
            if action not in legal_actions:
                raise ValueError(f"opponent selected illegal action: {action}")
            events.append(state.apply_action(action))
        return events

    def _select_opponent_action(self, state: GameState, legal_actions: list[int]) -> int:
        if self._opponent_policy is not None:
            return int(self._opponent_policy(state, legal_actions))
        return self._rng.choice(legal_actions)

    def _get_obs(self) -> np.ndarray:
        state = self._require_state()
        return encode_observation(
            state,
            self.learner_player,
            observation_mode=self.observation_mode,
        )

    def _terminal_reward(self) -> float:
        state = self._require_state()
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == self.learner_player else -1.0

    def _reward_components(
        self,
        learner_event: GameEvent,
        opponent_events: list[GameEvent],
    ) -> tuple[float, float]:
        terminal_reward = self._terminal_reward()
        if self.reward_mode == REWARD_MODE_TERMINAL:
            return terminal_reward, 0.0
        if self.reward_mode == REWARD_MODE_RF_SHAPED:
            return terminal_reward, self._project_rf_shaping_reward(
                learner_event,
                opponent_events,
            )
        raise RuntimeError(f"unknown reward_mode: {self.reward_mode}")

    def _project_rf_shaping_reward(
        self,
        learner_event: GameEvent,
        opponent_events: list[GameEvent],
    ) -> float:
        from yutnori.training.reward_shaping import project_rf_events_shaping_reward

        return project_rf_events_shaping_reward(
            learner_event,
            opponent_events,
            learner_player=self.learner_player,
        )

    def _base_info(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "learner_player": self.learner_player,
            "ruleset": RULESET,
            "reward_mode": self.reward_mode,
            "current_player": state.current_player,
            "winner": state.winner,
            "turn_count": state.turn_count,
            "decision_count": state.decision_count,
            "back_do_stats": state.back_do_stats(),
            "action_mask": self.action_masks(),
        }

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("environment must be reset before use")
        return self.state

    def _event_to_dict(self, event: GameEvent) -> dict[str, Any]:
        return {
            "actor": event.actor,
            "action": event.action,
            "piece_id": event.piece_id,
            "yut_result": event.yut_result.value,
            "moved_piece_ids": event.moved_piece_ids,
            "captured": event.captured,
            "captured_count": event.captured_count,
            "captured_piece_ids": event.captured_piece_ids,
            "stacked": event.stacked,
            "stack_size": event.stack_size,
            "finished_count": event.finished_count,
            "entered_shortcut": event.entered_shortcut,
            "landed_on_home": event.landed_on_home,
            "passed_home": event.passed_home,
            "moved_backward": event.yut_result.value == "BACK_DO",
            "bonus_rolls": [result.value for result in event.bonus_rolls],
            "auto_passes": [
                self._auto_pass_to_dict(auto_pass)
                for auto_pass in event.auto_passes
            ],
            "turn_changed": event.turn_changed,
            "winner": event.winner,
            "pool_counts": {
                result.value: count for result, count in event.pool_counts.items()
            },
        }

    @staticmethod
    def _auto_pass_to_dict(auto_pass) -> dict[str, Any]:
        return {
            "player": auto_pass.player,
            "rolls": [result.value for result in auto_pass.rolls],
            "pool_counts": {
                result.value: count
                for result, count in auto_pass.pool_counts.items()
            },
            "reason": auto_pass.reason,
        }


def observation_size(observation_mode: str = OBSERVATION_MODE_BASE) -> int:
    if observation_mode == OBSERVATION_MODE_BASE:
        return OBSERVATION_SIZE
    if observation_mode == OBSERVATION_MODE_TACTICAL:
        return TACTICAL_OBSERVATION_SIZE
    raise ValueError(
        f"observation_mode must be one of {', '.join(OBSERVATION_MODES)}"
    )


def encode_observation(
    state: GameState,
    player: int,
    *,
    observation_mode: str = OBSERVATION_MODE_BASE,
) -> np.ndarray:
    opponent = 1 - player
    values: list[float] = []
    values.extend(_position_values(state, player))
    values.extend(_status_values(state, player))
    values.extend(_track_values(state, player))
    values.extend(_stack_matrix_values(state, player))
    values.extend(_position_values(state, opponent))
    values.extend(_status_values(state, opponent))
    values.extend(_track_values(state, opponent))
    values.extend(_stack_matrix_values(state, opponent))
    values.extend(float(state.pool_counts[result]) for result in YUT_ORDER)
    observation = np.array(values, dtype=np.float32)
    if observation_mode == OBSERVATION_MODE_BASE:
        return observation
    if observation_mode == OBSERVATION_MODE_TACTICAL:
        return np.concatenate(
            [observation, tactical_action_features(state).reshape(-1)]
        ).astype(np.float32, copy=False)
    raise ValueError(
        f"observation_mode must be one of {', '.join(OBSERVATION_MODES)}"
    )


def _observation_space_low(observation_mode: str) -> float:
    if observation_mode == OBSERVATION_MODE_TACTICAL:
        return -1_000_000.0
    return 0.0


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
    return [float(_status_value(position.status)) for position in state.pieces[player]]


def _status_value(status: PieceStatus) -> int:
    if status == PieceStatus.WAITING:
        return 0
    if status == PieceStatus.ON_BOARD:
        return 1
    if status == PieceStatus.FINISHED:
        return 2
    raise ValueError(f"unknown piece status: {status}")


def _track_values(state: GameState, player: int) -> list[float]:
    return [float(_track_value(position)) for position in state.pieces[player]]


def _track_value(position: Position) -> int:
    if position.status != PieceStatus.ON_BOARD:
        return TRACK_NONE
    if position.route == Route.OUTER:
        return TRACK_OUTER
    if position.route == Route.C1_DIAGONAL:
        return TRACK_C1_VIA_C3
    if position.route == Route.C2_DIAGONAL:
        return TRACK_C2_VIA_CENTER_HOME
    if position.route == Route.CENTER_TO_HOME:
        if position.entry_route == Route.C1_DIAGONAL:
            return TRACK_C1_VIA_CENTER_HOME
        if position.entry_route == Route.C2_DIAGONAL:
            return TRACK_C2_VIA_CENTER_HOME
    raise ValueError(f"unknown logical track for position: {position}")


def _stack_matrix_values(state: GameState, player: int) -> list[float]:
    values: list[float] = []
    for left in state.pieces[player]:
        for right in state.pieces[player]:
            values.append(float(_same_stack(left, right)))
    return values


def _same_stack(left: Position, right: Position) -> bool:
    return (
        left.status == PieceStatus.ON_BOARD
        and right.status == PieceStatus.ON_BOARD
        and left.physical_cell is not None
        and left.physical_cell == right.physical_cell
    )
