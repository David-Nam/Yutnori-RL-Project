"""Baseline agents for rule validation and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from yutnori.core import (
    Cell,
    GameState,
    PieceStatus,
    Position,
    Route,
    YutResult,
    decode_action,
)
from yutnori.core.board import ROUTES
from yutnori.core.yut import steps_for


class Agent(Protocol):
    name: str

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        ...


@dataclass(frozen=True)
class ActionEvaluation:
    action: int
    piece_id: int
    yut_result: YutResult
    moved_count: int
    captured_count: int
    finished_count: int
    entered_shortcut: bool
    steps: int


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, _state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        return self._rng.choice(legal_actions)


class CaptureFirstAgent:
    name = "capture_first"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        evaluations = [evaluate_action(state, action) for action in legal_actions]
        capture_candidates = [
            evaluation
            for evaluation in evaluations
            if evaluation.captured_count > 0
        ]
        if capture_candidates:
            return max(capture_candidates, key=_capture_score).action
        return max(evaluations, key=_greedy_score).action


class GreedyFinishAgent:
    name = "greedy_finish"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        evaluations = [evaluate_action(state, action) for action in legal_actions]
        return max(evaluations, key=_greedy_score).action


class ProjectRFRuleBasedAgent:
    name = "project_rf_rule"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        return max(
            legal_actions,
            key=lambda action: (project_rf_action_score(state, action), action),
        )


class CommonRuleBasedAgent:
    """Frozen common evaluator rule agent with smallest-action tie breaking."""

    name = "common_rule_based"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        return max(
            legal_actions,
            key=lambda action: (project_rf_action_score(state, action), -action),
        )


def evaluate_action(state: GameState, action: int) -> ActionEvaluation:
    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be evaluated: {action}")

    actor = state.current_player
    opponent = 1 - actor
    piece_id, yut_result = decode_action(action)
    moving_piece_ids = state.stack_piece_ids(actor, piece_id)
    move_result = state.board.move(
        state.pieces[actor][piece_id],
        steps_for(yut_result),
    )

    captured_count = 0
    if (
        move_result.status == PieceStatus.ON_BOARD
        and move_result.physical_cell is not None
    ):
        captured_count = len(
            state.piece_ids_at_cell(opponent, move_result.physical_cell)
        )

    finished_count = (
        len(moving_piece_ids)
        if move_result.status == PieceStatus.FINISHED
        else 0
    )

    return ActionEvaluation(
        action=action,
        piece_id=piece_id,
        yut_result=yut_result,
        moved_count=len(moving_piece_ids),
        captured_count=captured_count,
        finished_count=finished_count,
        entered_shortcut=move_result.entered_shortcut,
        steps=steps_for(yut_result),
    )


def project_rf_action_score(state: GameState, action: int) -> float:
    """Score an action with the project-RF rule-based heuristic."""

    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be scored: {action}")

    actor = state.current_player
    opponent = 1 - actor
    piece_id, yut_result = decode_action(action)
    old_position = state.pieces[actor][piece_id]
    moving_piece_ids = state.stack_piece_ids(actor, piece_id)
    move_result = state.board.move(old_position, steps_for(yut_result))

    score = 0.0
    if move_result.status == PieceStatus.FINISHED:
        score += 100
    if (
        move_result.status == PieceStatus.ON_BOARD
        and move_result.physical_cell is not None
        and state.piece_ids_at_cell(opponent, move_result.physical_cell)
    ):
        score += 50
    if old_position.status == PieceStatus.WAITING:
        score += 5

    stack_size = (
        1 if old_position.status == PieceStatus.WAITING else len(moving_piece_ids)
    )
    score += 4 * max(0, stack_size - 1)
    score -= 0.5 * project_rf_distance_to_finish(move_result.position)
    return score


def project_rf_distance_to_finish(position: Position) -> int:
    """Return remaining positive-step distance under the local board rules."""

    if position.status == PieceStatus.FINISHED:
        return 0
    if position.status == PieceStatus.WAITING:
        return len(ROUTES[Route.OUTER])
    if position.status != PieceStatus.ON_BOARD:
        raise ValueError(f"unknown piece status: {position.status}")
    if position.physical_cell == Cell.HOME:
        return 1
    if position.route is None or position.index is None:
        raise ValueError("on-board position requires route and index")

    home_index = len(ROUTES[position.route]) - 1
    return home_index - position.index + 1


def _capture_score(evaluation: ActionEvaluation) -> tuple[int, int, int, int, int]:
    return (
        evaluation.captured_count,
        evaluation.finished_count,
        evaluation.moved_count,
        evaluation.steps,
        -evaluation.action,
    )


def _greedy_score(evaluation: ActionEvaluation) -> tuple[int, int, int, int, int]:
    return (
        evaluation.finished_count,
        evaluation.moved_count,
        int(evaluation.entered_shortcut),
        evaluation.steps,
        -evaluation.action,
    )
