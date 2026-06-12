"""Adapter for evaluating project-RF PyTorch checkpoints in the local engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from yutnori.agents.baseline import evaluate_action
from yutnori.core import (
    ACTION_SIZE,
    PIECES_PER_PLAYER,
    Cell,
    GameState,
    PieceStatus,
    Position,
    YutResult,
    decode_action,
    encode_action,
    empty_pool,
    is_bonus_result,
    steps_for,
)

PROJECT_RF_ACTION_SIZE = 20
PROJECT_RF_STATE_SIZE = 252
PROJECT_RF_YUT_ORDER = (
    YutResult.DO,
    YutResult.GAE,
    YutResult.GEOL,
    YutResult.YUT,
    YutResult.MO,
)
PROJECT_RF_START = -1
PROJECT_RF_FINISH = 99
PROJECT_RF_POSITION_VALUES = (
    PROJECT_RF_START,
    *range(27),
    PROJECT_RF_FINISH,
)
PROJECT_RF_POSITION_INDEX = {
    position: index for index, position in enumerate(PROJECT_RF_POSITION_VALUES)
}
PROJECT_RF_TACTICAL_WEIGHT = 2.5
PROJECT_RF_YUT_PROBABILITIES = {
    YutResult.DO: 0.153 / 0.991,
    YutResult.GAE: 0.346 / 0.991,
    YutResult.GEOL: 0.346 / 0.991,
    YutResult.YUT: 0.120 / 0.991,
    YutResult.MO: 0.026 / 0.991,
}

_PROJECT_RF_CELL_POSITION = {
    Cell.O1: 0,
    Cell.O2: 1,
    Cell.O3: 2,
    Cell.O4: 3,
    Cell.C1: 4,
    Cell.O6: 5,
    Cell.O7: 6,
    Cell.O8: 7,
    Cell.O9: 8,
    Cell.C2: 9,
    Cell.O11: 10,
    Cell.O12: 11,
    Cell.O13: 12,
    Cell.O14: 13,
    Cell.C3: 14,
    Cell.O16: 15,
    Cell.O17: 16,
    Cell.O18: 17,
    Cell.O19: 18,
    Cell.HOME: 19,
    Cell.A1: 20,
    Cell.A2: 21,
    Cell.B1: 22,
    Cell.B2: 23,
    Cell.CENTER: 24,
    Cell.B3: 25,
    Cell.B4: 26,
    # project-RF always turns toward B3 when the center is crossed. The local
    # engine can continue through A3/A4, so preserve remaining distance here.
    Cell.A3: 12,
    Cell.A4: 13,
}


class _ProjectRFPolicyNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, PROJECT_RF_ACTION_SIZE)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy(self.body(states))


@dataclass(frozen=True)
class _ImmediateOutcome:
    state: GameState
    captured_count: int
    finished_count: int
    entered_shortcut: bool
    turn_changed: bool


class ProjectRFCheckpointAgent:
    """Run a frozen project-RF policy against the common local rules."""

    name = "project_rf_checkpoint"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        use_tactical_prior: bool = True,
        tactical_weight: float = PROJECT_RF_TACTICAL_WEIGHT,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.use_tactical_prior = use_tactical_prior
        self.tactical_weight = float(tactical_weight)

        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        state_size = int(checkpoint.get("state_dim", PROJECT_RF_STATE_SIZE))
        body_state = checkpoint["body"]
        hidden_size = int(body_state["0.weight"].shape[0])
        if state_size != PROJECT_RF_STATE_SIZE:
            raise ValueError(
                f"unsupported project-RF state size {state_size}; "
                f"expected {PROJECT_RF_STATE_SIZE}"
            )

        self.network = _ProjectRFPolicyNetwork(state_size, hidden_size).to(self.device)
        self.network.body.load_state_dict(body_state)
        self.network.policy.load_state_dict(checkpoint["policy"])
        self.network.eval()

    @property
    def projection_metadata(self) -> dict[str, Any]:
        return {
            "adapter": "project_rf_checkpoint_v1",
            "source_state_size": PROJECT_RF_STATE_SIZE,
            "source_action_size": PROJECT_RF_ACTION_SIZE,
            "target_action_size": ACTION_SIZE,
            "action_mapping": (
                "project_rf=(yut_index*4)+piece_id; "
                "local=(piece_id*5)+yut_index"
            ),
            "a3_projection": 12,
            "a4_projection": 13,
            "projection_note": (
                "A3/A4 have no direct project-RF position because its shortcut "
                "always turns at center; they are mapped by remaining distance."
            ),
            "future_roll_policy": (
                "No evaluation RNG state is copied or sampled. Opponent "
                "counterplay uses project-RF's fixed single-roll probabilities."
            ),
            "use_tactical_prior": self.use_tactical_prior,
            "tactical_weight": (
                self.tactical_weight if self.use_tactical_prior else 0.0
            ),
        }

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        if state.current_player not in (0, 1):
            raise ValueError(f"invalid current player: {state.current_player}")
        if any(
            decode_action(action)[1] == YutResult.BACK_DO
            for action in legal_actions
        ):
            raise ValueError(
                "project-RF checkpoint has 20 forward-action logits and is "
                "incompatible with the full_backdo_v1 ruleset"
            )

        encoded_state = encode_project_rf_state(state)
        state_tensor = torch.as_tensor(
            encoded_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.network(state_tensor).squeeze(0)

        masked_logits = torch.full_like(logits, -1e9)
        for local_action in legal_actions:
            project_action = local_action_to_project_rf(local_action)
            score = logits[project_action]
            if self.use_tactical_prior:
                score = score + self.tactical_weight * project_rf_tactical_bonus(
                    state,
                    local_action,
                )
            masked_logits[project_action] = score

        project_action = int(torch.argmax(masked_logits).item())
        local_action = project_rf_action_to_local(project_action)
        if local_action not in legal_actions:
            raise RuntimeError(
                f"adapter selected illegal local action {local_action} "
                f"from project-RF action {project_action}"
            )
        return local_action


def encode_project_rf_state(state: GameState) -> np.ndarray:
    """Project the common GameState into project-RF's 252-value state."""

    player = state.current_player
    opponent = 1 - player
    features: list[float] = []
    mapped_positions: list[int] = []

    for position in state.pieces[player] + state.pieces[opponent]:
        mapped = project_rf_position(position)
        mapped_positions.append(mapped)
        encoded = [0.0] * len(PROJECT_RF_POSITION_VALUES)
        encoded[PROJECT_RF_POSITION_INDEX[mapped]] = 1.0
        features.extend(encoded)

    for yut_result in PROJECT_RF_YUT_ORDER:
        features.append(min(state.pool_counts[yut_result], 4) / 4)

    features.extend(
        [
            _finished_count(state, player) / PIECES_PER_PLAYER,
            _finished_count(state, opponent) / PIECES_PER_PLAYER,
            float(_can_capture(state, player)),
            float(_can_finish(state, player)),
            float(_can_enter_shortcut(state, player)),
            float(_capture_danger(state, player)),
            1.0,
        ]
    )
    features.extend(
        project_rf_distance_to_finish(position)
        / _PROJECT_RF_DISTANCE_NORMALIZER
        for position in mapped_positions
    )

    encoded_state = np.asarray(features, dtype=np.float32)
    if encoded_state.shape != (PROJECT_RF_STATE_SIZE,):
        raise RuntimeError(
            f"project-RF state has shape {encoded_state.shape}; "
            f"expected ({PROJECT_RF_STATE_SIZE},)"
        )
    return encoded_state


def project_rf_position(position: Position) -> int:
    if position.status == PieceStatus.WAITING:
        return PROJECT_RF_START
    if position.status == PieceStatus.FINISHED:
        return PROJECT_RF_FINISH
    if position.status != PieceStatus.ON_BOARD or position.physical_cell is None:
        raise ValueError(f"cannot project position: {position}")
    try:
        return _PROJECT_RF_CELL_POSITION[position.physical_cell]
    except KeyError as exc:
        raise ValueError(
            f"no project-RF projection for cell {position.physical_cell}"
        ) from exc


def local_action_to_project_rf(action: int) -> int:
    piece_id, yut_result = decode_action(action)
    if yut_result == YutResult.BACK_DO:
        raise ValueError("project-RF checkpoint does not support BACK_DO actions")
    yut_index = PROJECT_RF_YUT_ORDER.index(yut_result)
    return yut_index * PIECES_PER_PLAYER + piece_id


def project_rf_action_to_local(action: int) -> int:
    if action < 0 or action >= PROJECT_RF_ACTION_SIZE:
        raise ValueError(
            f"project-RF action must be in [0, {PROJECT_RF_ACTION_SIZE})"
        )
    piece_id = action % PIECES_PER_PLAYER
    yut_index = action // PIECES_PER_PLAYER
    return encode_action(piece_id, PROJECT_RF_YUT_ORDER[yut_index])


def project_rf_distance_to_finish(position: int) -> int:
    if position == PROJECT_RF_FINISH:
        return 0
    if position == PROJECT_RF_START:
        return 21
    route = _project_rf_route(position)
    return len(route) - route.index(position) - 1


def project_rf_tactical_bonus(state: GameState, action: int) -> float:
    """Apply project-RF's inference-time tactical prior under local rules."""

    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be scored: {action}")

    player = state.current_player
    piece_id, _yut_result = decode_action(action)
    before_position = state.pieces[player][piece_id]
    before_distance = project_rf_distance_to_finish(
        project_rf_position(before_position)
    )
    before_danger = _capture_danger(state, player)
    capture_available = _can_capture(state, player)
    near_goal_piece = _has_near_goal_piece(state, player)
    opponent_near_goal = _has_near_goal_piece(state, 1 - player)

    outcome = _apply_action_without_future_rolls(state, action)
    candidate = outcome.state
    after_position = candidate.pieces[player][piece_id]
    after_distance = project_rf_distance_to_finish(
        project_rf_position(after_position)
    )
    after_danger = (
        False if candidate.winner is not None else _capture_danger(candidate, player)
    )

    score = 0.0
    if candidate.winner == player:
        score += 20.0
    score += 10.0 * outcome.finished_count
    score += 9.0 * outcome.captured_count
    score += 0.12 * max(0, before_distance - after_distance)
    if capture_available and outcome.captured_count == 0:
        score -= 4.0
    if (
        before_position.status == PieceStatus.ON_BOARD
        and outcome.entered_shortcut
    ):
        score += 1.5
    if after_danger:
        score -= 7.0
    elif before_danger and not after_danger:
        score += 3.0
    if near_goal_piece and before_position.status == PieceStatus.WAITING:
        score -= 2.5
    if after_position.status != PieceStatus.FINISHED and after_distance <= 5:
        score += 1.0
    if opponent_near_goal and outcome.captured_count:
        score += 4.0
    if candidate.winner is None and outcome.turn_changed:
        score -= 2.5 * _opponent_counterplay(candidate, player)
    return score


_PROJECT_RF_SHORTCUT_A = (4, 20, 21, 24, 25, 26, PROJECT_RF_FINISH)
_PROJECT_RF_SHORTCUT_B = (9, 22, 23, 24, 25, 26, PROJECT_RF_FINISH)
_PROJECT_RF_CENTER_TO_FINISH = (24, 25, 26, PROJECT_RF_FINISH)
_PROJECT_RF_OUTER = (*range(20), PROJECT_RF_FINISH)
_PROJECT_RF_DISTANCE_NORMALIZER = 27


def _project_rf_route(position: int) -> tuple[int, ...]:
    if position in _PROJECT_RF_SHORTCUT_A:
        return _PROJECT_RF_SHORTCUT_A
    if position in _PROJECT_RF_SHORTCUT_B:
        return _PROJECT_RF_SHORTCUT_B
    if position in _PROJECT_RF_CENTER_TO_FINISH:
        return _PROJECT_RF_CENTER_TO_FINISH
    if position in _PROJECT_RF_OUTER:
        return _PROJECT_RF_OUTER
    raise ValueError(f"unknown project-RF position: {position}")


def _finished_count(state: GameState, player: int) -> int:
    return sum(
        position.status == PieceStatus.FINISHED for position in state.pieces[player]
    )


def _can_capture(state: GameState, player: int) -> bool:
    if state.current_player != player:
        return False
    return any(
        evaluate_action(state, action).captured_count > 0
        for action in state.get_legal_actions()
    )


def _can_finish(state: GameState, player: int) -> bool:
    if state.current_player != player:
        return False
    return any(
        evaluate_action(state, action).finished_count > 0
        for action in state.get_legal_actions()
    )


def _can_enter_shortcut(state: GameState, player: int) -> bool:
    if state.current_player != player:
        return False
    return any(
        evaluate_action(state, action).entered_shortcut
        for action in state.get_legal_actions()
    )


def _capture_danger(state: GameState, player: int) -> bool:
    my_cells = {
        position.physical_cell
        for position in state.pieces[player]
        if (
            position.status == PieceStatus.ON_BOARD
            and position.physical_cell is not None
        )
    }
    if not my_cells:
        return False

    for opponent_position in state.pieces[1 - player]:
        if opponent_position.status == PieceStatus.FINISHED:
            continue
        for steps in range(1, 6):
            move_result = state.board.move(opponent_position, steps)
            if (
                move_result.status == PieceStatus.ON_BOARD
                and move_result.physical_cell in my_cells
            ):
                return True
    return False


def _has_near_goal_piece(
    state: GameState,
    player: int,
    max_distance: int = 5,
) -> bool:
    return any(
        position.status == PieceStatus.ON_BOARD
        and project_rf_distance_to_finish(project_rf_position(position))
        <= max_distance
        for position in state.pieces[player]
    )


def _opponent_counterplay(state: GameState, player: int) -> float:
    opponent = 1 - player
    if state.current_player != opponent:
        return 0.0

    expected_value = 0.0
    for yut_result, probability in PROJECT_RF_YUT_PROBABILITIES.items():
        roll_state = _clone_without_rng(state)
        roll_state.pool_counts = empty_pool()
        roll_state.pool_counts[yut_result] = 1
        legal_actions = roll_state.get_legal_actions()
        best_for_roll = max(
            (
                _counterplay_gain(roll_state, action, opponent)
                for action in legal_actions
            ),
            default=0.0,
        )
        expected_value += probability * best_for_roll
    return expected_value


def _counterplay_gain(state: GameState, action: int, opponent: int) -> float:
    piece_id, _yut_result = decode_action(action)
    before_position = state.pieces[opponent][piece_id]
    before_distance = project_rf_distance_to_finish(
        project_rf_position(before_position)
    )
    outcome = _apply_action_without_future_rolls(state, action)
    after_position = outcome.state.pieces[opponent][piece_id]
    after_distance = project_rf_distance_to_finish(
        project_rf_position(after_position)
    )
    value = (
        2.5 * outcome.captured_count
        + 4.0 * outcome.finished_count
        + 0.03 * max(0, before_distance - after_distance)
    )
    if outcome.state.winner == opponent:
        value += 8.0
    return value


def _apply_action_without_future_rolls(
    state: GameState,
    action: int,
) -> _ImmediateOutcome:
    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be simulated: {action}")

    candidate = _clone_without_rng(state)
    actor = candidate.current_player
    opponent = 1 - actor
    piece_id, yut_result = decode_action(action)
    candidate.pool_counts[yut_result] -= 1

    moving_piece_ids = candidate.stack_piece_ids(actor, piece_id)
    move_result = candidate.board.move(
        candidate.pieces[actor][piece_id],
        steps_for(yut_result),
    )
    for moving_piece_id in moving_piece_ids:
        candidate.pieces[actor][moving_piece_id] = move_result.position

    captured_piece_ids: list[int] = []
    if (
        move_result.status == PieceStatus.ON_BOARD
        and move_result.physical_cell is not None
    ):
        captured_piece_ids = candidate.piece_ids_at_cell(
            opponent,
            move_result.physical_cell,
        )
        for captured_piece_id in captured_piece_ids:
            candidate.pieces[opponent][captured_piece_id] = Position.waiting()

        destination_piece_ids = candidate.piece_ids_at_cell(
            actor,
            move_result.physical_cell,
        )
        for destination_piece_id in destination_piece_ids:
            candidate.pieces[actor][destination_piece_id] = move_result.position

    finished_count = sum(
        candidate.pieces[actor][moving_piece_id].status
        == PieceStatus.FINISHED
        for moving_piece_id in moving_piece_ids
    )
    if all(
        position.status == PieceStatus.FINISHED
        for position in candidate.pieces[actor]
    ):
        candidate.winner = actor

    capture_bonus_keeps_turn = bool(captured_piece_ids) and not is_bonus_result(
        yut_result
    )
    has_remaining_action = bool(candidate.get_legal_actions(actor))
    turn_changed = (
        candidate.winner is None
        and not capture_bonus_keeps_turn
        and not has_remaining_action
    )
    if turn_changed:
        candidate.current_player = opponent
        candidate.pool_counts = empty_pool()

    return _ImmediateOutcome(
        state=candidate,
        captured_count=len(captured_piece_ids),
        finished_count=finished_count,
        entered_shortcut=move_result.entered_shortcut,
        turn_changed=turn_changed,
    )


def _clone_without_rng(state: GameState) -> GameState:
    clone = GameState(
        starting_player=state.current_player,
        board=state.board,
        yut_sampler=_NoRollSampler(),
    )
    clone.pieces = [positions[:] for positions in state.pieces]
    clone.pool_counts = state.pool_counts.copy()
    clone.winner = state.winner
    clone.turn_count = state.turn_count
    clone.decision_count = state.decision_count
    return clone


class _NoRollSampler:
    def sample(self) -> YutResult:
        raise RuntimeError("adapter simulations must not sample future rolls")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("project-RF checkpoint must contain a dictionary")
    for required_key in ("body", "policy"):
        if required_key not in checkpoint:
            raise ValueError(
                f"project-RF checkpoint is missing {required_key!r}"
            )
    return checkpoint
