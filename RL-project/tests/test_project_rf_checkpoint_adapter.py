from pathlib import Path

import pytest
import torch

from yutnori.agents.project_rf_checkpoint import (
    PROJECT_RF_STATE_SIZE,
    ProjectRFCheckpointAgent,
    encode_project_rf_state,
    local_action_to_project_rf,
    project_rf_action_to_local,
    project_rf_position,
)
from yutnori.core import (
    Cell,
    GameState,
    Position,
    Route,
    YutResult,
    encode_action,
)


def test_project_rf_action_mapping_round_trips_all_local_actions():
    for piece_id in range(4):
        for yut_result in (
            YutResult.DO,
            YutResult.GAE,
            YutResult.GEOL,
            YutResult.YUT,
            YutResult.MO,
        ):
            local_action = encode_action(piece_id, yut_result)
            assert project_rf_action_to_local(
                local_action_to_project_rf(local_action)
            ) == local_action


def test_project_rf_action_mapping_rejects_back_do():
    with pytest.raises(ValueError, match="BACK_DO"):
        local_action_to_project_rf(encode_action(0, YutResult.BACK_DO))


def test_project_rf_position_projection_covers_local_only_a3_a4_cells():
    assert project_rf_position(Position.at(Route.C1_DIAGONAL, 4)) == 12
    assert project_rf_position(Position.at(Route.C1_DIAGONAL, 5)) == 13
    assert project_rf_position(Position.home()) == 19


def test_project_rf_state_projection_has_expected_shape():
    state = GameState()
    state.set_pool(YutResult.DO, YutResult.YUT)

    encoded = encode_project_rf_state(state)

    assert encoded.shape == (PROJECT_RF_STATE_SIZE,)
    assert encoded.dtype.name == "float32"


def test_checkpoint_agent_maps_project_logits_back_to_local_actions(
    tmp_path: Path,
):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint, preferred_project_action=2)
    state = GameState()
    state.set_pool(YutResult.DO)
    agent = ProjectRFCheckpointAgent(
        checkpoint,
        use_tactical_prior=False,
    )

    action = agent.select_action(state, state.get_legal_actions())

    assert action == encode_action(2, YutResult.DO)


def test_capture_aware_adapter_prefers_available_capture(tmp_path: Path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    state = GameState()
    state.pieces[0][0] = Position.at(Route.OUTER, 1)
    state.pieces[0][1] = Position.at(Route.OUTER, 4)
    state.pieces[0][2] = Position.finished()
    state.pieces[0][3] = Position.finished()
    state.pieces[1][0] = Position.at(Route.OUTER, 3)
    state.set_pool(YutResult.GAE)
    agent = ProjectRFCheckpointAgent(checkpoint, use_tactical_prior=True)

    action = agent.select_action(state, state.get_legal_actions())

    assert action == encode_action(0, YutResult.GAE)


def test_capture_aware_adapter_does_not_sample_evaluation_rng(tmp_path: Path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    state = GameState(yut_sampler=_ExplodingSampler())
    state.set_pool(YutResult.DO)
    agent = ProjectRFCheckpointAgent(checkpoint, use_tactical_prior=True)

    action = agent.select_action(state, state.get_legal_actions())

    assert action in state.get_legal_actions()


def test_checkpoint_agent_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ProjectRFCheckpointAgent(tmp_path / "missing.pt")


def _write_zero_checkpoint(
    path: Path,
    *,
    preferred_project_action: int | None = None,
) -> None:
    hidden_size = 8
    body = torch.nn.Sequential(
        torch.nn.Linear(PROJECT_RF_STATE_SIZE, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
    )
    policy = torch.nn.Linear(hidden_size, 20)
    for parameter in list(body.parameters()) + list(policy.parameters()):
        torch.nn.init.zeros_(parameter)
    if preferred_project_action is not None:
        with torch.no_grad():
            policy.bias[preferred_project_action] = 1.0
    torch.save(
        {
            "state_dim": PROJECT_RF_STATE_SIZE,
            "body": body.state_dict(),
            "policy": policy.state_dict(),
        },
        path,
    )


class _ExplodingSampler:
    def sample(self):
        raise AssertionError("evaluation RNG must not be sampled")
