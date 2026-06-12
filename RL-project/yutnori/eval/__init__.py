"""Evaluation and analysis utilities."""

from yutnori.eval.legacy_head_to_head import (
    LEGACY_HEAD_TO_HEAD_PROTOCOL,
    LEGACY_RULESET,
    HeadToHeadGame,
    HeadToHeadResult,
    LegacyMaskablePPOAgent,
    LegacyNoBackdoSampler,
    encode_legacy_observation,
    evaluate_legacy_head_to_head,
)
from yutnori.eval.tournament import (
    GameResult,
    TournamentResult,
    play_game,
    run_tournament,
)

__all__ = [
    "LEGACY_HEAD_TO_HEAD_PROTOCOL",
    "LEGACY_RULESET",
    "GameResult",
    "HeadToHeadGame",
    "HeadToHeadResult",
    "LegacyMaskablePPOAgent",
    "LegacyNoBackdoSampler",
    "TournamentResult",
    "encode_legacy_observation",
    "evaluate_legacy_head_to_head",
    "play_game",
    "run_tournament",
]
