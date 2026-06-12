"""Tournament helpers for baseline and learned agents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from yutnori.agents import Agent
from yutnori.core import GameEvent, GameState, PLAYER_COUNT, YutSampler


@dataclass
class GameResult:
    winner: int
    starting_player: int
    turn_count: int
    decision_count: int
    events: list[GameEvent] = field(default_factory=list)


@dataclass
class TournamentResult:
    games: int
    wins: dict[int, int]
    starting_player_counts: dict[int, int]
    total_turns: int
    total_decisions: int

    def win_rate(self, player: int) -> float:
        if self.games == 0:
            return 0.0
        return self.wins.get(player, 0) / self.games

    @property
    def average_turns(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_turns / self.games

    @property
    def average_decisions(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_decisions / self.games


def play_game(
    agent0: Agent,
    agent1: Agent,
    *,
    seed: int | None = None,
    starting_player: int | None = None,
    max_decisions: int = 10_000,
    record_events: bool = False,
) -> GameResult:
    rng = random.Random(seed)
    resolved_starting_player = (
        rng.randrange(PLAYER_COUNT) if starting_player is None else starting_player
    )
    if resolved_starting_player < 0 or resolved_starting_player >= PLAYER_COUNT:
        raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")

    state = GameState(
        starting_player=resolved_starting_player,
        yut_sampler=YutSampler(rng=rng),
    )
    state.start_turn()
    agents = {0: agent0, 1: agent1}
    events: list[GameEvent] = []

    while state.winner is None:
        if state.decision_count >= max_decisions:
            raise RuntimeError(
                f"game exceeded max_decisions={max_decisions} without a winner"
            )
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("current player has no legal actions")
        agent = agents[state.current_player]
        action = int(agent.select_action(state, legal_actions))
        if action not in legal_actions:
            raise ValueError(
                f"{agent.name} selected illegal action {action}; "
                f"legal_actions={legal_actions}"
            )
        event = state.apply_action(action)
        if record_events:
            events.append(event)

    return GameResult(
        winner=state.winner,
        starting_player=resolved_starting_player,
        turn_count=state.turn_count,
        decision_count=state.decision_count,
        events=events,
    )


def run_tournament(
    agent0: Agent,
    agent1: Agent,
    *,
    games: int,
    seed: int | None = None,
    max_decisions: int = 10_000,
) -> TournamentResult:
    if games < 0:
        raise ValueError("games must be non-negative")

    rng = random.Random(seed)
    wins = {0: 0, 1: 0}
    starting_player_counts = {0: 0, 1: 0}
    total_turns = 0
    total_decisions = 0

    for _ in range(games):
        game_seed = rng.randrange(2**63)
        result = play_game(
            agent0,
            agent1,
            seed=game_seed,
            max_decisions=max_decisions,
        )
        wins[result.winner] += 1
        starting_player_counts[result.starting_player] += 1
        total_turns += result.turn_count
        total_decisions += result.decision_count

    return TournamentResult(
        games=games,
        wins=wins,
        starting_player_counts=starting_player_counts,
        total_turns=total_turns,
        total_decisions=total_decisions,
    )
