"""Seat-rotated cross-play evaluation."""

from __future__ import annotations

from collections import Counter

from simulation import run_game


def evaluate(agent_factory, opponent_factory, games: int = 30, seed: int = 1, max_turns: int = 3000) -> dict:
    results = Counter()
    turns: list[int] = []
    for index in range(games):
        neural_seat = index % 3
        opponents = [opponent_factory(index, slot) for slot in range(2)]
        agents = opponents[:]
        agents.insert(neural_seat, agent_factory(index))
        result = run_game(agents, seed=seed + index, max_turns=max_turns)
        turns.append(result.turns)
        if result.winner is None:
            results["draw" if result.draw else "unfinished_simulation"] += 1
        elif result.winner == neural_seat:
            results["agent"] += 1
        else:
            results["opponent"] += 1
    results["games"] = games
    results["avg_turns"] = round(sum(turns) / len(turns), 2)
    return dict(results)
