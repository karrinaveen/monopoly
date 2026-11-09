"""Seat-rotated simulation benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from agents import RandomAgent
from simulation import run_game


MAX_TURNS = 3000


def play_random_game(index: int) -> tuple[int | None, int, bool]:
    agents = [RandomAgent(100000 + index * 3 + position) for position in range(3)]
    result = run_game(agents, seed=700000 + index, max_turns=MAX_TURNS)
    return result.winner, result.turns, result.draw


def run_random_benchmark(games: int, workers: int) -> None:
    wins: Counter = Counter()
    turns: list[int] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for winner, length, is_draw in pool.map(play_random_game, range(games), chunksize=25):
            wins["draw" if is_draw else "unfinished_simulation" if winner is None else winner] += 1
            turns.append(length)
    print(f"games={games}")
    print(f"results={dict(wins)}")
    print(f"avg_turns={sum(turns) / len(turns):.2f}")
    print(f"max_turns={max(turns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_random_benchmark(args.games, args.workers)
