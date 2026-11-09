"""Run one reproducible search-guided self-play training generation."""

from __future__ import annotations

import argparse
import json

from neural_policy import train_from_search
from search_learning import collect_search_self_play, collect_search_vs_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--rollout-turns", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="neural_search_v1.pt")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--rollout-checkpoint", default=None)
    parser.add_argument("--opponent-population", action="store_true")
    args = parser.parse_args()

    collector = collect_search_vs_population if args.opponent_population else collect_search_self_play
    kwargs = {
        "games": args.games,
        "seed": args.seed,
        "simulations": args.simulations,
        "rollout_turns": args.rollout_turns,
        "max_turns": 1000,
    }
    if args.opponent_population:
        kwargs["rollout_checkpoint"] = args.rollout_checkpoint
    else:
        kwargs["rollout_checkpoint"] = args.rollout_checkpoint
    examples, game_stats = collector(**kwargs)
    training_stats = train_from_search(examples, epochs=args.epochs, seed=args.seed, checkpoint=args.checkpoint, init_checkpoint=args.init_checkpoint)
    print(json.dumps({"self_play": game_stats, "training": training_stats.__dict__, "checkpoint": args.checkpoint}, indent=2))


if __name__ == "__main__":
    main()
