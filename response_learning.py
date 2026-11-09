"""Outcome-based learning for Just Say No response decisions."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agents import Agent
from game_state import GameState, LegalMove
from simulation import run_game


def response_context(state: GameState, action_name: str, target_index: int, responder_index: int) -> tuple:
    target = state.players[target_index]
    responder = state.players[responder_index]
    target_rents = tuple(sorted(property_set.rent() for property_set in target.sets.values() if property_set.is_complete()))
    return (
        action_name,
        responder_index == state.current_player,
        state.actions_used,
        responder.bank_value(),
        responder.completed_sets(),
        target.completed_sets(),
        target_rents,
        len(responder.hand),
    )


@dataclass
class ResponseStats:
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.5


@dataclass
class TabularResponsePolicy:
    table: dict[str, list[ResponseStats]] = field(default_factory=lambda: defaultdict(lambda: [ResponseStats(), ResponseStats()]))
    epsilon: float = 0.15
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def choose(self, context: tuple) -> bool:
        key = repr(context)
        stats = self.table[key]
        if self.rng.random() < self.epsilon or stats[0].visits == 0 or stats[1].visits == 0:
            decision = self.rng.randrange(2)
        else:
            decision = 1 if stats[1].mean_reward > stats[0].mean_reward else 0
        return bool(decision)

    def update(self, context: tuple, decision: bool, reward: float) -> None:
        stats = self.table[repr(context)][int(decision)]
        stats.visits += 1
        stats.total_reward += reward

    def save(self, path: str | Path) -> None:
        payload = {key: [{"visits": item.visits, "total_reward": item.total_reward} for item in values] for key, values in self.table.items()}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path, epsilon: float = 0.0) -> "TabularResponsePolicy":
        policy = cls(epsilon=epsilon)
        payload = json.loads(Path(path).read_text())
        for key, values in payload.items():
            policy.table[key] = [ResponseStats(item["visits"], item["total_reward"]) for item in values]
        return policy


class LearningResponseAgent(Agent):
    def __init__(self, policy: TabularResponsePolicy, seed: int, name: str, player_index: int):
        self.policy = policy
        self.rng = random.Random(seed)
        self.name = name
        self.player_index = player_index
        self.response_decisions: list[tuple[tuple, bool]] = []

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        return self.rng.choice(moves)

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        context = response_context(state, action_name, target_index, self.player_index)
        decision = self.policy.choose(context)
        self.response_decisions.append((context, decision))
        return decision


def train_response_policy(episodes: int = 1000, seed: int = 1) -> tuple[TabularResponsePolicy, dict[str, float]]:
    policy = TabularResponsePolicy(epsilon=0.2, rng=random.Random(seed))
    wins = 0
    timeouts = 0
    decisions = 0
    for episode in range(episodes):
        agents = [LearningResponseAgent(policy, seed + episode * 3 + index, f"learner_{index}", index) for index in range(3)]
        result = run_game(agents, seed=seed + episode, max_turns=300)
        for player_index, agent in enumerate(agents):
            reward = 1.0 if result.winner == player_index else 0.5 if result.winner is None else 0.0
            for context, decision in agent.response_decisions:
                policy.update(context, decision, reward)
                decisions += 1
        if result.winner is not None:
            wins += 1
        elif result.draw:
            timeouts += 1
        else:
            timeouts += 1
    policy.epsilon = 0.0
    return policy, {"episodes": episodes, "completed_games": wins, "timeouts": timeouts, "response_decisions": decisions, "contexts": len(policy.table)}
