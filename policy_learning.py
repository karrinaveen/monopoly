"""Tabular self-play learning for the main action-selection policy."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agents import Agent
from game_state import GameState, LegalMove
from learning import action_id
from response_learning import TabularResponsePolicy, response_context
from simulation import run_game


def action_context(state: GameState, player_index: int, moves: list[LegalMove]) -> tuple:
    player = state.players[player_index]
    own_progress = tuple(sorted(len(property_set.cards) for property_set in player.sets.values()))
    opponent_summary = tuple(
        (len(opponent.hand), opponent.bank_value(), opponent.completed_sets())
        for index, opponent in enumerate(state.players)
        if index != player_index
    )
    move_kinds = tuple(sorted(move.kind for move in moves))
    return (
        state.actions_used,
        len(player.hand),
        min(player.bank_value(), 20),
        player.completed_sets(),
        own_progress,
        opponent_summary,
        move_kinds,
    )


@dataclass
class ActionStats:
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.5


@dataclass
class TabularActionPolicy:
    table: dict[str, dict[str, ActionStats]] = field(default_factory=dict)
    epsilon: float = 0.2
    rng: random.Random = field(default_factory=random.Random, repr=False)

    @staticmethod
    def key(context: tuple) -> str:
        return hashlib.blake2b(repr(context).encode(), digest_size=12).hexdigest()

    def choose(self, context: tuple, moves: list[LegalMove]) -> LegalMove:
        key = self.key(context)
        entries = self.table.setdefault(key, {})
        for move in moves:
            entries.setdefault(str(action_id(move)), ActionStats())
        if self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        return max(moves, key=lambda move: (entries[str(action_id(move))].mean_reward, entries[str(action_id(move))].visits))

    def update(self, context: tuple, move: LegalMove | int, reward: float) -> None:
        key = self.key(context)
        action_key = str(action_id(move)) if isinstance(move, LegalMove) else str(move)
        stats = self.table.setdefault(key, {}).setdefault(action_key, ActionStats())
        stats.visits += 1
        stats.total_reward += reward

    def save(self, path: str | Path) -> None:
        payload = {
            key: {action: {"visits": stats.visits, "total_reward": stats.total_reward} for action, stats in entries.items()}
            for key, entries in self.table.items()
        }
        Path(path).write_text(json.dumps(payload, indent=2))


class SelfPlayLearningAgent(Agent):
    def __init__(self, action_policy: TabularActionPolicy, response_policy: TabularResponsePolicy, seed: int, name: str, player_index: int):
        self.action_policy = action_policy
        self.response_policy = response_policy
        self.rng = random.Random(seed)
        self.name = name
        self.player_index = player_index
        self.action_decisions: list[tuple[tuple, int]] = []
        self.response_decisions: list[tuple[tuple, bool]] = []

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        context = action_context(state, self.player_index, moves)
        move = self.action_policy.choose(context, moves)
        self.action_decisions.append((context, action_id(move)))
        return move

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        context = response_context(state, action_name, target_index, self.player_index)
        decision = self.response_policy.choose(context)
        self.response_decisions.append((context, decision))
        return decision


def train_action_policy(episodes: int = 1000, seed: int = 1, response_policy: TabularResponsePolicy | None = None) -> tuple[TabularActionPolicy, TabularResponsePolicy, dict[str, float]]:
    action_policy = TabularActionPolicy(rng=random.Random(seed))
    response_policy = response_policy or TabularResponsePolicy(epsilon=0.2, rng=random.Random(seed + 1))
    wins = 0
    timeouts = 0
    action_decisions = 0
    response_decisions = 0
    for episode in range(episodes):
        agents = [SelfPlayLearningAgent(action_policy, response_policy, seed + episode * 3 + index, f"learner_{index}", index) for index in range(3)]
        result = run_game(agents, seed=seed + episode, max_turns=300)
        for player_index, agent in enumerate(agents):
            reward = 1.0 if result.winner == player_index else 0.5 if result.winner is None else 0.0
            for context, chosen_action in agent.action_decisions:
                action_policy.update(context, chosen_action, reward)
                action_decisions += 1
            for context, decision in agent.response_decisions:
                response_policy.update(context, decision, reward)
                response_decisions += 1
        if result.winner is not None:
            wins += 1
        else:
            timeouts += 1
    action_policy.epsilon = 0.0
    response_policy.epsilon = 0.0
    return action_policy, response_policy, {"episodes": episodes, "completed_games": wins, "timeouts": timeouts, "action_decisions": action_decisions, "response_decisions": response_decisions, "contexts": len(action_policy.table)}
