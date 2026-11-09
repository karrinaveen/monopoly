"""Collect search-improved self-play targets and train a policy/value model."""

from __future__ import annotations

import random
from dataclasses import dataclass

from agents import Agent
from agents import CashFirstAgent, PropertyFirstAgent, RandomAgent, TacticalAgent
from game_state import GameState, LegalMove
from learning import action_id, encode_observation
from mcts import ISMCTSAgent
from response_learning import TabularResponsePolicy
from simulation import apply_move, run_game
from neural_agent import NeuralAgent


@dataclass(frozen=True)
class SearchTrainingStep:
    player_index: int
    observation: tuple[int, ...]
    legal_action_ids: tuple[int, ...]
    policy_target: tuple[float, ...]
    legal_action_features: tuple[tuple[float, ...], ...]
    winner: int | None
    draw: bool


class SearchSelfPlayAgent(ISMCTSAgent):
    name = "search_self_play"

    def __init__(self, simulations: int, rollout_turns: int, seed: int, response_policy: TabularResponsePolicy | None = None, rollout_checkpoint: str | None = None):
        super().__init__(simulations=simulations, rollout_turns=rollout_turns, seed=seed, response_policy=response_policy, rollout_checkpoint=rollout_checkpoint)
        self.steps: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[tuple[float, ...], ...]]] = []

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        result = self.search(state, moves)
        total = sum(result.visit_counts)
        if total:
            target = tuple(count / total for count in result.visit_counts)
        else:
            target = tuple(1 / len(moves) for _ in moves)
        self.steps.append((self.player_index, encode_observation(state, self.player_index), result.action_ids, target, result.action_features))
        return result.move


def collect_search_self_play(
    games: int,
    seed: int = 1,
    simulations: int = 16,
    rollout_turns: int = 20,
    response_policy: TabularResponsePolicy | None = None,
    rollout_checkpoint: str | None = None,
    max_turns: int = 1000,
) -> tuple[list[SearchTrainingStep], dict[str, int]]:
    examples: list[SearchTrainingStep] = []
    wins = [0, 0, 0]
    draws = 0
    unfinished = 0
    for game in range(games):
        agents = [
            SearchSelfPlayAgent(simulations, rollout_turns, seed + game * 3 + player, response_policy, rollout_checkpoint)
            for player in range(3)
        ]
        result = run_game(agents, seed=seed + game, max_turns=max_turns)
        if result.winner is not None:
            wins[result.winner] += 1
        elif result.draw:
            draws += 1
        else:
            unfinished += 1
        for agent in agents:
            for player_index, observation, legal_ids, target, features in agent.steps:
                examples.append(SearchTrainingStep(player_index, observation, legal_ids, target, features, result.winner, result.draw))
    return examples, {"games": games, "wins_player_0": wins[0], "wins_player_1": wins[1], "wins_player_2": wins[2], "draws": draws, "unfinished": unfinished, "examples": len(examples)}


def collect_search_vs_population(
    games: int,
    seed: int = 1,
    simulations: int = 16,
    rollout_turns: int = 20,
    rollout_checkpoint: str | None = None,
    max_turns: int = 1000,
) -> tuple[list[SearchTrainingStep], dict[str, int]]:
    """Train the search policy against a rotating population of opponents."""
    examples: list[SearchTrainingStep] = []
    stats = {"games": games, "candidate_wins": 0, "opponent_wins": 0, "draws": 0, "unfinished": 0, "examples": 0}
    for game in range(games):
        candidate_seat = game % 3
        candidate = SearchSelfPlayAgent(simulations, rollout_turns, seed + game, rollout_checkpoint=rollout_checkpoint)
        opponents: list[Agent] = []
        for slot in range(2):
            member = (game * 2 + slot) % 7
            member_seed = seed + 10000 + game * 2 + slot
            if member == 0:
                opponent = NeuralAgent("neural_search_semantic_v1.pt", seed=member_seed)
            elif member == 1:
                opponent = NeuralAgent("neural_search_v2.pt", seed=member_seed)
            elif member == 2:
                opponent = RandomAgent(member_seed)
            elif member == 3:
                opponent = CashFirstAgent()
            elif member == 4:
                opponent = PropertyFirstAgent()
            elif member == 5:
                opponent = TacticalAgent()
            else:
                opponent = NeuralAgent("neural_search_semantic_v2.pt", seed=member_seed)
            opponents.append(opponent)
        agents = opponents[:]
        agents.insert(candidate_seat, candidate)
        result = run_game(agents, seed=seed + game, max_turns=max_turns)
        if result.winner is None:
            stats["draws" if result.draw else "unfinished"] += 1
        elif result.winner == candidate_seat:
            stats["candidate_wins"] += 1
        else:
            stats["opponent_wins"] += 1
        for player_index, observation, legal_ids, target, features in candidate.steps:
            examples.append(SearchTrainingStep(player_index, observation, legal_ids, target, features, result.winner, result.draw))
    stats["examples"] = len(examples)
    return examples, stats
