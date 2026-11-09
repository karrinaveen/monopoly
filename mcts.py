"""Determinized Information-Set Monte Carlo search.

This is a deliberately small first searcher. It samples plausible opponent
hands from the unseen deck, evaluates legal root actions with random rollouts,
and returns the action with the best estimated win rate.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

from agents import Agent
from game_state import GameState, LegalMove
from learning import action_id
from monopoly_engine import build_deck
from simulation import apply_move
from response_learning import TabularResponsePolicy, response_context
from learning import encode_action_features, encode_observation
from neural_policy import PolicyValueNet, action_features, normalized_observation, torch


def _visible_card_names(state: GameState, observer: int) -> set[str]:
    visible: set[str] = set(card.name for card in state.players[observer].hand)
    visible.update(card.name for card in state.discard_pile)
    for player in state.players:
        visible.update(card.name for card in player.bank)
        for property_set in player.sets.values():
            visible.update(card.name for card in property_set.cards)
            if property_set.house:
                visible.add(property_set.house.name)
            if property_set.hotel:
                visible.add(property_set.hotel.name)
    return visible


def determinize(state: GameState, observer: int, rng: random.Random) -> GameState:
    """Sample hidden hands and the draw pile while preserving public state."""
    sampled = copy.deepcopy(state)
    visible = _visible_card_names(state, observer)
    unknown = [card for card in build_deck() if card.name not in visible]
    rng.shuffle(unknown)
    for index, player in enumerate(sampled.players):
        if index == observer:
            continue
        hand_size = len(state.players[index].hand)
        player.hand = [unknown.pop() for _ in range(hand_size)]
    sampled.draw_pile = unknown
    sampled.rng = random.Random(rng.randrange(1 << 30))
    return sampled


@dataclass
class SearchStats:
    visits: int = 0
    wins: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    move: LegalMove
    action_ids: tuple[int, ...]
    action_features: tuple[tuple[float, ...], ...]
    visit_counts: tuple[int, ...]
    values: tuple[float, ...]


class ISMCTSAgent(Agent):
    name = "ismcts"

    def __init__(self, simulations: int = 64, rollout_turns: int = 80, seed: int | None = None, response_policy: TabularResponsePolicy | None = None, rollout_checkpoint: str | None = None):
        self.simulations = simulations
        self.rollout_turns = rollout_turns
        self.rng = random.Random(seed)
        self.response_policy = response_policy
        self.player_index = 0
        self.rollout_model = None
        if rollout_checkpoint:
            if torch is None:
                raise RuntimeError("PyTorch is required for neural-guided rollouts")
            saved = torch.load(rollout_checkpoint, map_location="cpu")
            self.rollout_model = PolicyValueNet(saved["observation_dim"], action_feature_dim=saved.get("action_feature_dim", 64))
            self.rollout_model.load_state_dict(saved["model"])
            self.rollout_model.eval()

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        return self.search(state, moves).move

    def search(self, state: GameState, moves: list[LegalMove]) -> SearchResult:
        if len(moves) == 1:
            return SearchResult(moves[0], (action_id(moves[0]),), (encode_action_features(moves[0]),), (1,), (0.5,))
        observer = state.current_player
        stats = {action_id(move): SearchStats() for move in moves}
        for _ in range(self.simulations):
            root_move = self._select_root_move(moves, stats)
            sampled = determinize(state, observer, self.rng)
            sampled_moves = sampled.legal_moves()
            selected_id = action_id(root_move)
            sampled_move = next((move for move in sampled_moves if action_id(move) == selected_id), None)
            if sampled_move is None:
                continue
            try:
                apply_move(sampled, sampled_move)
                self._rollout(sampled)
            except (RuntimeError, ValueError):
                result = 0.0
            else:
                result = 1.0 if sampled.winner == observer else 0.0
            entry = stats[selected_id]
            entry.visits += 1
            entry.wins += result
        chosen = max(moves, key=lambda move: (stats[action_id(move)].visits, stats[action_id(move)].win_rate))
        return SearchResult(
            chosen,
            tuple(action_id(move) for move in moves),
            tuple(encode_action_features(move) for move in moves),
            tuple(stats[action_id(move)].visits for move in moves),
            tuple(stats[action_id(move)].win_rate for move in moves),
        )

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        """Expose Just Say No as a search-agent decision.

        The first search baseline samples this response decision. It is
        intentionally not a hand-coded tactical rule; later policy training
        will replace this probability with a learned response head.
        """
        if self.response_policy is not None:
            context = response_context(state, action_name, target_index, self.player_index)
            return self.response_policy.choose(context)
        return self.rng.random() < 0.5

    def _select_root_move(self, moves: list[LegalMove], stats: dict[int, SearchStats]) -> LegalMove:
        unexplored = [move for move in moves if stats[action_id(move)].visits == 0]
        if unexplored:
            return self.rng.choice(unexplored)
        total = sum(entry.visits for entry in stats.values())
        exploration = math.sqrt(math.log(max(total, 1)))
        return max(
            moves,
            key=lambda move: stats[action_id(move)].win_rate + 1.4 * exploration / math.sqrt(stats[action_id(move)].visits),
        )

    def _rollout(self, state: GameState) -> None:
        for _ in range(self.rollout_turns):
            if state.winner is not None:
                return
            state.begin_turn()
            while state.turn_active and state.winner is None:
                moves = state.legal_moves()
                if not moves:
                    state.end_turn()
                    break
                move = self._rollout_move(state, moves)
                rollout_agents = [self, self, self]
                apply_move(state, move, rollout_agents)
                if move.kind == "end_turn":
                    break

    def _rollout_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        if self.rollout_model is None:
            return self.rng.choice(moves)
        player_index = state.current_player
        observation = torch.tensor([normalized_observation(encode_observation(state, player_index))], dtype=torch.float32)
        feature_dim = self.rollout_model.action_encoder[0].in_features
        features = [action_features(action_id(move)) for move in moves] if feature_dim == 64 else [encode_action_features(move) for move in moves]
        action_batch = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.rollout_model(observation, action_batch)
        return moves[int(logits[0].argmax().item())]
