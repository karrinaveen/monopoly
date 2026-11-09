"""Strategy-neutral self-play data collection.

This module deliberately contains no hand-written strategy. The initial policy
chooses uniformly from legal moves and records complete trajectories for later
MCTS and policy/value training.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from agents import Agent
from game_state import GameState, LegalMove
from simulation import apply_move
from monopoly_engine import CardKind, SET_SPECS, build_deck


CARD_IDS = {card.name: index for index, card in enumerate(build_deck())}
CARD_BY_NAME = {card.name: card for card in build_deck()}
MOVE_KINDS = ("end_turn", "bank", "pass_go", "property", "rent", "birthday", "debt_collector", "deal_breaker", "sly_deal", "forced_deal", "house", "hotel", "discard")
ACTION_NAMES = tuple(sorted({card.action for card in build_deck() if card.action}))
COLORS = tuple(sorted(SET_SPECS))


def encode_move(move: LegalMove) -> str:
    fields = (
        move.kind,
        move.card_name,
        move.color,
        move.target_index,
        move.target_color,
        move.target_card_name,
        move.destination_color,
        move.destination_color_2,
        move.card_name_2,
    )
    return "|".join("" if value is None else str(value) for value in fields)


def action_id(move: LegalMove) -> int:
    digest = hashlib.blake2b(encode_move(move).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def encode_action_features(move: LegalMove) -> tuple[float, ...]:
    """Semantic move representation shared across cards and game states."""
    features = [0.0] * 97

    def one_hot(start: int, labels: tuple[str, ...], value: str | None) -> None:
        if value in labels:
            features[start + labels.index(value)] = 1.0

    one_hot(0, MOVE_KINDS, move.kind)
    card = CARD_BY_NAME.get(move.card_name or "")
    if card:
        one_hot(13, ACTION_NAMES, card.action)
        features[30] = card.value / 10.0
        features[31 + list(CardKind).index(card.kind)] = 1.0
    one_hot(37, COLORS, move.color)
    one_hot(47, COLORS, move.target_color)
    one_hot(57, COLORS, move.destination_color)
    one_hot(67, COLORS, move.destination_color_2)
    if move.target_index is not None:
        features[77 + move.target_index % 3] = 1.0
    second = CARD_BY_NAME.get(move.card_name_2 or "")
    if second:
        one_hot(80, ACTION_NAMES, second.action)
        features[95] = second.value / 10.0
        features[96] = 1.0
    return tuple(features)


def _card_vector(cards) -> list[int]:
    vector = [0] * len(CARD_IDS)
    for card in cards:
        vector[CARD_IDS[card.name]] += 1
    return vector


def encode_observation(state: GameState, player_index: int) -> tuple[int, ...]:
    """Encode a player's information set without revealing opponent hands."""
    player = state.players[player_index]
    features: list[int] = []
    features.extend(_card_vector(player.hand))
    features.extend(_card_vector(player.bank))
    features.extend(_card_vector(player.all_property_cards()))
    features.extend(_card_vector(state.discard_pile))

    for opponent_index, opponent in enumerate(state.players):
        if opponent_index == player_index:
            continue
        features.extend((len(opponent.hand), opponent.bank_value(), opponent.completed_sets()))
        for color in sorted(SET_SPECS):
            property_set = opponent.sets.get(color)
            features.extend((len(property_set.cards) if property_set else 0, int(property_set.is_complete()) if property_set else 0, property_set.rent() if property_set else 0))

    features.extend((state.current_player, state.actions_used, int(state.property_locked), len(state.draw_pile), len(state.discard_pile)))
    return tuple(features)


@dataclass(frozen=True)
class EpisodeStep:
    player_index: int
    observation: tuple[int, ...]
    legal_action_ids: tuple[int, ...]
    chosen_action_id: int
    winner: int | None = None
    legal_action_features: tuple[tuple[float, ...], ...] = ()


class RandomPolicyAgent(Agent):
    name = "random_policy"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        return self.rng.choice(moves)

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        return self.rng.random() < 0.5


def generate_episode(seed: int, max_turns: int = 500) -> list[EpisodeStep]:
    rng = random.Random(seed)
    agents = [RandomPolicyAgent(rng.randrange(1 << 30)) for _ in range(3)]
    state = GameState.new([f"Random {index + 1}" for index in range(3)], seed=seed)
    pending: list[tuple[int, tuple[int, ...], tuple[int, ...], int, tuple[tuple[float, ...], ...]]] = []
    for _ in range(max_turns):
        if state.winner is not None:
            break
        state.begin_turn()
        while state.turn_active and state.winner is None:
            moves = state.legal_moves()
            if not moves:
                state.end_turn()
                break
            player_index = state.current_player
            observation = encode_observation(state, player_index)
            legal_ids = tuple(action_id(move) for move in moves)
            legal_features = tuple(encode_action_features(move) for move in moves)
            move = agents[player_index].choose_move(state, moves)
            pending.append((player_index, observation, legal_ids, action_id(move), legal_features))
            apply_move(state, move, agents)
            if move.kind == "end_turn":
                break

    return [EpisodeStep(player, observation, legal, chosen, state.winner, features) for player, observation, legal, chosen, features in pending]


def collect_episodes(count: int, seed: int = 1) -> list[EpisodeStep]:
    records: list[EpisodeStep] = []
    for index in range(count):
        records.extend(generate_episode(seed + index))
    return records
