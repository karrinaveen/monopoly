"""State invariants used by large simulation runs."""

from __future__ import annotations

from collections import Counter

from game_state import GameState
from monopoly_engine import build_deck


def card_locations(state: GameState) -> list[str]:
    locations: list[str] = []
    locations.extend(card.name for card in state.draw_pile)
    locations.extend(card.name for card in state.discard_pile)
    for player in state.players:
        locations.extend(card.name for card in player.hand)
        locations.extend(card.name for card in player.bank)
        for property_set in player.sets.values():
            locations.extend(card.name for card in property_set.cards)
            if property_set.house:
                locations.append(property_set.house.name)
            if property_set.hotel:
                locations.append(property_set.hotel.name)
    return locations


def validate_state(state: GameState) -> None:
    expected = Counter(card.name for card in build_deck())
    actual = Counter(card_locations(state))
    if actual != expected:
        missing = list((expected - actual).elements())
        extra = list((actual - expected).elements())
        raise AssertionError(f"Card conservation failed; missing={missing[:3]}, extra={extra[:3]}")
    if state.winner is not None and state.players[state.winner].completed_sets() < 3:
        raise AssertionError("Winner does not have three complete sets")
    for player in state.players:
        for color, property_set in player.sets.items():
            if not property_set.cards and property_set.house is None and property_set.hotel is None:
                raise AssertionError("Empty property set retained")
            if len(property_set.cards) > 0:
                for card in property_set.cards:
                    if not card.is_universal_wild and color not in card.colors:
                        raise AssertionError(f"Incompatible card {card.name} in {color}")
