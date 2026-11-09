"""Mutable game state and basic legal moves for the custom ruleset.

This layer intentionally separates state transitions from strategy. It is the
foundation for random play, heuristic agents, and Monte Carlo search.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from monopoly_engine import (
    Card,
    CardKind,
    PropertySet,
    SET_SPECS,
    build_deck,
    complete_set_count,
)


@dataclass
class PlayerState:
    name: str
    hand: list[Card] = field(default_factory=list)
    bank: list[Card] = field(default_factory=list)
    sets: dict[str, PropertySet] = field(default_factory=dict)

    def completed_sets(self) -> int:
        return complete_set_count(self.sets.values())

    def all_property_cards(self) -> list[Card]:
        return [card for property_set in self.sets.values() for card in property_set.cards]

    def bank_value(self) -> int:
        return sum(card.value for card in self.bank)


@dataclass(frozen=True)
class LegalMove:
    kind: str
    card_name: str | None = None
    color: str | None = None
    target_index: int | None = None
    target_color: str | None = None
    target_card_name: str | None = None
    destination_color: str | None = None
    destination_color_2: str | None = None
    card_name_2: str | None = None
    description: str = ""


@dataclass
class GameState:
    players: list[PlayerState]
    draw_pile: list[Card]
    discard_pile: list[Card] = field(default_factory=list)
    current_player: int = 0
    actions_used: int = 0
    property_locked: bool = False
    turn_active: bool = False
    winner: int | None = None
    draw: bool = False
    rng: random.Random = field(default_factory=random.Random, repr=False)

    @classmethod
    def new(cls, player_names: Iterable[str] = ("Player 1", "Player 2", "Player 3"), seed: int | None = None) -> "GameState":
        names = list(player_names)
        if len(names) != 3:
            raise ValueError("The first engine version supports exactly three players")
        rng = random.Random(seed)
        deck = build_deck()
        rng.shuffle(deck)
        players = [PlayerState(name) for name in names]
        state = cls(players, deck, rng=rng)
        for player in players:
            player.hand.extend(state.draw_cards(5))
        return state

    @property
    def player(self) -> PlayerState:
        return self.players[self.current_player]

    def check_winner(self) -> int | None:
        """Set and return the winner once any player owns three sets."""
        for index, player in enumerate(self.players):
            if player.completed_sets() >= 3:
                self.winner = index
                return index
        return self.winner

    def check_exhaustion_draw(self) -> bool:
        """The user-defined rare draw: both recyclable piles are exhausted."""
        if self.winner is None and not self.draw_pile and not self.discard_pile:
            self.draw = True
        return self.draw

    def draw_cards(self, count: int) -> list[Card]:
        drawn: list[Card] = []
        for _ in range(count):
            if not self.draw_pile:
                if not self.discard_pile:
                    break
                self.draw_pile, self.discard_pile = self.discard_pile, []
                self.rng.shuffle(self.draw_pile)
            drawn.append(self.draw_pile.pop())
        return drawn

    def begin_turn(self) -> None:
        if self.turn_active:
            raise RuntimeError("A turn is already active")
        if self.winner is not None:
            raise RuntimeError("The game has already ended")
        self.turn_active = True
        self.actions_used = 0
        self.property_locked = False
        draw_count = 5 if not self.player.hand else 2
        self.player.hand.extend(self.draw_cards(draw_count))

    def _require_turn(self) -> None:
        if not self.turn_active:
            raise RuntimeError("Begin the turn before playing a card")
        if self.actions_used >= 3:
            raise RuntimeError("This player has used all three actions")

    def _remove_from_hand(self, card_name: str) -> Card:
        for index, card in enumerate(self.player.hand):
            if card.name == card_name:
                return self.player.hand.pop(index)
        raise ValueError(f"Card is not in hand: {card_name}")

    def play_to_bank(self, card_name: str) -> Card:
        self._require_turn()
        card = self._remove_from_hand(card_name)
        if card.kind == CardKind.PROPERTY or card.kind == CardKind.WILD_PROPERTY:
            self.player.hand.append(card)
            raise ValueError("Property cards cannot be banked as cash")
        self.player.bank.append(card)
        self.actions_used += 1
        self.property_locked = True
        return card

    def play_property(self, card_name: str, color: str | None = None) -> PropertySet:
        self._require_turn()
        card = self._remove_from_hand(card_name)
        if card.kind not in (CardKind.PROPERTY, CardKind.WILD_PROPERTY):
            self.player.hand.append(card)
            raise ValueError("Only property cards can be placed in a property set")

        if card.is_universal_wild:
            if color not in SET_SPECS:
                self.player.hand.append(card)
                raise ValueError("A universal wild requires a target color")
        elif color is None:
            if len(card.colors) != 1:
                self.player.hand.append(card)
                raise ValueError("A double-color wild requires a chosen color")
            color = card.colors[0]
        elif color not in card.colors:
            self.player.hand.append(card)
            raise ValueError("Card cannot represent the selected color")

        property_set = self.player.sets.setdefault(color, PropertySet(color))
        if not property_set.can_accept(card):
            self.player.hand.append(card)
            raise ValueError("Card cannot be added to this property set")
        if len(property_set.cards) >= SET_SPECS[color].required:
            self.player.hand.append(card)
            raise ValueError("This property set is already full")
        property_set.cards.append(card)
        self.actions_used += 1
        self.property_locked = True
        if self.player.completed_sets() >= 3:
            self.winner = self.current_player
        return property_set

    def build_house(self, card_name: str, color: str) -> PropertySet:
        self._require_turn()
        property_set = self.player.sets.get(color)
        if property_set is None or not property_set.is_complete():
            raise ValueError("A House requires a complete property set")
        if property_set.house is not None:
            raise ValueError("This set already has a House")
        card = self._remove_from_hand(card_name)
        if card.kind != CardKind.HOUSE:
            self.player.hand.append(card)
            raise ValueError("The selected card is not a House")
        property_set.house = card
        self.actions_used += 1
        self.property_locked = True
        return property_set

    def build_hotel(self, card_name: str, color: str) -> PropertySet:
        self._require_turn()
        property_set = self.player.sets.get(color)
        if property_set is None or not property_set.is_complete():
            raise ValueError("A Hotel requires a complete property set")
        if property_set.house is None:
            raise ValueError("A Hotel requires a House first")
        if property_set.hotel is not None:
            raise ValueError("This set already has a Hotel")
        card = self._remove_from_hand(card_name)
        if card.kind != CardKind.HOTEL:
            self.player.hand.append(card)
            raise ValueError("The selected card is not a Hotel")
        property_set.hotel = card
        self.actions_used += 1
        self.property_locked = True
        return property_set

    def play_pass_go(self, card_name: str) -> list[Card]:
        self._require_turn()
        card = self._remove_from_hand(card_name)
        if card.action != "pass_go":
            self.player.hand.append(card)
            raise ValueError("Card is not Pass Go")
        self.discard_pile.append(card)
        drawn = self.draw_cards(2)
        self.player.hand.extend(drawn)
        self.actions_used += 1
        self.property_locked = True
        return drawn

    def play_action_to_discard(self, card_name: str, expected_action: str | None = None) -> Card:
        """Play an action card whose effect is resolved by a higher layer.

        The card is placed in the middle immediately. It is never available
        as cash after being played, and it will be reshuffled later when the
        draw pile is exhausted.
        """
        self._require_turn()
        card = self._remove_from_hand(card_name)
        if card.kind != CardKind.ACTION:
            self.player.hand.append(card)
            raise ValueError("Only action cards can be played to the discard pile")
        if expected_action is not None and card.action != expected_action:
            self.player.hand.append(card)
            raise ValueError(f"Expected {expected_action}, got {card.action}")
        self.discard_pile.append(card)
        self.actions_used += 1
        self.property_locked = True
        return card

    def discard_from_hand(self, card_name: str) -> Card:
        """Discard freely only after all three actions have been used."""
        if not self.turn_active:
            raise RuntimeError("There is no active turn")
        if self.actions_used < 3:
            raise RuntimeError("Free discard is available only after three actions")
        card = self._remove_from_hand(card_name)
        self.discard_pile.append(card)
        return card

    def can_end_turn(self) -> bool:
        return self.actions_used == 3 or len(self.player.hand) <= 7

    def end_turn(self) -> None:
        if not self.turn_active:
            raise RuntimeError("There is no active turn")
        if not self.can_end_turn():
            raise RuntimeError("The player must use an action before ending with more than seven cards")
        self.turn_active = False
        self.current_player = (self.current_player + 1) % len(self.players)

    def legal_moves(self) -> list[LegalMove]:
        if not self.turn_active or self.winner is not None:
            return []
        moves: list[LegalMove] = []
        if self.can_end_turn():
            moves.append(LegalMove("end_turn", description="End the turn"))
        if self.actions_used < 3:
            for card in self.player.hand:
                if card.kind in (CardKind.MONEY, CardKind.ACTION, CardKind.HOUSE, CardKind.HOTEL):
                    moves.append(LegalMove("bank", card.name, description=f"Bank {card.name}"))
                if card.action == "pass_go":
                    moves.append(LegalMove("pass_go", card.name, description="Play Pass Go"))
                if card.action in ("birthday", "debt_collector"):
                    if card.action == "birthday":
                        moves.append(LegalMove("birthday", card.name, description="Charge Birthday"))
                    else:
                        for target_index in range(len(self.players)):
                            if target_index != self.current_player:
                                moves.append(LegalMove("debt_collector", card.name, target_index=target_index, description="Play Debt Collector"))
                if card.action == "any_color_rent":
                    for color, property_set in self.player.sets.items():
                        if property_set.is_complete():
                            for target_index in range(len(self.players)):
                                if target_index != self.current_player:
                                    moves.append(LegalMove("rent", card.name, color, target_index=target_index, description="Play Any-Color Rent"))
                rent_colors = {
                    "light_green_black_rent": ("light_green", "black"),
                    "orange_pink_rent": ("orange", "pink"),
                    "dark_green_dark_blue_rent": ("dark_green", "dark_blue"),
                    "red_yellow_rent": ("red", "yellow"),
                    "light_blue_brown_rent": ("light_blue", "brown"),
                }
                if card.action in rent_colors:
                    for candidate in rent_colors[card.action]:
                        if candidate in self.player.sets and self.player.sets[candidate].is_complete():
                            double_card = next((hand_card for hand_card in self.player.hand if hand_card.action == "double_rent"), None)
                            moves.append(LegalMove("rent", card.name, candidate, description="Play Color Rent"))
                            if double_card is not None and self.actions_used + 2 <= 3:
                                moves.append(LegalMove("rent", card.name, candidate, card_name_2=double_card.name, description="Play Double Color Rent"))
                if card.action == "deal_breaker":
                    for target_index, opponent in enumerate(self.players):
                        if target_index == self.current_player:
                            continue
                        for color, property_set in opponent.sets.items():
                            if property_set.is_complete() and color not in self.player.sets:
                                moves.append(LegalMove("deal_breaker", card.name, target_index=target_index, target_color=color, description="Play Deal Breaker"))
                if card.action == "forced_deal":
                    for target_index, opponent in enumerate(self.players):
                        if target_index == self.current_player:
                            continue
                        for own_color, own_set in self.player.sets.items():
                            if own_set.is_complete():
                                continue
                            for own_card in own_set.cards:
                                for target_color, target_set in opponent.sets.items():
                                    if target_set.is_complete():
                                        continue
                                    for target_card in target_set.cards:
                                        own_destination = next((candidate for candidate in target_card.colors if candidate not in self.player.sets or len(self.player.sets[candidate].cards) < SET_SPECS[candidate].required), None)
                                        target_destination = next((candidate for candidate in own_card.colors if candidate not in opponent.sets or len(opponent.sets[candidate].cards) < SET_SPECS[candidate].required), None)
                                        if own_destination is not None and target_destination is not None:
                                            moves.append(LegalMove("forced_deal", card.name, color=own_color, target_index=target_index, target_color=target_color, target_card_name=target_card.name, destination_color=own_destination, destination_color_2=target_destination, card_name_2=own_card.name, description="Play Forced Deal"))
                if card.action == "sly_deal":
                    for target_index, opponent in enumerate(self.players):
                        if target_index == self.current_player:
                            continue
                        for target_color, property_set in opponent.sets.items():
                            if property_set.is_complete():
                                continue
                            for target_card in property_set.cards:
                                if target_card.is_universal_wild:
                                    continue
                                destination = next((candidate for candidate in target_card.colors if candidate not in self.player.sets or len(self.player.sets[candidate].cards) < SET_SPECS[candidate].required), None)
                                if destination is not None:
                                    moves.append(LegalMove("sly_deal", card.name, target_index=target_index, target_color=target_color, target_card_name=target_card.name, destination_color=destination, description="Play Sly Deal"))
                if card.kind in (CardKind.PROPERTY, CardKind.WILD_PROPERTY):
                    colors = card.colors if card.colors else tuple(SET_SPECS)
                    for color in colors:
                        existing = self.player.sets.get(color)
                        if existing is None or len(existing.cards) < SET_SPECS[color].required:
                            moves.append(LegalMove("property", card.name, color, description=f"Place {card.name} as {color}"))
            if len(self.player.hand) > 7 and self.actions_used == 3:
                for card in self.player.hand:
                    moves.append(LegalMove("discard", card.name, description=f"Discard {card.name}"))
            for card in self.player.hand:
                if card.kind == CardKind.HOUSE:
                    for color, property_set in self.player.sets.items():
                        if property_set.is_complete() and property_set.house is None:
                            moves.append(LegalMove("house", card.name, color, description=f"Build House on {color}"))
                if card.kind == CardKind.HOTEL:
                    for color, property_set in self.player.sets.items():
                        if property_set.is_complete() and property_set.house is not None and property_set.hotel is None:
                            moves.append(LegalMove("hotel", card.name, color, description=f"Build Hotel on {color}"))
        return moves
