"""Core rules model for the user's three-player Monopoly card variant.

This first milestone deliberately focuses on deterministic rules and card data.
Search and self-play can be layered on top of these pure functions later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class CardKind(str, Enum):
    PROPERTY = "property"
    WILD_PROPERTY = "wild_property"
    MONEY = "money"
    ACTION = "action"
    HOUSE = "house"
    HOTEL = "hotel"


@dataclass(frozen=True)
class SetSpec:
    color: str
    required: int
    value: int
    rents: tuple[int, ...]


SET_SPECS: dict[str, SetSpec] = {
    "brown": SetSpec("brown", 2, 1, (1, 2)),
    "light_blue": SetSpec("light_blue", 3, 1, (1, 2, 3)),
    "pink": SetSpec("pink", 3, 2, (1, 2, 4)),
    "orange": SetSpec("orange", 3, 2, (1, 3, 5)),
    "red": SetSpec("red", 3, 3, (2, 3, 6)),
    "yellow": SetSpec("yellow", 3, 3, (2, 4, 6)),
    "dark_green": SetSpec("dark_green", 3, 4, (2, 4, 7)),
    "light_green": SetSpec("light_green", 2, 2, (1, 2)),
    "dark_blue": SetSpec("dark_blue", 2, 4, (3, 8)),
    "black": SetSpec("black", 4, 2, (1, 2, 3, 4)),
}


DOUBLE_WILDS: dict[str, tuple[str, str]] = {
    "wild_red_yellow": ("red", "yellow"),
    "wild_dark_green_black": ("dark_green", "black"),
    "wild_light_green_black": ("light_green", "black"),
    "wild_light_blue_brown": ("light_blue", "brown"),
    "wild_orange_pink": ("orange", "pink"),
    "wild_dark_green_dark_blue": ("dark_green", "dark_blue"),
    "wild_light_blue_black": ("light_blue", "black"),
}


@dataclass(frozen=True)
class Card:
    name: str
    kind: CardKind
    value: int = 0
    colors: tuple[str, ...] = ()
    action: str | None = None

    @property
    def is_universal_wild(self) -> bool:
        return self.kind == CardKind.WILD_PROPERTY and not self.colors


ACTION_COUNTS: dict[str, tuple[int, int]] = {
    "sly_deal": (3, 3),
    "forced_deal": (3, 3),
    "deal_breaker": (2, 5),
    "just_say_no": (3, 4),
    "pass_go": (10, 1),
    "debt_collector": (3, 3),
    "any_color_rent": (3, 3),
    "birthday": (3, 2),
    "double_rent": (2, 1),
    "light_green_black_rent": (2, 1),
    "orange_pink_rent": (2, 1),
    "dark_green_dark_blue_rent": (2, 1),
    "red_yellow_rent": (2, 1),
    "light_blue_brown_rent": (1, 1),
}


def build_deck() -> list[Card]:
    """Build the exact card inventory supplied so far (105 cards)."""
    deck: list[Card] = []

    for spec in SET_SPECS.values():
        for index in range(spec.required):
            deck.append(Card(f"{spec.color}_{index + 1}", CardKind.PROPERTY, spec.value, (spec.color,)))

    for name, colors in DOUBLE_WILDS.items():
        count = {"wild_red_yellow": 2, "wild_orange_pink": 2}.get(name, 1)
        for index in range(count):
            deck.append(Card(f"{name}_{index + 1}", CardKind.WILD_PROPERTY, 1, colors))

    for index in range(2):
        deck.append(Card(f"universal_wild_{index + 1}", CardKind.WILD_PROPERTY, 0))

    for value, count in ((5, 2), (10, 1), (4, 3), (2, 5), (3, 3), (1, 6)):
        for index in range(count):
            deck.append(Card(f"money_{value}_{index + 1}", CardKind.MONEY, value))

    for action, (count, value) in ACTION_COUNTS.items():
        for index in range(count):
            deck.append(Card(f"{action}_{index + 1}", CardKind.ACTION, value, action=action))

    for kind, count, value in ((CardKind.HOUSE, 3, 3), (CardKind.HOTEL, 2, 4)):
        for index in range(count):
            deck.append(Card(f"{kind.value}_{index + 1}", kind, value))

    return deck


@dataclass
class PropertySet:
    color: str
    cards: list[Card] = field(default_factory=list)
    house: Card | None = None
    hotel: Card | None = None

    def can_accept(self, card: Card) -> bool:
        return card.kind in (CardKind.PROPERTY, CardKind.WILD_PROPERTY) and (
            self.color in card.colors or card.is_universal_wild
        )

    def is_complete(self) -> bool:
        return len(self.cards) >= SET_SPECS[self.color].required and any(
            self.color in card.colors for card in self.cards
        )

    def rent(self) -> int:
        if not self.is_complete():
            return 0
        spec = SET_SPECS[self.color]
        level = min(len(self.cards), len(spec.rents))
        return spec.rents[level - 1] + (self.house.value if self.house else 0) + (self.hotel.value if self.hotel else 0)


def complete_set_count(sets: Iterable[PropertySet]) -> int:
    return sum(property_set.is_complete() for property_set in sets)


def can_take_with_sly_or_forced(property_set: PropertySet, card: Card) -> bool:
    """User rules: complete sets are protected; universal wilds are always protected."""
    return not property_set.is_complete() and not card.is_universal_wild


def deal_breaker_cards(property_set: PropertySet) -> list[Card]:
    if not property_set.is_complete():
        raise ValueError("Deal Breaker can only take a complete property set")
    return [*property_set.cards, *([property_set.house] if property_set.house else []), *([property_set.hotel] if property_set.hotel else [])]


def payment_subsets(cards: Sequence[Card], amount: int) -> list[tuple[Card, ...]]:
    """Return legal whole-card payments with value at least the amount.

    Property cards are deliberately accepted here because the user's version
    permits cash/property combinations and voluntary property payment.
    """
    results: list[tuple[Card, ...]] = []
    for mask in range(1, 1 << len(cards)):
        chosen = tuple(card for index, card in enumerate(cards) if mask & (1 << index))
        if sum(card.value for card in chosen) >= amount:
            results.append(chosen)
    return results


def rent_targets(action: str, owner_sets: Sequence[PropertySet], player_count: int = 3) -> int:
    """Number of opponents charged by a rent action."""
    if action == "any_color_rent":
        return 1
    return player_count - 1


def validate_configuration() -> list[str]:
    errors: list[str] = []
    for color, spec in SET_SPECS.items():
        if len(spec.rents) != spec.required:
            errors.append(f"{color}: rent levels must equal required property count")
    for name, colors in DOUBLE_WILDS.items():
        if len(colors) != 2:
            errors.append(f"{name}: double wild must have exactly two colors")
    return errors


if __name__ == "__main__":
    deck = build_deck()
    print(f"cards={len(deck)}")
    print(f"configuration_errors={validate_configuration()}")
