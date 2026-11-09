"""Pure action-card rules used by agents and the future game resolver."""

from __future__ import annotations

from dataclasses import dataclass, field


RENT_ACTIONS = {
    "any_color_rent",
    "light_green_black_rent",
    "orange_pink_rent",
    "dark_green_dark_blue_rent",
    "red_yellow_rent",
    "light_blue_brown_rent",
}


def action_cost(action: str, paired_with: str | None = None) -> int:
    """Return the number of turn actions consumed by a play."""
    if action == "double_rent":
        if paired_with not in RENT_ACTIONS:
            raise ValueError("Double the Rent must be played with a rent card")
        return 2
    return 1


def rent_targets(action: str, player_count: int = 3) -> int:
    if action not in RENT_ACTIONS:
        raise ValueError(f"Not a rent action: {action}")
    return 1 if action == "any_color_rent" else player_count - 1


def just_say_no_action_cost(responder_index: int, active_player_index: int) -> int:
    """Cost of a Just Say No response in the user's rules.

    The defending player responds for free. If the active player counters the
    response, that counter consumes one of the active player's actions.
    """
    return 1 if responder_index == active_player_index else 0


@dataclass(frozen=True)
class RentRequest:
    attacker: int
    rent_action: str
    amount_by_target: dict[int, int]
    double_rent: bool = False

    def targets(self) -> tuple[int, ...]:
        return tuple(self.amount_by_target)


@dataclass
class JustSayNoChain:
    """Finite response chain for one targeted action.

    The defending player's response is free. A counter played by the active
    player consumes one active-turn action. With three copies in the deck, no
    real chain can exceed three responses.
    """

    original_action: str
    responses: list[int] = field(default_factory=list)
    max_cards: int = 3

    def add_response(self, player_index: int) -> None:
        if self.is_resolved:
            raise RuntimeError("The Just Say No chain is already resolved")
        if len(self.responses) >= self.max_cards:
            raise RuntimeError("No Just Say No cards remain in this deck")
        self.responses.append(player_index)

    @property
    def is_resolved(self) -> bool:
        return len(self.responses) >= self.max_cards

    def action_succeeds(self, chain_ended: bool = True) -> bool:
        if not chain_ended:
            raise ValueError("Specify whether the response chain has ended")
        return len(self.responses) % 2 == 0
