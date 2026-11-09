"""State-changing effects for property-stealing action cards."""

from __future__ import annotations

from dataclasses import dataclass

from game_state import GameState
from monopoly_engine import Card, CardKind, PropertySet, SET_SPECS, can_take_with_sly_or_forced


RENT_COLORS = {
    "light_green_black_rent": ("light_green", "black"),
    "orange_pink_rent": ("orange", "pink"),
    "dark_green_dark_blue_rent": ("dark_green", "dark_blue"),
    "red_yellow_rent": ("red", "yellow"),
    "light_blue_brown_rent": ("light_blue", "brown"),
}


@dataclass(frozen=True)
class Debt:
    payer: int
    payee: int
    amount: int
    source_action: str


def _property_card_owner(player, card_name: str) -> tuple[PropertySet, Card] | None:
    for property_set in player.sets.values():
        for card in property_set.cards:
            if card.name == card_name:
                return property_set, card
    return None


def legal_payment_cards(state: GameState, payer_index: int) -> list[Card]:
    """Cards that can be offered as payment.

    Banked cards and properties are legal. Houses/hotels attached to a set
    are not separate payment cards; they remain attached to that set.
    """
    payer = state.players[payer_index]
    cards = list(payer.bank)
    cards.extend(payer.all_property_cards())
    return cards


def pay_debt(state: GameState, debt: Debt, card_names: list[str], destination_colors: dict[str, str] | None = None) -> list[Card]:
    """Transfer selected whole cards from payer to payee.

    The selected cards must cover the debt with no change. If a property is
    removed from a set, any fixed House/Hotel stays attached to that set.
    """
    destination_colors = destination_colors or {}
    payer = state.players[debt.payer]
    payee = state.players[debt.payee]
    if len(card_names) != len(set(card_names)):
        raise ValueError("A payment cannot use the same card twice")

    selected: list[tuple[Card, PropertySet | None]] = []
    for card_name in card_names:
        bank_card = next((card for card in payer.bank if card.name == card_name), None)
        if bank_card is not None:
            selected.append((bank_card, None))
            continue
        owned = _property_card_owner(payer, card_name)
        if owned is None:
            raise ValueError(f"Card is not available for payment: {card_name}")
        source_set, property_card = owned
        selected.append((property_card, source_set))

    if sum(card.value for card, _ in selected) < debt.amount:
        raise ValueError("Selected payment does not cover the debt")

    transferred: list[Card] = []
    for card, source_set in selected:
        if source_set is None:
            payer.bank.remove(card)
            payee.bank.append(card)
        else:
            source_set.cards.remove(card)
            if not source_set.cards and source_set.house is None and source_set.hotel is None:
                del payer.sets[source_set.color]
            color = _destination_color(card, destination_colors.get(card.name))
            _add_to_property_area(payee, card, color)
        transferred.append(card)
    state.check_winner()
    return transferred


def prepare_rent(
    state: GameState,
    rent_card_name: str,
    target_color: str,
    target_index: int | None = None,
    double_rent_card_name: str | None = None,
) -> list[Debt]:
    """Play a rent card and return debts for affected opponents.

    The actual payment is a separate decision, allowing an agent to choose
    cash, properties, or a mixture and to model Just Say No responses.
    """
    attacker = state.player
    rent_card = next((card for card in attacker.hand if card.name == rent_card_name), None)
    if rent_card is None or rent_card.action not in (*RENT_COLORS, "any_color_rent"):
        raise ValueError("The selected card is not a rent card")
    if rent_card.action != "any_color_rent" and target_color not in RENT_COLORS[rent_card.action]:
        raise ValueError("The rent card cannot target that color")
    owned_set = attacker.sets.get(target_color)
    if owned_set is None or not owned_set.is_complete():
        raise ValueError("Rent requires the attacker to have a complete matching set")
    required_actions = 2 if double_rent_card_name is not None else 1
    if state.actions_used + required_actions > 3:
        raise RuntimeError("Not enough actions remain for this rent play")
    if double_rent_card_name is not None:
        double_card = next((card for card in attacker.hand if card.name == double_rent_card_name), None)
        if double_card is None or double_card.action != "double_rent":
            raise ValueError("The second card must be Double the Rent")
        state.play_action_to_discard(double_rent_card_name, "double_rent")
    state.play_action_to_discard(rent_card_name, rent_card.action)
    amount = owned_set.rent() * (2 if double_rent_card_name else 1)
    if rent_card.action == "any_color_rent":
        if target_index is None or target_index == state.current_player:
            raise ValueError("Any-color rent requires one opponent target")
        targets = [target_index]
    else:
        targets = [index for index in range(len(state.players)) if index != state.current_player]
    return [Debt(index, state.current_player, amount, rent_card.action) for index in targets]


def prepare_birthday(state: GameState, card_name: str) -> list[Debt]:
    card = next((card for card in state.player.hand if card.name == card_name), None)
    if card is None or card.action != "birthday":
        raise ValueError("The selected card is not Birthday")
    state.play_action_to_discard(card_name, "birthday")
    return [Debt(index, state.current_player, 2, "birthday") for index in range(len(state.players)) if index != state.current_player]


def prepare_debt_collector(state: GameState, card_name: str, target_index: int) -> Debt:
    if target_index == state.current_player:
        raise ValueError("Debt Collector requires another player")
    card = next((card for card in state.player.hand if card.name == card_name), None)
    if card is None or card.action != "debt_collector":
        raise ValueError("The selected card is not Debt Collector")
    state.play_action_to_discard(card_name, "debt_collector")
    return Debt(target_index, state.current_player, 5, "debt_collector")


def _card_in_set(property_set: PropertySet, card_name: str) -> Card:
    for card in property_set.cards:
        if card.name == card_name:
            return card
    raise ValueError(f"Card is not in the selected property set: {card_name}")


def _remove_card(property_set: PropertySet, card_name: str) -> Card:
    for index, card in enumerate(property_set.cards):
        if card.name == card_name:
            return property_set.cards.pop(index)
    raise ValueError(f"Card is not in the selected property set: {card_name}")


def _destination_color(card: Card, requested: str | None) -> str:
    if card.is_universal_wild:
        if requested not in SET_SPECS:
            raise ValueError("A universal wild requires a destination color")
        return requested
    if requested is None:
        if len(card.colors) != 1:
            raise ValueError("A double-color wild requires a destination color")
        return card.colors[0]
    if requested not in card.colors:
        raise ValueError("The selected card cannot represent that color")
    return requested


def _add_to_property_area(player, card: Card, color: str) -> None:
    destination = player.sets.setdefault(color, PropertySet(color))
    if len(destination.cards) >= SET_SPECS[color].required:
        raise ValueError("The destination property set is already full")
    if not destination.can_accept(card):
        raise ValueError("The card cannot be placed in the destination set")
    destination.cards.append(card)


def play_sly_deal(
    state: GameState,
    card_name: str,
    target_index: int,
    target_color: str,
    target_card_name: str,
    destination_color: str | None = None,
) -> Card:
    """Take one card from an opponent's incomplete set."""
    attacker = state.player
    if target_index == state.current_player:
        raise ValueError("Sly Deal requires another player")
    target = state.players[target_index]
    source = target.sets.get(target_color)
    if source is None:
        raise ValueError("Target property set does not exist")
    card = _card_in_set(source, target_card_name)
    if not can_take_with_sly_or_forced(source, card):
        raise ValueError("That card is protected from Sly Deal")
    color = _destination_color(card, destination_color)
    _add_to_property_area(attacker, card, color)
    _remove_card(source, target_card_name)
    if not source.cards and source.house is None and source.hotel is None:
        del target.sets[target_color]
    return state.play_action_to_discard(card_name, "sly_deal")


def play_forced_deal(
    state: GameState,
    card_name: str,
    target_index: int,
    own_color: str,
    own_card_name: str,
    target_color: str,
    target_card_name: str,
    own_destination_color: str | None = None,
    target_destination_color: str | None = None,
) -> Card:
    """Exchange one card from each player's incomplete property set."""
    if target_index == state.current_player:
        raise ValueError("Forced Deal requires another player")
    attacker = state.player
    target = state.players[target_index]
    own_set = attacker.sets.get(own_color)
    target_set = target.sets.get(target_color)
    if own_set is None or target_set is None:
        raise ValueError("Both selected property sets must exist")
    own_card = _card_in_set(own_set, own_card_name)
    target_card = _card_in_set(target_set, target_card_name)
    if not can_take_with_sly_or_forced(own_set, own_card) or not can_take_with_sly_or_forced(target_set, target_card):
        raise ValueError("Forced Deal cannot use a card from a complete set")
    own_destination = _destination_color(target_card, target_destination_color)
    target_destination = _destination_color(own_card, own_destination_color)
    _add_to_property_area(attacker, target_card, own_destination)
    _add_to_property_area(target, own_card, target_destination)
    _remove_card(own_set, own_card_name)
    _remove_card(target_set, target_card_name)
    if not own_set.cards and own_set.house is None and own_set.hotel is None:
        del attacker.sets[own_color]
    if not target_set.cards and target_set.house is None and target_set.hotel is None:
        del target.sets[target_color]
    return state.play_action_to_discard(card_name, "forced_deal")


def play_deal_breaker(state: GameState, card_name: str, target_index: int, target_color: str) -> Card:
    """Take an entire completed set, including its fixed buildings."""
    if target_index == state.current_player:
        raise ValueError("Deal Breaker requires another player")
    attacker = state.player
    target = state.players[target_index]
    stolen = target.sets.get(target_color)
    if stolen is None or not stolen.is_complete():
        raise ValueError("Deal Breaker requires a complete target set")
    if target_color in attacker.sets:
        raise ValueError("The attacker already has a property area for this color")
    attacker.sets[target_color] = target.sets.pop(target_color)
    if attacker.completed_sets() >= 3:
        state.winner = state.current_player
    return state.play_action_to_discard(card_name, "deal_breaker")
