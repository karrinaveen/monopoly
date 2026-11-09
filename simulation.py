"""Small tournament runner for baseline agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agents import Agent
from action_effects import pay_debt, play_deal_breaker, play_forced_deal, play_sly_deal, prepare_birthday, prepare_debt_collector, prepare_rent, legal_payment_cards
from game_state import GameState, LegalMove
from monopoly_engine import Card, CardKind, SET_SPECS


@dataclass(frozen=True)
class GameResult:
    winner: int | None
    turns: int
    action_count: int
    draw: bool = False
    diagnostics: dict | None = None


def _game_diagnostics(state: GameState) -> dict:
    return {
        "draw_pile": len(state.draw_pile),
        "discard_pile": len(state.discard_pile),
        "players": [
            {
                "name": player.name,
                "hand": len(player.hand),
                "hand_kinds": {kind.value: sum(card.kind == kind for card in player.hand) for kind in CardKind},
                "bank_value": player.bank_value(),
                "bank_actions": sum(card.kind == CardKind.ACTION for card in player.bank),
                "properties": len(player.all_property_cards()),
                "complete_sets": player.completed_sets(),
                "incomplete_sets": sum(bool(prop.cards) and not prop.is_complete() for prop in player.sets.values()),
            }
            for player in state.players
        ],
    }


def _has_just_say_no(state: GameState, player_index: int) -> bool:
    return any(card.action == "just_say_no" for card in state.players[player_index].hand)


def _spend_just_say_no(state: GameState, player_index: int) -> None:
    player = state.players[player_index]
    card_index = next(index for index, card in enumerate(player.hand) if card.action == "just_say_no")
    state.discard_pile.append(player.hand.pop(card_index))


def _action_survives(state: GameState, action_name: str, target_index: int, agents, original_cost: int = 1) -> bool:
    """Resolve one target's Just Say No chain before the action is played."""
    if agents is None or not _has_just_say_no(state, target_index):
        return True
    active = state.current_player
    responder = target_index
    responses = 0
    while responses < 3:
        if not _has_just_say_no(state, responder):
            break
        if responder == active:
            # The original action still needs its action budget.
            if state.actions_used + original_cost + 1 > 3:
                break
        if not agents[responder].choose_just_say_no(state, action_name, target_index):
            break
        _spend_just_say_no(state, responder)
        responses += 1
        if responder == active:
            state.actions_used += 1
            state.property_locked = True
        responder = active if responder == target_index else target_index
    return responses % 2 == 0


def apply_move(state: GameState, move: LegalMove, agents=None) -> None:
    if move.kind == "end_turn":
        state.end_turn()
    elif move.kind == "bank":
        state.play_to_bank(move.card_name)
    elif move.kind == "property":
        state.play_property(move.card_name, move.color)
    elif move.kind == "pass_go":
        state.play_pass_go(move.card_name)
    elif move.kind == "house":
        state.build_house(move.card_name, move.color)
    elif move.kind == "hotel":
        state.build_hotel(move.card_name, move.color)
    elif move.kind == "discard":
        state.discard_from_hand(move.card_name)
    elif move.kind == "birthday":
        affected = [index for index in range(len(state.players)) if index != state.current_player]
        survivors = {index: _action_survives(state, "birthday", index, agents) for index in affected}
        for debt in prepare_birthday(state, move.card_name):
            if survivors[debt.payer]:
                _auto_pay(state, debt)
    elif move.kind == "debt_collector":
        if _action_survives(state, "debt_collector", move.target_index, agents):
            _auto_pay(state, prepare_debt_collector(state, move.card_name, move.target_index))
        else:
            state.play_action_to_discard(move.card_name, "debt_collector")
    elif move.kind == "rent":
        if move.target_index is None:
            affected = [index for index in range(len(state.players)) if index != state.current_player]
        else:
            affected = [move.target_index]
        original_cost = 2 if move.card_name_2 else 1
        survivors = {index: _action_survives(state, "rent", index, agents, original_cost) for index in affected}
        for debt in prepare_rent(state, move.card_name, move.color, move.target_index, move.card_name_2):
            if survivors[debt.payer]:
                _auto_pay(state, debt)
    elif move.kind == "deal_breaker":
        if _action_survives(state, "deal_breaker", move.target_index, agents):
            play_deal_breaker(state, move.card_name, move.target_index, move.target_color)
        else:
            state.play_action_to_discard(move.card_name, "deal_breaker")
    elif move.kind == "sly_deal":
        if _action_survives(state, "sly_deal", move.target_index, agents):
            play_sly_deal(state, move.card_name, move.target_index, move.target_color, move.target_card_name, move.destination_color)
        else:
            state.play_action_to_discard(move.card_name, "sly_deal")
    elif move.kind == "forced_deal":
        if _action_survives(state, "forced_deal", move.target_index, agents):
            play_forced_deal(state, move.card_name, move.target_index, move.color, move.card_name_2, move.target_color, move.target_card_name, move.destination_color_2, move.destination_color)
        else:
            state.play_action_to_discard(move.card_name, "forced_deal")
    else:
        raise ValueError(f"Unsupported baseline move: {move.kind}")
    state.check_winner()


def _select_payment_cards(cards: list[Card], amount: int) -> list[Card] | None:
    """Choose whole-card payment with least overpayment, then fewest cards."""
    # Dynamic programming avoids enumerating every subset. For each subtotal,
    # keep the smallest-card-count combination; stable input order breaks ties.
    best_for_total: dict[int, tuple[Card, ...]] = {0: ()}
    for card in cards:
        for subtotal, chosen in list(best_for_total.items()):
            new_total = subtotal + card.value
            candidate = (*chosen, card)
            current = best_for_total.get(new_total)
            if current is None or len(candidate) < len(current):
                best_for_total[new_total] = candidate
    payable = [total for total in best_for_total if total >= amount]
    if not payable:
        return None
    total = min(payable, key=lambda value: (value - amount, len(best_for_total[value])))
    return list(best_for_total[total])


def _auto_pay(state: GameState, debt) -> None:
    cards = legal_payment_cards(state, debt.payer)
    chosen = _select_payment_cards(cards, debt.amount)
    if chosen is None:
        return
    destination_colors = {}
    payee = state.players[debt.payee]
    reserved: dict[str, int] = {}
    for card in chosen:
        if card.kind in (CardKind.PROPERTY, CardKind.WILD_PROPERTY):
            candidates = card.colors if card.colors else tuple(SET_SPECS)
            destination = next(
                (
                    color for color in candidates
                    if (len(payee.sets[color].cards) if color in payee.sets else 0) + reserved.get(color, 0) < SET_SPECS[color].required
                ),
                None,
            )
            if destination is None:
                return
            reserved[destination] = reserved.get(destination, 0) + 1
            if len(candidates) != 1 or card.is_universal_wild:
                destination_colors[card.name] = destination
    pay_debt(state, debt, [card.name for card in chosen], destination_colors)


def run_game(agents: list[Agent], seed: int | None = None, max_turns: int = 500, validator=None) -> GameResult:
    if len(agents) != 3:
        raise ValueError("The baseline runner requires three agents")
    for index, agent in enumerate(agents):
        agent.player_index = index
    state = GameState.new([agent.name for agent in agents], seed=seed)
    if validator:
        validator(state)
    action_count = 0
    action_flow: Counter = Counter()
    for turn_number in range(1, max_turns + 1):
        if state.winner is not None or state.draw:
            break
        state.begin_turn()
        if state.winner is None:
            state.check_exhaustion_draw()
        if state.draw:
            break
        if validator:
            validator(state)
        while state.turn_active and state.winner is None:
            moves = state.legal_moves()
            if not moves:
                state.end_turn()
                break
            move = agents[state.current_player].choose_move(state, moves)
            state.last_move = move
            if move.kind == "bank" and any(card.name == move.card_name and card.kind == CardKind.ACTION for card in state.player.hand):
                action_flow["banked_action"] += 1
            elif move.kind in {"birthday", "debt_collector", "rent", "deal_breaker", "sly_deal", "forced_deal", "pass_go"}:
                action_flow[move.kind] += 1
            apply_move(state, move, agents)
            if validator:
                validator(state)
            action_count += move.kind not in ("end_turn", "discard")
            if move.kind == "end_turn":
                break
        if state.winner is not None:
            break
    diagnostics = _game_diagnostics(state)
    diagnostics["action_flow"] = dict(action_flow)
    return GameResult(state.winner, turn_number, action_count, state.draw, diagnostics)


def tournament(agent_factories: list, games: int = 100, seed: int = 1) -> Counter:
    wins: Counter = Counter()
    for game_index in range(games):
        agents = [factory(game_index * 3 + index) for index, factory in enumerate(agent_factories)]
        result = run_game(agents, seed=seed + game_index)
        if result.winner is not None:
            wins[agents[result.winner].name] += 1
    return wins


if __name__ == "__main__":
    from agents import CashFirstAgent, PropertyFirstAgent, RandomAgent

    wins = tournament([
        lambda seed: CashFirstAgent(),
        lambda seed: PropertyFirstAgent(),
        lambda seed: RandomAgent(seed),
    ], games=100)
    print(dict(wins))
