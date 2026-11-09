"""Baseline policies for the first tournament experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game_state import GameState, LegalMove
from monopoly_engine import CardKind


class Agent:
    name = "agent"

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        raise NotImplementedError

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        """Whether this player spends a Just Say No on the pending action."""
        return False


@dataclass
class RandomAgent(Agent):
    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        return self.rng.choice(moves)

    def choose_just_say_no(self, state: GameState, action_name: str, target_index: int) -> bool:
        return self.rng.random() < 0.5


class CashFirstAgent(Agent):
    name = "cash_first"

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        player = state.player
        if len(player.hand) > 7:
            forced = [move for move in moves if move.kind == "discard"]
            if forced:
                return min(forced, key=lambda move: next(card.value for card in player.hand if card.name == move.card_name))
        bank_cash = [move for move in moves if move.kind == "bank" and next(card for card in player.hand if card.name == move.card_name).kind in (CardKind.MONEY, CardKind.ACTION)]
        if bank_cash:
            return min(bank_cash, key=lambda move: next(card.value for card in player.hand if card.name == move.card_name))
        properties = [move for move in moves if move.kind == "property"]
        if properties:
            return properties[0]
        return next((move for move in moves if move.kind == "end_turn"), moves[0])


class PropertyFirstAgent(Agent):
    name = "property_first"

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        properties = [move for move in moves if move.kind == "property"]
        if properties:
            return properties[0]
        buildings = [move for move in moves if move.kind in ("house", "hotel")]
        if buildings:
            return buildings[0]
        return next((move for move in moves if move.kind == "end_turn"), moves[0])


class TacticalAgent(CashFirstAgent):
    """Uses available attacks and rent before falling back to cash-first play."""

    name = "tactical"

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        attacks = [move for move in moves if move.kind in ("deal_breaker", "sly_deal")]
        if attacks:
            return attacks[0]
        income = [move for move in moves if move.kind in ("rent", "birthday", "debt_collector")]
        if income:
            return income[0]
        return super().choose_move(state, moves)


class CashReserveAgent(Agent):
    """Evaluation opponent that banks toward a configured cash reserve."""

    def __init__(self, reserve_target: int, name: str | None = None):
        self.reserve_target = reserve_target
        self.name = name or f"reserve_{reserve_target}"
        self.opening_banks: list[int] = []
        self.banked_cards: list[tuple[str, int]] = []
        self._turns_seen = 0

    def choose_move(self, state: GameState, moves: list[LegalMove]) -> LegalMove:
        player = state.player
        if state.actions_used == 0 and self._turns_seen < 5:
            self.opening_banks.append(player.bank_value())
            self._turns_seen += 1

        forced = [move for move in moves if move.kind == "discard"]
        if forced:
            return max(forced, key=lambda move: next(card.value for card in player.hand if card.name == move.card_name))

        wins = [move for move in moves if move.kind == "property" and move.color and _would_complete(player, move)]
        if wins:
            return wins[0]

        attacks = [move for move in moves if move.kind == "deal_breaker"]
        if attacks:
            return attacks[0]
        attacks = [move for move in moves if move.kind in ("forced_deal", "sly_deal")]
        if attacks:
            return attacks[0]
        income = [move for move in moves if move.kind in ("rent", "birthday", "debt_collector")]
        if income:
            return income[0]

        properties = [move for move in moves if move.kind == "property"]
        if properties:
            return max(properties, key=lambda move: len(player.sets[move.color].cards) if move.color in player.sets else 0)

        if len(player.hand) < 7:
            pass_go = [move for move in moves if move.kind == "pass_go"]
            if pass_go:
                return pass_go[0]

        if player.bank_value() < self.reserve_target:
            bankable = [move for move in moves if move.kind == "bank"]
            if bankable:
                cash = [move for move in bankable if next(card.kind for card in player.hand if card.name == move.card_name) == CardKind.MONEY]
                pool = cash or bankable
                chosen = min(pool, key=lambda move: next(card.value for card in player.hand if card.name == move.card_name))
                card = next(card for card in player.hand if card.name == chosen.card_name)
                self.banked_cards.append((card.name, card.value))
                return chosen

        buildings = [move for move in moves if move.kind in ("house", "hotel")]
        if buildings:
            return buildings[0]
        return next((move for move in moves if move.kind == "end_turn"), moves[0])


def _would_complete(player, move: LegalMove) -> bool:
    from monopoly_engine import SET_SPECS
    current = player.sets.get(move.color)
    return current is not None and len(current.cards) + 1 >= SET_SPECS[move.color].required
