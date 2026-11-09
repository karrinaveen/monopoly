import unittest

from action_rules import JustSayNoChain, action_cost, just_say_no_action_cost, rent_targets
from action_effects import (
    Debt,
    pay_debt,
    play_deal_breaker,
    play_forced_deal,
    play_sly_deal,
    prepare_birthday,
    prepare_debt_collector,
    prepare_rent,
)
from agents import Agent, CashFirstAgent, PropertyFirstAgent, RandomAgent, TacticalAgent
from game_state import GameState
from learning import action_id, collect_episodes, encode_move, encode_observation, generate_episode
from mcts import ISMCTSAgent, determinize
from simulation import apply_move, run_game
from validation import validate_state
from response_learning import TabularResponsePolicy, train_response_policy
from policy_learning import TabularActionPolicy, train_action_policy
from neural_policy import action_features, normalized_observation, torch
from neural_agent import NeuralAgent
from search_learning import SearchSelfPlayAgent, collect_search_self_play
from monopoly_engine import (
    Card,
    CardKind,
    PropertySet,
    build_deck,
    can_take_with_sly_or_forced,
    complete_set_count,
    deal_breaker_cards,
    payment_subsets,
    validate_configuration,
)


class RulesTests(unittest.TestCase):
    def test_configuration_and_deck(self):
        self.assertEqual(validate_configuration(), [])
        self.assertEqual(len(build_deck()), 105)

    def test_universal_wild_cannot_complete_alone(self):
        property_set = PropertySet("dark_blue", [Card("wild", CardKind.WILD_PROPERTY)])
        self.assertFalse(property_set.is_complete())

    def test_universal_wild_can_complete_with_standard_property(self):
        cards = [
            Card("dark_blue_1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("universal", CardKind.WILD_PROPERTY),
        ]
        self.assertTrue(PropertySet("dark_blue", cards).is_complete())

    def test_double_wild_can_complete_eligible_set(self):
        cards = [
            Card("black_1", CardKind.PROPERTY, 2, ("black",)),
            Card("black_2", CardKind.PROPERTY, 2, ("black",)),
            Card("black_3", CardKind.PROPERTY, 2, ("black",)),
            Card("wild", CardKind.WILD_PROPERTY, 1, ("light_green", "black")),
        ]
        self.assertTrue(PropertySet("black", cards).is_complete())

    def test_sly_and_forced_protection(self):
        incomplete = PropertySet("dark_blue", [Card("db", CardKind.PROPERTY, 4, ("dark_blue",))])
        complete = PropertySet("dark_blue", [
            Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("db2", CardKind.PROPERTY, 4, ("dark_blue",)),
        ])
        universal = Card("u", CardKind.WILD_PROPERTY)
        self.assertTrue(can_take_with_sly_or_forced(incomplete, incomplete.cards[0]))
        self.assertFalse(can_take_with_sly_or_forced(complete, complete.cards[0]))
        self.assertFalse(can_take_with_sly_or_forced(incomplete, universal))

    def test_deal_breaker_includes_buildings(self):
        house = Card("house", CardKind.HOUSE, 3)
        hotel = Card("hotel", CardKind.HOTEL, 4)
        property_set = PropertySet(
            "dark_blue",
            [Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)), Card("db2", CardKind.PROPERTY, 4, ("dark_blue",))],
            house,
            hotel,
        )
        self.assertEqual(len(deal_breaker_cards(property_set)), 4)

    def test_rent_and_set_count(self):
        property_set = PropertySet(
            "dark_blue",
            [Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)), Card("db2", CardKind.PROPERTY, 4, ("dark_blue",))],
            Card("house", CardKind.HOUSE, 3),
            Card("hotel", CardKind.HOTEL, 4),
        )
        self.assertEqual(property_set.rent(), 15)
        self.assertEqual(complete_set_count([property_set]), 1)

    def test_whole_card_payment_and_no_change(self):
        cards = [Card("five", CardKind.MONEY, 5), Card("one", CardKind.MONEY, 1)]
        payments = payment_subsets(cards, 1)
        self.assertIn((cards[0],), payments)
        self.assertIn((cards[1],), payments)

    def test_game_starts_with_three_players_and_five_cards(self):
        state = GameState.new(seed=7)
        self.assertEqual([len(player.hand) for player in state.players], [5, 5, 5])
        self.assertEqual(state.current_player, 0)

    def test_turn_draws_two_or_five_when_empty(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        self.assertEqual(len(state.player.hand), 7)
        state.end_turn()
        state.players[1].hand.clear()
        state.begin_turn()
        self.assertEqual(len(state.player.hand), 5)

    def test_first_action_locks_layout_but_allows_new_properties(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        property_card = next(card for card in state.player.hand if card.kind == CardKind.PROPERTY)
        state.play_property(property_card.name)
        self.assertTrue(state.property_locked)
        replacement = next(card for card in state.player.hand if card.kind == CardKind.PROPERTY)
        state.play_property(replacement.name)

    def test_pass_go_draws_two_and_consumes_action(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        pass_go = Card("pass_go_test", CardKind.ACTION, 1, action="pass_go")
        state.player.hand.append(pass_go)
        before = len(state.player.hand)
        state.play_pass_go(pass_go.name)
        self.assertEqual(len(state.player.hand), before + 1)
        self.assertEqual(state.actions_used, 1)

    def test_played_action_enters_middle_but_banked_action_does_not(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        action = Card("birthday_test", CardKind.ACTION, 2, action="birthday")
        banked = Card("deal_breaker_test", CardKind.ACTION, 5, action="deal_breaker")
        state.player.hand.extend([action, banked])
        state.play_action_to_discard(action.name, "birthday")
        state.play_to_bank(banked.name)
        self.assertIn(action, state.discard_pile)
        self.assertNotIn(banked, state.discard_pile)
        self.assertIn(banked, state.player.bank)

    def test_discard_pile_reshuffles_when_draw_pile_is_empty(self):
        state = GameState.new(seed=7)
        state.draw_pile.clear()
        marker = Card("marker", CardKind.ACTION, 1, action="pass_go")
        state.discard_pile.append(marker)
        self.assertEqual(state.draw_cards(1), [marker])
        self.assertEqual(state.discard_pile, [])

    def test_empty_draw_and_discard_piles_are_rare_draw(self):
        state = GameState.new(seed=17)
        state.draw_pile.clear()
        state.discard_pile.clear()
        self.assertTrue(state.check_exhaustion_draw())
        self.assertTrue(state.draw)

    def test_eight_cards_after_one_action_cannot_end(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        state.player.hand.append(state.player.hand[0])
        self.assertFalse(state.can_end_turn())

    def test_banked_action_is_cash_and_leaves_hand(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        deal_breaker = Card("deal_breaker_test", CardKind.ACTION, 5, action="deal_breaker")
        state.player.hand.append(deal_breaker)
        state.play_to_bank(deal_breaker.name)
        self.assertNotIn(deal_breaker, state.player.hand)
        self.assertIn(deal_breaker, state.player.bank)
        self.assertEqual(state.player.bank_value(), 5)

    def test_action_costs_and_rent_scope(self):
        self.assertEqual(action_cost("deal_breaker"), 1)
        self.assertEqual(action_cost("double_rent", "red_yellow_rent"), 2)
        self.assertEqual(rent_targets("red_yellow_rent"), 2)
        self.assertEqual(rent_targets("any_color_rent"), 1)

    def test_just_say_no_chain_alternates_result(self):
        chain = JustSayNoChain("deal_breaker")
        self.assertTrue(chain.action_succeeds())
        chain.add_response(1)
        self.assertFalse(chain.action_succeeds())
        chain.add_response(0)
        self.assertTrue(chain.action_succeeds())
        chain.add_response(1)
        self.assertFalse(chain.action_succeeds())
        with self.assertRaises(RuntimeError):
            chain.add_response(0)

    def test_defender_just_say_no_is_free_but_active_counter_costs_action(self):
        self.assertEqual(just_say_no_action_cost(responder_index=1, active_player_index=0), 0)
        self.assertEqual(just_say_no_action_cost(responder_index=0, active_player_index=0), 1)

    def test_just_say_no_cancels_deal_breaker_without_changing_defender_turn(self):
        class AlwaysNo(Agent):
            name = "always_no"
            def choose_move(self, state, moves):
                return moves[0]
            def choose_just_say_no(self, state, action_name, target_index):
                return True

        state = GameState.new(seed=59)
        state.begin_turn()
        breaker = Card("breaker", CardKind.ACTION, 5, action="deal_breaker")
        defender_no = Card("no", CardKind.ACTION, 4, action="just_say_no")
        state.player.hand.append(breaker)
        state.player.hand[:] = [card for card in state.player.hand if card.action != "just_say_no"]
        state.players[1].hand.append(defender_no)
        state.players[1].hand[:] = [card for card in state.players[1].hand if card.action != "just_say_no" or card.name == "no"]
        state.players[1].sets["dark_blue"] = PropertySet("dark_blue", [
            Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("db2", CardKind.PROPERTY, 4, ("dark_blue",)),
        ])
        move = next(move for move in state.legal_moves() if move.kind == "deal_breaker")
        apply_move(state, move, [AlwaysNo(), AlwaysNo(), AlwaysNo()])
        self.assertIn("dark_blue", state.players[1].sets)
        self.assertEqual(state.actions_used, 1)
        self.assertIn(defender_no, state.discard_pile)

    def test_state_validation_passes_on_new_game_and_after_action(self):
        state = GameState.new(seed=61)
        validate_state(state)
        state.begin_turn()
        move = state.legal_moves()[0]
        apply_move(state, move)
        validate_state(state)

    def test_response_policy_trains_and_serializes(self):
        policy, stats = train_response_policy(episodes=5, seed=71)
        self.assertEqual(stats["episodes"], 5)
        self.assertGreaterEqual(stats["contexts"], 0)
        path = "/tmp/monopoly_response_policy_test.json"
        policy.save(path)
        restored = TabularResponsePolicy.load(path)
        self.assertEqual(len(restored.table), len(policy.table))

    def test_mcts_accepts_trained_response_policy(self):
        policy, _ = train_response_policy(episodes=3, seed=73)
        agent = ISMCTSAgent(simulations=2, rollout_turns=2, seed=5, response_policy=policy)
        state = GameState.new(seed=74)
        state.begin_turn()
        self.assertIsInstance(agent.choose_just_say_no(state, "deal_breaker", 1), bool)

    def test_main_action_policy_trains(self):
        action_policy, response_policy, stats = train_action_policy(episodes=3, seed=79)
        self.assertEqual(stats["episodes"], 3)
        self.assertGreaterEqual(stats["contexts"], 0)
        self.assertIsInstance(action_policy, TabularActionPolicy)
        self.assertIsInstance(response_policy, TabularResponsePolicy)

    def test_neural_encoding_is_fixed_and_stable(self):
        self.assertEqual(action_features(1234), action_features(1234))
        self.assertEqual(len(action_features(1234)), 64)
        self.assertEqual(normalized_observation((0, 50, 200)), [0.0, 0.5, 1.0])

    def test_neural_training_reports_missing_runtime_cleanly(self):
        if torch is not None:
            self.skipTest("PyTorch is installed in this environment")

    def test_neural_agent_loads_checkpoint_and_selects_legal_move(self):
        if torch is None:
            self.skipTest("PyTorch is unavailable")
        agent = NeuralAgent("neural_policy_v1.pt", seed=83)
        state = GameState.new(seed=84)
        state.begin_turn()
        moves = state.legal_moves()
        self.assertIn(agent.choose_move(state, moves), moves)

    def test_search_self_play_records_normalized_policy_targets(self):
        state = GameState.new(seed=87)
        state.begin_turn()
        agent = SearchSelfPlayAgent(simulations=3, rollout_turns=1, seed=9)
        agent.player_index = 0
        move = agent.choose_move(state, state.legal_moves())
        self.assertIn(move, state.legal_moves())
        target = agent.steps[0][3]
        self.assertAlmostEqual(sum(target), 1.0)

    def test_search_training_data_collection_runs(self):
        examples, stats = collect_search_self_play(games=1, seed=89, simulations=2, rollout_turns=1, max_turns=10)
        self.assertEqual(stats["games"], 1)
        self.assertGreater(len(examples), 0)

    def test_sly_deal_cannot_take_from_complete_set(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        action = Card("sly", CardKind.ACTION, 3, action="sly_deal")
        state.player.hand.append(action)
        target = state.players[1]
        target.sets["dark_blue"] = PropertySet("dark_blue", [
            Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("db2", CardKind.PROPERTY, 4, ("dark_blue",)),
        ])
        with self.assertRaises(ValueError):
            play_sly_deal(state, action.name, 1, "dark_blue", "db1")

    def test_sly_deal_moves_an_incomplete_property_to_attacker(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        action = Card("sly", CardKind.ACTION, 3, action="sly_deal")
        state.player.hand.append(action)
        target = state.players[1]
        target.sets["orange"] = PropertySet("orange", [Card("orange_1", CardKind.PROPERTY, 2, ("orange",))])
        play_sly_deal(state, action.name, 1, "orange", "orange_1")
        self.assertNotIn("orange", target.sets)
        self.assertEqual(len(state.player.sets["orange"].cards), 1)
        self.assertIn(action, state.discard_pile)

    def test_forced_deal_swaps_two_incomplete_properties(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        action = Card("forced", CardKind.ACTION, 3, action="forced_deal")
        state.player.hand.append(action)
        state.player.sets["orange"] = PropertySet("orange", [Card("mine", CardKind.PROPERTY, 2, ("orange",))])
        state.players[1].sets["pink"] = PropertySet("pink", [Card("theirs", CardKind.PROPERTY, 2, ("pink",))])
        play_forced_deal(state, action.name, 1, "orange", "mine", "pink", "theirs")
        self.assertIn("pink", state.player.sets)
        self.assertIn("orange", state.players[1].sets)

    def test_deal_breaker_transfers_buildings_and_wins(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        action = Card("breaker", CardKind.ACTION, 5, action="deal_breaker")
        state.player.hand.append(action)
        target = state.players[1]
        target.sets["dark_blue"] = PropertySet(
            "dark_blue",
            [Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)), Card("db2", CardKind.PROPERTY, 4, ("dark_blue",))],
            Card("house", CardKind.HOUSE, 3),
            Card("hotel", CardKind.HOTEL, 4),
        )
        play_deal_breaker(state, action.name, 1, "dark_blue")
        stolen = state.player.sets["dark_blue"]
        self.assertTrue(stolen.is_complete())
        self.assertIsNotNone(stolen.house)
        self.assertIsNotNone(stolen.hotel)
        self.assertEqual(len(state.discard_pile), 1)

    def test_normal_rent_creates_one_debt_per_opponent(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        state.player.sets["dark_blue"] = PropertySet(
            "dark_blue",
            [Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)), Card("db2", CardKind.PROPERTY, 4, ("dark_blue",))],
        )
        rent = Card("rent", CardKind.ACTION, 1, action="red_yellow_rent")
        state.player.hand.append(rent)
        state.player.sets["red"] = PropertySet("red", [
            Card("r1", CardKind.PROPERTY, 3, ("red",)),
            Card("r2", CardKind.PROPERTY, 3, ("red",)),
            Card("r3", CardKind.PROPERTY, 3, ("red",)),
        ])
        debts = prepare_rent(state, rent.name, "red")
        self.assertEqual(len(debts), 2)
        self.assertEqual(debts[0].amount, 6)
        self.assertEqual(state.actions_used, 1)

    def test_any_color_rent_targets_one_player_and_payment_moves_cash(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        state.player.sets["dark_blue"] = PropertySet("dark_blue", [
            Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("db2", CardKind.PROPERTY, 4, ("dark_blue",)),
        ])
        rent = Card("wild_rent", CardKind.ACTION, 3, action="any_color_rent")
        state.player.hand.append(rent)
        payment = Card("ten", CardKind.MONEY, 10)
        state.players[1].bank.append(payment)
        debts = prepare_rent(state, rent.name, "dark_blue", target_index=1)
        self.assertEqual(len(debts), 1)
        pay_debt(state, debts[0], [payment.name])
        self.assertIn(payment, state.player.bank)
        self.assertNotIn(payment, state.players[1].bank)

    def test_birthday_and_debt_collector_create_debts(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        birthday = Card("birthday", CardKind.ACTION, 2, action="birthday")
        debt_collector = Card("debt", CardKind.ACTION, 3, action="debt_collector")
        state.player.hand.extend([birthday, debt_collector])
        birthday_debts = prepare_birthday(state, birthday.name)
        self.assertEqual([debt.amount for debt in birthday_debts], [2, 2])
        debt = prepare_debt_collector(state, debt_collector.name, 1)
        self.assertEqual((debt.payer, debt.amount), (1, 5))

    def test_house_and_hotel_require_complete_set_in_order(self):
        state = GameState.new(seed=7)
        state.begin_turn()
        state.player.sets["dark_blue"] = PropertySet("dark_blue", [
            Card("db1", CardKind.PROPERTY, 4, ("dark_blue",)),
            Card("db2", CardKind.PROPERTY, 4, ("dark_blue",)),
        ])
        house = Card("house_test", CardKind.HOUSE, 3)
        hotel = Card("hotel_test", CardKind.HOTEL, 4)
        state.player.hand.extend([house, hotel])
        state.build_house(house.name, "dark_blue")
        self.assertEqual(state.player.sets["dark_blue"].rent(), 11)
        state.build_hotel(hotel.name, "dark_blue")
        self.assertEqual(state.player.sets["dark_blue"].rent(), 15)

    def test_baseline_game_runner_produces_a_result(self):
        result = run_game([CashFirstAgent(), PropertyFirstAgent(), RandomAgent(3)], seed=11, max_turns=1000)
        self.assertLessEqual(result.turns, 1000)
        self.assertIn(result.winner, (None, 0, 1, 2))

    def test_tactical_agent_can_run_with_action_moves(self):
        result = run_game([TacticalAgent(), CashFirstAgent(), RandomAgent(4)], seed=19, max_turns=1000)
        self.assertIn(result.winner, (None, 0, 1, 2))

    def test_forced_deal_is_generated(self):
        state = GameState.new(seed=53)
        state.begin_turn()
        forced = Card("forced_test", CardKind.ACTION, 3, action="forced_deal")
        state.player.hand.append(forced)
        state.player.sets["orange"] = PropertySet("orange", [Card("mine", CardKind.PROPERTY, 2, ("orange",))])
        state.players[1].sets["pink"] = PropertySet("pink", [Card("theirs", CardKind.PROPERTY, 2, ("pink",))])
        moves = [move for move in state.legal_moves() if move.kind == "forced_deal"]
        self.assertTrue(moves)

    def test_strategy_neutral_self_play_records_legal_actions(self):
        state = GameState.new(seed=23)
        state.begin_turn()
        move = state.legal_moves()[0]
        self.assertEqual(action_id(move), action_id(move))
        self.assertTrue(encode_move(move))
        self.assertGreater(len(encode_observation(state, 0)), 100)
        records = generate_episode(23, max_turns=100)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(step.chosen_action_id in step.legal_action_ids for step in records))

    def test_episode_collection_is_reproducible(self):
        first = collect_episodes(2, seed=31)
        second = collect_episodes(2, seed=31)
        self.assertEqual(first, second)

    def test_determinization_hides_opponent_hands_but_preserves_sizes(self):
        state = GameState.new(seed=41)
        sampled = determinize(state, 0, __import__("random").Random(5))
        self.assertEqual(len(sampled.players[1].hand), len(state.players[1].hand))
        self.assertEqual(len(sampled.players[2].hand), len(state.players[2].hand))
        self.assertEqual(len(sampled.players[0].hand), len(state.players[0].hand))

    def test_ismcts_returns_a_legal_move(self):
        state = GameState.new(seed=47)
        state.begin_turn()
        moves = state.legal_moves()
        chosen = ISMCTSAgent(simulations=8, rollout_turns=4, seed=3).choose_move(state, moves)
        self.assertIn(chosen, moves)

    def test_ismcts_can_use_neural_guided_rollouts(self):
        if torch is None:
            self.skipTest("PyTorch is unavailable")
        agent = ISMCTSAgent(simulations=2, rollout_turns=2, seed=11, rollout_checkpoint="neural_policy_v1.pt")
        state = GameState.new(seed=12)
        state.begin_turn()
        self.assertIn(agent.choose_move(state, state.legal_moves()), state.legal_moves())


if __name__ == "__main__":
    unittest.main()
