# Monopoly Strategy Engine: Rules, Implementation, and Experiment Log

Last updated: 2026-11-09

Repository: [`karrinaveen/monopoly`](https://github.com/karrinaveen/monopoly)

Local initial project commit: `fc28437` (`Initial Monopoly strategy engine project`)

## 1. Purpose and current status

This project models a custom three-player Monopoly Deal variant and experiments with agents that aim to win by completing three property sets. The long-term objective is to learn strong play from large numbers of games and explain the choices that improve win rate.

The code now contains a rules/state engine, action resolution, baseline agents, simulation, imperfect-information Monte Carlo search, tabular response learning, and neural policy/value models. It also contains saved experimental policy and checkpoint files. **The work so far has not established an optimized strategy.** The learned models have been weak in cross-play, and the cash-reserve comparison is confounded by a substantial seat effect.

This document distinguishes:

- Rules the user explicitly specified for this home version.
- How the current code represents those rules.
- What the experiments measured, and what they do not establish.

## 2. User-confirmed rules for the simulated variant

These are the latest agreed rules from the conversation. They are custom-version rules, not claims about every official Monopoly Deal edition.

### Game and turns

- Three players are modeled first.
- A player wins immediately after completing three property sets.
- Each player receives five cards at setup.
- At the start of a turn, draw two cards. If the player starts the turn with an empty hand, draw five instead.
- A player may use up to three actions on their turn. A player may finish sooner when allowed by the hand-limit rule.
- A normal rent card charges both opponents. An Any-Color Rent card charges one selected opponent for a color the player has completed.
- Double the Rent must be played with a rent card and costs two actions total. It doubles the rent amount.
- Pass Go is an action: discard it to the center pile and draw two cards.

### Hand limit and discarding

- The hand limit is seven cards.
- A player with more than seven cards cannot simply replace an available play with a free discard. The player must use the three actions first; if still over the limit after the third action, they may discard freely to return to seven.
- The player may choose which card(s) to discard.

### Properties, wilds, and buildings

- Pink is an ordinary color set with the same general rules as the other colors; it was missing from the first screenshot and was added after clarification.
- Standard colored properties and two-color wild properties can be assigned to an allowed color. A two-color wild can represent either printed color.
- A universal wild has no printed color and cannot complete a set by itself. It must be placed into a color set that also contains at least one property of that color.
- Property/wild arrangements may be changed before the player's first action of the turn. Once the first action is taken, the arrangement is locked for that turn.
- A standard or two-color property in a complete set is protected from Sly Deal and Forced Deal. A two-color wild is otherwise eligible for those actions when it is in an incomplete set.
- A universal wild cannot be taken by Sly Deal or Forced Deal. It can be transferred as part of a complete set taken by Deal Breaker.
- Sly Deal takes one eligible property from an opponent's incomplete set. Forced Deal swaps one eligible property from each player's incomplete sets. Neither action can take from a complete set.
- Deal Breaker takes an opponent's complete set; the set's attached house/hotel transfer with it.
- Houses and hotels can be built on a complete set. A house adds 3M rent. A hotel adds a further 4M rent and requires a house first.
- For this project, assume houses and hotels cannot be moved once built, except that a Deal Breaker transfers them as part of the set. The official-edition question was not independently verified.

### Actions, counter-actions, and the center pile

- Every card except the universal wild has a cash value. A player may bank an action card, house, or hotel as cash when the card has a value.
- Once banked, a card is ordinary cash permanently; it loses its action/building function.
- Just Say No can respond to an action that affects its player, including rent, Birthday, Debt Collector, Sly Deal, Forced Deal, or Deal Breaker. It cancels the action only for the player who used that Just Say No. For an action affecting multiple opponents, each opponent's effect is resolved separately.
- Countering a Just Say No with another Just Say No is allowed. There are only three Just Say No cards in the listed deck, so a chain is finite. A Just Say No played by the active player to counter a response consumes one of that player's actions; the defending response is free under the agreed rules.
- Resolved action cards and Pass Go go into the center/discard pile. If the draw pile runs out, cards in the center pile are reshuffled to form a draw pile. The game is a draw only if both draw and discard piles are empty and there is no winner; the user expects this to be extremely rare.

### Payments

- A player asked to pay may give banked cash, property, or a mixture of both.
- Payments use whole cards and no change is returned. For example, if 1M is owed and the payer gives a 5M card, the full 5M transfers.
- The payer has no choice to split a single card's value. If asked for 3M and paying from cash, the preference is one 3M card over separate 1M and 2M cards because it transfers fewer cards.
- Properties and banked cards can be paid; cards still in hand cannot. A house/hotel attached to a set is not a separate payment card.

## 3. Card inventory encoded from the screenshots and clarifications

The current deck builder creates 105 cards: 28 single-color properties, 9 two-color wild properties, 2 universal wilds, 41 action cards, 20 cash cards, and 5 building cards. The inventory should be checked against the physical deck before treating any simulation as authoritative.

### Property sets

| Color | Cards needed | Value per card | Rent by set size |
|---|---:|---:|---|
| Brown | 2 | 1M | 1M, 2M |
| Light blue | 3 | 1M | 1M, 2M, 3M |
| Pink | 3 | 2M | 1M, 2M, 4M |
| Orange | 3 | 2M | 1M, 3M, 5M |
| Red | 3 | 3M | 2M, 3M, 6M |
| Yellow | 3 | 3M | 2M, 4M, 6M |
| Dark green | 3 | 4M | 2M, 4M, 7M |
| Light green | 2 | 2M | 1M, 2M |
| Dark blue | 2 | 4M | 3M, 8M |
| Black | 4 | 2M | 1M, 2M, 3M, 4M |

Blank rent cells in the original screenshot mean that set size does not exist. For example, Dark Green has three cards and no fourth-card rent.

### Wild properties

| Wild card colors | Count |
|---|---:|
| Red / Yellow | 2 |
| Dark Green / Black | 1 |
| Light Green / Black | 1 |
| Light Blue / Brown | 1 |
| Orange / Pink | 2 |
| Dark Green / Dark Blue | 1 |
| Light Blue / Black | 1 |
| Universal, any color | 2 |

Two-color wilds have a listed value of 1M in the engine. Universal wilds have no cash value in the supplied inventory and are not bankable.

### Action cards

| Action | Count | Cash value each | Implemented effect / note |
|---|---:|---:|---|
| Sly Deal | 3 | 3M | Take one eligible property from an incomplete opponent set |
| Forced Deal | 3 | 3M | Swap one eligible property from each incomplete set |
| Deal Breaker | 2 | 5M | Take a complete set, including its buildings |
| Just Say No | 3 | 4M | Cancel an action affecting the user of the card; finite counter-chain |
| Pass Go | 10 | 1M | Draw two cards |
| Debt Collector | 3 | 3M | Charge one opponent 5M |
| Any-Color Rent | 3 | 3M | Charge one opponent rent for a completed color set |
| Birthday | 3 | 2M | Charge each opponent 2M |
| Double the Rent | 2 | 1M | Pair with rent; doubles its amount; costs two actions total |
| Light Green / Black Rent | 2 | 1M | Rent for one completed matching set; charges both opponents |
| Orange / Pink Rent | 2 | 1M | Same |
| Dark Green / Dark Blue Rent | 2 | 1M | Same |
| Red / Yellow Rent | 2 | 1M | Same |
| Light Blue / Brown Rent | 1 | 1M | Same |

### Cash and buildings

| Card | Count | Value each |
|---|---:|---:|
| Cash | 2 | 5M |
| Cash | 1 | 10M |
| Cash | 3 | 4M |
| Cash | 5 | 2M |
| Cash | 3 | 3M |
| Cash | 6 | 1M |
| House | 3 | 3M |
| Hotel | 2 | 4M |

## 4. Implementation map

- [`monopoly_engine.py`](monopoly_engine.py): card inventory, set/rent specifications, property-set completion, and basic protection rules.
- [`game_state.py`](game_state.py): players, hand/bank/property state, turn draw, action budget, legal moves, winner checks, and deck reshuffle.
- [`action_effects.py`](action_effects.py): rent debts, payment transfer, Sly Deal, Forced Deal, Deal Breaker, and Just Say No support used by the simulator.
- [`action_rules.py`](action_rules.py): small pure helpers for action costs, rent targets, and finite Just Say No chains.
- [`simulation.py`](simulation.py): game loop, action resolution, automatic debt payment, results, and diagnostics.
- [`agents.py`](agents.py): random, cash-first, property-first, tactical, and cash-reserve baselines.
- [`mcts.py`](mcts.py), [`search_learning.py`](search_learning.py), [`learning.py`](learning.py), [`neural_policy.py`](neural_policy.py), and [`neural_agent.py`](neural_agent.py): imperfect-information search, action/state features, search-data collection, and neural policy/value inference/training.
- [`response_learning.py`](response_learning.py) and [`policy_learning.py`](policy_learning.py): tabular response/action learning experiments.
- [`evaluation.py`](evaluation.py) and [`benchmark.py`](benchmark.py): cross-play and simulation benchmark runners.
- `neural_search_*.pt`, `neural_policy*.pt`, `response_policy*.json`, and `action_policy.json.gz`: saved experimental checkpoints or policy tables. Their existence does not mean they are strong or validated.

## 5. Issues found and fixes or remaining gaps

### Rules interpretation

- **Pink was initially omitted.** The user clarified its set size (three), card value (2M), and rents (1M/2M/4M). It is now a normal color in `SET_SPECS`.
- **Two different kinds of wilds were initially conflated.** The rules and model now distinguish stealable two-color wilds from universal wilds, which cannot complete a set alone and cannot be taken by Sly/Forced.
- **“Blank rent” entries caused ambiguity.** They mean that rent level is not applicable because that set size is impossible.
- **Action cards as cash and payment denomination matter.** Banking permanently removes an action's function. The automatic payment selector was changed to use dynamic programming and prefer the least overpayment, then the fewest cards. Thus a single 3M card wins over 1M+2M for a 3M debt. It does not yet value future strategic use of a particular bank card or property when two payment choices tie.
- **Draws and simulation timeouts were at risk of being conflated.** Results now distinguish a rule-defined exhaustion draw from a game that simply reached a simulation turn cap without a winner. The user-defined draw requires both draw and discard piles to be empty.
- **Wins can occur during property/payment transfers.** Winner checks are performed after moves and payment transfers so a player completing their third set by receiving cards is recognized.
- **A Player B turn-progression issue was reported during early interactive play.** The current state transition advances the active player in `end_turn`. The present repository is a Python engine/simulator; it does not include a separate interactive UI, so UI turn progression is not currently validated here.
- **Subset search could become expensive.** Payment originally used exhaustive combinations for smaller states and a greedy fallback for larger ones. The simulator now uses a subtotal dynamic program for both, avoiding the fallback's undesirable choice of multiple small cards when a single exact card is available.

### Known rule-model gaps to resolve before relying on results

- The agreed pre-action property/wild rearrangement rule is **not fully implemented**: `property_locked` is reset at turn start and set after actions, but there is no legal move/API to reassign cards between sets before the first action.
- The current Deal Breaker legal-move generator only offers a target color not already present in the attacker's property area. The user only specified that Deal Breaker takes a complete set; confirm whether duplicate colors should be legal, then align the engine.
- Automatic rent/debt payment is deterministic and does not learn from the payer's position. It minimizes overpayment/card count and assigns property cards to the payee's first available eligible set. If that chosen property combination cannot be assigned, the automatic payment may stop instead of searching the next-best feasible combination.
- Sly/Forced destination colors for wilds are generated by a simple first-available-color rule. A strategic agent should choose these deliberately.
- House/hotel immovability is a selected custom assumption. The requested official-edition behavior has not been independently checked.
- Exact card-inventory transcription should be confirmed against the physical deck.

## 6. Strategy-learning approach attempted so far

The user explicitly asked for a process that learns by playing many games rather than starting from a fixed “cash first” or “property first” rule. The code has several generations of learning components, but the current training loop is still experimental:

1. Define a deterministic rules/state model, then generate legal moves from the state.
2. Use imperfect-information Monte Carlo search to compare legal moves through sampled hidden states and rollouts.
3. Save the search's normalized action visit counts as policy targets and the eventual winner as a value target.
4. Train a neural policy/value model on those targets, then evaluate it in seat-rotated cross-play.
5. Include random, heuristic, and prior-model opponents in some training/evaluation pools to reduce narrow self-play overfitting.

This is **search-imitation/self-play training**, not yet a mature reinforcement-learning system that blindly trains millions of games and automatically promotes stronger generations. Search rollouts and the available opponents shape the targets; the model has not yet demonstrated a reliable improvement over a broad benchmark. The baseline policies are useful controls, not assumed optimal play.

An important strategic hypothesis came from the user: a low early bank can make a player an easy target, and lost property/cash can compound (“poor becomes poor”). Cash flexibility also matters: a bank made of several small denominations can pay exact debts, while a 10M+5M bank may force a large overpayment for a small bill. These are hypotheses to test, not conclusions established by the current simulations.

## 7. Experiments and results recorded so far

### Search-guided model generations

- A semantic action-feature model was trained from an earlier search generation. An earlier self-play run used 40 games, 16 search simulations, 20 rollout turns, and two epochs; it produced 4,706 training examples and 12 draws.
- A later mixed-opponent generation used 20 games, 8 simulations, 12 rollout turns, and two epochs, initialized from the prior semantic checkpoint. It produced 792 examples. During collection, the candidate won 7 games, opponents won 11, and 2 were draws.
- A 300-game cross-play of the later model against a mixed pool recorded 72 candidate wins, 175 opponent wins, 53 draws, and a mean length of 41.2 turns. This is weak performance in that evaluation, not a strategy recommendation. The mixed pool included older learned models and random/cash-first/property-first/tactical baselines.
- The results above do not include a complete run manifest with every seed, checkpoint hash, opponent count, or raw game trace. Preserve those in future evaluations.

### Cash-reserve target comparison

The last completed experiment compared simple `CashReserveAgent` variants targeting 8M, 12M, or 15M. The three variants played one another; only the reserve target differed. There were 300 games, the target agents were rotated equally across seats (100 appearances in each seat per target), seeds began at 73000, and the turn cap was 500.

| Reserve target | Wins | Win rate |
|---:|---:|---:|
| 8M | 84 | 28.0% |
| 12M | 112 | 37.3% |
| 15M | 104 | 34.7% |

There were no draws or unfinished games in this run; average length was 25.5 turns. Seat totals across all target variants were seat 0: 71 wins, seat 1: 102, seat 2: 127. The seat skew is large enough that the apparent 12M lead is only preliminary evidence; investigate the seat/deal-order effect and repeat evaluation against a diverse pool before selecting a target.

Average bank value at the start of each agent's first five turns:

| Target | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 |
|---:|---:|---:|---:|---:|---:|
| 8M | 0.00M | 0.26M | 1.39M | 3.69M | 5.70M |
| 12M | 0.00M | 0.29M | 1.51M | 3.85M | 6.57M |
| 15M | 0.00M | 0.36M | 1.56M | 3.79M | 6.34M |

The reserve policy prioritizes an immediate winning property play, then Deal Breaker, Forced/Sly Deal, income actions, and property placement. It plays Pass Go while the hand is below seven. Only afterward does it bank toward its target; it chooses a cash card over an action card if one is available, and chooses the lowest-value card in that pool. This preference is hand-coded. Therefore the experiment does **not** establish which action card is best to bank, nor does it compare learned cash-denomination policies. The results provide a first threshold comparison only.

## 8. Current artifacts and reproduction

The root includes the Python modules, existing test suite, saved policies, and neural checkpoints. Training entry point:

```bash
python3 train_search.py \
  --games 20 \
  --simulations 16 \
  --rollout-turns 20 \
  --epochs 3 \
  --seed 2026 \
  --checkpoint neural_search_next.pt
```

For a meaningful comparison, record the exact command, seed, checkpoint SHA, game rules version, opponent mix, max-turn behavior, seat assignment, wins/draws/unfinished counts, turn lengths, and per-seat results. Rotate the candidate through seats and use paired seeds where possible.

The prior public push included the initial project source and the existing policy/checkpoint artifacts. This file records the current research history; future experiments should add a dated entry and raw machine-readable result summaries rather than relying only on conversational notes.
