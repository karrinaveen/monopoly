# Monopoly Strategy Engine

A Python simulator and experimental learning framework for a custom three-player Monopoly Deal ruleset. The project aims to compare policies through reproducible self-play and tournament evaluation, then explain the decisions that improve win rate.

## Project status

The repository contains:

- A game rules engine for the agreed custom rules, including property sets, wild properties, rent, payment, attacks, Just Say No, and buildings.
- Baseline agents, simulation, and benchmark utilities.
- Search-guided self-play, tabular response policies, and neural policy/value checkpoints.
- Saved experimental policy tables and checkpoints. They are research artifacts, not evidence that a best strategy has been found.

The card inventory currently reflects the cards transcribed from the provided screenshots. Confirm the physical deck inventory before treating simulation results as definitive.

## Requirements

- Python 3
- PyTorch for neural policies and search training

## Run

Run the existing rules and policy test suite:

```bash
python3 -m unittest -v
```

Run a random-agent benchmark:

```bash
python3 benchmark.py --games 1000 --workers 4
```

Train a search-guided policy generation:

```bash
python3 train_search.py \
  --games 20 \
  --simulations 16 \
  --rollout-turns 20 \
  --epochs 3 \
  --seed 2026 \
  --checkpoint neural_search_next.pt
```

Use `--init-checkpoint` to fine-tune a compatible checkpoint, or `--opponent-population` to collect games against the configured opponent pool. Record the seed and training arguments with each result so generations can be compared fairly.

## Strategy evaluation

Win rate is the objective. Evaluation should rotate seats and use a varied opponent population; draws and unfinished simulations should be reported separately. Keep rule changes and policy changes identifiable so a result can be attributed to the correct change.
