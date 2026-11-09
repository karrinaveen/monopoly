"""Optional neural policy/value model for the game.

The model scores only the currently legal actions, so it naturally supports
the game's variable action set. PyTorch is optional at import time; training
requires installing it separately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from learning import EpisodeStep, encode_action_features


def action_features(action, buckets: int = 64) -> list[float]:
    """Semantic features for a legal move, independent of card instance ID."""
    if isinstance(action, int):
        # Compatibility for existing experimental checkpoints trained on IDs.
        digest = hashlib.blake2b(str(action).encode(), digest_size=16).digest()
        vector = [0.0] * buckets
        for offset in (0, 4, 8, 12):
            vector[int.from_bytes(digest[offset:offset + 4], "big") % buckets] += 1.0
        return vector
    return list(encode_action_features(action))


def normalized_observation(observation: tuple[int, ...]) -> list[float]:
    return [min(float(value), 100.0) / 100.0 for value in observation]


def outcome_for(step: EpisodeStep) -> float:
    if step.winner is None:
        return 0.0
    return 1.0 if step.winner == step.player_index else 0.0


try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by runtime availability
    torch = None
    nn = None


if nn is not None:

    class PolicyValueNet(nn.Module):
        def __init__(self, observation_dim: int, action_feature_dim: int = 97, hidden_dim: int = 128):
            super().__init__()
            self.observation_encoder = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(action_feature_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, 1)
            self.value_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        def forward(self, observations, action_features_batch):
            state_embedding = self.observation_encoder(observations)
            action_embedding = self.action_encoder(action_features_batch)
            logits = (state_embedding.unsqueeze(1) * action_embedding).sum(dim=-1)
            values = self.value_head(state_embedding).squeeze(-1)
            return logits, values


@dataclass(frozen=True)
class NeuralTrainingStats:
    examples: int
    epochs: int
    final_loss: float


def train_policy_value(records: list[EpisodeStep], epochs: int = 3, learning_rate: float = 1e-3, seed: int = 1, checkpoint: str | None = None) -> NeuralTrainingStats:
    if torch is None:
        raise RuntimeError("PyTorch is required for neural training; install torch or mlx first")
    if not records:
        raise ValueError("At least one replay record is required")
    torch.manual_seed(seed)
    observation_dim = len(records[0].observation)
    model = PolicyValueNet(observation_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        total = 0.0
        for record in records:
            observation = torch.tensor([normalized_observation(record.observation)], dtype=torch.float32)
            action_tensor = torch.tensor(record.legal_action_features, dtype=torch.float32)
            logits, value = model(observation, action_tensor.unsqueeze(0))
            chosen_index = record.legal_action_ids.index(record.chosen_action_id)
            policy_loss = nn.functional.cross_entropy(logits, torch.tensor([chosen_index]))
            value_loss = nn.functional.mse_loss(value, torch.tensor([outcome_for(record)], dtype=torch.float32))
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        final_loss = total / len(records)
    if checkpoint:
        torch.save({"model": model.state_dict(), "observation_dim": observation_dim, "action_feature_dim": 97}, checkpoint)
    return NeuralTrainingStats(len(records), epochs, final_loss)


def train_from_search(records, epochs: int = 3, learning_rate: float = 3e-4, seed: int = 1, checkpoint: str | None = None, init_checkpoint: str | None = None) -> NeuralTrainingStats:
    """Fit policy to normalized MCTS visit counts and value to game outcome."""
    if torch is None:
        raise RuntimeError("PyTorch is required for neural training")
    if not records:
        raise ValueError("At least one search record is required")
    torch.manual_seed(seed)
    observation_dim = len(records[0].observation)
    action_feature_dim = len(records[0].legal_action_features[0])
    model = PolicyValueNet(observation_dim, action_feature_dim=action_feature_dim)
    if init_checkpoint:
        saved = torch.load(init_checkpoint, map_location="cpu")
        if saved["observation_dim"] != observation_dim:
            raise ValueError("Initial checkpoint observation size does not match the current game encoder")
        if saved.get("action_feature_dim", 64) != action_feature_dim:
            raise ValueError("Initial checkpoint uses an incompatible action encoder; train this generation from scratch")
        model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(records)).tolist()
        total_loss = 0.0
        for index in order:
            record = records[index]
            observation = torch.tensor([normalized_observation(record.observation)], dtype=torch.float32)
            actions = torch.tensor(record.legal_action_features, dtype=torch.float32).unsqueeze(0)
            target_policy = torch.tensor([record.policy_target], dtype=torch.float32)
            target_value = 1.0 if record.winner == record.player_index else 0.0
            logits, value = model(observation, actions)
            policy_loss = -(target_policy * nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(value, torch.tensor([target_value], dtype=torch.float32))
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        final_loss = total_loss / len(records)
    if checkpoint:
        torch.save({"model": model.state_dict(), "observation_dim": observation_dim, "action_feature_dim": action_feature_dim}, checkpoint)
    return NeuralTrainingStats(len(records), epochs, final_loss)
