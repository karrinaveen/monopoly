"""Inference agent for trained policy/value checkpoints."""

from __future__ import annotations

import random

from agents import Agent
from learning import encode_action_features, encode_observation
from neural_policy import PolicyValueNet, action_features, torch
from response_learning import TabularResponsePolicy, response_context


class NeuralAgent(Agent):
    name = "neural"

    def __init__(self, checkpoint: str, response_policy: TabularResponsePolicy | None = None, seed: int | None = None, temperature: float = 0.0):
        if torch is None:
            raise RuntimeError("PyTorch is required to load the neural policy")
        saved = torch.load(checkpoint, map_location="cpu")
        self.model = PolicyValueNet(saved["observation_dim"], action_feature_dim=saved.get("action_feature_dim", 64))
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.response_policy = response_policy
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.player_index = 0

    def choose_move(self, state, moves):
        observation = torch.tensor([[
            min(float(value), 100.0) / 100.0 for value in encode_observation(state, self.player_index)
        ]], dtype=torch.float32)
        feature_dim = self.model.action_encoder[0].in_features
        encoded = [self._legacy_action_features(move) for move in moves] if feature_dim == 64 else [encode_action_features(move) for move in moves]
        features = torch.tensor([encoded], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(observation, features)
        values = logits[0].tolist()
        if self.temperature <= 0:
            return moves[max(range(len(moves)), key=lambda index: values[index])]
        weights = [pow(2.718281828, value / self.temperature) for value in values]
        return self.rng.choices(moves, weights=weights, k=1)[0]

    @staticmethod
    def _action_id(move):
        from learning import action_id
        return action_id(move)

    @classmethod
    def _legacy_action_features(cls, move):
        return action_features(cls._action_id(move))

    def choose_just_say_no(self, state, action_name: str, target_index: int) -> bool:
        if self.response_policy is not None:
            return self.response_policy.choose(response_context(state, action_name, target_index, self.player_index))
        return False
