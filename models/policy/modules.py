"""
Politica (actor-critic) treinada via PPO sobre rollouts imaginados pelo
World Model congelado (models/dynamics/gptdynamics.py::WorldModel).

Entrada: state_repr (B, n_embd) -- a representacao compacta do contexto que
o World Model ja usa pra prever reward/done (WorldModel.encode_state).
Saida: distribuicao categorica sobre as 5 acoes discretas + valor escalar.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim=256, n_actions=5, hidden_dim=128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        h = self.trunk(state)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, state):
        """Usado no rollout (sem grad, dentro do no_grad do encode_state/imagine_next_frame)."""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, state, action):
        """Usado nas epocas de update do PPO: reavalia log_prob/valor/entropia
        para acoes ja amostradas, com grad. Nao toca no World Model."""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), value, dist.entropy()
