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


@torch.no_grad()
def soft_update_(target, source, decay):
    """Critico-alvo suavizado por EMA (DreamerV3): apos cada update de PPO,
    target = decay*target + (1-decay)*source, no lugar de copiar o peso
    inteiro de uma vez. Usado pra estabilizar os alvos de valor (values do
    rollout/bootstrap) contra a rede que esta sendo atualizada a cada passo
    de gradiente -- ver techreport.tex, secao de estabilizadores do
    DreamerV3/IRIS."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


class ReturnNormalizer(nn.Module):
    """Escala vantagens por uma faixa de percentil dos retornos (DreamerV3:
    "return normalization by percentile range"), suavizada por EMA entre
    updates -- substitui a normalizacao z-score (media/desvio) recalculada do
    zero a cada minibatch. Duas diferencas deliberadas: (1) so ESCALA, nao
    centraliza -- a vantagem ja e ~zero-media por construcao (GAE); (2) a
    faixa (percentil alto - percentil baixo) e suave ao longo de MUITOS
    updates, entao um unico retorno extremo (ex.: done fora da pista, ordens
    de grandeza acima do resto) nao domina a escala de um minibatch inteiro
    como o desvio padrao por lote fazia."""

    def __init__(self, decay=0.99, low=0.05, high=0.95):
        super().__init__()
        self.decay = decay
        self.low = low
        self.high = high
        self.register_buffer("p_low", torch.zeros(1))
        self.register_buffer("p_high", torch.zeros(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update_and_get_scale(self, returns, active_mask):
        active = active_mask.reshape(-1).bool()
        vals = returns.reshape(-1)[active]
        if vals.numel() == 0:
            return max((self.p_high - self.p_low).item(), 1.0)

        lo = torch.quantile(vals, self.low)
        hi = torch.quantile(vals, self.high)
        if not bool(self.initialized.item()):
            self.p_low.copy_(lo.reshape(1))
            self.p_high.copy_(hi.reshape(1))
            self.initialized.fill_(True)
        else:
            self.p_low.mul_(self.decay).add_(lo.reshape(1) * (1 - self.decay))
            self.p_high.mul_(self.decay).add_(hi.reshape(1) * (1 - self.decay))
        return max((self.p_high - self.p_low).item(), 1.0)
