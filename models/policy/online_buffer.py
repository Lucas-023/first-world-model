"""
Replay buffer online pro loop de treino de verdade (models/policy/train_online.py):
guarda episodios reais completos coletados ao vivo (mesmo formato de
dataset_tokens -- tokens/actions/rewards/dones) e amostra janelas de
contexto pra treinar o World Model (compute_loss) ou semear rollouts
imaginados (collect_rollout), com a MESMA distribuicao que
models/dynamics/dataset.py::CarRacingTokenDataset usa sobre o dataset
offline: uma amostra uniforme sobre TODAS as janelas validas de TODOS os
episodios (nao "episodio uniforme, depois offset uniforme dentro dele" --
isso sub-representaria episodios longos).

Capacidade limitada em passos (FIFO): quando estoura, descarta o episodio
mais antigo inteiro. Como as janelas de cada episodio sao inseridas em
`window_index` em ordem de chegada, as do episodio mais antigo estao sempre
no INICIO da lista -- descartar e O(k) (k = janelas daquele episodio), nunca
uma varredura O(N) do buffer inteiro.
"""

import numpy as np
import torch
from collections import deque

from models.dynamics.dataset import symlog


class OnlineReplayBuffer:
    def __init__(self, capacity_steps, context_len, device):
        self.capacity_steps = capacity_steps
        self.context_len = context_len
        self.device = device

        self.episodes_by_id = {}
        self.episode_order = deque()   # ids, mais antigo primeiro
        self.window_index = []         # [(episode_id, start_offset), ...] em ordem de insercao
        self.total_steps = 0
        self._next_id = 0

    def add_episode(self, data):
        """data: dict com tokens (T,64)/actions (T,)/rewards (T,)/dones (T,)
        -- mesmo formato que collect_from_policy.py::RealEpisodeStepper.episode_arrays()
        produz. So episodios COMPLETOS devem ser passados aqui (o episodio em
        andamento fica de fora ate terminar)."""
        tokens = data["tokens"].astype(np.int64)
        actions = data["actions"].astype(np.int64)
        dones = data["dones"].astype(np.int64)
        rewards = data["rewards"].astype(np.float32)
        rewards_symlog = symlog(rewards)

        T = tokens.shape[0]
        eid = self._next_id
        self._next_id += 1
        self.episodes_by_id[eid] = {
            "tokens": tokens, "actions": actions, "dones": dones,
            "rewards_symlog": rewards_symlog,
        }
        self.episode_order.append(eid)
        self.total_steps += T

        if T >= self.context_len + 1:
            n_windows = T - self.context_len
            self.window_index.extend((eid, s) for s in range(n_windows))

        self._evict_if_needed()

    def _evict_if_needed(self):
        while self.total_steps > self.capacity_steps and len(self.episode_order) > 1:
            oldest_id = self.episode_order.popleft()
            oldest = self.episodes_by_id.pop(oldest_id)
            self.total_steps -= oldest["tokens"].shape[0]

            i = 0
            while i < len(self.window_index) and self.window_index[i][0] == oldest_id:
                i += 1
            if i > 0:
                del self.window_index[:i]

    def __len__(self):
        """Numero de janelas validas disponiveis pra amostrar (nao numero de episodios)."""
        return len(self.window_index)

    def num_episodes(self):
        return len(self.episode_order)

    def _sample_indices(self, batch_size):
        if len(self.window_index) == 0:
            raise RuntimeError("OnlineReplayBuffer vazio -- nenhuma janela valida ainda (buffer precisa de pelo menos 1 episodio completo com T >= context_len+1)")
        idx = np.random.randint(0, len(self.window_index), size=batch_size)
        return [self.window_index[i] for i in idx]

    def sample_training_windows(self, batch_size, device=None):
        """Pro compute_loss do World Model:
        (obs_ctx, act_ctx, obs_target, reward_target, done_target)."""
        device = device or self.device
        picks = self._sample_indices(batch_size)

        obs_ctx_b, act_ctx_b, obs_tgt_b, rew_tgt_b, done_tgt_b = [], [], [], [], []
        for eid, start in picks:
            ep = self.episodes_by_id[eid]
            end = start + self.context_len
            obs_ctx_b.append(ep["tokens"][start:end])
            act_ctx_b.append(ep["actions"][start:end])
            obs_tgt_b.append(ep["tokens"][end])
            rew_tgt_b.append(ep["rewards_symlog"][end])
            done_tgt_b.append(ep["dones"][end])

        return (
            torch.from_numpy(np.stack(obs_ctx_b)).long().to(device),
            torch.from_numpy(np.stack(act_ctx_b)).long().to(device),
            torch.from_numpy(np.stack(obs_tgt_b)).long().to(device),
            torch.tensor(rew_tgt_b, dtype=torch.float32, device=device),
            torch.tensor(done_tgt_b, dtype=torch.long, device=device),
        )

    def sample_seed_contexts(self, batch_size, device=None):
        """Pras sementes do rollout imaginado (collect_rollout): (obs_ctx, act_ctx)."""
        device = device or self.device
        picks = self._sample_indices(batch_size)

        obs_ctx_b, act_ctx_b = [], []
        for eid, start in picks:
            ep = self.episodes_by_id[eid]
            end = start + self.context_len
            obs_ctx_b.append(ep["tokens"][start:end])
            act_ctx_b.append(ep["actions"][start:end])

        return (
            torch.from_numpy(np.stack(obs_ctx_b)).long().to(device),
            torch.from_numpy(np.stack(act_ctx_b)).long().to(device),
        )
