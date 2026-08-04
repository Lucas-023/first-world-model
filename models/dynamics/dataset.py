"""
Dataset para Dynamics GPT com contexto -> proximo passo.

Cada janela devolve:
    obs_ctx      : (context_len, 64)
    act_ctx      : (context_len,)
    obs_target   : (64,)
    reward_target: ()  float32, reward real transformado por symlog
    done_target  : ()  em {0,1}
"""

import os
import glob
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


def symlog(rewards):
    """Reward continuo -> alvo de regressao em escala symlog:
    sign(x)*log(1+|x|). Comprime magnitude preservando o sinal, igual
    DreamerV3 -- estabiliza a regressao mesmo com rewards de escalas bem
    diferentes (ex.: -100 ao sair da pista vs. +0.1 por passo em pista).
    Unica fonte de verdade dessa conversao -- reaproveitada por
    models/policy/online_buffer.py pra garantir que o replay buffer online
    rotule reward exatamente como o dataset offline sempre rotulou."""
    return (np.sign(rewards) * np.log1p(np.abs(rewards))).astype(np.float32)


def symexp(y):
    """Inversa de symlog, em torch: sign(y)*(exp(|y|)-1). Usada pelo
    WorldModel (gptdynamics.py) pra converter a predicao da cabeca de reward
    (que vive em escala symlog, a mesma do alvo de treino) de volta pra
    escala real de reward antes de devolver pra quem consome (rollout.py,
    scripts de avaliacao/visualizacao)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class CarRacingTokenDataset(Dataset):
    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        folder: str,
        split: str = "train",
        context_len: int = 19,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        stride: int = 1,
    ):
        assert split in self.SPLITS, f"split deve ser um de {self.SPLITS}"
        assert train_ratio + val_ratio < 1.0
        assert stride >= 1

        self.context_len = context_len
        self.chunks = []

        files = sorted(glob.glob(os.path.join(folder, "*.npz")))

        # bucket por hash do NOME do arquivo (nao por indice/contagem) -- assim
        # a divisao de um episodio ja existente nunca muda quando novos
        # episodios sao adicionados na pasta depois (ex.: coleta com uma
        # politica treinada, pra fine-tunar o World Model). Com o esquema
        # antigo (rng.permutation(len(files))) o split inteiro era
        # recalculado toda vez que o numero de arquivos mudava, entao um
        # episodio podia migrar de test/val pra train (ou vice-versa) entre
        # duas rodadas de traingpt.py -- inutilizava best_val_loss como
        # medida de "melhorou de verdade" ao comparar checkpoints.
        def bucket(fname):
            h = hashlib.md5(f"{seed}:{os.path.basename(fname)}".encode()).hexdigest()
            return int(h[:8], 16) / 0xFFFFFFFF  # em [0, 1), estavel por nome+seed

        n_train_frac = train_ratio
        n_val_frac = train_ratio + val_ratio

        if split == "train":
            selected = [f for f in files if bucket(f) < n_train_frac]
        elif split == "val":
            selected = [f for f in files if n_train_frac <= bucket(f) < n_val_frac]
        else:
            selected = [f for f in files if bucket(f) >= n_val_frac]

        n = len(files)

        print(f"[{split:>5}] {len(selected)}/{n} episodios")

        for f in selected:
            d = np.load(f, allow_pickle=False)
            tokens = d["tokens"].astype(np.int64)        # (T, 64)
            actions = d["actions"].astype(np.int64)      # (T,)
            dones = d["dones"].astype(np.int64)          # (T,)
            rewards = d["rewards"].astype(np.float32)    # (T,)
            rewards_symlog = symlog(rewards)              # escala symlog

            T = tokens.shape[0]
            if T < context_len + 1:
                continue

            for i in range(0, T - context_len, stride):
                obs_ctx = tokens[i:i + context_len]                   # (context_len, 64)
                act_ctx = actions[i:i + context_len]                  # (context_len,)
                obs_target = tokens[i + context_len]                  # (64,)
                reward_target = rewards_symlog[i + context_len]       # ()
                done_target = dones[i + context_len]                  # ()
                self.chunks.append((obs_ctx, act_ctx, obs_target, reward_target, done_target))

        print(f"[{split:>5}] {len(self.chunks)} janelas (contexto={context_len}, stride={stride})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        obs_ctx, act_ctx, obs_target, reward_target, done_target = self.chunks[idx]
        return (
            torch.from_numpy(obs_ctx).long(),
            torch.from_numpy(act_ctx).long(),
            torch.from_numpy(obs_target).long(),
            torch.tensor(reward_target, dtype=torch.float32),
            torch.tensor(done_target, dtype=torch.long),
        )


DynamicsDataset = CarRacingTokenDataset