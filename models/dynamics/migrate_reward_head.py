"""
Migra um checkpoint do World Model salvo ANTES da regressao symlog
(head_rewards de 3 classes) para o formato novo (head_rewards escalar).

Reaproveita todo peso cujo nome E forma batem (tronco, obs_emb/head_obs,
head_dones) -- so head_rewards.weight/head_rewards.bias ficam com a
inicializacao aleatoria de um WorldModel novo (formato 1xN, sem equivalente
no checkpoint antigo). NAO pre-treina a cabeca de reward -- ela reaprende do
zero no proximo treino/fine-tune.

Aceita tanto checkpoints de traingpt.py (chave "model_state_dict") quanto de
train_online.py (chave "world_model_state_dict") -- sempre salva a saida com
"model_state_dict", que e o que traingpt.py --resume_ckpt e
train_online.py --dynamics_ckpt esperam.

Uso:
    python -m models.dynamics.migrate_reward_head <checkpoint_antigo.pt> <checkpoint_novo.pt>
"""

import sys
import torch

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig


def migrate(src, dst):
    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    if "model_state_dict" in ckpt:
        old_sd = ckpt["model_state_dict"]
    elif "world_model_state_dict" in ckpt:
        old_sd = ckpt["world_model_state_dict"]
    else:
        raise KeyError(f"Checkpoint sem model_state_dict/world_model_state_dict: {list(ckpt.keys())}")

    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    fresh = WorldModel(config)

    new_sd = fresh.state_dict()
    kept, reinit = [], []
    for k in new_sd:
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
            kept.append(k)
        else:
            reinit.append(k)

    torch.save({"model_state_dict": new_sd, "config": config.__dict__}, dst)
    print(f"Migrado: {dst}")
    print(f"Pesos reaproveitados: {len(kept)}")
    print(f"Pesos reinicializados (aleatorios, formato novo): {reinit}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python -m models.dynamics.migrate_reward_head <checkpoint_antigo.pt> <checkpoint_novo.pt>")
        sys.exit(1)
    migrate(sys.argv[1], sys.argv[2])
