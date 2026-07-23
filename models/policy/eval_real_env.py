"""
Avalia a politica treinada via imaginacao (train_dream.py) no ambiente REAL
CarRacing-v3. Nada aqui passa pelo World Model gerando frames -- e o teste
que prova (ou nao) que o aprendizado dentro do sonho transferiu pra pratica.

Pipeline por passo: frame real -> VQ-VAE (encode_indices) -> token entra na
janela deslizante -> WorldModel.encode_state (mesma leitura usada no treino)
-> ActorCritic.act -> acao aplicada de verdade no gym -> reward real (continuo,
nao a classe {-1,0,+1} que o world model preve).

Uso:
    python -m models.policy.eval_real_env --n_episodes 10
    python -m models.policy.eval_real_env --n_episodes 3 --deterministic
    python -m models.policy.eval_real_env --n_episodes 1 --render   (so com display local)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import ResizeObservation

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.modules import VQVAE
from models.policy.modules import ActorCritic


class CropBlackBar(gym.ObservationWrapper):
    """Identica a agente_coleta/coleta.py -- remove a faixa preta do HUD."""
    def __init__(self, env, crop_rows=12):
        super().__init__(env)
        self._crop_rows = crop_rows
        h, w, c = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(h - crop_rows, w, c), dtype=np.uint8)

    def observation(self, obs):
        return obs[: -self._crop_rows, :, :]


class FrameSkip(gym.Wrapper):
    """Identica a agente_coleta/coleta.py -- o dataset foi coletado com esse
    mesmo skip, entao cada "frame" que o world model conhece corresponde a
    `skip` passos reais agregados, nao a 1 passo bruto do gym."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = info = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_eval_env(frame_skip, img_size, crop_rows, render):
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human" if render else None)
    env = FrameSkip(env, skip=frame_skip)
    env = CropBlackBar(env, crop_rows=crop_rows)
    env = ResizeObservation(env, shape=(img_size, img_size))
    return env


def frame_to_tensor(frame, device):
    # (H,W,3) uint8 -> (1,3,H,W) float32 [0,1] -- mesma normalizacao de NormalizeAndTranspose
    x = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return x.to(device)


def load_world_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    model = WorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


@torch.no_grad()
def run_episode(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, deterministic, seed):
    obs, _ = env.reset(seed=seed)

    # pula a introducao (zoom-in) do CarRacing -- mesmo skip_frames do extract_tokens.py,
    # usado no dataset real que treinou o world model
    for _ in range(skip_frames):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            return 0.0, 0

    # monta a janela de contexto inicial andando com acao "nada" (0), gerando
    # historico sequencial real em vez de duplicar um unico frame
    obs_ctx_list, act_ctx_list = [], []
    for _ in range(context_len):
        tok = vqvae.encode_indices(frame_to_tensor(obs, device)).view(1, -1)  # (1,64)
        obs_ctx_list.append(tok)
        act_ctx_list.append(torch.zeros(1, dtype=torch.long, device=device))
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            return 0.0, 0

    obs_ctx = torch.stack(obs_ctx_list, dim=1)   # (1,T,64)
    act_ctx = torch.stack(act_ctx_list, dim=1)   # (1,T)

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        state, _, _ = world_model.encode_state(obs_ctx, act_ctx)
        if deterministic:
            logits, _ = actor_critic.forward(state)
            action = logits.argmax(dim=-1)
        else:
            action, _, _, _ = actor_critic.act(state)

        a = int(action.item())
        obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        next_tok = vqvae.encode_indices(frame_to_tensor(obs, device)).view(1, -1)
        obs_ctx = torch.cat([obs_ctx[:, 1:], next_tok.unsqueeze(1)], dim=1)
        act_ctx = torch.cat([act_ctx[:, 1:], action.unsqueeze(1)], dim=1)

    return total_reward, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_ckpt", type=str, default="models/policy/policy_best.pt")
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt.pt")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_rows", type=int, default=12)
    p.add_argument("--skip_frames", type=int, default=12, help="frames de introducao (zoom) ignorados, igual extract_tokens.py")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--deterministic", action="store_true", help="usa argmax da politica em vez de amostrar da categorica")
    p.add_argument("--render", action="store_true", help="so funciona com display local, nao numa VM headless")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device

    world_model, wm_config = load_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado (context_len={wm_config.context_len}, congelado).")

    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=wm_config.obs_vocab_size).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"])
    vqvae.eval()
    for p_ in vqvae.parameters():
        p_.requires_grad_(False)
    print("VQ-VAE carregado.")

    actor_critic = ActorCritic(
        state_dim=wm_config.n_embd, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)
    ckpt = torch.load(args.policy_ckpt, map_location=device, weights_only=True)
    actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
    actor_critic.eval()
    print(
        f"Politica carregada de {args.policy_ckpt} "
        f"(update {ckpt.get('update', '?')}, best_reward imaginado {ckpt.get('best_reward', float('nan')):.4f})"
    )

    env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, args.render)

    print(f"\nRodando {args.n_episodes} episodios no CarRacing-v3 REAL "
          f"({'deterministico (argmax)' if args.deterministic else 'estocastico (sample)'})...\n")

    rewards, steps_log = [], []
    for ep in range(args.n_episodes):
        reward, steps = run_episode(
            env, world_model, vqvae, actor_critic, wm_config.context_len, device,
            args.skip_frames, args.deterministic, seed=args.seed + ep,
        )
        rewards.append(reward)
        steps_log.append(steps)
        print(f"[Ep {ep:>3}] reward: {reward:>8.2f} | steps: {steps:>4}")

    env.close()

    rewards = np.array(rewards)
    print(f"\n{'='*60}")
    print(f"  Reward medio: {rewards.mean():.2f}  |  Std: {rewards.std():.2f}")
    print(f"  Reward max  : {rewards.max():.2f}  |  Min: {rewards.min():.2f}")
    print(f"  Steps medio : {np.mean(steps_log):.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
