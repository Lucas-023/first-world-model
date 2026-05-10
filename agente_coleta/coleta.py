"""
Coleta de dados para World Model — CarRacing-v3
===============================================
Roda o agente PPO treinado e salva episodios completos em disco.

Cada episodio e salvo como um arquivo .npz contendo:
    obs     : (T, 3, 64, 64) float32  — frames RGB normalizados [0,1]
    actions : (T,)            int      — acao discreta tomada
    rewards : (T,)            float32  — recompensa recebida
    dones   : (T,)            bool     — terminal/truncado

Os arquivos ficam em:
    data_dir/
        episode_0000.npz
        episode_0001.npz
        ...

Uso:
    python collect.py --model_path models/best_model.zip --n_episodes 500
    python collect.py --model_path models/best_model.zip --n_episodes 200 --epsilon 0.05
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces
from gymnasium.wrappers import ResizeObservation
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Wrappers — identicos ao train.py para garantir consistencia
# ---------------------------------------------------------------------------

class CropBlackBar(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, crop_rows: int = 12):
        super().__init__(env)
        self._crop_rows = crop_rows
        h, w, c = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(h - crop_rows, w, c),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return obs[: -self._crop_rows, :, :]


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
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


class NormalizeAndTranspose(gym.ObservationWrapper):
    """(H, W, C) uint8  ->  (C, H, W) float32 [0, 1]"""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(c, h, w),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0


class FrameStackChannels(gym.Wrapper):
    """Empilha n_stack frames no eixo de canais -> (C*n_stack, H, W)"""
    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self._n_stack = n_stack
        c, h, w = self.observation_space.shape
        self._frames: deque = deque(maxlen=n_stack)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(c * n_stack, h, w),
            dtype=np.float32,
        )

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n_stack):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Criacao do ambiente de coleta
# Diferente do treino: sem EarlyTermination (queremos episodios completos)
# ---------------------------------------------------------------------------

def make_collection_env(
    frame_skip: int = 4,
    img_size: int = 64,
    crop_rows: int = 12,
    n_stack: int = 4,
    seed: int = 0,
):
    env = gym.make("CarRacing-v3", continuous=False)
    env = FrameSkip(env, skip=frame_skip)
    env = CropBlackBar(env, crop_rows=crop_rows)
    env = ResizeObservation(env, shape=(img_size, img_size))
    env = NormalizeAndTranspose(env)
    env = FrameStackChannels(env, n_stack=n_stack)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------

def collect(args):
    os.makedirs(args.data_dir, exist_ok=True)

    # Verifica quantos episodios ja existem (para retomar coleta)
    existing = [f for f in os.listdir(args.data_dir) if f.endswith(".npz")]
    start_ep = len(existing)
    if start_ep > 0:
        print(f"[INFO] {start_ep} episodios ja existem em '{args.data_dir}'. Continuando a partir do ep {start_ep}.")

    env = make_collection_env(
        frame_skip=args.frame_skip,
        img_size=args.img_size,
        crop_rows=args.crop_rows,
        n_stack=args.frame_stack,
        seed=args.seed,
    )

    # Carrega o agente — sem env pois vamos chamar predict() manualmente
    model = PPO.load(args.model_path, device="cuda")

    print(f"\nColetando {args.n_episodes} episodios -> '{args.data_dir}'")
    print(f"Epsilon (exploracao aleatoria): {args.epsilon:.0%}")
    print(f"Frame skip: {args.frame_skip}  |  Frame stack: {args.frame_stack}\n")

    total_steps = 0
    rewards_log = []

    for ep in range(args.n_episodes):
        ep_idx = start_ep + ep

        # Buffers do episodio
        # Salvamos o frame ATUAL (sem stack) para o world model:
        # shape (C, H, W) = (3, 64, 64) — mais leve e mais util
        obs_buf     = []   # frames individuais (3, 64, 64)
        action_buf  = []
        reward_buf  = []
        done_buf    = []

        stacked_obs, _ = env.reset(seed=args.seed + ep_idx)

        # Extrai o frame atual (ultimos C canais do stack)
        # O stack e (C*n_stack, H, W); os ultimos C canais sao o frame mais recente
        C = 3  # canais RGB
        done = False
        ep_reward = 0.0

        while not done:
            # Frame atual = ultimos 3 canais do stack
            current_frame = stacked_obs[-C:, :, :]   # (3, 64, 64)

            # Acao: epsilon-greedy sobre a politica do agente
            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                # predict() espera (1, obs_shape) — adiciona batch dim
                action, _ = model.predict(
                    stacked_obs[np.newaxis],
                    deterministic=True,
                )
                action = int(action[0])

            next_stacked, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_buf.append(current_frame)
            action_buf.append(action)
            reward_buf.append(reward)
            done_buf.append(done)

            stacked_obs = next_stacked
            ep_reward  += reward

        # Salva episodio
        ep_steps = len(action_buf)
        total_steps += ep_steps
        rewards_log.append(ep_reward)

        save_path = os.path.join(args.data_dir, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(
            save_path,
            obs     = np.array(obs_buf,    dtype=np.float32),   # (T, 3, 64, 64)
            actions = np.array(action_buf, dtype=np.int32),     # (T,)
            rewards = np.array(reward_buf, dtype=np.float32),   # (T,)
            dones   = np.array(done_buf,   dtype=bool),         # (T,)
        )

        print(
            f"[Ep {ep_idx:04d}]  "
            f"Steps: {ep_steps:>4}  |  "
            f"Reward: {ep_reward:>8.2f}  |  "
            f"Total steps: {total_steps:>8,}  |  "
            f"Salvo: {save_path}"
        )

    env.close()

    print(f"\n{'='*60}")
    print(f"  Coleta concluida!")
    print(f"  Episodios coletados : {args.n_episodes}")
    print(f"  Total de steps      : {total_steps:,}")
    print(f"  Reward medio        : {np.mean(rewards_log):.2f}")
    print(f"  Reward max          : {np.max(rewards_log):.2f}")
    print(f"  Reward min          : {np.min(rewards_log):.2f}")
    print(f"  Dados salvos em     : {args.data_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Inspecao rapida do dataset
# ---------------------------------------------------------------------------

def inspect(data_dir: str):
    """Mostra um resumo dos dados ja coletados."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    if not files:
        print("Nenhum episodio encontrado.")
        return

    total_steps = 0
    rewards = []
    for f in files:
        d = np.load(os.path.join(data_dir, f))
        total_steps += len(d["actions"])
        rewards.append(d["rewards"].sum())

    first = np.load(os.path.join(data_dir, files[0]))
    print(f"\n{'='*60}")
    print(f"  Dataset: {data_dir}")
    print(f"  Episodios  : {len(files)}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Reward medio: {np.mean(rewards):.2f}  |  Std: {np.std(rewards):.2f}")
    print(f"  Reward max  : {np.max(rewards):.2f}  |  Min: {np.min(rewards):.2f}")
    print(f"\n  Shape por episodio (exemplo '{files[0]}'):")
    for k, v in first.items():
        print(f"    {k:10s}: {v.shape}  dtype={v.dtype}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Coleta de dados para World Model")

    p.add_argument("--model_path",  type=str, required=True,
                   help="Caminho para o modelo PPO (.zip)")
    p.add_argument("--n_episodes",  type=int, default=500,
                   help="Numero de episodios a coletar (default: 500)")
    p.add_argument("--data_dir",    type=str, default="./dataset",
                   help="Pasta onde salvar os episodios (default: ./dataset)")
    p.add_argument("--epsilon",     type=float, default=0.02,
                   help="Fracao de acoes aleatorias para diversidade (default: 0.02)")

    # Devem ser identicos ao treino
    p.add_argument("--frame_skip",  type=int, default=4)
    p.add_argument("--frame_stack", type=int, default=4)
    p.add_argument("--img_size",    type=int, default=64)
    p.add_argument("--crop_rows",   type=int, default=12)
    p.add_argument("--seed",        type=int, default=0)

    # Inspecao
    p.add_argument("--inspect",     action="store_true",
                   help="So inspeciona o dataset existente, sem coletar")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.inspect:
        inspect(args.data_dir)
    else:
        collect(args)