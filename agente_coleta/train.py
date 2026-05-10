"""
Treinamento PPO para CarRacing-v3 com Frame Skip 4
===================================================
- Observacao: RGB 64x64 (barra preta inferior removida)
- Frame skip: 4  (cada acao repetida por 4 frames)
- Frame stack: 4 (empilha 4 observacoes consecutivas -> 12 canais)

Dependencias:
    pip install gymnasium[box2d] stable-baselines3[extra] shimmy

Uso:
    python train.py --timesteps 2_000_000 --n_envs 8
    python train.py --timesteps 1_000_000 --n_envs 8 --resume
    python train.py --timesteps 1_000_000 --n_envs 8 --resume --resume_path models/ppo_carracing_500000_steps.zip
    python train.py --eval models/best_model.zip
"""

import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class CropBlackBar(gym.ObservationWrapper):
    """
    Remove a barra preta do painel inferior do CarRacing.
    96x96x3 -> 84x96x3
    """
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
    """Repete a acao por `skip` frames, acumula recompensa."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
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
    """
    (H, W, C) uint8  ->  (C, H, W) float32 [0, 1]

    Junta normalizacao + transposicao num wrapper so para evitar
    dependencia do VecTransposeImage, que rejeita float32.
    """
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
    """
    Empilha os ultimos `n_stack` frames no eixo de canais (eixo 0).
    Resultado: (C*n_stack, H, W)

    Usamos este wrapper no lugar do VecFrameStack para garantir que o
    empilhamento aconteca DEPOIS da transposicao e no eixo correto.
    """
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
        # Repete o frame inicial para preencher o buffer no reset
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


class EarlyTermination(gym.Wrapper):
    """
    Trunca episodios ruins (media de reward baixa).
    Usar apenas no treino.
    """
    def __init__(self, env: gym.Env, patience: int = 100, threshold: float = -0.1):
        super().__init__(env)
        self._patience  = patience
        self._threshold = threshold
        self._buf: deque = deque(maxlen=patience)

    def reset(self, **kwargs):
        self._buf.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(reward)
        if len(self._buf) == self._patience and np.mean(self._buf) < self._threshold:
            truncated = True
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Fabrica de ambiente
# ---------------------------------------------------------------------------

def make_env(
    rank: int = 0,
    seed: int = 42,
    frame_skip: int = 4,
    img_size: int = 64,
    crop_rows: int = 12,
    n_stack: int = 4,
    early_termination: bool = True,
):
    """
    Pipeline:
        CarRacing-v3 (96x96x3 uint8)
          -> FrameSkip(4)            12.5 decisoes/s
          -> CropBlackBar(12)        84x96x3
          -> ResizeObservation       64x64x3
          -> NormalizeAndTranspose   (3,64,64) float32
          -> FrameStackChannels(4)   (12,64,64) float32
          -> EarlyTermination        so no treino
          -> Monitor
    """
    def _init():
        env = gym.make("CarRacing-v3", continuous=False)
        env = FrameSkip(env, skip=frame_skip)
        env = CropBlackBar(env, crop_rows=crop_rows)
        env = ResizeObservation(env, shape=(img_size, img_size))
        env = NormalizeAndTranspose(env)
        env = FrameStackChannels(env, n_stack=n_stack)
        if early_termination:
            env = EarlyTermination(env, patience=100, threshold=-0.1)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ---------------------------------------------------------------------------
# Callback de progresso
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, print_freq: int = 50):
        super().__init__(verbose=0)
        self._freq   = print_freq
        self._buf    = deque(maxlen=print_freq)
        self._ep_cnt = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._buf.append(info["episode"]["r"])
                self._ep_cnt += 1
                if self._ep_cnt % self._freq == 0:
                    print(
                        f"[Ep {self._ep_cnt:>6}]  "
                        f"Steps: {self.num_timesteps:>10,}  |  "
                        f"Reward medio ({self._freq} ep): {np.mean(self._buf):>8.2f}"
                    )
        return True


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.log_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # -- Resume ---------------------------------------------------------------
    resume_path = None
    if args.resume:
        if args.resume_path and os.path.exists(args.resume_path):
            resume_path = args.resume_path
        else:
            auto = os.path.join(args.model_dir, "ppo_carracing_final.zip")
            if os.path.exists(auto):
                resume_path = auto
            else:
                print("[AVISO] --resume ativado mas nenhum modelo encontrado. Iniciando do zero.")

    print("=" * 60)
    print(f"  CarRacing PPO  |  Frame skip: {args.frame_skip}")
    print(f"  Obs: RGB {args.img_size}x{args.img_size}  |  Frame stack: {args.frame_stack}")
    print(f"  Envs paralelos : {args.n_envs}")
    print(f"  Timesteps: +{args.timesteps:,}" if resume_path else f"  Timesteps total: {args.timesteps:,}")
    print(f"  Retomando de  : {resume_path}" if resume_path else "  Iniciando do zero")
    print("=" * 60)

    # -- Envs -----------------------------------------------------------------
    env_fn = make_env(
        frame_skip=args.frame_skip, img_size=args.img_size,
        crop_rows=args.crop_rows, n_stack=args.frame_stack,
        early_termination=True,
    )
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)

    eval_fn = make_env(
        frame_skip=args.frame_skip, img_size=args.img_size,
        crop_rows=args.crop_rows, n_stack=args.frame_stack,
        early_termination=False,
    )
    eval_env = make_vec_env(eval_fn, n_envs=1, seed=args.seed + 999)

    # -- Learning rate --------------------------------------------------------
    # -- Learning rate --------------------------------------------------------
    def linear_schedule(initial_value: float, final_value: float = 0.0):
        """
        progress_remaining:
            1.0 -> inicio do treino
            0.0 -> fim do treino
        """
        def func(progress_remaining: float) -> float:
            return final_value + (
                progress_remaining * (initial_value - final_value)
            )
        return func

    lr_fn = linear_schedule(
        initial_value=2.5e-4,
        final_value=5e-5,
    )

    # -- Modelo ---------------------------------------------------------------
    policy_kwargs = dict(
        normalize_images=False,  # imagens ja normalizadas e transpostas
    )

    if resume_path:
        print(f"\nCarregando modelo de: {resume_path}")
        model = PPO.load(
            resume_path,
            env=vec_env,
            device="cuda",
            custom_objects={
                "learning_rate": lr_fn,
                "clip_range":    0.2,
                "n_steps":       512,
                "batch_size":    512,
                "policy_kwargs": policy_kwargs,
            },
            tensorboard_log=args.log_dir,
        )
    else:
        model = PPO(
            policy          = "CnnPolicy",
            env             = vec_env,
            learning_rate   = lr_fn,
            n_steps         = 128,
            batch_size      = 512,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            verbose         = 0,
            device          = "cuda",
            tensorboard_log = args.log_dir,
            seed            = args.seed,
            policy_kwargs   = policy_kwargs,
        )

    # -- Callbacks ------------------------------------------------------------
    ckpt_freq = max(args.timesteps // 20, 10_000)
    eval_freq = max(args.timesteps // 40,  5_000)

    callbacks = [
        ProgressCallback(print_freq=50),
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=args.model_dir,
            name_prefix="ppo_carracing",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.model_dir,
            log_path=args.log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
    ]

    # -- Treino ---------------------------------------------------------------
    print("\nIniciando treinamento...\n")
    model.learn(
        total_timesteps     = args.timesteps,
        callback            = callbacks,
        tb_log_name         = "ppo_carracing_fs4_rgb64",
        reset_num_timesteps = not args.resume,
        progress_bar        = True,
    )

    final_path = os.path.join(args.model_dir, "ppo_carracing_final")
    model.save(final_path)
    print(f"\nModelo final salvo em: {final_path}.zip")

    vec_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------

def evaluate(model_path, args):
    env_fn = make_env(
        frame_skip=args.frame_skip, img_size=args.img_size,
        crop_rows=args.crop_rows, n_stack=args.frame_stack,
        early_termination=False,
    )
    env = DummyVecEnv([env_fn])
    model = PPO.load(model_path, env=env, device="cuda")
    print(f"\nAvaliando: {model_path}  ({args.n_eval_episodes} episodios)\n")

    rewards = []
    for ep in range(1, args.n_eval_episodes + 1):
        obs   = env.reset()
        done  = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward[0]
        rewards.append(total)
        print(f"  Ep {ep:>3}: {total:>8.2f}")

    print(f"\nMedia: {np.mean(rewards):.2f}  |  Std: {np.std(rewards):.2f}  |  Max: {np.max(rewards):.2f}")
    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",       type=int,  default=2_000_000)
    p.add_argument("--n_envs",          type=int,  default=24)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--resume_path",     type=str,  default=None)
    p.add_argument("--frame_skip",      type=int,  default=4)
    p.add_argument("--frame_stack",     type=int,  default=4)
    p.add_argument("--img_size",        type=int,  default=64)
    p.add_argument("--crop_rows",       type=int,  default=12)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--log_dir",         type=str,  default="./logs")
    p.add_argument("--model_dir",       type=str,  default="./models")
    p.add_argument("--eval",            type=str,  default=None)
    p.add_argument("--n_eval_episodes", type=int,  default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args.eval, args)
    else:
        train(args)