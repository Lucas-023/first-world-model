import os
import gc
import random
import numpy as np

# =========================
# CARREGA PPO PRIMEIRO
# =========================
from stable_baselines3 import PPO

print("📦 Carregando modelo PPO...")
model = PPO.load("ppo_breakout", device="cpu")
print("✅ Modelo carregado!\n")

# =========================
# RESTO DOS IMPORTS DEPOIS
# =========================
import cv2
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


# =========================
# CONFIG
# =========================
SAVE_DIR = "dataset_carracing_rl"
MIN_REWARD = 75
MAX_EPISODES = 2000
MAX_STEPS = 2000
EPSILON = 0.05

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# PREPROCESS
# =========================
def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return gray


# =========================
# ENV
# =========================
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = Monitor(env)
        return env
    return _init


# =========================
# MAIN
# =========================
def main():
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    ep_total = 0
    ep_saved = 0
    ep_discarded = 0

    while ep_saved < MAX_EPISODES:
        obs = env.reset()

        frames = []
        actions = []
        rewards = []
        dones = []
        truncateds = []

        total_reward = 0.0

        for _ in range(MAX_STEPS):
            if random.random() < EPSILON:
                action = np.array([env.action_space.sample()])
            else:
                action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, done, info = env.step(action)

            reward = float(reward[0])
            done = bool(done[0])

            rgb_last = obs[0, :, :, -3:]
            gray = preprocess(rgb_last)

            frames.append(gray)
            actions.append(action[0].astype(np.float32))
            rewards.append(reward)
            dones.append(done)
            truncateds.append(bool(info[0].get("TimeLimit.truncated", False)))

            total_reward += reward
            obs = next_obs

            if done:
                break

        ep_total += 1

        if total_reward < MIN_REWARD:
            ep_discarded += 1
            if ep_discarded % 10 == 0:
                print(f"[{ep_discarded} descartados | último reward={total_reward:.1f}]")

            del frames, actions, rewards, dones, truncateds
            gc.collect()
            continue

        filename = os.path.join(SAVE_DIR, f"ep_{ep_saved:05d}.npz")

        np.savez_compressed(
            filename,
            obs=np.array(frames, dtype=np.uint8),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=bool),
            truncated=np.array(truncateds, dtype=bool),
            episode_reward=np.float32(total_reward),
        )

        del frames, actions, rewards, dones, truncateds
        gc.collect()

        ep_saved += 1
        print(f"EP {ep_total} SALVO [{ep_saved}/{MAX_EPISODES}] | reward={total_reward:.1f}")

    env.close()
    print(f"\n✅ Dataset completo! {ep_saved} episódios salvos em '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()