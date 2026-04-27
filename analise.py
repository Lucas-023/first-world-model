import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

def make_env():
    def _init():
        env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
        env = AtariWrapper(env)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

obs = env.reset()
print("obs shape:", obs.shape)
print("obs[0] shape:", obs[0].shape)
print("obs[0, 0] min/max:", obs[0, 0].min(), obs[0, 0].max())
print("obs[0, -1] min/max:", obs[0, -1].min(), obs[0, -1].max())

# testa alguns steps
for _ in range(100):
    obs, _, _, _ = env.step(np.array([1]))

print("\nApós 100 steps:")
print("obs shape:", obs.shape)
print("obs[0, :, :, 0] min/max:", obs[0, :, :, 0].min(), obs[0, :, :, 0].max())
print("obs[0, :, :, -1] min/max:", obs[0, :, :, -1].min(), obs[0, :, :, -1].max())

import matplotlib.pyplot as plt
plt.imshow(obs[0, :, :, -1], cmap='gray')
plt.savefig("teste_frame.png")
print("Salvo em teste_frame.png")
env.close()