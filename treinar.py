import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

gym.register_envs(ale_py)

NUM_ENVS = 8

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        return env
    return _init

if __name__ == "__main__":
    print("🎮 Treinando PPO com RTX 3060 Ti...")

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",        # usa GPU
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.01,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(
        total_timesteps=10_000_000,
        log_interval=50,
    )

    model.save("ppo_breakout")
    env.close()
    print("✅ Modelo salvo!")