from stable_baselines3 import PPO

print("abrindo...")
model = PPO.load("ppo_breakout", device="cpu")
print("carregou")