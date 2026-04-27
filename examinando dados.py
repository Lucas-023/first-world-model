import numpy as np
import glob
import os

DATASET_DIR = "dataset_breakout_rl"

npz_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
print(f"Total de arquivos .npz: {len(npz_files)}\n")

if len(npz_files) == 0:
    print("Nenhum arquivo encontrado!")
    exit()

rewards_total = []
frames_total = []

for npz_path in npz_files:
    data = np.load(npz_path)
    total_reward = data['rewards'].sum()
    n_frames = len(data['obs'])
    rewards_total.append(total_reward)
    frames_total.append(n_frames)
    print(f"{os.path.basename(npz_path)} | frames={n_frames} | reward={total_reward:.1f} | obs_shape={data['obs'][0].shape}")

print(f"\n--- RESUMO ({len(npz_files)} episódios) ---")
print(f"Reward médio:      {np.mean(rewards_total):.1f}")
print(f"Reward max:        {np.max(rewards_total):.1f}")
print(f"Reward min:        {np.min(rewards_total):.1f}")
print(f"Frames médio:      {np.mean(frames_total):.0f}")
print(f"Frames total:      {sum(frames_total):,}")
print(f"Tamanho estimado:  {sum(frames_total) * 64 * 64 / 1024 / 1024:.1f} MB")