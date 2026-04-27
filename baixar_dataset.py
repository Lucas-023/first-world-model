from datasets import load_dataset
import numpy as np
import cv2
import os

SAVE_DIR = "dataset_breakout_rl"
os.makedirs(SAVE_DIR, exist_ok=True)

MIN_REWARD = 20
MAX_EPISODES = 500  # breakout só tem 36 episódios, pong tem 303

# Baixa o dataset de pong que tem mais episódios e rewards bons
print("📦 Baixando dataset...")
ds = load_dataset("jat-project/jat-dataset", "atari-pong", split="train")
print(f"✅ {len(ds)} episódios carregados!")
print("Colunas:", ds.column_names)
print("Exemplo:", {k: type(v) for k, v in ds[0].items()})