import gymnasium as gym
import ale_py
import cv2
import numpy as np
import os
import csv

gym.register_envs(ale_py)

pasta_dataset = "dataset_jogo"
os.makedirs(pasta_dataset, exist_ok=True)

env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
estado, info = env.reset()

frames_salvos = 0
meta_frames = 100000

print("🎮 Iniciando o Pong e coletando dados...")

# Abre o arquivo CSV para salvar os metadados
csv_path = os.path.join(pasta_dataset, "metadata.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Novo cabeçalho!
    writer.writerow(["frame", "acao", "recompensa", "terminado"])

    while frames_salvos < meta_frames:
        acao = env.action_space.sample()
        estado, recompensa, terminado, truncado, info = env.step(acao)
        
        # Reduzindo para 64x64 e salvando a imagem
        frame_64 = cv2.resize(estado, (64, 64))
        frame_bgr = cv2.cvtColor(frame_64, cv2.COLOR_RGB2BGR)
        nome_arquivo = f"frame_{frames_salvos:06d}.png"
        cv2.imwrite(os.path.join(pasta_dataset, nome_arquivo), frame_bgr)
        
        # Salvando no CSV (Convertendo boolean 'terminado' para 1 ou 0)
        writer.writerow([nome_arquivo, acao, recompensa, int(terminado)])
        
        frames_salvos += 1
        if frames_salvos % 200 == 0:
            print(f"Progresso: {frames_salvos} / {meta_frames} frames salvos.")

        if terminado or truncado:
            estado, info = env.reset()

print("✅ Coleta finalizada! Imagens e metadata.csv gerados.")