import gymnasium as gym
import ale_py # Importando o emulador do Atari
import cv2
import numpy as np
import os

# Mágica necessária no Gymnasium >= 1.0.0 para reconhecer a família "ALE"
gym.register_envs(ale_py)

pasta_dataset = "dataset_jogo"
os.makedirs(pasta_dataset, exist_ok=True)

# Agora sim, iniciando o Pong!
env = gym.make("ALE/Pong-v5")
estado, info = env.reset()

frames_salvos = 0
meta_frames = 100000

print("Iniciando o Pong e coletando dados...")

while frames_salvos < meta_frames:
    acao = env.action_space.sample()
    estado, recompensa, terminado, truncado, info = env.step(acao)
    
    # Reduzindo para 64x64
    frame_64 = cv2.resize(estado, (64, 64))
    
    # Salvando a imagem
    frame_bgr = cv2.cvtColor(frame_64, cv2.COLOR_RGB2BGR)
    nome_arquivo = f"frame_{frames_salvos:06d}.png"
    cv2.imwrite(os.path.join(pasta_dataset, nome_arquivo), frame_bgr)
    
    frames_salvos += 1
    if frames_salvos % 200 == 0:
        print(f"Progresso: {frames_salvos} / {meta_frames} frames salvos.")

    if terminado or truncado:
        estado, info = env.reset()

env.close()
print("Sucesso absoluto! Verifique a pasta 'dataset_jogo'.")