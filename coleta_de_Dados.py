import gymnasium as gym
import numpy as np
import cv2
import os

# Configurações
NUM_EPISODES = 10000         # Quantos vídeos/trajetórias coletar
MAX_STEPS = 300           # Passos por episódio
ACTION_REPEAT = 3         # Quantos frames pular repetindo a mesma ação
SAVE_DIR = "dataset_carracing"
os.makedirs(SAVE_DIR, exist_ok=True)

class BrownianPolicy:
    """Gera ações com transições suaves (Random Walk/Brownian Motion)"""
    def __init__(self):
        self.action = np.array([0.0, 0.0, 0.0]) # Volante, Acelerador, Freio

    def get_action(self):
        # Adiciona um pequeno ruído à ação anterior
        noise = np.random.randn(3) * 0.1
        self.action += noise
        
        # Limita os valores aos padrões do CarRacing
        self.action[0] = np.clip(self.action[0], -1.0, 1.0) # Volante: -1 a 1
        self.action[1] = np.clip(self.action[1], 0.0, 1.0)  # Acelerador: 0 a 1
        self.action[2] = np.clip(self.action[2] - 0.05, 0.0, 0.2) # Freio: tende a zero para o carro andar
        
        return self.action.copy()

def process_frame(obs):
    """Corta o painel e redimensiona para 64x64"""
    # O frame original é 96x96. O painel inferior ocupa os últimos 12 pixels.
    cropped_obs = obs[:84, :, :] 
    # Redimensiona para 64x64 (padrão World Models)
    resized_obs = cv2.resize(cropped_obs, (64, 64), interpolation=cv2.INTER_AREA)
    return resized_obs

def main():
    # Usando render_mode="rgb_array" para capturar os pixels sem abrir janela
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        policy = BrownianPolicy()
        
        frames = []
        actions = []
        rewards = [] # NOVA LISTA: Para guardar as recompensas
        dones = []   # NOVA LISTA: Para guardar se o jogo acabou
        
        # O carro começa parado e a pista dá um "zoom" inicial. 
        # Vamos pular os primeiros 50 passos do ambiente (apenas acelerando levemente).
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
            
        for step in range(MAX_STEPS):
            action = policy.get_action()
            
            # Action Repeat (Pular frames)
            reward_sum = 0
            is_done = False # Variável para capturar o fim de jogo
            
            for _ in range(ACTION_REPEAT):
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                if terminated or truncated:
                    is_done = True
                    break
            
            processed_img = process_frame(obs)
            
            # Salvando TUDO que o Transformer vai precisar
            frames.append(processed_img)
            actions.append(action)
            rewards.append(reward_sum) # Salva a recompensa acumulada nos 3 frames
            dones.append(is_done)      # Salva True ou False
            
            if is_done:
                break
                
        # Salva o episódio como um array numpy compactado com as 4 colunas!
        filename = os.path.join(SAVE_DIR, f"episode_{ep:04d}.npz")
        np.savez_compressed(
            filename, 
            obs=np.array(frames, dtype=np.uint8), 
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32), # Adicionado
            dones=np.array(dones, dtype=bool)            # Adicionado
        )
        print(f"Episódio {ep} salvo: {len(frames)} frames.")

    env.close()

if __name__ == "__main__":
    main()