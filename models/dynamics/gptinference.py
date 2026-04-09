import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ==============================================================================
# 1. IMPORTS DOS SEUS MODELOS (Ajuste esses caminhos conforme o seu projeto)
# ==============================================================================
# Assumindo que o seu GPT está em models.dynamics.gptdynamics
# e o seu VQ-VAE está em uma pasta semelhante.
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig  # Substitua pelo nome exato da sua classe
from models.encoder.modules import VQVAE      # Descomente e ajuste para o seu VQ-VAE

def load_models(vqvae_path, gpt_path, device):
    print("⏳ Carregando arquiteturas e pesos...")
    
    # IMPORTANTE: Instancie as classes com os mesmos parâmetros do treinamento!
    # Exemplo (ajuste os argumentos para os que você usou):
    vqvae = VQVAE().to(device) 
    # Primeiro criamos o objeto de configuração
    config = WorldModelConfig(vocab_size=512, block_size=320, action_vocab_size=5)
    # Depois passamos ele para o WorldModel
    gpt = WorldModel(config).to(device) 

    # Carrega os dicionários de pesos salvos
    ckpt_vqvae = torch.load(vqvae_path, map_location=device, weights_only=True)
    ckpt_gpt = torch.load(gpt_path, map_location=device, weights_only=True)
    
    # Se você salvou um dicionário (ex: checkpoint_dict no seu traingpt.py), extraia o model_state_dict
    # Se salvou apenas os pesos direto, passe o ckpt_gpt direto.
    gpt_state = ckpt_gpt.get("model_state_dict", ckpt_gpt) 
    vqvae_state = ckpt_vqvae.get("model_state_dict", ckpt_vqvae)

    gpt.load_state_dict(gpt_state)
    vqvae.load_state_dict(vqvae_state)

    # Modo de avaliação (desliga Dropout e BatchNorm)
    gpt.eval()
    vqvae.eval()
    
    print("✅ Modelos carregados com sucesso!")
    return vqvae, gpt

def get_action_from_keyboard(key):
    """
    Mapeia as teclas do teclado para as 5 ações do seu CarRacing:
    0: Nada | 1: Esquerda (A) | 2: Direita (D) | 3: Acelerar (W) | 4: Frear (S)
    """
    if key == ord('w'): return 3
    if key == ord('a'): return 1
    if key == ord('d'): return 2
    if key == ord('s'): return 4
    return 0 # Ação padrão se não apertar nada relevante

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Caminhos dos pesos baseados no seu log
    vqvae_ckpt_path = "models/VQVAE_PONG/ckpt.pt"
    gpt_ckpt_path = "models/GPT_CARRACING/gpt_ckpt.pt"
    
    # 1. Carrega os modelos
    vqvae, gpt = load_models(vqvae_ckpt_path, gpt_ckpt_path, device)
    
    # 2. Prepara a imagem inicial ("A Faísca")
    # Pegue qualquer frame do seu dataset para iniciar o sonho
    start_image_path = "dataset_jogo/frame_000000.png" # MUDE para um arquivo que exista na pasta
    if not os.path.exists(start_image_path):
        raise FileNotFoundError(f"Coloque uma imagem inicial válida em {start_image_path}")

    img_pil = Image.open(start_image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 3. Buffer de Contexto (O GPT precisa lembrar dos últimos N frames)
    # Supondo que sua sequência seja de 5 frames (5 * 64 = 320 tokens)
    frames_per_seq = 5
    tokens_per_frame = 64
    max_seq_len = frames_per_seq * tokens_per_frame

    with torch.no_grad():
        # Transforma a imagem inicial em tokens
        z = vqvae.encoder(img_tensor)
        _, _, initial_tokens = vqvae.vq(z)
        
        # Inicia a lista de tokens do contexto (achata para 1D)
        context_tokens = initial_tokens.flatten().tolist()
        
        print("🚗 Pressione W, A, S, D na janela do OpenCV para dirigir. ESC para sair.")

        # Inicia o contexto de imagens e ações
        context_tokens = initial_tokens.flatten().tolist()
        context_actions = [0] # Ação inicial neutra

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27: # Tecla ESC
                break
            
            action_int = get_action_from_keyboard(key)
            context_actions.append(action_int)

            # O Segredo: Manter até 4 frames no passado (256 tokens) 
            # para que ele tenha espaço para gerar o 5º frame (64 tokens), 
            # respeitando o limite do block_size (320).
            if len(context_tokens) > (4 * 64):
                context_tokens = context_tokens[-(4 * 64):]
            
            # Puxa o número exato de ações correspondente aos frames no contexto
            num_frames_in_context = len(context_tokens) // 64
            actions_to_pass = context_actions[-num_frames_in_context:]

            x_input = torch.tensor([context_tokens], dtype=torch.long).to(device)
            action_tensor = torch.tensor([actions_to_pass], dtype=torch.long).to(device)

            # O GPT gera olhando para todo o contexto e histórico de ações
            next_frame_tokens = gpt.generate(x_input, action_tensor, num_tokens_to_generate=64, temperature=0.5, top_k=50)
            
            # Pega APENAS os 64 tokens recém-gerados (o novo frame T+1)
            new_tokens = next_frame_tokens[0, -64:].tolist()
            context_tokens.extend(new_tokens)

            # Usa o VQVAE para decodificar apenas os tokens do novo frame
            novo_frame_tensor = torch.tensor([new_tokens], dtype=torch.long).to(device)
            predicted_img_tensor = vqvae.decode_indices(novo_frame_tensor)
            
            # Converte para OpenCV e mostra na tela
            img_np = predicted_img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1) 
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            img_show = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("CarRacing - World Model Imagination", img_show)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()