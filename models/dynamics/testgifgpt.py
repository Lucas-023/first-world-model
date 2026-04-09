import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# IMPORTS DAS SUAS ARQUITETURAS
# Ajuste se o caminho das suas pastas for diferente
# ==============================================================================
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.modules import VQVAE

def save_tensor_sequence_as_gif(tensor_list, filename="resultado_gpt.gif", duration=100):
    """Transforma uma lista de tensores gerados em um GIF animado."""
    frames_pil = []
    for img_tensor in tensor_list:
        # Desnormaliza se o tensor estiver entre [-1, 1]
        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1.0) / 2.0
            
        # Garante limites seguros e converte para numpy
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0).detach().cpu()
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(-1)
            
        frames_pil.append(Image.fromarray(img_np))
        
    frames_pil[0].save(
        filename, save_all=True, append_images=frames_pil[1:], duration=duration, loop=0
    )
    print(f"\n🎬 GIF salvo com sucesso: {filename}")

def load_models(vqvae_path, gpt_path, device):
    print("⏳ Carregando arquiteturas e pesos...")
    
    # 1. Carrega VQ-VAE
    vqvae = VQVAE().to(device) 
    
    # 2. Carrega GPT com a sua configuração do CarRacing
    config = WorldModelConfig(vocab_size=512, block_size=320, action_vocab_size=5)
    gpt = WorldModel(config).to(device) 

    # 3. Carrega os checkpoints
    ckpt_vqvae = torch.load(vqvae_path, map_location=device, weights_only=True)
    ckpt_gpt = torch.load(gpt_path, map_location=device, weights_only=True)
    
    # 4. Extrai os state_dicts de forma segura (caso tenha salvo dicionários inteiros)
    gpt_state = ckpt_gpt.get("model_state_dict", ckpt_gpt) 
    vqvae_state = ckpt_vqvae.get("model_state_dict", ckpt_vqvae)

    gpt.load_state_dict(gpt_state)
    vqvae.load_state_dict(vqvae_state)

    gpt.eval()
    vqvae.eval()
    
    print("✅ Modelos carregados com sucesso!")
    return vqvae, gpt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Rodando em: {device.upper()}")
    
    # ==========================================================================
    # CAMINHOS DOS ARQUIVOS (Ajuste se necessário)
    # ==========================================================================
    vqvae_ckpt_path = "models/VQVAE_PONG/ckpt.pt"
    gpt_ckpt_path = "models/GPT_CARRACING/gpt_ckpt.pt"
    dataset_folder = "dataset_jogo" 
    
    vqvae, gpt = load_models(vqvae_ckpt_path, gpt_ckpt_path, device)

    # ==========================================================================
    # LER METADATA DO DATASET
    # ==========================================================================
    csv_path = os.path.join(dataset_folder, "metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado!")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Pula o cabeçalho
        rows = list(reader)

    # ==========================================================================
    # CONFIGURAÇÃO DO TESTE
    # ==========================================================================
    # Pula os primeiros 150 frames (animação de zoom da câmera no CarRacing)
    start_index = 150 
    
    seed_frames = 4         # Dá 4 frames reais de "memória" inicial
    frames_to_generate = 50 # Força o GPT a prever os próximos 50 frames

    context_tokens = []
    context_actions = []
    generated_images = []

    # ==========================================================================
    # 1. PREPARAR A MEMÓRIA INICIAL (SEED)
    # ==========================================================================
    print(f"\n🧠 Preparando memória inicial a partir do frame {start_index}...")
    for i in range(seed_frames):
        # Lendo a estrutura: img_name, action, reward, done
        img_name, action, _, _ = rows[start_index + i] 
        img_path = os.path.join(dataset_folder, img_name)
        
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = vqvae.encoder(img_tensor)
            _, _, indices = vqvae.vq(z)
            context_tokens.extend(indices.flatten().tolist())
            
        context_actions.append(int(float(action)))
        
        # Salva os frames originais no GIF para termos referência visual
        generated_images.append(img_tensor.squeeze(0)) 

    # ==========================================================================
    # 2. LOOP DE GERAÇÃO AUTORREGRESSIVA (IMAGINAÇÃO DO GPT)
    # ==========================================================================
    print(f"\n🚀 Iniciando imaginação do GPT ({frames_to_generate} frames)...")
    for step in tqdm(range(frames_to_generate)):
        
        # Pega a ação REAL do CSV que o jogador tomou neste exato momento futuro
        future_action = int(float(rows[start_index + seed_frames + step][1]))
        context_actions.append(future_action)
        
        # Mantém apenas os últimos 4 frames no contexto (4 * 64 = 256 tokens)
        if len(context_tokens) > (4 * 64):
            context_tokens = context_tokens[-(4 * 64):]
            
        num_frames_in_context = len(context_tokens) // 64
        actions_to_pass = context_actions[-num_frames_in_context:]

        x_input = torch.tensor([context_tokens], dtype=torch.long).to(device)
        a_input = torch.tensor([actions_to_pass], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Usando os mesmos parâmetros de amostragem do seu inference.py original
            next_tokens = gpt.generate(
                x_input, 
                a_input, 
                num_tokens_to_generate=64, 
                temperature=1.8, 
                top_k=50
            )
            
            # Extrai apenas os 64 tokens gerados (o frame T+1)
            new_frame_tokens = next_tokens[0, -64:].tolist()
            
        # Adiciona o novo frame à memória para o próximo loop
        context_tokens.extend(new_frame_tokens)
        
        # Decodifica os tokens imaginados de volta para imagem usando o VQVAE
        new_frame_tensor = torch.tensor([new_frame_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            pred_img = vqvae.decode_indices(new_frame_tensor)
            
        generated_images.append(pred_img.squeeze(0))

    # ==========================================================================
    # 3. SALVAR RESULTADO
    # ==========================================================================
    save_tensor_sequence_as_gif(generated_images, "teste_isolado_gpt_carracing.gif", duration=100)

if __name__ == '__main__':
    main()