import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import random
import re
from torchvision import transforms

# Importa os módulos que criamos
from models.encoder.modules import VQVAE
from models.dynamics.modules import GPTDynamicsModel

# Configurações
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
VOCAB_SIZE = 512
D_MODEL = 256
SEQ_LEN = 320  # Contexto de 5 frames (5 * 64 tokens)
NUM_ACTIONS = 6 # Os 6 botões do Atari

def extract_number(filename):
    s = re.findall(r'\d+', filename)
    return int(s[0]) if s else -1

@torch.no_grad()
def load_models(vqvae_path, dynamics_path):
    print("👁️  Carregando VQ-VAE (Os olhos)...")
    vqvae = VQVAE(in_channels=3, latent_dim=LATENT_DIM, num_embeddings=VOCAB_SIZE).to(DEVICE)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=DEVICE, weights_only=False)["model_state_dict"])
    vqvae.eval()

    print("🧠 Carregando Transformer (O cérebro)...")
    # NOVO: Precisamos avisar que ele tem 6 botões!
    dynamics = GPTDynamicsModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_actions=NUM_ACTIONS).to(DEVICE)
    dynamics.load_state_dict(torch.load(dynamics_path, map_location=DEVICE, weights_only=False)["model_state_dict"])
    dynamics.eval()
    
    return vqvae, dynamics

@torch.no_grad()
def get_priming_sequence(dataset_path, num_frames_priming=5):
    images = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    images.sort(key=extract_number)
    
    start_idx = random.randint(0, len(images) - num_frames_priming - 1)
    sampled_files = images[start_idx : start_idx + num_frames_priming]
    
    print(f"🎬 Iniciando com frames reais: {sampled_files[0]} até {sampled_files[-1]}")
    
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    tensors = []
    for img_name in sampled_files:
        img = Image.open(os.path.join(dataset_path, img_name)).convert('RGB')
        tensors.append(preprocess(img))
    
    return torch.stack(tensors).to(DEVICE)

@torch.no_grad()
def generate_dream(vqvae, dynamics, priming_images, frames_to_generate=30):
    _, _, indices = vqvae(priming_images)
    current_sequence = indices.view(-1)
    
    tokens_to_generate = frames_to_generate * 64
    
    # O PLANO DE JOYSTICK: 2 = Cima
    total_frames = 5 + frames_to_generate
    joystick_plan = [2] * total_frames 
    plan_tensor = torch.tensor(joystick_plan, dtype=torch.long).to(DEVICE)
    
    print(f"🔮 Imaginando {frames_to_generate} frames de futuro com Top-K...")
    pbar = tqdm(range(tokens_to_generate))
    
    for i in pbar:
        total_len = len(current_sequence)
        
        # --- FIX DO BUG ESPACIAL: Pular frames inteiros ---
        if total_len <= SEQ_LEN:
            start_token_idx = 0
            start_frame_idx = 0
        else:
            # A Mágica: Garante que o recorte COMECE SEMPRE no início de um frame (múltiplo de 64)
            frames_to_skip = (total_len - SEQ_LEN + 63) // 64
            start_token_idx = frames_to_skip * 64
            start_frame_idx = frames_to_skip
        # --------------------------------------------------
            
        context = current_sequence[start_token_idx:].unsqueeze(0) 
        
        # Recortando as ações corretas
        num_frames_in_context = (context.size(1) + 63) // 64
        current_actions = plan_tensor[start_frame_idx : start_frame_idx + num_frames_in_context].unsqueeze(0)
        
        logits = dynamics(context, actions=current_actions)
        next_token_logits = logits[0, -1, :]
        
        # Pode usar Temperatura 0.8 e Top-K 10 (O modelo treinou bem, não precisa ser tão rígido)
        temperature = 0.5 
        logits_scaled = next_token_logits / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        
        top_k = 5
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[sampled_idx]
        
        current_sequence = torch.cat([current_sequence, next_token], dim=0)
        
    return current_sequence

@torch.no_grad()
def save_dream_gif(tokens, vqvae, filename="dream.gif"):
    print("🎨 Renderizando GIF final...")
    
    num_total_tokens = tokens.size(0)
    num_frames = num_total_tokens // 64
    
    valid_tokens = num_frames * 64
    tokens = tokens[:valid_tokens]
    
    indices_2d = tokens.view(num_frames, 8, 8)
    reconstructed_frames = vqvae.decode_indices(indices_2d)
    
    to_pil = transforms.ToPILImage()
    pil_images = []
    for i in range(reconstructed_frames.size(0)):
        frame = reconstructed_frames[i].cpu().clamp(0, 1)
        pil_images.append(to_pil(frame))
    
    pil_images[0].save(
        filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=50,
        loop=0
    )
    print(f"✅ GIF salvo com sucesso: {filename}")

if __name__ == "__main__":
    DATASET_PATH = "./dataset_jogo"
    VQVAE_CKPT = "models/VQVAE_PONG/ckpt.pt"
    DYNAMICS_CKPT = "models/DYNAMICS_PONG/dynamics_ckpt.pt" # Certifique-se que o treino novo terminou!
    
    if not os.path.exists(DYNAMICS_CKPT):
        raise FileNotFoundError("Checkpoints não encontrados. Treine o Transformer primeiro!")

    vqvae, dynamics = load_models(VQVAE_CKPT, DYNAMICS_CKPT)
    priming_imgs = get_priming_sequence(DATASET_PATH, num_frames_priming=5)
    
    # Gera 30 frames de futuro baseados no plano de joystick
    dream_tokens = generate_dream(vqvae, dynamics, priming_imgs, frames_to_generate=30)
    
    save_dream_gif(dream_tokens, vqvae, filename="models/DYNAMICS_PONG/dream.gif")