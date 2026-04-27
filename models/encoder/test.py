import os
from xml.parsers.expat import model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Ajuste os imports abaixo para bater com a estrutura de pastas do seu projeto
from models.encoder.dataset import CarRacingDataset
from models.encoder.modules import VQVAE

def evaluate_vqvae(args):
    # Configurar dispositivo
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    # 1. Inicializar o modelo
    model = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings
    ).to(device)

    # 2. Carregar o Checkpoint
    ckpt_path = os.path.join("models", args.run_name, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint não encontrado em: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])   
    model.eval() # Modo de avaliação (desliga dropout, etc)
    print(f"✅ Modelo carregado da época {checkpoint.get('epoch', 'N/A')}")

    # 3. Carregar um batch de teste
    dataset = CarRacingDataset(args.dataset_path, max_files=200)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    images = next(iter(dataloader)).to(device)

    # 4. Fazer a inferência
    with torch.no_grad():
        x_recon, vq_loss, indices = model(images)
        # Calcula o erro de reconstrução bruto (MSE)
        mse_loss = nn.MSELoss()(x_recon, images).item()

    # 5. Analisar a saúde do Codebook
    # Quantos vetores únicos foram usados neste batch?
    unique_indices = torch.unique(indices)
    active_tokens = len(unique_indices)
    usage_percentage = (active_tokens / args.num_embeddings) * 100

    print("\n📊 --- RESULTADOS DA AVALIAÇÃO ---")
    print(f"🔹 Loss de Reconstrução (MSE): {mse_loss:.6f}")
    print(f"🔹 Loss do Quantizador (VQ):   {vq_loss.item():.6f}")
    print(f"🔹 Tokens ativos no Codebook:  {active_tokens} / {args.num_embeddings} ({usage_percentage:.2f}%)")
    
    # Diagnóstico
    if usage_percentage < 5:
        print("⚠️ ALERTA: Uso muito baixo do codebook! O modelo provavelmente sofreu 'Codebook Collapse'.")
    elif usage_percentage > 30:
        print("🌟 EXCELENTE: O modelo está distribuindo bem as informações no catálogo!")
    else:
        print("👍 NORMAL: O uso do codebook está aceitável, mas pode melhorar com mais treino ou ajustes de hyperparâmetros.")

    # 6. Salvar Imagens Lado a Lado
    os.makedirs("results/tests", exist_ok=True)
    
    # Pegamos até 8 imagens para não criar uma grade gigante
    n = min(8, images.size(0))
    
    # Concatena: linha de cima (originais) / linha de baixo (reconstruídas)
    comparison = torch.cat([images[:n], x_recon[:n]], dim=0)
    
    save_path = os.path.join("results/tests", "comparacao_teste.jpg")
    save_image(comparison, save_path, nrow=n, normalize=True, value_range=(0, 1))
    
    print(f"\n🖼️ Imagem de comparação salva em: {save_path}")
    print("👉 A linha superior mostra as imagens originais.")
    print("👉 A linha inferior mostra as reconstruções do VQ-VAE.")

if __name__ == '__main__':
    print("🚀 Iniciando o script de teste...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',     type=str, default="VQVAE_PONG")
    # Coloque o caminho real da sua pasta de imagens aqui no 'default'
    parser.add_argument('--dataset_path', type=str, default="./dataset_carracing", help="Caminho para a pasta com os .npz") 
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--image_size',   type=int, default=64)
    parser.add_argument('--latent_dim',   type=int, default=128)
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--device',       type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate_vqvae(args)
    print("✅ Fim da execução!")