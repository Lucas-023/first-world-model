import os
from PIL import Image
from tqdm import tqdm
import shutil

pasta_origem = "dataset_jogo"      # Coloque o nome da sua pasta original aqui
pasta_destino = "dataset_cortado"  # Pasta onde as imagens cortadas vão ficar

os.makedirs(pasta_destino, exist_ok=True)

# Lista todos os arquivos da pasta original
arquivos = [f for f in os.listdir(pasta_origem) if f.endswith('.png')]

print(f"Cortando {len(arquivos)} imagens...")

for arquivo in tqdm(arquivos):
    caminho_antigo = os.path.join(pasta_origem, arquivo)
    caminho_novo = os.path.join(pasta_destino, arquivo)
    
    # 1. Abre a imagem original (96x96)
    img = Image.open(caminho_antigo).convert('RGB')
    
    # 2. Corta a imagem: crop((esquerda, topo, direita, baixo))
    # Mantemos do pixel 0 até a linha 84
    img_cortada = img.crop((0, 0, 96, 84))
    
    # 3. Redimensiona para o tamanho do VQ-VAE (64x64)
    img_final = img_cortada.resize((64, 64), Image.Resampling.LANCZOS)
    
    # 4. Salva na nova pasta
    img_final.save(caminho_novo)

# Copia o arquivo metadata.csv (se ele existir) para a nova pasta não quebrar o treino do Transformer
csv_origem = os.path.join(pasta_origem, "metadata.csv")
if os.path.exists(csv_origem):
    shutil.copy(csv_origem, os.path.join(pasta_destino, "metadata.csv"))

print("Pronto! Agora você pode treinar apontando para a pasta 'dataset_cortado'.")