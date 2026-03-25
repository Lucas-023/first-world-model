#!/bin/bash

# COLOQUE SEU USUARIO E NOME DO MODELO AQUI
REPO_NAME="Lucas-023/meu-pong-vqvae" 

echo "Iniciando o vigia de backup..."

while true; do
    echo "Sincronizando modelos com o Hugging Face..."
    
    # Sobe a pasta models/VQVAE_PONG (onde estão os ckpt.pt)
    huggingface-cli upload $REPO_NAME ./models/VQVAE_PONG /models --repo-type model
    
    # Sobe a pasta results/VQVAE_PONG (onde estão as imagens de teste)
    huggingface-cli upload $REPO_NAME ./results/VQVAE_PONG /results --repo-type model
    
    echo "Upload feito! Dormindo por 1 hora..."
    sleep 3600
done
