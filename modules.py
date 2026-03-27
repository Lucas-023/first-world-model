import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # pe shape: [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Soma a codificação posicional aos embeddings originais
        x = x + self.pe[:, :x.size(1), :]
        return x

class GPTDynamicsModel(nn.Module):
    # Adicionamos num_actions=6 (O Atari Pong tem 6 botões possíveis)
    def __init__(self, vocab_size=512, d_model=256, n_heads=8, n_layers=4, num_actions=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding da Visão (As pecinhas do VQ-VAE)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding (Onde a pecinha está na tela)
        # 320 é o nosso seq_len (5 frames * 64 tokens)
        self.pos_emb = nn.Parameter(torch.zeros(1, 320, d_model))
        
        # --- NOVO: O "Receptor" do Joystick ---
        self.action_emb = nn.Embedding(num_actions, d_model)
        # -------------------------------------

        # O Coração do Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # A Boca (Adivinha o próximo token)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, actions=None):
        b, t = x.size()
        
        # 1. Mistura Visão + Posição
        token_embeddings = self.tok_emb(x) # [Batch, Tempo, Dimensão]
        position_embeddings = self.pos_emb[:, :t, :]
        
        x_emb = token_embeddings + position_embeddings
        
        # 2. --- NOVO: Injetando a Intenção do Jogador ---
        if actions is not None:
            # actions shape: [Batch, Frames]
            act_embeddings = self.action_emb(actions) 
            
            # MÁGICA: Cada frame tem 64 tokens visuais. 
            # Nós clonamos a ação 64 vezes para ela cobrir o frame inteiro!
            act_embeddings = act_embeddings.repeat_interleave(64, dim=1) 
            
            # Garante que o tamanho não passe do limite (segurança)
            act_embeddings = act_embeddings[:, :t, :]
            
            # Soma o joystick na mente do modelo
            x_emb = x_emb + act_embeddings
        # ------------------------------------------------

        # 3. Máscara Causal (Para ele não "roubar" olhando o futuro)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        
        # 4. Pensa...
        x_out = self.transformer(x_emb, mask=mask, is_causal=True)
        
        # 5. Responde!
        logits = self.lm_head(x_out)
        return logits