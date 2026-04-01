import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injeta a informação de tempo/posição. 
    Sem isso, o Transformer não saberia o que é passado e o que é futuro.
    """
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
        x = x + self.pe[:, :x.size(1), :]
        return x

class GPTDynamicsModel(nn.Module):
    """
    O Modelo de Mundo (World Model) baseado em Transformer.
    Arquitetura decoder-only (estilo GPT): usa TransformerEncoder com máscara causal.
    Prevê o próximo token condicionado ao estado visual atual E à ação tomada.

    NOTA ARQUITETURAL: Usamos TransformerEncoder com is_causal=True para simular
    um decoder-only (GPT). Isso é equivalente funcionalmente a um decoder sem
    cross-attention. Se você quiser condicionar em um estado externo (ex: goal embedding),
    migre para TransformerDecoder com cross-attention.
    """
    def __init__(self, vocab_size=512, d_model=256, nhead=8, num_layers=6, dropout=0.1, max_seq_len=1024, num_actions=6):
        super().__init__()
        self.d_model = d_model
        
        # 1. Dicionário visual: mapeia token_id → vetor d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Dicionário de ações: mapeia action_id → vetor d_model
        self.action_embedding = nn.Embedding(num_actions, d_model)
        
        # 3. Codificação posicional: informa ao modelo a ordem dos tokens no tempo
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 4. Transformer causal (decoder-only style)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 5. Cabeça de projeção: vetor d_model → logits sobre vocab_size tokens
        self.fc_out = nn.Linear(d_model, vocab_size)


        self.done_head = nn.Linear(d_model, 2)

    def forward(self, x, actions=None):
        """
        Args:
            x:       [batch_size, seq_len] — índices dos tokens visuais.
            actions: [batch_size, num_frames] — ação por frame (opcional).
        
        Returns:
            logits:  [batch_size, seq_len, vocab_size] — probabilidades do próximo token.
        """
        device = x.device
        seq_len = x.size(1)
        
        # Máscara causal para que cada token só pode ver tokens anteriores
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # Embedding visual escalado pela dimensao
        x_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        
        if actions is not None:
            # Mapeia qual frame cada token pertence: tokens 0-63 → frame 0, 64-127 → frame 1, etc.
            frame_indices = torch.div(torch.arange(seq_len, device=device), 64, rounding_mode='trunc')
            
            # evita out of bounds
            frame_indices = torch.clamp(frame_indices, max=actions.size(1) - 1)
            
            token_actions = torch.gather(
                actions, 1,
                frame_indices.unsqueeze(0).expand(actions.size(0), -1)
            )
            
            a_emb = self.action_embedding(token_actions)
            x_emb = x_emb + a_emb 
            
        x_emb = self.pos_encoder(x_emb)
        
        out = self.transformer(x_emb, mask=mask)

        logits_pixels = self.fc_out(out)
        logits_dones = self.done_head(out) 
        
        return logits_pixels, logits_dones