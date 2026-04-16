import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash attention nativo do PyTorch (muito mais rápido e usa menos VRAM)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class WorldModelConfig:
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, 
                 tokens_per_frame=64, frames_per_seq=5, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.tokens_per_frame = tokens_per_frame
        self.frames_per_seq = frames_per_seq
        self.block_size = (tokens_per_frame + 1) * frames_per_seq
        self.bias = True
class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_size, config.n_embd),
            pos_emb = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        
        # 1. Action Encoder: Transforma [Volante, Acel, Freio] -> n_embd
        self.action_encoder = nn.Linear(3, config.n_embd)

        # 2. Cabeças de Saída
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.reward_head = nn.Linear(config.n_embd, 1) # Preve um número (recompensa)
        self.done_head = nn.Linear(config.n_embd, 1)   # Preve probabilidade (fim de jogo)

        # Peso amigável: amarra os pesos da lm_head com a tok_emb (padrão GPT)
        self.transformer.tok_emb.weight = self.lm_head.weight 

    def forward(self, img_tokens, actions, targets_img=None, targets_reward=None, targets_done=None):
        device = img_tokens.device
        b, t_seq, f_tokens = img_tokens.size() # [Batch, Frames_Seq, 64]
        
        # --- PASSO 1: EMBEDDINGS ---
        # Imagens: [B, T_seq, 64, Emb]
        tok_emb = self.transformer.tok_emb(img_tokens) 
        # Ações: [B, T_seq, 1, Emb]
        act_emb = self.action_encoder(actions).unsqueeze(2) 
        
        # --- PASSO 2: INTERCALAR (A MÁGICA) ---
        # Concatenamos a ação após os 64 tokens de cada frame
        # Resultado: [B, T_seq, 65, Emb]
        combined = torch.cat([tok_emb, act_emb], dim=2)
        # Planificamos para a sequência longa que o GPT gosta: [B, T_seq * 65, Emb]
        x = combined.view(b, -1, self.config.n_embd)

        # Adiciona posição
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
        x = self.transformer.drop(x + self.transformer.pos_emb(pos))

        # Passa pelos blocos de Atenção
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # --- PASSO 3: CABEÇAS E LOSS ---
        if targets_img is not None:
            # 3.1 Previsão de Imagem (Próximo Token)
            logits = self.lm_head(x)
            
            # Ajustamos as saídas para prever o PRÓXIMO item da sequência
            # targets_img deve vir no formato intercalado também ou calculamos o shift
            # Para simplificar: loss de reconstrução dos tokens de imagem
            loss_img = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), 
                                      # Aqui você precisará de uma lógica para alinhar targets
                                      targets_img.reshape(-1))

            # 3.2 Previsão de Reward e Done (Baseadas no token da Ação)
            # Pegamos a saída do Transformer na posição onde entrou a ação (índices 64, 129...)
            action_indices = torch.arange(self.config.tokens_per_frame, x.size(1), 
                                          self.config.tokens_per_frame + 1, device=device)
            
            action_outputs = x[:, action_indices, :] # [B, T_seq, Emb]
            
            reward_preds = self.reward_head(action_outputs).squeeze(-1)
            done_preds = self.done_head(action_outputs).squeeze(-1)

            loss_reward = F.mse_loss(reward_preds, targets_reward)
            loss_done = F.binary_cross_entropy_with_logits(done_preds, targets_done)

            # A loss total é a soma balanceada
            loss = loss_img + (10.0 * loss_reward) + (5.0 * loss_done)
            return logits, (loss, loss_img, loss_reward, loss_done)
        
        else:
            # Em vez de apenas o último [-1], retorne os últimos 64 para a imagem
            return self.lm_head(x[:, -64:, :]), None
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    @torch.no_grad()
    def generate(self, img_tokens, actions, max_steps=20):
        model_device = next(self.parameters()).device
        
        # Garante formato [1, 1, 64]
        curr_img_tokens = img_tokens.to(model_device)
        if curr_img_tokens.dim() == 2:
            curr_img_tokens = curr_img_tokens.unsqueeze(1)
            
        all_frames = [curr_img_tokens]

        for t in range(max_steps):
            # 1. Filtra as ações para o tamanho atual da sequência
            curr_actions = actions[:, :len(all_frames), :].to(model_device)
            
            # 2. Concatena os frames: [1, T_atual, 64]
            input_imgs = torch.cat(all_frames, dim=1) 
            
            # 3. Forward pass (usa o modo inferência do seu modelo)
            # O seu forward sem targets retorna: self.lm_head(x[:, [-1], :])
            # MAS para gerar 64 tokens, precisamos de mais do que o índice [-1]!
            
            # --- MUDANÇA CRUCIAL AQUI ---
            # Vamos chamar o processamento do transformer diretamente ou 
            # ajustar o forward para não retornar apenas o último token.
            logits, _ = self.forward(input_imgs, curr_actions)
            
            # Se o seu lm_head retorna [B, Seq, Vocab], queremos os últimos 64
            # que correspondem à previsão do próximo frame de imagem.
            next_frame_logits = logits[:, -64:, :] 
            
            # 4. Argmax para pegar os IDs dos tokens
            next_frame_tokens = torch.argmax(next_frame_logits, dim=-1) # [1, 64]
            
            # 5. Garantir que o formato seja [1, 1, 64] para o próximo torch.cat
            if next_frame_tokens.dim() == 2:
                next_frame_tokens = next_frame_tokens.unsqueeze(1)
            
            all_frames.append(next_frame_tokens)

        return torch.cat(all_frames, dim=1)