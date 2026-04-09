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
    def __init__(self, **kwargs):
        self.block_size = kwargs.get('block_size', 320)       # 5 frames * 64 tokens
        self.vocab_size = kwargs.get('vocab_size', 512)       # Igual ao num_embeddings do VQ-VAE
        self.action_vocab_size = kwargs.get('action_vocab_size', 4) # Quantidade de ações do jogo
        self.n_layer = kwargs.get('n_layer', 6)
        self.n_head = kwargs.get('n_head', 8)
        self.n_embd = kwargs.get('n_embd', 512)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', False)

class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            w_act = nn.Embedding(config.action_vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight Tying
        
        # CABEÇA DO DONE (Prevê apenas 1 valor: 0 ou 1)
        self.done_head = nn.Linear(config.n_embd, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, actions=None, targets=None, targets_done=None):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) # Visão: [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos) # Tempo/Espaço: [T, n_embd]
        
        x = tok_emb + pos_emb
        
        if actions is not None:
            act_emb = self.transformer.w_act(actions) # [B, frames_per_seq, n_embd]
            
            # TRUQUE MÁGICO: Espalha a ação do frame para todos os 64 tokens dele
            frames_per_seq = actions.size(1)
            tokens_per_frame = t // frames_per_seq
            act_emb_expanded = act_emb.repeat_interleave(tokens_per_frame, dim=1) # [B, T, n_embd]
            
            x = x + act_emb_expanded

        # Passa pelos blocos do Transformer
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 1. Erro da Imagem (Previsão de Tokens)
            logits = self.lm_head(x)
            loss_img = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss_img
            
            # 2. Erro do Fim do Episódio (Done)
            if targets_done is not None:
                done_logits = self.done_head(x).squeeze(-1) # [B, T]
                
                frames_per_seq = targets_done.size(1)
                tokens_per_frame = t // frames_per_seq
                
                # Pegamos apenas o último token de cada frame (ex: índices 63, 127, 191...)
                last_token_indices = torch.arange(
                    tokens_per_frame - 1, t, tokens_per_frame, device=device
                )
                
                done_preds = done_logits[:, last_token_indices] # [B, frames_per_seq]
                
                # Calcula o erro usando Binary Cross Entropy
                loss_done = F.binary_cross_entropy_with_logits(done_preds, targets_done)
                
                # Junta os dois erros
                loss = loss_img + loss_done
        else:
            # Modo de imaginação (Inferência)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

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