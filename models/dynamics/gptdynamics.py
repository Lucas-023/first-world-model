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
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class WorldModelConfig:
    """
    Layout do vocabulário (offsets):
        [0 .. vocab_img-1]                          → tokens de imagem
        [vocab_img .. vocab_img+vocab_action-1]      → tokens de ação
        [vocab_img+vocab_action .. +vocab_reward-1]  → tokens de reward discreto
        [vocab_img+vocab_action+vocab_reward .. +1]  → token de done

    Layout por frame na sequência (67 tokens):
        z(0..63) | a(64) | r(65) | d(66)
    """
    def __init__(
        self,
        vocab_img       = 512,
        vocab_action    = 5,    # CarRacing-v3 discrete: 0-4
        vocab_reward    = 21,   # 21 bins entre -1 e +1 (padrao IRIS)
        n_embd          = 256,
        n_head          = 8,
        n_layer         = 6,
        tokens_per_frame = 64,  # tokens visuais por frame (8x8 grid)
        frames_per_seq  = 20,
        dropout         = 0.1,
        bias            = True,
    ):
        self.vocab_img    = vocab_img
        self.vocab_action = vocab_action
        self.vocab_reward = vocab_reward
        self.vocab_done   = 2   # 0=nao-terminal, 1=terminal

        # Offsets para cada tipo de token no vocabulário unificado
        self.img_offset    = 0
        self.act_offset    = vocab_img
        self.rew_offset    = vocab_img + vocab_action
        self.done_offset   = vocab_img + vocab_action + vocab_reward

        self.vocab_total   = vocab_img + vocab_action + vocab_reward + self.vocab_done

        # Estrutura temporal
        # Cada frame = 64 tokens visuais + 1 ação + 1 reward + 1 done = 67
        self.img_tokens_per_frame = tokens_per_frame
        self.tokens_per_frame     = tokens_per_frame + 1 + 1 + 1   # 67
        self.frames_per_seq       = frames_per_seq
        self.block_size           = self.tokens_per_frame * frames_per_seq

        # Transformer
        self.n_embd   = n_embd
        self.n_head   = n_head
        self.n_layer  = n_layer
        self.dropout  = dropout
        self.bias     = bias


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_total, config.n_embd),
            pos_emb = nn.Embedding(config.block_size,  config.n_embd),
            drop    = nn.Dropout(config.dropout),
            h       = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f    = LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_total, bias=False)
        # Weight tying
        self.transformer.tok_emb.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        B, T   = idx.size()
        assert T <= self.config.block_size, \
            f"Sequencia {T} maior que block_size {self.config.block_size}"

        pos = torch.arange(0, T, device=device)
        x   = self.transformer.drop(
            self.transformer.tok_emb(idx) + self.transformer.pos_emb(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # idx e targets chegam com o mesmo shape (B, T)
            # o shift ja foi feito em traingpt.py: inputs=seq[:,:-1], targets=seq[:,1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return logits, loss

        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict    = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups  = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):
        cfg = self.config
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -cfg.block_size:]
            logits, _    = self.forward(idx_cond)
            logits       = logits[:, -1, :] / temperature

            # Mascara posicional: restringe o vocabulario ao tipo correto
            # Layout por frame: z(0..63) | a(64) | r(65) | d(66)
            pos_in_frame = idx_cond.shape[1] % cfg.tokens_per_frame
            mask = torch.full_like(logits, -float('inf'))

            if pos_in_frame < cfg.img_tokens_per_frame:   # posicoes 0-63: imagem
                mask[:, cfg.img_offset : cfg.img_offset + cfg.vocab_img] = 0.0
            elif pos_in_frame == cfg.img_tokens_per_frame:  # posicao 64: acao
                mask[:, cfg.act_offset : cfg.act_offset + cfg.vocab_action] = 0.0
            elif pos_in_frame == cfg.img_tokens_per_frame + 1:  # posicao 65: reward
                mask[:, cfg.rew_offset : cfg.rew_offset + cfg.vocab_reward] = 0.0
            else:  # posicao 66: done
                mask[:, cfg.done_offset : cfg.done_offset + cfg.vocab_done] = 0.0

            logits = logits + mask

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat((idx, next_token), dim=1)

        return idx
