"""
World Model estilo IRIS com heads separadas.

Layout por bloco (tokens_per_block = 17):
    [z_0, z_1, ..., z_63, a]   — 16 tokens visuais + 1 token de acao

Heads:
    head_obs     — prediz tokens visuais (obs_vocab_size classes)
    head_rewards — prediz reward com 3 classes {-1, 0, +1}
    head_ends    — prediz done  com 2 classes {0, 1}

Reward e done sao inferidos da representacao interna na posicao
da acao — nao sao tokens explicitos na sequencia (padrao IRIS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
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
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Config

class WorldModelConfig:
    """
    tokens_per_block = img_tokens + 1  (16 visuais + 1 acao = 17)

    Layout por bloco:
        [z_0 .. z_63 | a]

    Reward e done NAO sao tokens — sao preditos pelas heads
    a partir da representacao na posicao da acao.
    """
    def __init__(
        self,
        obs_vocab_size: int   = 512,
        act_vocab_size: int   = 5,
        img_tokens:     int   = 1,
        frames_per_seq: int   = 20,
        n_embd:         int   = 512,
        n_head:         int   = 8,
        n_layer:        int   = 8,
        dropout:        float = 0.1,
        bias:           bool  = True,
    ):
        self.obs_vocab_size   = obs_vocab_size
        self.act_vocab_size   = act_vocab_size
        self.reward_vocab     = 3   # {-1, 0, +1}
        self.done_vocab       = 2   # {0, 1}

        self.img_tokens       = img_tokens
        self.tokens_per_block = img_tokens + 1              
        self.frames_per_seq   = frames_per_seq
        self.block_size       = self.tokens_per_block * frames_per_seq  

        self.n_embd   = n_embd
        self.n_head   = n_head
        self.n_layer  = n_layer
        self.dropout  = dropout
        self.bias     = bias


# World Model

class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Embeddings separados por tipo, o modelo aprende representacoes distintas para tokens visuais e tokens de acao
        self.obs_emb = nn.Embedding(config.obs_vocab_size, config.n_embd)
        self.act_emb = nn.Embedding(config.act_vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size,     config.n_embd)
        self.drop    = nn.Dropout(config.dropout)

        # Transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f   = LayerNorm(config.n_embd, config.bias)

        # Head de observacao: prediz proximo token visual
        self.head_obs = nn.Linear(config.n_embd, config.obs_vocab_size, bias=False)
        self.head_obs.weight = self.obs_emb.weight

        # Head de reward: 3 classes {-1, 0, +1}
        self.head_rewards = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.reward_vocab),
        )

        # Head de done: 2 classes {0, 1}
        self.head_ends = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.done_vocab),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, obs_tokens, act_tokens):
        """
        Monta sequencia intercalando embeddings por bloco.

        obs_tokens : (B, T, 64)
        act_tokens : (B, T)

        Retorna: (B, T*65, n_embd)
        """
        B, T, K = obs_tokens.shape
        device   = obs_tokens.device

        obs_e = self.obs_emb(obs_tokens)               # (B, T, 64, n_embd)
        act_e = self.act_emb(act_tokens).unsqueeze(2)  # (B, T,  1, n_embd)

        # (B, T, 65, n_embd) -> (B, T*65, n_embd)
        x = torch.cat([obs_e, act_e], dim=2).view(
            B, T * self.config.tokens_per_block, self.config.n_embd
        )

        pos = torch.arange(x.size(1), device=device)
        return self.drop(x + self.pos_emb(pos))

    def forward(self, obs_tokens, act_tokens):
        """
        obs_tokens : (B, T, 64)  long
        act_tokens : (B, T)      long

        Retorna:
            logits_obs     : (B, T, 64, obs_vocab_size)
            logits_rewards : (B, T, 3)
            logits_ends    : (B, T, 2)
        """
        B, T, K = obs_tokens.shape
        tpb      = self.config.tokens_per_block  

        x = self._embed(obs_tokens, act_tokens) 

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Reorganiza em blocos: (B, T, 17, n_embd)
        x = x.view(B, T, tpb, self.config.n_embd)

        # Posicoes 0..15 de cada bloco: tokens visuais
        logits_obs = self.head_obs(x[:, :, :K, :])    # (B, T, 16, obs_vocab)

        # Posicao 16 de cada bloco: acao — usada para reward e done
        act_repr       = x[:, :, K, :]                 # (B, T, n_embd)
        logits_rewards = self.head_rewards(act_repr)    # (B, T, 3)
        logits_ends    = self.head_ends(act_repr)       # (B, T, 2)

        return logits_obs, logits_rewards, logits_ends

    def compute_loss(self, obs_tokens, act_tokens, rewards_sign, dones):
        """
        obs_tokens   : (B, T, 64)  long
        act_tokens   : (B, T)      long
        rewards_sign : (B, T)      long — {0=negativo, 1=neutro, 2=positivo}
        dones        : (B, T)      long — {0, 1}

        Retorna: loss total e losses individuais para logging
        """
        B, T, K = obs_tokens.shape

        logits_obs, logits_rewards, logits_ends = self.forward(obs_tokens, act_tokens)

        obs_stream = obs_tokens.reshape(B, T * K)                   
        log_stream = logits_obs.reshape(B, T * K, -1)       

        loss_obs = F.cross_entropy(
            log_stream[:, :-1].reshape(-1, self.config.obs_vocab_size),
            obs_stream[:, 1:].reshape(-1),
        )

        #  Loss reward 
        loss_rewards = F.cross_entropy(
            logits_rewards.reshape(-1, self.config.reward_vocab),
            rewards_sign.reshape(-1),
        )

        #  Loss done
        loss_ends = F.cross_entropy(
            logits_ends.reshape(-1, self.config.done_vocab),
            dones.reshape(-1),
        )

        loss = loss_obs + loss_rewards + loss_ends
        return loss, loss_obs, loss_rewards, loss_ends

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay   = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups  = [
            {'params': decay,   'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def imagine_next_frame(self, obs_tokens, act_token, temperature=1.0, top_k=50):
        """
        Dado historico de frames e a proxima acao, imagina os 16 tokens
        do proximo frame autorregressivamente.

        obs_tokens : (1, T, 16)  long  — historico de tokens visuais
        act_token  : (1,)        long  — acao a tomar no proximo passo

        Retorna:
            next_obs   : (1, 16)  — tokens do frame imaginado
            reward_pred: int      — reward predito {-1, 0, +1}
            done_pred  : bool     — done predito
        """
        config = self.config
        device = obs_tokens.device
        B, T, K = obs_tokens.shape

        next_obs   = torch.zeros(B, 1, K, dtype=torch.long, device=device)
        act_tokens = torch.cat([
            torch.zeros(B, T, dtype=torch.long, device=device),
            act_token.unsqueeze(1)
        ], dim=1)  # (B, T+1)

        ctx_obs = torch.cat([obs_tokens, next_obs], dim=1)  

        # Gera token por token de forma autorregressiva
        for k in range(K):
            logits_obs, logits_rew, logits_end = self.forward(ctx_obs, act_tokens)

            # Logits do token k do ultimo bloco
            next_logits = logits_obs[:, -1, k, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            ctx_obs[:, -1, k] = token

        next_obs    = ctx_obs[:, -1, :]  

        # Reward e done do ultimo bloco
        _, logits_rew, logits_end = self.forward(ctx_obs, act_tokens)
        reward_bin  = logits_rew[:, -1, :].argmax(dim=-1)   # {0,1,2}
        reward_pred = (reward_bin.float() - 1)               # {-1,0,+1}
        done_pred   = logits_end[:, -1, :].argmax(dim=-1).bool()

        return next_obs, reward_pred, done_pred