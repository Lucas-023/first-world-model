"""
World Dynamics Model — previsao do proximo frame condicionada em acoes.

Recebe contexto de context_len frames e prevê os 64 tokens
do frame seguinte.

Layout por bloco (65 tokens):
    [z_0..z_63 | a]

Durante o treino:
    input  : context_len blocos  → (context_len * 65) tokens
    output : 64 logits para os tokens do proximo frame

Durante a inferencia (imagine_next_frame):
    mesma logica — contexto entra, 1 frame sai autorregressivamente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsConfig:
    def __init__(
        self,
        obs_vocab_size: int   = 512,
        act_vocab_size: int   = 5,
        img_tokens:     int   = 64,     # 8x8 grid
        context_len:    int   = 19,     # frames de contexto
        n_embd:         int   = 256,
        n_head:         int   = 4,
        n_layer:        int   = 6,
        dropout:        float = 0.1,
    ):
        self.obs_vocab_size   = obs_vocab_size
        self.act_vocab_size   = act_vocab_size
        self.img_tokens       = img_tokens
        self.context_len      = context_len
        self.tokens_per_block = img_tokens + 1                  # 65
        self.block_size       = self.tokens_per_block * context_len  # 65*19=1235
        self.n_embd   = n_embd
        self.n_head   = n_head
        self.n_layer  = n_layer
        self.dropout  = dropout


class DynamicsModel(nn.Module):
    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        self.obs_emb = nn.Embedding(config.obs_vocab_size, config.n_embd)
        self.act_emb = nn.Embedding(config.act_vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size,     config.n_embd)
        self.drop    = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.n_embd,
            nhead           = config.n_head,
            dim_feedforward = 4 * config.n_embd,
            dropout         = config.dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = config.n_layer,
        )

        # Head de previsao: a partir da representacao do ultimo token
        # do contexto, prediz cada um dos 64 tokens do proximo frame
        self.head_obs = nn.Linear(config.n_embd, config.obs_vocab_size, bias=False)
        self.head_obs.weight = self.obs_emb.weight

        # Mascara causal
        mask = torch.triu(
            torch.ones(config.block_size, config.block_size), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _embed(self, obs_tokens, act_tokens):
        """
        obs_tokens : (B, T, 64)  — T frames de contexto
        act_tokens : (B, T)
        Retorna    : (B, T*65, n_embd)
        """
        B, T, K = obs_tokens.shape
        device   = obs_tokens.device

        obs_e = self.obs_emb(obs_tokens)               # (B, T, 64, n_embd)
        act_e = self.act_emb(act_tokens).unsqueeze(2)  # (B, T,  1, n_embd)

        x = torch.cat([obs_e, act_e], dim=2).view(
            B, T * self.config.tokens_per_block, self.config.n_embd
        )
        pos = torch.arange(x.size(1), device=device)
        return self.drop(x + self.pos_emb(pos))

    def encode_context(self, obs_tokens, act_tokens):
        """
        Processa o contexto pelo transformer e retorna a representacao
        do ultimo token — usada para prever o proximo frame.

        obs_tokens : (B, T, 64)
        act_tokens : (B, T)
        Retorna    : (B, n_embd)  representacao do ultimo token do contexto
        """
        x       = self._embed(obs_tokens, act_tokens)   # (B, T*65, n_embd)
        seq_len = x.size(1)
        mask    = self.causal_mask[:seq_len, :seq_len]
        x       = self.transformer(x, mask=mask, is_causal=True)
        # Pega a representacao do ultimo token (ultimo token da acao do ultimo bloco)
        return x[:, -1, :]   # (B, n_embd)

    def compute_loss(self, obs_ctx, act_ctx, obs_target):
        """
        obs_ctx    : (B, context_len, 64)  — frames de contexto
        act_ctx    : (B, context_len)      — acoes do contexto
        obs_target : (B, 64)               — frame a prever

        Prediz cada token do target autorregressivamente dentro do bloco:
        usa o contexto + tokens ja gerados do target para prever o proximo.

        Para simplificar o treino (teacher forcing):
        concatena o target ao contexto e prediz token a token com shift.
        """
        B, T, K = obs_ctx.shape
        device   = obs_ctx.device

        # Concatena contexto + target como mais um bloco
        # act do target: usa acao 0 como placeholder (nao importa pois
        # o target e o que queremos prever, nao condicionar)
        act_target  = torch.zeros(B, 1, dtype=torch.long, device=device)
        full_obs    = torch.cat([obs_ctx, obs_target.unsqueeze(1)], dim=1)  # (B, T+1, 64)
        full_act    = torch.cat([act_ctx, act_target],              dim=1)  # (B, T+1)

        # Processa tudo junto
        x       = self._embed(full_obs, full_act)       # (B, (T+1)*65, n_embd)
        seq_len = x.size(1)
        mask    = self.causal_mask[:seq_len, :seq_len]
        x       = self.transformer(x, mask=mask, is_causal=True)
        x       = x.view(B, T + 1, self.config.tokens_per_block, self.config.n_embd)

        # Pega as representacoes nas posicoes visuais do ULTIMO bloco (o target)
        # posicoes 0..63 do bloco T
        target_repr   = x[:, -1, :K, :]                           # (B, 64, n_embd)
        logits_target = self.head_obs(target_repr)                 # (B, 64, vocab)

        # Teacher forcing: posicao k prediz token k do target
        # logits[:, :-1] prediz tokens[:, 1:] — shift de 1 dentro do bloco
        # Mas aqui queremos que a representacao DO CONTEXTO prediga token 0,
        # e token 0 prediga token 1, etc.
        #
        # Solucao: usa a representacao do ultimo token do CONTEXTO (posicao -1
        # do bloco T-1) para prever o token 0 do target, e assim por diante.
        ctx_last_repr = x[:, -2, -1, :]                            # (B, n_embd) — ultimo token do contexto
        ctx_logits    = self.head_obs(ctx_last_repr).unsqueeze(1)  # (B, 1, vocab)

        # Sequencia de logits: [ctx_last → z0, z0 → z1, ..., z62 → z63]
        all_logits = torch.cat([ctx_logits, logits_target[:, :-1, :]], dim=1)  # (B, 64, vocab)

        # Target: todos os 64 tokens do frame alvo
        loss = F.cross_entropy(
            all_logits.reshape(-1, self.config.obs_vocab_size),
            obs_target.reshape(-1),
        )
        return loss

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95)):
        decay   = [p for n, p in self.named_parameters()
                   if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in self.named_parameters()
                   if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([
            {'params': decay,   'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ], lr=learning_rate, betas=betas)

    @torch.no_grad()
    def imagine_next_frame(self, obs_tokens, act_token, temperature=1.0, top_k=50):
        """
        Dado historico de context_len frames, imagina o proximo frame
        autorregressivamente token por token.

        obs_tokens : (1, context_len, 64)
        act_token  : (1,)  — acao a tomar no proximo passo

        Retorna: (1, 64)
        """
        B, T, K = obs_tokens.shape
        device   = obs_tokens.device

        # Placeholder para o frame a imaginar
        next_frame = torch.zeros(B, 1, K, dtype=torch.long, device=device)
        ctx_obs    = torch.cat([obs_tokens, next_frame], dim=1)    # (B, T+1, 64)
        ctx_act    = torch.zeros(B, T + 1, dtype=torch.long, device=device)
        ctx_act[:, -1] = act_token

        # Gera os 64 tokens um por um (autorregressivo dentro do bloco)
        for k in range(K):
            x       = self._embed(ctx_obs, ctx_act)
            seq_len = x.size(1)
            mask    = self.causal_mask[:seq_len, :seq_len]
            x       = self.transformer(x, mask=mask, is_causal=True)
            x       = x.view(B, T + 1, self.config.tokens_per_block, self.config.n_embd)

            # Representacao na posicao k do ultimo bloco
            if k == 0:
                # Token 0: condicionado no ultimo token do contexto
                repr_k = x[:, -2, -1, :]       # (B, n_embd)
            else:
                repr_k = x[:, -1, k - 1, :]    # (B, n_embd)

            logits = self.head_obs(repr_k) / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            ctx_obs[:, -1, k] = token

        return ctx_obs[:, -1, :]   # (1, 64)