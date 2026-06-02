"""
Dynamics GPT para prever:
1) tokens da proxima imagem
2) classe de reward {-1, 0, +1}
3) done {0, 1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelConfig:
    def __init__(
        self,
        obs_vocab_size: int = 512,
        act_vocab_size: int = 5,
        img_tokens: int = 64,
        context_len: int = 19,
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 6,
        dropout: float = 0.1,
    ):
        self.obs_vocab_size = obs_vocab_size
        self.act_vocab_size = act_vocab_size
        self.reward_vocab = 3
        self.done_vocab = 2
        self.img_tokens = img_tokens
        self.context_len = context_len
        self.tokens_per_block = img_tokens + 1
        self.block_size = self.tokens_per_block * (context_len + 1)
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.obs_emb = nn.Embedding(config.obs_vocab_size, config.n_embd)
        self.act_emb = nn.Embedding(config.act_vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        self.head_obs = nn.Linear(config.n_embd, config.obs_vocab_size, bias=False)
        self.head_obs.weight = self.obs_emb.weight
        self.head_rewards = nn.Linear(config.n_embd, config.reward_vocab)
        self.head_dones = nn.Linear(config.n_embd, config.done_vocab)

        mask = torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()
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
        B, T, _ = obs_tokens.shape
        obs_e = self.obs_emb(obs_tokens)
        act_e = self.act_emb(act_tokens).unsqueeze(2)
        x = torch.cat([obs_e, act_e], dim=2).view(
            B, T * self.config.tokens_per_block, self.config.n_embd
        )
        pos = torch.arange(x.size(1), device=obs_tokens.device)
        return self.drop(x + self.pos_emb(pos))

    def _transform(self, obs_tokens, act_tokens):
        x = self._embed(obs_tokens, act_tokens)
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=mask, is_causal=True)
        return x

    def compute_loss(self, obs_ctx, act_ctx, obs_target, reward_target, done_target):
        B, T, K = obs_ctx.shape
        device = obs_ctx.device

        act_target = torch.zeros(B, 1, dtype=torch.long, device=device)
        full_obs = torch.cat([obs_ctx, obs_target.unsqueeze(1)], dim=1)
        full_act = torch.cat([act_ctx, act_target], dim=1)

        x = self._transform(full_obs, full_act)
        x = x.view(B, T + 1, self.config.tokens_per_block, self.config.n_embd)

        target_repr = x[:, -1, :K, :]
        logits_target = self.head_obs(target_repr)
        ctx_last_repr = x[:, -2, -1, :]
        ctx_logits = self.head_obs(ctx_last_repr).unsqueeze(1)
        obs_logits = torch.cat([ctx_logits, logits_target[:, :-1, :]], dim=1)

        loss_obs = F.cross_entropy(
            obs_logits.reshape(-1, self.config.obs_vocab_size),
            obs_target.reshape(-1),
        )
        loss_rewards = F.cross_entropy(self.head_rewards(ctx_last_repr), reward_target)
        loss_dones = F.cross_entropy(self.head_dones(ctx_last_repr), done_target)
        loss = loss_obs + loss_rewards + loss_dones
        return loss, loss_obs, loss_rewards, loss_dones

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95)):
        decay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )

    @torch.no_grad()
    def imagine_next_frame(self, obs_tokens, act_tokens, act_token, temperature=1.0, top_k=50):
        B, T, K = obs_tokens.shape
        device = obs_tokens.device

        next_frame = torch.zeros(B, 1, K, dtype=torch.long, device=device)
        ctx_obs = torch.cat([obs_tokens, next_frame], dim=1)
        ctx_act = torch.cat([act_tokens, act_token.unsqueeze(1)], dim=1)

        x_ctx = self._transform(obs_tokens, act_tokens).view(
            B, T, self.config.tokens_per_block, self.config.n_embd
        )
        ctx_last_repr = x_ctx[:, -1, -1, :]
        reward_pred = self.head_rewards(ctx_last_repr).argmax(dim=-1)
        done_pred = self.head_dones(ctx_last_repr).argmax(dim=-1)

        for k in range(K):
            x = self._transform(ctx_obs, ctx_act)
            x = x.view(B, T + 1, self.config.tokens_per_block, self.config.n_embd)

            if k == 0:
                repr_k = x[:, -2, -1, :]
            else:
                repr_k = x[:, -1, k - 1, :]

            logits = self.head_obs(repr_k) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            ctx_obs[:, -1, k] = token

        return ctx_obs[:, -1, :], reward_pred, done_pred