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
    def encode_state(self, obs_tokens, act_tokens):
        """
        Representacao compacta do contexto (B,n_embd) + logits de reward/done.

        obs_tokens: (B,T,64), act_tokens: (B,T).

        Importante: reward_logits/done_logits aqui descrevem a transicao JA
        COMMITADA por act_tokens[:, -1] -- nao uma acao futura. E a mesma
        leitura que compute_loss usa pra treinar head_rewards/head_dones
        (ctx_last_repr = posicao da ultima acao do contexto). act_tokens[:,-1]
        so tem efeito causal na chamada SEGUINTE, depois que ele entra na
        janela deslizante -- ele e causalmente inerte dentro desta mesma
        chamada (nenhuma posicao usada aqui consegue "ver" alem da propria
        ultima acao do contexto).
        """
        B, T, K = obs_tokens.shape
        x = self._transform(obs_tokens, act_tokens).view(
            B, T, self.config.tokens_per_block, self.config.n_embd
        )
        state_repr = x[:, -1, -1, :]
        return state_repr, self.head_rewards(state_repr), self.head_dones(state_repr)

    @torch.no_grad()
    def encode_state_and_frame(self, obs_tokens, act_tokens):
        """
        Como encode_state, mas tambem retorna frame_repr (B,n_embd): a media
        das 64 posicoes de token visual do ULTIMO frame do contexto (antes da
        posicao de acao), pooled num vetor so. Existe pra aproximar o desenho
        do controlador do paper original (Ha & Schmidhuber 2018): la, C recebe
        z_t (latente da VAE do frame atual) CONCATENADO com h_t (estado oculto
        da RNN, usado pra prever o proximo z) -- aqui, state_repr faz o papel
        de h_t (ja e o mesmo vetor usado pra prever reward/done/proximo frame)
        e frame_repr aproxima z_t (representacao do frame atual). Ressalva:
        frame_repr passa pela mascara causal, entao carrega alguma mistura do
        contexto anterior tambem -- nao e um z "puro" isolado do frame, so uma
        aproximacao pratica reaproveitando o forward que ja existe.
        """
        B, T, K = obs_tokens.shape
        x = self._transform(obs_tokens, act_tokens).view(
            B, T, self.config.tokens_per_block, self.config.n_embd
        )
        state_repr = x[:, -1, -1, :]
        frame_repr = x[:, -1, :-1, :].mean(dim=1)
        return state_repr, frame_repr, self.head_rewards(state_repr), self.head_dones(state_repr)

    def _project_kv(self, layer, normed_x):
        """Projeta K,V de uma camada a partir do input ja normalizado (norm_first=True e
        o mesmo input que layer.self_attn recebe internamente) -- usa os MESMOS pesos
        (in_proj_weight/bias) que nn.MultiheadAttention usa, sem duplicar parametros."""
        B, L, d_model = normed_x.shape
        n_head = layer.self_attn.num_heads
        head_dim = d_model // n_head
        _, w_k, w_v = layer.self_attn.in_proj_weight.chunk(3, dim=0)
        _, b_k, b_v = layer.self_attn.in_proj_bias.chunk(3, dim=0)
        k = F.linear(normed_x, w_k, b_k).view(B, L, n_head, head_dim).transpose(1, 2)
        v = F.linear(normed_x, w_v, b_v).view(B, L, n_head, head_dim).transpose(1, 2)
        return k, v

    def _incremental_layer_step(self, layer, x_new, k_cache, v_cache):
        """Roda 1 camada do transformer para UM token novo, atendendo ao cache de K,V das
        posicoes anteriores (em vez de recomputar a sequencia inteira). Replica exatamente
        TransformerEncoderLayer.forward (norm_first=True) usando os mesmos submodulos/pesos
        da camada -- so a atencao e feita manualmente pra poder usar o cache.
        x_new: (B,1,n_embd):  embedding (+pos) do token novo, antes da camada.
        Retorna (x_out, k_cache, v_cache) com o cache ja estendido."""
        B, _, d_model = x_new.shape
        n_head = layer.self_attn.num_heads
        head_dim = d_model // n_head

        normed = layer.norm1(x_new)
        w_q, w_k, w_v = layer.self_attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = layer.self_attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(normed, w_q, b_q).view(B, 1, n_head, head_dim).transpose(1, 2)
        k_new = F.linear(normed, w_k, b_k).view(B, 1, n_head, head_dim).transpose(1, 2)
        v_new = F.linear(normed, w_v, b_v).view(B, 1, n_head, head_dim).transpose(1, 2)

        k_cache = torch.cat([k_cache, k_new], dim=2)
        v_cache = torch.cat([v_cache, v_new], dim=2)

        attn = F.scaled_dot_product_attention(q, k_cache, v_cache)
        attn = attn.transpose(1, 2).reshape(B, 1, d_model)
        attn = layer.self_attn.out_proj(attn)

        x = x_new + layer.dropout1(attn)
        normed2 = layer.norm2(x)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed2))))
        x = x + layer.dropout2(ff)
        return x, k_cache, v_cache

    @torch.no_grad()
    def imagine_next_frame(self, obs_tokens, act_tokens, act_token, temperature=1.0, top_k=50):
        """Gera os `img_tokens` do proximo frame com KV-cache: 1 forward completo sobre o
        contexto conhecido (T frames) semeia o cache de cada camada, depois cada token novo
        so processa a si mesmo (atendendo ao cache), em vez de recomputar a sequencia inteira
        a cada um dos `img_tokens` passos. Equivalente ao `imagine_next_frame` anterior (ver
        models/dynamics/test_kv_cache.py), so que sem o trabalho redundante.

        `act_token` (a acao recem-escolhida) nao entra em nenhum calculo aqui de proposito --
        ela e causalmente inerte dentro desta mesma chamada (ver docstring de encode_state);
        so importa na chamada SEGUINTE, depois de entrar na janela deslizante. Fica como
        parametro so por compatibilidade de interface com quem chama."""
        B, T, K = obs_tokens.shape
        device = obs_tokens.device

        x = self._embed(obs_tokens, act_tokens)  # (B, T*tokens_per_block, n_embd), sem o frame novo
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        kv_cache = []
        for layer in self.transformer.layers:
            kv_cache.append(self._project_kv(layer, layer.norm1(x)))
            x = layer(x, src_mask=mask, is_causal=True)

        h = x[:, -1, :]  # representacao na ultima posicao do contexto conhecido
        reward_logits = self.head_rewards(h)
        done_logits = self.head_dones(h)
        reward_pred = reward_logits.argmax(dim=-1)
        done_pred = done_logits.argmax(dim=-1)

        generated = torch.zeros(B, K, dtype=torch.long, device=device)
        h_cur = h

        for k in range(K):
            logits = self.head_obs(h_cur) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            generated[:, k] = token

            if k < K - 1:
                pos = torch.full((1,), seq_len + k, device=device, dtype=torch.long)
                x_new = self.drop(self.obs_emb(token).unsqueeze(1) + self.pos_emb(pos))
                for i, layer in enumerate(self.transformer.layers):
                    k_cache, v_cache = kv_cache[i]
                    x_new, k_cache, v_cache = self._incremental_layer_step(layer, x_new, k_cache, v_cache)
                    kv_cache[i] = (k_cache, v_cache)
                h_cur = x_new.squeeze(1)

        return generated, reward_pred, done_pred