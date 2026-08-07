"""
Teste de equivalencia: confirma que o imagine_next_frame com KV-cache (em
gptdynamics.py) da o mesmo resultado que a implementacao anterior, sem cache
(que recomputava a sequencia inteira a cada um dos img_tokens passos). A
versao antiga fica reimplementada aqui SO pra comparar -- nao roda em
producao, nao mantem estado nenhum do modelo real.

Uso: python -m models.dynamics.test_kv_cache
"""

import time
import torch
import torch.nn.functional as F

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig


@torch.no_grad()
def _imagine_next_frame_no_cache(model, obs_tokens, act_tokens, act_token, temperature=1.0, top_k=50):
    """Copia fiel da implementacao anterior (pre-KV-cache) de imagine_next_frame."""
    B, T, K = obs_tokens.shape
    device = obs_tokens.device

    next_frame = torch.zeros(B, 1, K, dtype=torch.long, device=device)
    ctx_obs = torch.cat([obs_tokens, next_frame], dim=1)
    ctx_act = torch.cat([act_tokens, act_token.unsqueeze(1)], dim=1)

    _, reward_logits, done_logits = model.encode_state(obs_tokens, act_tokens)
    reward_pred = reward_logits.argmax(dim=-1)
    done_pred = done_logits.argmax(dim=-1)

    for k in range(K):
        x = model._transform(ctx_obs, ctx_act)
        x = x.view(B, T + 1, model.config.tokens_per_block, model.config.n_embd)

        if k == 0:
            repr_k = x[:, -2, -1, :]
        else:
            repr_k = x[:, -1, k - 1, :]

        logits = model.head_obs(repr_k) / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        ctx_obs[:, -1, k] = token

    return ctx_obs[:, -1, :], reward_pred, done_pred


def _tiny_model():
    config = WorldModelConfig(
        obs_vocab_size=32, act_vocab_size=5, img_tokens=8,
        context_len=4, n_embd=32, n_head=4, n_layer=3, dropout=0.0,
    )
    model = WorldModel(config)
    model.eval()
    return model, config


def _random_inputs(config, B):
    T, K = config.context_len, config.img_tokens
    obs_tokens = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_tokens = torch.randint(0, config.act_vocab_size, (B, T))
    act_token = torch.randint(0, config.act_vocab_size, (B,))
    return obs_tokens, act_tokens, act_token


def test_reward_done_identical():
    """reward/done nao passam por multinomial -- vem do mesmo forward nativo
    (nn.TransformerEncoder cai no path sem nested-tensor aqui, entao meu loop
    manual por camada e literalmente o mesmo calculo). Deve bater EXATO."""
    model, config = _tiny_model()
    obs_tokens, act_tokens, act_token = _random_inputs(config, B=3)

    _, rew_old, done_old = _imagine_next_frame_no_cache(model, obs_tokens, act_tokens, act_token)
    _, rew_new, done_new = model.imagine_next_frame(obs_tokens, act_tokens, act_token)

    assert torch.equal(rew_old, rew_new), "reward_pred deveria ser identico com/sem cache"
    assert torch.equal(done_old, done_new), "done_pred deveria ser identico com/sem cache"
    print("OK: 1) reward/done identicos com e sem KV-cache")


def test_greedy_tokens_match():
    """Troca multinomial por argmax (decodificacao gulosa) nas duas implementacoes,
    pra isolar a logica numerica da ordem de consumo do RNG -- se a matematica do
    cache estiver certa, os tokens gerados tem que bater token a token."""
    model, config = _tiny_model()
    obs_tokens, act_tokens, act_token = _random_inputs(config, B=2)

    orig_multinomial = torch.multinomial

    def greedy_multinomial(probs, num_samples, **kwargs):
        return probs.argmax(dim=-1, keepdim=True)

    torch.multinomial = greedy_multinomial
    try:
        frame_old, rew_old, done_old = _imagine_next_frame_no_cache(model, obs_tokens, act_tokens, act_token)
        frame_new, rew_new, done_new = model.imagine_next_frame(obs_tokens, act_tokens, act_token)
    finally:
        torch.multinomial = orig_multinomial

    assert torch.equal(frame_old, frame_new), f"tokens gerados divergiram:\n{frame_old}\nvs\n{frame_new}"
    assert torch.equal(rew_old, rew_new)
    assert torch.equal(done_old, done_new)
    print("OK: 2) tokens gerados (decodificacao gulosa) identicos com e sem KV-cache")


def test_seeded_reproducible():
    """Com amostragem estocastica real (multinomial de verdade), o mesmo seed
    antes da chamada tem que dar o mesmo sonho -- confirma que o cache nao
    introduziu nenhuma fonte de nao-determinismo escondida."""
    model, config = _tiny_model()
    obs_tokens, act_tokens, act_token = _random_inputs(config, B=2)

    torch.manual_seed(123)
    frame_a, rew_a, done_a = model.imagine_next_frame(obs_tokens, act_tokens, act_token)
    torch.manual_seed(123)
    frame_b, rew_b, done_b = model.imagine_next_frame(obs_tokens, act_tokens, act_token)

    assert torch.equal(frame_a, frame_b), "mesmo seed deveria dar o mesmo sonho"
    assert torch.equal(rew_a, rew_b) and torch.equal(done_a, done_b)
    print("OK: 3) mesmo seed => mesmo resultado (reprodutibilidade do KV-cache)")


def test_timing_info():
    """So informativo -- nesse tamanho de modelo (brinquedo) o overhead do loop
    Python pode dominar e mascarar o ganho real; o ganho de verdade so aparece
    no tamanho real (context_len=19, img_tokens=64) e deve ser medido com
    --benchmark_only na GPU, nao aqui."""
    model, config = _tiny_model()
    obs_tokens, act_tokens, act_token = _random_inputs(config, B=4)

    n = 10
    t0 = time.time()
    for _ in range(n):
        _imagine_next_frame_no_cache(model, obs_tokens, act_tokens, act_token)
    t_old = time.time() - t0

    t0 = time.time()
    for _ in range(n):
        model.imagine_next_frame(obs_tokens, act_tokens, act_token)
    t_new = time.time() - t0

    print(f"OK: 4) info -- sem cache {t_old:.3f}s vs com cache {t_new:.3f}s nesse modelo brinquedo")


if __name__ == "__main__":
    test_reward_done_identical()
    test_greedy_tokens_match()
    test_seeded_reproducible()
    test_timing_info()
    print("\nTodos os testes de equivalencia do KV-cache passaram.")
