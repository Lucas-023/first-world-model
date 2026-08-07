"""
Teste de encanamento (nao de qualidade) do pipeline de RL via imaginacao.
Modelos minusculos, pesos aleatorios, roda em segundos na CPU. Sem pytest
(repo nao usa), so asserts + prints, igual aos test_*.py existentes.

Uso: python -m models.policy.smoke_test
"""

import numpy as np
import torch

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.dynamics.dataset import reward_to_class
from models.policy.modules import ActorCritic, ReturnNormalizer, soft_update_
from models.policy.rollout import collect_rollout, compute_active_mask, compute_gae
from models.policy.train_dream import ppo_update
from models.policy.train_real_dreamer import pad_and_concat_batches
from models.policy.online_buffer import OnlineReplayBuffer


def tiny_world_model():
    config = WorldModelConfig(
        obs_vocab_size=16, act_vocab_size=5, img_tokens=8,
        context_len=3, n_embd=32, n_head=2, n_layer=1, dropout=0.0,
    )
    model = WorldModel(config)
    model.eval()
    return model, config


def test_shapes():
    model, config = tiny_world_model()
    ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)
    B, T, K = 3, config.context_len, config.img_tokens

    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    state, rlog, dlog = model.encode_state(obs_ctx, act_ctx)
    assert state.shape == (B, config.n_embd)
    assert rlog.shape == (B, 3)
    assert dlog.shape == (B, 2)

    action, log_prob, value, entropy = ac.act(state)
    assert action.shape == (B,) and action.dtype == torch.long
    assert bool((action >= 0).all()) and bool((action < config.act_vocab_size).all())
    assert log_prob.shape == (B,) and value.shape == (B,) and entropy.shape == (B,)

    next_frame, reward_pred, done_pred = model.imagine_next_frame(obs_ctx, act_ctx, action)
    assert next_frame.shape == (B, K) and next_frame.dtype == torch.long
    assert bool((next_frame >= 0).all()) and bool((next_frame < config.obs_vocab_size).all())
    assert reward_pred.shape == (B,) and done_pred.shape == (B,)
    print("OK: 1) formas")


def test_act_token_inertness():
    """Trava de regressao: act_token nao tem efeito causal DENTRO da mesma
    chamada de imagine_next_frame -- so passa a importar na chamada seguinte,
    depois de entrar na janela deslizante (ver docstring de encode_state)."""
    model, config = tiny_world_model()
    B, T, K = 2, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    action_A = torch.zeros(B, dtype=torch.long)
    action_B = torch.full((B,), config.act_vocab_size - 1, dtype=torch.long)

    torch.manual_seed(7)
    frame_A, rew_A, done_A = model.imagine_next_frame(obs_ctx, act_ctx, action_A)
    torch.manual_seed(7)
    frame_B, rew_B, done_B = model.imagine_next_frame(obs_ctx, act_ctx, action_B)

    assert torch.equal(frame_A, frame_B), "next_frame nao deveria depender de act_token"
    assert torch.equal(rew_A, rew_B), "reward_pred nao deveria depender de act_token"
    assert torch.equal(done_A, done_B), "done_pred nao deveria depender de act_token"
    print("OK: 2) act_token e causalmente inerte dentro da mesma chamada (confirmado)")


def test_shift_correctness():
    """A consequencia de action_0 (reward/done) so aparece no passo t=1
    (via encode_state pos-slide), nunca no proprio t=0."""
    model, config = tiny_world_model()
    B, T, K = 2, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    action_A = torch.zeros(B, dtype=torch.long)
    action_B = torch.full((B,), config.act_vocab_size - 1, dtype=torch.long)

    torch.manual_seed(11)
    next_frame, _, _ = model.imagine_next_frame(obs_ctx, act_ctx, action_A)  # inerte a acao, so precisa de 1

    obs_ctx_1 = torch.cat([obs_ctx[:, 1:], next_frame.unsqueeze(1)], dim=1)
    act_ctx_1_A = torch.cat([act_ctx[:, 1:], action_A.unsqueeze(1)], dim=1)
    act_ctx_1_B = torch.cat([act_ctx[:, 1:], action_B.unsqueeze(1)], dim=1)

    _, rlog_1_A, dlog_1_A = model.encode_state(obs_ctx_1, act_ctx_1_A)
    _, rlog_1_B, dlog_1_B = model.encode_state(obs_ctx_1, act_ctx_1_B)
    assert not torch.equal(rlog_1_A, rlog_1_B), "reward_logits do passo seguinte deveria depender de action_0"

    # determinismo: mesma chamada, mesma entrada -> mesma saida (sem no_grad/dropout escondendo ruido)
    _, rlog_1_A2, dlog_1_A2 = model.encode_state(obs_ctx_1, act_ctx_1_A)
    assert torch.equal(rlog_1_A, rlog_1_A2) and torch.equal(dlog_1_A, dlog_1_A2)
    print("OK: 3) deslocamento reward/done por 1 passo confirmado")


def test_gradient_isolation():
    model, config = tiny_world_model()
    ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)

    B, T, K = 4, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    buffer = collect_rollout(model, ac, obs_ctx, act_ctx, horizon=4)
    ppo_update(ac, opt, buffer, n_epochs=1, minibatch_size=4, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)

    for p in model.parameters():
        assert p.grad is None, "gradiente vazou pro World Model (deveria estar congelado)"
    has_grad = False
    for p in ac.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
    assert has_grad, "ActorCritic deveria ter recebido gradiente"
    print("OK: 4) isolamento de gradiente (World Model intocado, ActorCritic atualizado)")


def test_gae_masking():
    H, B = 5, 2
    dones = torch.zeros(H, B)
    dones[2, 0] = 1.0  # linha 0 termina em t=2; linha 1 nunca termina

    def make_buffer(garbage_reward, garbage_value):
        rewards = torch.full((H, B), 0.1)
        values = torch.full((H, B), 0.5)
        rewards[3, 0], rewards[4, 0] = garbage_reward, -garbage_reward
        values[3, 0], values[4, 0] = garbage_value, -garbage_value
        return rewards, values

    bootstrap_value = torch.zeros(B)
    active_mask = compute_active_mask(dones)
    assert torch.equal(active_mask[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(active_mask[:, 1], torch.ones(H))

    rewards1, values1 = make_buffer(1e6, 1e6)
    rewards2, values2 = make_buffer(999.0, 42.0)

    adv1, ret1 = compute_gae(rewards1, values1, dones, active_mask, bootstrap_value)
    adv2, ret2 = compute_gae(rewards2, values2, dones, active_mask, bootstrap_value)

    assert torch.allclose(adv1, adv2), "advantage nao deveria mudar com lixo pos-done (firewall do GAE + mascara)"
    assert torch.allclose(ret1 * active_mask, ret2 * active_mask), "returns ponderados pela mascara deveriam ser identicos"
    # linha 1 (nunca termina) deve continuar contribuindo normalmente nos 5 passos
    assert bool((active_mask[:, 1] == 1.0).all())
    print("OK: 5) mascara pos-done protege GAE/returns de lixo (linha ativa continua intacta)")


def test_pack_and_concat():
    """train_real_dreamer.py::pad_and_concat_batches combina o buffer real
    (H,B variavel por episodio) com o buffer imaginado (H=horizon fixo) --
    trava de regressao pra garantir que o padding do lado mais raso fica
    marcado como inativo (active_mask=0) e nao vaza lixo pra loss."""
    device = torch.device("cpu")
    state_dim = 4

    def fake_buffer(H, B, fill_value):
        return {
            "states": torch.full((H, B, state_dim), fill_value),
            "actions": torch.zeros(H, B, dtype=torch.long),
            "log_probs": torch.full((H, B), fill_value),
            "advantages": torch.full((H, B), fill_value),
            "returns": torch.full((H, B), fill_value),
            "active_mask": torch.ones(H, B),
            "rewards": torch.full((H, B), fill_value),
        }

    buf_a = fake_buffer(H=3, B=2, fill_value=1.0)  # ex.: buffer "real"
    buf_b = fake_buffer(H=2, B=3, fill_value=2.0)  # ex.: buffer "imaginado" (horizon menor)

    combined = pad_and_concat_batches(buf_a, buf_b, device)

    assert combined["states"].shape == (3, 5, state_dim)
    assert combined["active_mask"].shape == (3, 5)

    # colunas 0-1 vieram de buf_a (H=3) -- todas as 3 linhas ativas
    assert torch.equal(combined["active_mask"][:, :2], torch.ones(3, 2))
    assert torch.equal(combined["states"][:, :2], torch.full((3, 2, state_dim), 1.0))

    # colunas 2-4 vieram de buf_b (H=2) -- so as 2 primeiras linhas ativas,
    # a 3a linha (padding) tem active_mask=0 e estado zerado
    assert torch.equal(combined["active_mask"][:2, 2:], torch.ones(2, 3))
    assert torch.equal(combined["active_mask"][2, 2:], torch.zeros(3))
    assert torch.equal(combined["states"][:2, 2:], torch.full((2, 3, state_dim), 2.0))
    assert torch.equal(combined["states"][2, 2:], torch.zeros(3, state_dim))
    print("OK: 6) padding+concat real/imaginado marca preenchimento como inativo")


def test_target_critic_and_return_normalizer():
    """Trava de regressao pra soft_update_ (EMA do critico-alvo) e
    ReturnNormalizer (escala de advantage por percentil de retorno,
    DreamerV3): (1) soft_update_ move o alvo em direcao a fonte pela fracao
    exata de (1-decay), nunca copia tudo de uma vez; (2) ReturnNormalizer SO
    escala (nao centraliza) e a faixa aprendida bate com os percentis reais
    dos retornos de entrada; (3) collect_rollout aceita target_actor_critic e
    ppo_update aceita return_normalizer sem quebrar formas/gradiente."""
    torch.manual_seed(0)
    model, config = tiny_world_model()
    ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)
    target_ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)

    p_before = [p.clone() for p in target_ac.parameters()]
    decay = 0.9
    soft_update_(target_ac, ac, decay)
    for p_old, p_new, p_src in zip(p_before, target_ac.parameters(), ac.parameters()):
        expected = decay * p_old + (1 - decay) * p_src
        assert torch.allclose(p_new, expected, atol=1e-6), "soft_update_ deveria mover o alvo por exatamente (1-decay) em direcao a fonte"

    normalizer = ReturnNormalizer(decay=0.5, low=0.05, high=0.95)
    returns = torch.cat([torch.linspace(-1, 1, 98), torch.tensor([100.0, -100.0])])  # outliers extremos
    active_mask = torch.ones_like(returns)
    scale1 = normalizer.update_and_get_scale(returns, active_mask)
    lo = torch.quantile(returns, 0.05).item()
    hi = torch.quantile(returns, 0.95).item()
    assert abs(scale1 - (hi - lo)) < 1e-3, "primeira chamada deveria inicializar direto na faixa real (sem EMA ainda)"
    assert scale1 < 50, "outliers isolados (100/-100) nao deveriam dominar a escala (isso e o ponto de usar percentil, nao min/max)"

    returns2 = returns * 10  # faixa bem maior
    scale2 = normalizer.update_and_get_scale(returns2, active_mask)
    assert scale1 < scale2 < (torch.quantile(returns2, 0.95) - torch.quantile(returns2, 0.05)).item() + 1e-3, \
        "segunda chamada deveria suavizar (EMA) em direcao a nova faixa, nao pular direto pra ela"

    B, T, K = 4, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))
    buffer = collect_rollout(model, ac, obs_ctx, act_ctx, horizon=4, target_actor_critic=target_ac)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    ppo_update(
        ac, opt, buffer, n_epochs=1, minibatch_size=4, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, return_normalizer=normalizer,
    )
    for p in model.parameters():
        assert p.grad is None, "gradiente vazou pro World Model"
    print("OK: 8) critico-alvo EMA e normalizacao de advantage por percentil")


def test_online_buffer():
    """models/policy/online_buffer.py::OnlineReplayBuffer -- trava de
    regressao pra 3 coisas: (1) amostragem pondera por numero de janelas
    por episodio (nao por episodio uniformemente -- reproduz a mesma
    distribuicao de CarRacingTokenDataset); (2) conversao de reward pra
    classe bate com dataset.py::reward_to_class; (3) FIFO por capacidade
    descarta o episodio mais antigo por inteiro, sem deixar janela orfa em
    window_index."""
    context_len = 5
    device = torch.device("cpu")

    def fake_episode(T, seed):
        rng = np.random.default_rng(seed)
        return {
            "tokens": rng.integers(0, 512, size=(T, 64)).astype(np.uint16),
            "actions": rng.integers(0, 5, size=(T,)).astype(np.int32),
            "rewards": rng.normal(size=(T,)).astype(np.float32),
            "dones": np.array([0] * (T - 1) + [1], dtype=np.uint8),
        }

    buf = OnlineReplayBuffer(capacity_steps=1000, context_len=context_len, device=device)
    ep_a = fake_episode(10, seed=1)   # 10 - 5 = 5 janelas
    ep_b = fake_episode(20, seed=2)   # 20 - 5 = 15 janelas
    ep_c = fake_episode(3, seed=3)    # curto demais (T < context_len+1=6): 0 janelas, mas conta pra capacidade

    buf.add_episode(ep_a)
    buf.add_episode(ep_b)
    buf.add_episode(ep_c)

    assert buf.total_steps == 10 + 20 + 3
    assert len(buf) == 5 + 15 + 0, "amostragem tem que ponderar por janelas/episodio, nao por episodio"
    assert buf.num_episodes() == 3

    obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt = buf.sample_training_windows(16, device)
    assert obs_ctx.shape == (16, context_len, 64) and obs_ctx.dtype == torch.int64
    assert act_ctx.shape == (16, context_len) and act_ctx.dtype == torch.int64
    assert obs_tgt.shape == (16, 64) and obs_tgt.dtype == torch.int64
    assert rew_tgt.shape == (16,) and rew_tgt.dtype == torch.int64
    assert done_tgt.shape == (16,) and done_tgt.dtype == torch.int64
    assert bool(((rew_tgt >= 0) & (rew_tgt <= 2)).all()), "classe de reward tem que estar em {0,1,2}"

    seed_obs_ctx, seed_act_ctx = buf.sample_seed_contexts(8, device)
    assert seed_obs_ctx.shape == (8, context_len, 64)
    assert seed_act_ctx.shape == (8, context_len)

    # conversao de reward bate com a MESMA funcao que o dataset offline usa --
    # reconfirma comparando ep.rewards_class (calculado dentro do buffer) com
    # uma nova chamada de reward_to_class sobre o reward bruto do episodio
    # original (identificado pelo tamanho: T=10 -> ep_a, T=20 -> ep_b)
    eid, start = buf.window_index[0]
    ep = buf.episodes_by_id[eid]
    target_idx = start + context_len
    raw = ep_a["rewards"] if ep["tokens"].shape[0] == 10 else ep_b["rewards"]
    assert ep["rewards_class"][target_idx] == reward_to_class(raw[target_idx:target_idx + 1])[0]

    # FIFO: capacidade estourada descarta o episodio mais antigo por inteiro,
    # sem deixar janela orfa em window_index
    buf2 = OnlineReplayBuffer(capacity_steps=25, context_len=context_len, device=device)
    buf2.add_episode(fake_episode(10, seed=10))   # eid=0
    buf2.add_episode(fake_episode(10, seed=11))   # eid=1
    buf2.add_episode(fake_episode(10, seed=12))   # eid=2, total=30 > 25 -> descarta eid=0
    assert buf2.num_episodes() == 2
    assert buf2.total_steps == 20
    assert len(buf2) == 10  # 2 episodios de 10 passos, 5 janelas cada
    assert 0 not in buf2.episode_order
    assert all(eid != 0 for eid, _ in buf2.window_index), "janela orfa apontando pro episodio descartado"

    print("OK: 7) OnlineReplayBuffer -- amostragem ponderada, classe de reward e FIFO corretos")


if __name__ == "__main__":
    test_shapes()
    test_act_token_inertness()
    test_shift_correctness()
    test_gradient_isolation()
    test_gae_masking()
    test_pack_and_concat()
    test_target_critic_and_return_normalizer()
    test_online_buffer()
    print("\nTodos os smoke tests passaram.")
