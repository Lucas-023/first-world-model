"""
Loop online de verdade (estilo IRIS/DreamerV3): o World Model DEIXA DE SER
CONGELADO durante o treino da politica. Um unico processo intercala, em
ciclos continuos:
  1) alguns passos reais no CarRacing-v3 (politica atual, estocastica, com
     pequeno epsilon de exploracao) -- alimentam um replay buffer online
     (models/policy/online_buffer.py::OnlineReplayBuffer);
  2) alguns passos de gradiente no World Model (WorldModel.compute_loss)
     sobre janelas amostradas desse buffer;
  3) alguns updates de PPO na politica, TREINADA SO POR IMAGINACAO (igual
     train_dream.py) -- as sementes do rollout imaginado vem do buffer, mas
     o sinal de treino do PPO e inteiramente o reward previsto pelo World
     Model dentro da imaginacao, nunca o reward real diretamente.

Diferenca central pros outros scripts deste projeto: em todos eles
(train_dream.py, train_real_latent.py, train_real_dreamer.py) o World Model
e treinado uma vez, offline, e depois usado congelado pra sempre. Aqui ele
continua aprendendo com a experiencia que a propria politica (em treino)
esta gerando -- fecha o loop de verdade, nao em rodadas grandes como
collect_from_policy.py + traingpt.py --resume_ckpt faziam.

Cuidado que nao existia nos outros scripts: como o World Model alterna
entre .eval() (escolher acao real / rollout imaginado) e .train() (passo de
gradiente), e ele tem dropout=0.1, e preciso ser explicito sobre qual modo
esta ativo em cada chamada -- nos scripts anteriores isso nunca importava
porque o WM ficava em .eval() pra sempre (congelado).

Uso:
    python -m models.policy.train_online --benchmark_only
    python -m models.policy.train_online --cycles 2000
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import time
from collections import deque
import torch
from torch.amp import autocast, GradScaler

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.policy.modules import ActorCritic, ReturnNormalizer, soft_update_
from models.policy.rollout import collect_rollout
from models.policy.train_dream import ppo_update, save_policy_dream
from models.policy.eval_real_env import make_eval_env, run_episode_policy
from models.policy.collect_from_policy import RealEpisodeStepper
from models.policy.online_buffer import OnlineReplayBuffer
from models.encoder.modules import VQVAE
from models.encoder.board import Board

EVAL_SEED_BASE = 10_000_000


def load_trainable_world_model(ckpt_path, device):
    """Como eval_real_env.py/train_dream.py::load_world_model, mas NAO
    congela os parametros -- aqui o World Model precisa continuar
    treinavel. Warm-start: comeca de um checkpoint ja pre-treinado
    offline, nao do zero (ele ja aprendeu fisica/visao basica; o loop
    online adapta, nao redescobre)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    model = WorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config


def wm_grad_step(world_model, wm_optimizer, scaler, buffer, batch_size, device_type, device):
    obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt = buffer.sample_training_windows(batch_size, device)
    wm_optimizer.zero_grad(set_to_none=True)
    with autocast(device_type=device_type):
        loss, l_obs, l_rew, l_done = world_model.compute_loss(obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt)
    scaler.scale(loss).backward()
    scaler.unscale_(wm_optimizer)
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
    scaler.step(wm_optimizer)
    scaler.update()
    return loss.item(), l_obs.item(), l_rew.item(), l_done.item()


def policy_update_step(world_model, actor_critic, target_actor_critic, return_normalizer, policy_optimizer, buffer, args, device):
    obs_ctx, act_ctx = buffer.sample_seed_contexts(args.seed_batch_size, device)
    rollout_buf = collect_rollout(
        world_model, actor_critic, obs_ctx, act_ctx, args.horizon, args.gamma, args.gae_lambda,
        target_actor_critic=target_actor_critic,
    )
    stats = ppo_update(
        actor_critic, policy_optimizer, rollout_buf,
        n_epochs=args.n_epochs, minibatch_size=args.minibatch_size,
        clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, return_normalizer=return_normalizer,
    )
    soft_update_(target_actor_critic, actor_critic, args.critic_ema_decay)
    active_sum = rollout_buf["active_mask"].sum().clamp(min=1).item()
    mean_reward = (rollout_buf["rewards"] * rollout_buf["active_mask"]).sum().item() / active_sum
    return stats, mean_reward


def _full_checkpoint_dict(cycle, world_model, wm_optimizer, actor_critic, policy_optimizer,
                           target_actor_critic, return_normalizer, best_eval_reward):
    """Estado completo (pesos + otimizadores + estabilizadores) -- usado tanto
    pro checkpoint de resume (online_ckpt.pt) quanto pro melhor (online_best.pt),
    pra retomar de qualquer um dos dois sem reiniciar o momentum do Adam."""
    return {
        "cycle": cycle,
        "world_model_state_dict": world_model.state_dict(),
        "wm_optimizer_state_dict": wm_optimizer.state_dict(),
        "actor_critic_state_dict": actor_critic.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "target_actor_critic_state_dict": target_actor_critic.state_dict(),
        "return_normalizer_state_dict": return_normalizer.state_dict(),
        "best_eval_reward": best_eval_reward,
    }


def train(args):
    device = args.device
    device_type = device.split(":")[0]
    os.makedirs(args.save_dir, exist_ok=True)
    img_dir = os.path.join(args.save_dir, "dreams")
    os.makedirs(img_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "online_ckpt.pt")
    best_path = os.path.join(args.save_dir, "online_best.pt")

    world_model, wm_config = load_trainable_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado de {args.dynamics_ckpt} (TREINAVEL, warm-start).")

    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=wm_config.obs_vocab_size).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print(f"VQ-VAE carregado de {args.vqvae_path} (congelado).")

    actor_critic = ActorCritic(
        state_dim=wm_config.n_embd, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)

    # critico-alvo suavizado por EMA + normalizacao de advantage por percentil
    # de retorno (DreamerV3) -- ver modules.py::soft_update_/ReturnNormalizer.
    target_actor_critic = copy.deepcopy(actor_critic)
    target_actor_critic.eval()
    for p in target_actor_critic.parameters():
        p.requires_grad_(False)
    return_normalizer = ReturnNormalizer(decay=args.return_norm_decay).to(device)

    wm_optimizer = world_model.configure_optimizers(weight_decay=args.wm_weight_decay, learning_rate=args.wm_lr)
    policy_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.policy_lr)
    scaler = GradScaler(device_type)

    # env separado do usado pra eval periodico -- o stepper mantem um
    # episodio real em andamento ATRAVES de varios ciclos; rodar eval no
    # MESMO env no meio de um episodio resetaria ele por baixo do stepper.
    env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, render=False)
    eval_env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, render=False)
    n_actions = env.action_space.n

    buffer = OnlineReplayBuffer(args.buffer_capacity_steps, wm_config.context_len, device)
    stepper = RealEpisodeStepper(
        env, world_model, vqvae, actor_critic, wm_config.context_len, device,
        args.skip_frames, args.epsilon, deterministic=False, use_zh=False, n_actions=n_actions,
    )

    seed_counter = [args.seed]

    def start_new_episode():
        world_model.eval()
        ok = stepper.reset(seed_counter[0])
        while not ok:
            seed_counter[0] += 1
            ok = stepper.reset(seed_counter[0])
        seed_counter[0] += 1

    start_new_episode()
    real_episode_count = 0

    def take_real_step():
        nonlocal real_episode_count
        world_model.eval()
        done = stepper.step()
        if done:
            buffer.add_episode(stepper.episode_arrays())
            real_episode_count += 1
            start_new_episode()

    if args.benchmark_only:
        print("Aquecendo buffer (precisa de pelo menos 1 episodio completo)...")
        while len(buffer) == 0:
            take_real_step()

        n_bench = 50
        start = time.time()
        for _ in range(n_bench):
            take_real_step()
        real_elapsed = time.time() - start
        print(f"Real       : {n_bench} passos em {real_elapsed:.2f}s -> {n_bench / real_elapsed:.2f} passos/s")

        world_model.train()
        n_bench_wm = 10
        start = time.time()
        for _ in range(n_bench_wm):
            wm_grad_step(world_model, wm_optimizer, scaler, buffer, args.wm_batch_size, device_type, device)
        wm_elapsed = time.time() - start
        print(f"World Model: {n_bench_wm} passos de gradiente em {wm_elapsed:.2f}s -> {n_bench_wm / wm_elapsed:.2f} passos/s")

        world_model.eval()
        n_bench_policy = 5
        start = time.time()
        for _ in range(n_bench_policy):
            policy_update_step(world_model, actor_critic, target_actor_critic, return_normalizer, policy_optimizer, buffer, args, device)
        policy_elapsed = time.time() - start
        print(f"Politica   : {n_bench_policy} updates de PPO em {policy_elapsed:.2f}s -> {n_bench_policy / policy_elapsed:.2f} updates/s")
        return

    board = Board(args.run_name)
    print(f"TensorBoard: tensorboard --logdir runs/{args.run_name}")

    start_cycle = 0
    best_eval_reward = float("-inf")
    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        world_model.load_state_dict(ckpt["world_model_state_dict"])
        actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        if "wm_optimizer_state_dict" in ckpt:
            wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
        if "policy_optimizer_state_dict" in ckpt:
            policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        if "target_actor_critic_state_dict" in ckpt:
            target_actor_critic.load_state_dict(ckpt["target_actor_critic_state_dict"])
        else:
            target_actor_critic.load_state_dict(actor_critic.state_dict())
        if "return_normalizer_state_dict" in ckpt:
            return_normalizer.load_state_dict(ckpt["return_normalizer_state_dict"])
        start_cycle = ckpt["cycle"] + 1
        best_eval_reward = float("-inf") if args.reset_best else ckpt.get("best_eval_reward", float("-inf"))
        print(f"Continuando do ciclo {start_cycle} | melhor eval: {best_eval_reward:.4f}"
              + (" (resetado com --reset_best)" if args.reset_best else ""))
        print("Buffer de replay NAO e persistido -- comeca vazio, reenche com interacao real.")
    else:
        print("Iniciando do zero.")

    recent_evals = deque(maxlen=max(1, args.best_eval_window))

    if start_cycle >= args.cycles:
        print(
            f"\nNada a fazer: checkpoint ja esta no ciclo {start_cycle - 1}, "
            f"que e >= --cycles {args.cycles}. Para continuar, use --cycles maior "
            f"que {start_cycle - 1}; para recomecar do zero com hiperparametros "
            f"novos (ex.: apos mudar --vf_coef), use um --save_dir novo em vez de "
            f"retomar deste."
        )
        return

    print("\nIniciando loop online (World Model + politica juntos)...")
    for cycle in range(start_cycle, args.cycles):
        for _ in range(args.env_steps_per_cycle):
            take_real_step()

        if len(buffer) == 0:
            print(f"[Ciclo {cycle:>5}] aquecendo buffer ({buffer.total_steps} passos reais, {real_episode_count} episodios, 0 janelas validas)...")
            continue

        wm_losses = [
            wm_grad_step(world_model, wm_optimizer, scaler, buffer, args.wm_batch_size, device_type, device)
            for _ in range(args.wm_updates_per_cycle)
        ]

        policy_results = [
            policy_update_step(world_model, actor_critic, target_actor_critic, return_normalizer, policy_optimizer, buffer, args, device)
            for _ in range(args.policy_updates_per_cycle)
        ]
        policy_stats = [r[0] for r in policy_results]
        imagined_rewards = [r[1] for r in policy_results]

        mean_wm_loss = sum(l[0] for l in wm_losses) / len(wm_losses)
        mean_policy_loss = sum(s["policy_loss"] for s in policy_stats) / len(policy_stats)
        mean_value_loss = sum(s["value_loss"] for s in policy_stats) / len(policy_stats)
        mean_entropy = sum(s["entropy"] for s in policy_stats) / len(policy_stats)
        mean_imag_reward = sum(imagined_rewards) / len(imagined_rewards)

        print(
            f"[Ciclo {cycle:>5}] wm_loss {mean_wm_loss:.4f} | policy_loss {mean_policy_loss:.4f} | "
            f"value_loss {mean_value_loss:.4f} | entropy {mean_entropy:.4f} | "
            f"reward_imaginado {mean_imag_reward:.4f} | buffer {len(buffer)}j/{buffer.num_episodes()}ep | "
            f"episodios_reais {real_episode_count}"
        )
        board.log_scalar("wm/loss", mean_wm_loss, cycle)
        board.log_scalar("train/policy_loss", mean_policy_loss, cycle)
        board.log_scalar("train/value_loss", mean_value_loss, cycle)
        board.log_scalar("train/entropy", mean_entropy, cycle)
        board.log_scalar("train/imagined_mean_reward", mean_imag_reward, cycle)
        board.log_scalar("buffer/n_windows", len(buffer), cycle)
        board.log_scalar("buffer/n_episodes", buffer.num_episodes(), cycle)
        board.log_scalar("real/n_episodes", real_episode_count, cycle)

        if cycle % args.eval_every == 0 or cycle == args.cycles - 1:
            world_model.eval()
            eval_rewards = []
            for i in range(args.eval_episodes):
                r, _ = run_episode_policy(
                    eval_env, world_model, vqvae, actor_critic, wm_config.context_len, device,
                    args.skip_frames, deterministic=True, seed=EVAL_SEED_BASE + i,
                )
                eval_rewards.append(r)
            eval_reward = sum(eval_rewards) / len(eval_rewards)
            print(f"  [Eval seeds fixas] reward_medio {eval_reward:.2f} ({args.eval_episodes} episodios)")
            board.log_scalar("eval/mean_reward", eval_reward, cycle)

            # Media movel das ultimas --best_eval_window avaliacoes como criterio de
            # "melhor" -- um unico eval_reward e o maximo de dezenas/centenas de
            # sorteios ruidosos ao longo do treino (winner's-curse: ver
            # docs/DECISIONS.md "Council consultation" -- o pico historico de 747
            # ficou ~2 desvios acima da media real do processo). best_eval_window=1
            # (default) preserva o comportamento antigo.
            recent_evals.append(eval_reward)
            smoothed_eval_reward = sum(recent_evals) / len(recent_evals)
            if len(recent_evals) == recent_evals.maxlen and smoothed_eval_reward > best_eval_reward:
                best_eval_reward = smoothed_eval_reward
                torch.save(
                    _full_checkpoint_dict(
                        cycle, world_model, wm_optimizer, actor_critic, policy_optimizer,
                        target_actor_critic, return_normalizer, best_eval_reward,
                    ),
                    best_path,
                )
                print(f"  -> Novo melhor checkpoint (eval suavizado: {best_eval_reward:.2f}, janela={len(recent_evals)})")

        if args.viz_every > 0 and cycle % args.viz_every == 0:
            try:
                world_model.eval()
                save_path = os.path.join(img_dir, f"online_dream_{cycle:05d}.png")
                save_policy_dream(world_model, vqvae, actor_critic, stepper.obs_ctx, stepper.act_ctx, save_path, args.viz_frames, board, cycle)
                print(f"  -> Imagem: {save_path}")
            except Exception as e:
                print(f"  Aviso viz: {e}")

        if args.ckpt_every > 0 and cycle % args.ckpt_every == 0:
            torch.save(
                {
                    "cycle": cycle,
                    "world_model_state_dict": world_model.state_dict(),
                    "actor_critic_state_dict": actor_critic.state_dict(),
                    "best_eval_reward": best_eval_reward,
                },
                os.path.join(args.save_dir, f"online_cycle{cycle}.pt"),
            )

        torch.save(
            _full_checkpoint_dict(
                cycle, world_model, wm_optimizer, actor_critic, policy_optimizer,
                target_actor_critic, return_normalizer, best_eval_reward,
            ),
            ckpt_path,
        )

    print("\nTreino finalizado!")
    env.close()
    eval_env.close()
    board.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt", help="warm-start -- o World Model comeca DESTE checkpoint pre-treinado offline, nao do zero")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt_best.pt")
    p.add_argument("--save_dir", type=str, default="models/policy_online")
    p.add_argument("--run_name", type=str, default="POLICY_ONLINE")
    p.add_argument("--cycles", type=int, default=2000)
    p.add_argument("--env_steps_per_cycle", type=int, default=32, help="passos reais dados por ciclo antes de intercalar gradiente do WM/PPO")
    p.add_argument("--wm_updates_per_cycle", type=int, default=4)
    p.add_argument("--wm_batch_size", type=int, default=32)
    p.add_argument("--wm_lr", type=float, default=1e-4, help="mesmo default de traingpt.py")
    p.add_argument("--wm_weight_decay", type=float, default=0.01, help="mesmo default de traingpt.py")
    p.add_argument("--policy_updates_per_cycle", type=int, default=1)
    p.add_argument("--seed_batch_size", type=int, default=32, help="sementes reais amostradas do buffer por update de PPO (batch do rollout imaginado)")
    p.add_argument("--horizon", type=int, default=8, help="passos de imaginacao por rollout -- eval_rollout.py mostrou reward confiavel so ate ~passo 10-11")
    p.add_argument("--policy_lr", type=float, default=3e-4)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_rows", type=int, default=12)
    p.add_argument("--skip_frames", type=int, default=12, help="frames de introducao (zoom) ignorados, igual extract_tokens.py")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--epsilon", type=float, default=0.01, help="exploracao aleatoria na coleta real")
    p.add_argument("--buffer_capacity_steps", type=int, default=100_000, help="capacidade do replay buffer online em passos (FIFO) -- ~poucas centenas de episodios")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.05, help="default menor que train_dream.py (0.5) -- diagnostico de colapso de entropia no run original ja motivou essa mudanca (ver techreport.tex), reaproveitado aqui de train_real_latent.py/train_real_dreamer.py")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--critic_ema_decay", type=float, default=0.98, help="suavizacao do critico-alvo (modules.py::soft_update_)")
    p.add_argument("--return_norm_decay", type=float, default=0.99, help="suavizacao EMA da faixa de percentil usada pra escalar advantage (modules.py::ReturnNormalizer)")
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--best_eval_window", type=int, default=1, help="tamanho da media movel de eval/mean_reward usada pra escolher 'melhor checkpoint' -- 1 (default) preserva o comportamento antigo (ponto unico); >1 reduz a inflacao por sorte de amostragem (ver docs/DECISIONS.md 'Council consultation': o pico de 747 ficou ~2 desvios acima da media real do processo)")
    p.add_argument("--reset_best", action="store_true", help="ao retomar de um checkpoint de outro save_dir (ex.: copiado de outro experimento), nao herdar o best_eval_reward dele -- sem isso, um resume so consegue salvar um novo 'best' se superar a marca do experimento de origem, o que pode nunca acontecer e mascarar silenciosamente que o mecanismo de best parou de funcionar")
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=20)
    p.add_argument("--viz_frames", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)
