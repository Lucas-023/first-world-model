"""
Coleta episodios reais usando uma politica JA TREINADA (ActorCritic sobre
state_repr do World Model) e escreve .npz direto no formato que
models/dynamics/dataset.py::CarRacingTokenDataset espera (tokens/actions/
rewards/dones) -- o mesmo formato que extract_tokens.py produz a partir do
dataset bruto de agente_coleta/coleta.py, so sem o passo intermediario de
salvar pixels crus e re-tokenizar depois (o token ja sai computado durante o
proprio rollout).

Fecha o loop real -> World Model que faltava: hoje o World Model
(models/dynamics/traingpt.py) e treinado UMA VEZ, offline, num dataset
coletado por um PPO de pixels antes de qualquer politica via World Model
existir -- ele nunca ve os estados que as politicas treinadas aqui
(train_real_latent.py, train_real_dreamer.py) realmente visitam. Rodar este
script com uma politica ja treinada e depois continuar o treino do World
Model nos episodios novos + antigos juntos (traingpt.py --resume_ckpt ...
--dataset_path dataset_tokens) fecha esse loop.

IMPORTANTE -- convencao de indices dos arrays salvos (ver
agente_coleta/coleta.py e models/dynamics/dataset.py): tokens[t]/actions[t]/
rewards[t]/dones[t] tem que ficar alinhados exatamente como coleta.py sempre
gravou -- actions[t] e a acao ESCOLHIDA a partir do frame tokens[t] (nao a
acao que produziu tokens[t]); rewards[t]/dones[t] sao o resultado de aplicar
essa acao. A janela deslizante (obs_ctx/act_ctx) usada aqui so pra ESCOLHER
a acao a cada passo e outra coisa, com outro pareamento (mesma logica de
train_real_latent.py::collect_episode: "quadro que chegou / acao que
causou") -- os dois buffers sao mantidos em paralelo, mas so o buffer no
formato coleta.py e salvo em disco. Nao reaproveitar a janela deslizante
como se fosse o buffer de arquivo -- os dois tem pareamento diferente.

Uso:
    python -m models.policy.collect_from_policy \
        --policy_ckpt models/policy_real_latent/policy_best.pt \
        --n_episodes 200 --tag real_latent
    python -m models.policy.collect_from_policy \
        --policy_ckpt models/policy_real_latent_zh/policy_best.pt \
        --zh --n_episodes 200 --tag real_latent_zh
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import torch

from models.policy.modules import ActorCritic
from models.policy.eval_real_env import make_eval_env, frame_to_tensor, _warmup, load_world_model
from models.encoder.modules import VQVAE


class RealEpisodeStepper:
    """Anda 1 passo real de cada vez, mantendo em paralelo (a) a janela
    deslizante (obs_ctx/act_ctx) usada pra ESCOLHER a acao via World
    Model/ActorCritic -- pareamento "quadro que chegou / acao que causou",
    igual train_real_latent.py::collect_episode -- e (b) os buffers
    achatados (tokens/actions/rewards/dones) no formato dataset_tokens, na
    convencao de indice de agente_coleta/coleta.py (actions[t] e a acao
    ESCOLHIDA a partir de tokens[t], nao a que produziu tokens[t]). Os dois
    pareamentos sao diferentes de proposito -- nao confundir um com o outro
    (ver docstring do modulo).

    Usado tanto por collect_episode_for_dataset (chama .step() em loop ate
    o episodio acabar) quanto por train_online.py (chama .step() um passo
    de cada vez, intercalado com ciclos de treino do World Model/politica)."""

    def __init__(self, env, world_model, vqvae, actor_critic, context_len, device,
                 skip_frames, epsilon, deterministic, use_zh, n_actions):
        self.env = env
        self.world_model = world_model
        self.vqvae = vqvae
        self.actor_critic = actor_critic
        self.context_len = context_len
        self.device = device
        self.skip_frames = skip_frames
        self.epsilon = epsilon
        self.deterministic = deterministic
        self.use_zh = use_zh
        self.n_actions = n_actions

        self.obs_ctx = None
        self.act_ctx = None
        self.tok_buf = self.act_buf = self.rew_buf = self.done_buf = None

    @torch.no_grad()
    def reset(self, seed):
        """Pula os frames de introducao (zoom-in) e monta a janela de
        contexto inicial (context_len passos com acao 'nada'=0) -- SAO
        passos reais, entao entram no buffer de arquivo tambem (nao sao so
        descartados como em train_real_latent.py::collect_episode, que so
        precisa da janela pronta pra comecar o PPO). Devolve False se o
        episodio terminou durante esse warmup (raro, mas possivel)."""
        obs, _ = self.env.reset(seed=seed)
        obs, ended = _warmup(self.env, self.skip_frames)
        if ended:
            return False

        self.tok_buf, self.act_buf, self.rew_buf, self.done_buf = [], [], [], []
        obs_ctx_list, act_ctx_list = [], []
        for _ in range(self.context_len):
            tok = self.vqvae.encode_indices(frame_to_tensor(obs, self.device)).view(1, -1)
            obs_ctx_list.append(tok)
            act_ctx_list.append(torch.zeros(1, dtype=torch.long, device=self.device))

            tok_np = tok.view(-1).cpu().numpy().astype(np.uint16)
            next_obs, reward, terminated, truncated, _ = self.env.step(0)

            self.tok_buf.append(tok_np)
            self.act_buf.append(0)
            self.rew_buf.append(float(reward))
            self.done_buf.append(bool(terminated or truncated))

            obs = next_obs
            if terminated or truncated:
                return False

        self.obs_ctx = torch.stack(obs_ctx_list, dim=1)
        self.act_ctx = torch.stack(act_ctx_list, dim=1)
        return True

    @torch.no_grad()
    def step(self):
        """1 passo real: escolhe a acao (epsilon-aleatoria / argmax
        deterministico / amostra estocastica), aplica no ambiente, atualiza
        a janela deslizante e os buffers achatados. Devolve `done` (bool)."""
        if self.use_zh:
            h, z, _, _ = self.world_model.encode_state_and_frame(self.obs_ctx, self.act_ctx)
            state = torch.cat([h, z], dim=-1)
        else:
            state, _, _ = self.world_model.encode_state(self.obs_ctx, self.act_ctx)

        if np.random.rand() < self.epsilon:
            action_id = int(np.random.randint(self.n_actions))
        elif self.deterministic:
            logits, _ = self.actor_critic.forward(state)
            action_id = int(logits.argmax(dim=-1).item())
        else:
            action, _, _, _ = self.actor_critic.act(state)
            action_id = int(action.item())

        current_tok = self.obs_ctx[:, -1, :].view(-1).cpu().numpy().astype(np.uint16)
        next_obs, reward, terminated, truncated, _ = self.env.step(action_id)
        done = bool(terminated or truncated)

        self.tok_buf.append(current_tok)
        self.act_buf.append(action_id)
        self.rew_buf.append(float(reward))
        self.done_buf.append(done)

        next_tok = self.vqvae.encode_indices(frame_to_tensor(next_obs, self.device)).view(1, -1)
        self.obs_ctx = torch.cat([self.obs_ctx[:, 1:], next_tok.unsqueeze(1)], dim=1)
        self.act_ctx = torch.cat([self.act_ctx[:, 1:], torch.tensor([[action_id]], device=self.device)], dim=1)
        return done

    def episode_arrays(self):
        """Devolve o episodio acumulado ate agora no formato dataset_tokens."""
        return {
            "tokens": np.stack(self.tok_buf).astype(np.uint16),   # (T, 64)
            "actions": np.array(self.act_buf, dtype=np.int32),     # (T,)
            "rewards": np.array(self.rew_buf, dtype=np.float32),   # (T,)
            "dones": np.array(self.done_buf, dtype=np.uint8),      # (T,)
        }


def collect_episode_for_dataset(env, world_model, vqvae, actor_critic, context_len, device,
                                 skip_frames, epsilon, deterministic, seed, use_zh, n_actions):
    """Roda 1 episodio completo via RealEpisodeStepper e devolve os 4 arrays
    no formato dataset_tokens. Devolve None se o episodio terminou durante
    o warmup (zoom-in ou montagem da janela de contexto) -- raro, descartado
    sem contar pro total (CarRacingTokenDataset tambem ignora episodios com
    T < context_len+1, entao um episodio assim nao renderia nenhuma janela
    de treino de qualquer forma)."""
    stepper = RealEpisodeStepper(
        env, world_model, vqvae, actor_critic, context_len, device,
        skip_frames, epsilon, deterministic, use_zh, n_actions,
    )
    if not stepper.reset(seed):
        return None

    done = False
    while not done:
        done = stepper.step()

    return stepper.episode_arrays()


def collect(args):
    device = args.device

    world_model, wm_config = load_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado (context_len={wm_config.context_len}, congelado).")

    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=wm_config.obs_vocab_size).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print(f"VQ-VAE carregado de {args.vqvae_path} (congelado).")

    state_dim = wm_config.n_embd * 2 if args.zh else wm_config.n_embd
    actor_critic = ActorCritic(
        state_dim=state_dim, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)
    ckpt = torch.load(args.policy_ckpt, map_location=device, weights_only=True)
    actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
    actor_critic.eval()
    print(f"Politica carregada de {args.policy_ckpt} (update {ckpt.get('update', '?')}).")

    env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, render=False)

    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.startswith(f"{args.tag}_") and f.endswith(".npz")]
    start_idx = len(existing)
    if start_idx > 0:
        print(f"[INFO] {start_idx} episodios com tag '{args.tag}' ja existem em '{args.out_dir}'. Continuando do indice {start_idx}.")

    n_actions = env.action_space.n
    saved = 0
    tries = 0
    ep_lengths, ep_rewards = [], []
    while saved < args.n_episodes:
        idx = start_idx + saved
        data = collect_episode_for_dataset(
            env, world_model, vqvae, actor_critic, wm_config.context_len, device,
            args.skip_frames, args.epsilon, args.deterministic,
            seed=args.seed + start_idx + tries, use_zh=args.zh, n_actions=n_actions,
        )
        tries += 1
        if data is None:
            continue

        save_path = os.path.join(args.out_dir, f"{args.tag}_{idx:04d}.npz")
        np.savez_compressed(save_path, **data)

        T = data["tokens"].shape[0]
        ep_reward = float(data["rewards"].sum())
        ep_lengths.append(T)
        ep_rewards.append(ep_reward)
        saved += 1
        print(f"[Ep {idx:04d}] steps: {T:>4} | reward: {ep_reward:>8.2f} | salvo: {save_path}")

    env.close()
    print(f"\n{'='*60}")
    print(f"  Coleta concluida! {saved} episodios salvos em '{args.out_dir}' (tag='{args.tag}')")
    print(f"  Steps medio : {np.mean(ep_lengths):.1f}")
    print(f"  Reward medio: {np.mean(ep_rewards):.2f}  |  Std: {np.std(ep_rewards):.2f}")
    print(f"{'='*60}\n")
    print(
        f"Proximo passo: continuar o treino do World Model nos episodios novos + antigos:\n"
        f"  python -m models.dynamics.traingpt --dataset_path {args.out_dir} "
        f"--resume_ckpt models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy_ckpt", type=str, required=True, help="checkpoint da politica (policy_best.pt ou policy_ckpt.pt de train_real_latent.py/train_real_latent_zh.py/train_real_dreamer.py)")
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt_best.pt", help="tem que ser o MESMO checkpoint que extract_tokens.py usou pra gerar dataset_tokens/ (default ckpt_best.pt)")
    p.add_argument("--out_dir", type=str, default="dataset_tokens", help="pasta de saida -- default e a MESMA pasta usada pelo dataset original, pra treinar o World Model com os episodios antigos + novos juntos")
    p.add_argument("--tag", type=str, required=True, help="prefixo do nome dos arquivos (ex.: 'real_latent', 'real_dreamer') -- nunca colide com episode_XXXX.npz do dataset original, e permite retomar/misturar coletas de politicas diferentes na mesma pasta")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_rows", type=int, default=12)
    p.add_argument("--skip_frames", type=int, default=12, help="frames de introducao (zoom) ignorados, igual extract_tokens.py")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--epsilon", type=float, default=0.0, help="fracao de acoes aleatorias pra diversidade -- baixo por default, ja que o objetivo aqui e capturar a distribuicao real que a POLITICA visita, nao maximizar cobertura como agente_coleta/coleta.py fazia do zero")
    p.add_argument("--deterministic", action="store_true", help="usa argmax da politica em vez de amostrar da categorica (so tem efeito quando epsilon nao dispara)")
    p.add_argument("--zh", action="store_true", help="politica treinada com estado [h, frame_repr] (512-d) em vez de so h (256-d) -- ver train_real_latent_zh.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    collect(args)
