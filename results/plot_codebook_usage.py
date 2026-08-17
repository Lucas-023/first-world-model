#!/usr/bin/env python3
"""
Gera o gráfico de uso do codebook do VQ-VAE para a apresentação.

Uso típico (a partir da raiz do projeto), com o dataset já tokenizado:

    python results/plot_codebook_usage.py \
        --tokens_dir dataset_tokens \
        --out results/slides_assets/codebook_usage.png

Cada .npz em --tokens_dir deve conter o array `tokens` (T, 64) uint16 com os
índices do codebook (é exatamente o que `extract_tokens.py` produz).

Se você ainda NÃO tokenizou o dataset, use --encode para codificar os frames
crus na hora, informando o checkpoint do VQ-VAE treinado e a pasta dos episódios:

    python results/plot_codebook_usage.py --encode \
        --data_dir dataset --ckpt models/VQVAE/ckpt_best.pt \
        --out results/slides_assets/codebook_usage.png

Depois recompile a apresentação — o slide "VQ-VAE — Uso do Codebook" já aponta
para results/slides_assets/codebook_usage.png.
"""
import argparse, glob, os
import numpy as np

# paleta do deck
ACC2 = "#1F497D"; ACC = "#2E8B57"; INK = "#333333"; MUTED = "#8a8a8a"


def counts_from_tokens(tokens_dir, codebook_size):
    files = sorted(glob.glob(os.path.join(tokens_dir, "*.npz")))
    if not files:
        raise SystemExit(f"Nenhum .npz encontrado em {tokens_dir!r}.")
    counts = np.zeros(codebook_size, dtype=np.int64)
    for f in files:
        tok = np.load(f)["tokens"].reshape(-1).astype(np.int64)
        counts += np.bincount(tok, minlength=codebook_size)
    print(f"{len(files)} episódios · {counts.sum()} tokens")
    return counts


def counts_from_encode(data_dir, ckpt, codebook_size, base, latent, max_frames):
    import sys, torch
    sys.path.insert(0, "models/encoder")
    from modules import VQVAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = VQVAE(in_channels=3, latent_dim=latent, num_embeddings=codebook_size).to(dev).eval()
    sd = torch.load(ckpt, map_location=dev)
    m.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    counts = np.zeros(codebook_size, dtype=np.int64); seen = 0
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            obs = np.load(f)["obs"].astype(np.float32)
            if obs.max() > 1.0:
                obs /= 255.0
            for i in range(0, len(obs), 128):
                x = torch.from_numpy(obs[i:i + 128]).to(dev)
                ind = m.encode_indices(x).reshape(-1).cpu().numpy()
                counts += np.bincount(ind, minlength=codebook_size)
                seen += x.shape[0]
            if seen >= max_frames:
                break
    print(f"{seen} frames codificados")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", default="dataset_tokens")
    ap.add_argument("--out", default="results/slides_assets/codebook_usage.png")
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--encode", action="store_true", help="codifica frames crus em vez de ler tokens")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--ckpt", default="models/VQVAE/ckpt_best.pt")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=20000)
    args = ap.parse_args()

    if args.encode:
        counts = counts_from_encode(args.data_dir, args.ckpt, args.codebook_size,
                                    args.base_channels, args.latent_dim, args.max_frames)
    else:
        counts = counts_from_tokens(args.tokens_dir, args.codebook_size)

    K = args.codebook_size
    p = counts / counts.sum()
    used = int((counts > 0).sum())
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    ppl = float(np.exp(entropy))
    order = np.argsort(counts)[::-1]
    top10 = 100 * p[order[:10]].sum()
    print(f"codes usados: {used}/{K} ({100*used/K:.1f}%) · "
          f"perplexity: {ppl:.0f}/{K} · top-10 cobre {top10:.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freq = 100 * p[order]
    x = np.arange(K)
    fig, ax = plt.subplots(figsize=(7.8, 3.4), dpi=200)
    ax.fill_between(x, freq, 0, color=ACC, alpha=0.12, zorder=1)
    ax.plot(x, freq, color=ACC2, lw=2, zorder=3)
    ax.axvline(used, color=MUTED, ls="--", lw=1.1, zorder=2)
    ax.annotate(f"{used}/{K} códigos usados", (used, freq.max() * 0.6),
                xytext=(8, 0), textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("Códigos do codebook (ordenados por frequência)", fontsize=10, color=INK)
    ax.set_ylabel("Uso (% dos tokens)", fontsize=10, color=INK)
    ax.set_title(f"Perplexity {ppl:.0f}/{K}   ·   top-10 cobre {top10:.0f}%",
                 fontsize=10, color=MUTED, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#cccccc")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9); ax.margins(x=0.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print("salvo em", args.out)


if __name__ == "__main__":
    main()
