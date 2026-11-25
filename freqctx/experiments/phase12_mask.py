"""Phase 12 -- que fait l'encodeur frequentiel ? Trois figures du masque, base figee v2, sans entrainement.

L'encodeur est LINEAIRE en les pilotes une fois ses coefficients choisis : pour chaque utilisateur,
retards g = IDFT(LS du peigne), puis g_hat[t] = c0[t] g[t] + cm[t] g[t-1] + cp[t] g[t+1], puis DFT.
Seuls les coefficients (c0, cm, cp) dependent des donnees (attention entre retards). On les compare a
un Wiener ORACLE par retard, w(t) = P(t) / (P(t) + s2), avec P(t) le profil sans bruit du vrai canal sur
la meme grille (puissance moyenne sur les antennes) et s2 le bruit par retard (mesure sur la vraie
realisation du bruit). Puissances reelles + controle de puissance montant (cadre de la base v2).

1. Exemples : profil des retards (vrai, LS, bruit) et masque |c0| contre Wiener oracle (UMi/UMa/RMa).
   L'echelle de |c0| est libre (le precodeur normalise chaque canal) : elle est ramenee a 1 en moyenne
   sur les retards a plus de 20 dB de SNR (la ou tout bon filtre vaut ~1).
2. Masque selon le SNR pilote (0-20 dB ; le modele ne le recoit pas) : |c0| en fonction du SNR par
   retard P(t)/s2, et nombre effectif de retards gardes (somme de |c0|^2), contre le Wiener oracle.
3. Noyau equivalent en frequence de la SC centrale (poids de chaque pilote dans h_hat(f0)) : appris,
   Wiener oracle, LMMSE (covariance du melange, fixe).

    python -u -m experiments.phase12_mask            # calcul + figures
    python -u -m experiments.phase12_mask --plot     # figures seules, depuis le JSON
"""
import argparse
import json
import os
import time

import numpy as np

from utils import RESULTS, git_commit

OUT = os.path.join(RESULTS, "phase12")
PILOTS, DROPS, BATCH, F0 = [0, 5, 10, 15, 20], 64, 16, 306
SCEN_NAMES = {"umi": "UMi", "uma": "UMa", "rma": "RMa"}
BINS = np.arange(-30, 42, 2.0)                       # SNR par retard (dB)


def coeffs(est, ls):
    """Reprend DelayEstimator.call (banded) : retards g [B,K,T,M] et coefficients c0, cm, cp [B,K,T]."""
    import tensorflow as tf
    B = ls.shape[0]
    g = est.delays(ls)
    p = tf.reduce_mean(tf.abs(g) ** 2, axis=-1, keepdims=True)
    feats = tf.concat([tf.math.real(g), tf.math.imag(g), tf.math.log(p + 1e-9) / 10.0], axis=-1)
    K, T = g.shape[1], g.shape[2]
    x = est.embed(tf.reshape(feats, [B * K, T, feats.shape[-1]])) + est.pos(tf.range(T))[None]
    for blk in est.blocks:
        x = blk(x)
    o = tf.reshape(est.gain(x), [B, K, T, 6]).numpy()
    c0 = 1 / (1 + np.exp(-o[..., 0])) + 1j * o[..., 1]
    return g.numpy(), c0, o[..., 2] + 1j * o[..., 3], o[..., 4] + 1j * o[..., 5]


def kernel_row(v, c0, cm, cp, idft_k):
    """Poids des N pilotes dans h_hat(f0) : v = DFT[f0, :] (T,), filtre a bande, idft_k [T, N]."""
    r = v * c0 + np.roll(v * cm, -1) + np.roll(v * cp, 1)
    return r @ idft_k


def width(k):
    """Nombre effectif de pilotes utilises (rapport de participation)."""
    a = np.abs(k) ** 2
    return float(a.sum() ** 2 / (a ** 2).sum())


def compute():
    import tensorflow as tf
    from sionna.phy.config import config as sionna_config
    from channel import ChannelGenerator
    from config import SystemConfig
    from csi import UplinkSounder
    from experiments.phase7_context import SCENARIOS, cov_mixture, sound_power_controlled
    from experiments.phase11_power import load

    net = load("pilot_rand_ph32_norm_relpow", "norm")
    est = net.estimator
    T, Kc = est.T, est.pilot_idx.shape[0]
    idft, dft, pidx = est.idft.numpy(), est.dft.numpy(), est.pilot_idx.numpy()
    delays_ns = (np.where(np.arange(T) < T - 8, np.arange(T), np.arange(T) - T) * 1e9
                 / (2 * 612 * 30e3)).tolist()           # grille 2x : 27 ns ; 8 retards "negatifs"
    cov = cov_mixture().numpy()
    cov = cov / np.real(np.mean(np.diag(cov)))
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "weights": "phase7/pilot_rand_ph32_norm_relpow", "drops": DROPS, "f0": F0},
           "pilots_db": PILOTS, "delays_ns": delays_ns, "bins_db": BINS.tolist(), "stats": {}, "examples": {}}
    check = 0.0
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen, num_slots=1, relative_power=True)
        sionna_config.seed = 4026 + 10 * i                  # memes tirages que la phase 11
        gen, lin = ChannelGenerator(cfg), UplinkSounder(cfg, "lin")
        h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
        gt = np.concatenate([est.delays(h[j:j + BATCH]).numpy() for j in range(0, DROPS, BATCH)])
        P = np.mean(np.abs(gt) ** 2, axis=-1)               # [B, K, T] profil sans bruit sur la grille
        st, ex = {}, {}
        for pdb in PILOTS:
            sionna_config.seed = 9000 + 10 * i + pdb
            ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], float(pdb))
                            for j in range(0, DROPS, BATCH)], 0)
            parts = [coeffs(est, ls[j:j + BATCH]) for j in range(0, DROPS, BATCH)]
            g, c0, cm, cp = (np.concatenate([q[n] for q in parts]) for n in range(4))
            s2 = np.mean(np.abs(g - gt) ** 2, axis=(2, 3), keepdims=True)[..., 0]   # [B, K, 1]
            w = P / (P + s2)
            snr_t = 10 * np.log10(P / s2 + 1e-12)
            m = np.abs(c0)                                  # echelle libre (le precodeur normalise) :
            strong = snr_t > 20                              # ramenee a 1 en moyenne sur les retards forts
            ref = np.where(strong.any(-1), np.sum(m * strong, -1) / np.maximum(strong.sum(-1), 1), m.max(-1))
            m = m / ref[..., None]
            idx = np.digitize(snr_t.ravel(), BINS) - 1
            ok = (idx >= 0) & (idx < len(BINS) - 1)
            curve = [float(m.ravel()[ok][idx[ok] == b].mean()) if np.any(idx[ok] == b) else None
                     for b in range(len(BINS) - 1)]
            # noyaux de la SC centrale ; LMMSE : R[f0, P] (R[P, P] + s2_rel I)^-1
            wl, ww, wlm = [], [], []
            lsn, hn = ls.numpy(), h.numpy()
            for b in range(DROPS):
                for k in range(Kc):
                    kl = kernel_row(dft[F0], c0[b, k], cm[b, k], cp[b, k], idft[k])
                    kw = kernel_row(dft[F0], w[b, k], 0 * w[b, k], 0 * w[b, k], idft[k])
                    pk = pidx[k]
                    s2r = float(np.mean(np.abs(lsn[b, pk, k] - hn[b, pk, k]) ** 2) / np.mean(np.abs(hn[b, pk, k]) ** 2))
                    klm = cov[F0, pk] @ np.linalg.inv(cov[np.ix_(pk, pk)] + s2r * np.eye(len(pk)))
                    wl.append(width(kl)), ww.append(width(kw)), wlm.append(width(klm))
                    if b == 0 and k == 0:
                        y = lsn[0, pk, 0, :]
                        ref = net.encode(ls[:1]).numpy()[0, F0, 0, :]
                        check = max(check, float(np.abs(kl @ y - ref).max() / np.abs(ref).max()))
                        if pdb == 10:
                            ex["kernels"] = {"offsets_sc": (pk - F0).tolist(),
                                             **{n: [z.real.tolist(), z.imag.tolist()]
                                                for n, z in [("learned", kl), ("wiener", kw), ("lmmse", klm)]}}
            st[str(pdb)] = {"taps_learned": float(np.mean(np.sum(m ** 2, -1))),
                       "taps_wiener": float(np.mean(np.sum(w ** 2, -1))),
                       "corr_mask_wiener": float(np.corrcoef(m.ravel(), w.ravel())[0, 1]),
                       "kernel_width_pilots": {"learned": float(np.mean(wl)), "wiener": float(np.mean(ww)),
                                               "lmmse": float(np.mean(wlm))},
                       "curve_mask_vs_snr_tap": curve}
            ex[f"mask_p{pdb}"] = m[0, 0].tolist()
            if pdb == 10:
                ex.update(P_true_db=(10 * np.log10(P[0, 0] + 1e-15)).tolist(),
                          P_ls_db=(10 * np.log10(np.mean(np.abs(g[0, 0]) ** 2, -1))).tolist(),
                          noise_db=float(10 * np.log10(s2[0, 0, 0])), wiener=w[0, 0].tolist())
            print(f"{scen} pilote {pdb} dB : retards gardes {st[str(pdb)]['taps_learned']:.1f} (Wiener "
                  f"{st[str(pdb)]['taps_wiener']:.1f}), corr {st[str(pdb)]['corr_mask_wiener']:.2f}, largeur noyau "
                  f"{ {n: round(v, 1) for n, v in st[str(pdb)]['kernel_width_pilots'].items()} }", flush=True)
        out["stats"][scen], out["examples"][scen] = st, ex
    out["meta"]["linearity_check_rel_err"] = check
    print("verification linearite (noyau x pilotes = sortie encodeur) :", check)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "mask.json"), "w"), indent=1)
    return out


# --- figures -----------------------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]       # SNR pilote 0 -> 20 dB


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                         "legend.frameon": False, "savefig.dpi": 200})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    d = np.array(out["delays_ns"]) / 1e3
    order = np.argsort(d)
    scens = list(out["examples"])

    # 1. exemples
    fig, ax = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True, sharey="row")
    for j, s in enumerate(scens):
        e = out["examples"][s]
        a = ax[0, j]
        a.plot(d[order], np.array(e["P_ls_db"])[order], color="#b8b7ae", lw=0.8, label="LS (bruité)")
        a.plot(d[order], np.array(e["P_true_db"])[order], color=BLUE, lw=1.1, label="vrai canal (sans bruit)")
        a.axhline(e["noise_db"], color=MUTED, ls="--", lw=1, label="niveau du bruit")
        a.set_title(SCEN_NAMES[s], color=INK)
        b = ax[1, j]
        b.plot(d[order], np.array(e["wiener"])[order], color=ORANGE, ls="--", label="Wiener oracle")
        b.plot(d[order], np.array(e["mask_p10"])[order], color=BLUE, label="masque appris |c₀| (normalisé)")
        b.set_xlabel("retard (µs)")
        b.set_xlim(-0.3, 6)
    ax[0, 0].set_ylabel("puissance par retard (dB)")
    ax[1, 0].set_ylabel("gain du retard")
    ax[0, 0].set_ylim(np.array(out["examples"][scens[0]]["P_true_db"]).max() - 55, None)
    ax[0, 2].legend(loc="upper right", fontsize=8)
    ax[1, 2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Le masque appris garde les retards des trajets et coupe le plancher de bruit (pilote 10 dB, un lien par scénario)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "mask_examples")

    # 2. selon le SNR pilote
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = (np.array(out["bins_db"][:-1]) + np.array(out["bins_db"][1:])) / 2
    for c, p in zip(RAMP, out["pilots_db"]):
        curves = np.array([[np.nan if v is None else v for v in out["stats"][s][str(p)]["curve_mask_vs_snr_tap"]]
                           for s in scens])
        a.plot(x, np.nanmean(curves, 0), color=c, label=f"pilote {p} dB")
    xl = 10 ** (x / 10)
    a.plot(x, xl / (1 + xl), color=ORANGE, ls="--", label="Wiener P/(P+σ²)")
    a.set_xlabel("SNR du retard, P(τ)/σ² (dB)")
    a.set_ylabel("masque appris |c₀| moyen")
    a.set_title("Le seuil bouge avec le bruit (non donné au modèle), plus sévère que Wiener", color=INK, fontsize=9)
    a.legend(fontsize=8, loc="upper left")
    for c, s in zip([BLUE, ORANGE, AQUA], scens):
        p = out["pilots_db"]
        b.plot(p, [out["stats"][s][str(q)]["taps_learned"] for q in p], color=c, marker="o", ms=4,
               label=f"{SCEN_NAMES[s]}, appris")
        b.plot(p, [out["stats"][s][str(q)]["taps_wiener"] for q in p], color=c, ls="--", marker="s", ms=4,
               mfc="white", label=f"{SCEN_NAMES[s]}, Wiener oracle")
    b.set_xlabel("SNR pilote (dB)")
    b.set_ylabel("retards gardés (Σ gain²)")
    b.set_title("Plus de retards quand le bruit baisse, mais bien moins que Wiener", color=INK, fontsize=9)
    b.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, "mask_vs_pilot")

    # 3. noyau equivalent
    fig, ax = plt.subplots(2, 3, figsize=(10, 4.6), sharey="row")
    for j, s in enumerate(scens):
        e = out["examples"][s]["kernels"]
        off = np.array(e["offsets_sc"])
        o = np.argsort(off)
        for n, c, ls, lab in [("lmmse", MUTED, ":", "LMMSE (covariance fixe)"),
                              ("wiener", ORANGE, "--", "Wiener oracle"), ("learned", BLUE, "-", "encodeur appris")]:
            z = np.array(e[n][0]) + 1j * np.array(e[n][1])
            z = z * np.exp(-1j * np.angle(z[np.argmax(np.abs(z))])) / np.abs(z).max()
            ax[0, j].plot(off[o], 20 * np.log10(np.abs(z[o]) + 1e-6), color=c, ls=ls, lw=1.2, label=lab)
            ax[1, j].plot(off[o], np.abs(z[o]), color=c, ls=ls, lw=1.2, marker="." if n == "learned" else None,
                          ms=3, label=lab)
        ax[0, j].set_title(SCEN_NAMES[s], color=INK)
        ax[0, j].set_ylim(-60, 3)
        ax[1, j].set_xlim(-60, 60)
        ax[1, j].set_xlabel("écart à la SC estimée (SC)")
    ax[0, 0].set_ylabel("|poids| (dB)")
    ax[1, 0].set_ylabel("|poids|, zoom ±60 SC")
    ax[0, 2].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Poids de chaque pilote dans l'estimation de la SC centrale : l'encodeur moyenne plus large que LMMSE, et plus sur RMa",
                 color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "mask_kernel")
    print("figures ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    out = json.load(open(os.path.join(OUT, "mask.json"))) if args.plot else compute()
    plot(out)
