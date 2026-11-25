"""Phase 12 -- deux figures de synthese (sans calcul) a partir des resultats existants :
1. fig_pilot_sweep : debit (moyenne UMi/UMa/RMa) selon le SNR pilote, base v2 contre les references
   classiques (results/phase12/robust.json, puissances reelles).
2. fig_stecath_energy : debit quantifie selon l'energie par inference sur St-Catherine, notre
   precodeur contre BitAdapt (results/stecath/energy_comparison.json, modele d'energie corrige).

    python -m experiments.phase12_figs
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.phase12_mask import AQUA, BLUE, INK, MUTED, ORANGE, save, style  # noqa: E402
from utils import RESULTS  # noqa: E402


def pilot_sweep():
    d = json.load(open(os.path.join(RESULTS, "phase12", "robust.json")))["scenarios"]
    series = [("base v2", "notre modèle (base v2)", BLUE, "-", "o"),
              ("Wiener formule fermee + WMMSE 100 it.", "Wiener (retards) + WMMSE", ORANGE, "--", "s"),
              ("LMMSE + WMMSE robuste 100 it.", "LMMSE + WMMSE robuste", AQUA, ":", "^")]
    pilots = sorted(int(p) for p in d["umi"]["pilot_sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, j, snr in [(axes[0], 2, 20), (axes[1], 3, 25)]:
        for key, lab, c, ls, mk in series:
            y = [np.mean([d[s]["pilot_sweep"][str(p)][key][j] for s in d]) for p in pilots]
            ax.plot(pilots, y, color=c, ls=ls, marker=mk, ms=5, label=lab)
        ax.set_title(f"SNR de données {snr} dB", color=INK, fontsize=9)
        ax.set_xlabel("SNR pilote (dB)")
        ax.set_xticks(pilots)
    axes[0].set_ylabel("débit somme (bps/Hz), moyenne UMi/UMa/RMa")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Depuis les pilotes : M=8, K=4, 612 SC, puissances réelles", color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_pilot_sweep")


def stecath_energy():
    d = json.load(open(os.path.join(RESULTS, "stecath", "energy_comparison.json")))
    norm = json.load(open(os.path.join(RESULTS, "stecath", "art_pair_h32_lam5_qprod_norm.json")))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, mk in [("tiny", "v"), ("small", "D"), ("medium", "s")]:
        q = d["labo"][name]["quant"]
        lams = sorted(q, key=float)
        ax.plot([q[l]["energy_uJ"] for l in lams], [q[l]["sr"] for l in lams], color=MUTED, marker=mk, ms=5,
                lw=1, label=f"BitAdapt {name} ({d['labo'][name]['params'] / 1e6:.1f} M param.)")
    ours = [d["nous"][k] for k in ("art_pair_h32_lam1", "art_pair_h32_lam5", "art_pair_h32_lam20")]
    ax.plot([o["energy_uJ"] for o in ours], [o["sr"] for o in ours], color="#86b6ef", marker="o", ms=5, lw=1,
            label="notre précodeur, entrée brute")
    ax.plot([norm["energy_pJ"]["quant_corrige"] / 1e6], [norm["sr_quant"]], color=BLUE, marker="*", ms=13, ls="none",
            label="notre précodeur, entrée normalisée (44 k param.)")
    ax.axhline(26.88, color=ORANGE, ls="--", lw=1)
    ax.text(0.9, 26.95, "WMMSE 300 it. (32 bits)", color=ORANGE, fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("énergie par inférence (µJ, modèle d'énergie corrigé)")
    ax.set_ylabel("débit somme quantifié (bps/Hz)")
    ax.set_title("St-Catherine, tout quantifié (ART)", color=INK, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_stecath_energy")


def raytracing():
    """(phase 13) % du WMMSE en CSI parfait sur les canaux de lancer de rayons jamais vus, puissances reelles."""
    sets = [("Boston", json.load(open(os.path.join(RESULTS, "phase13", "deepmimo_boston.json")))
             ["rates"]["puissances reelles"])]
    for s, lab in [("stecath", "St-Catherine"), ("decarie", "Décarie"), ("ericsson", "Ericsson")]:
        sets.append((lab, json.load(open(os.path.join(RESULTS, "phase13", f"lab_{s}_psi2az.json")))["rates"]))
    series = [("encodeur convolution", "notre modèle (encodeur conv)", BLUE),
              ("Wiener formule fermee + WMMSE 100 it.", "Wiener (retards) + WMMSE", ORANGE),
              ("LMMSE + WMMSE robuste 100 it.", "LMMSE + WMMSE robuste", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(sets))
    for ax, p in zip(axes, ("0", "10")):
        for n, (key, lab, c) in enumerate(series):
            y = [100 * r[p][key][2] / r[p]["WMMSE 100 it. (CSI parfait)"][2] for _, r in sets]
            ax.bar(x + (n - 1) * 0.26, y, 0.24, color=c, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([s for s, _ in sets], fontsize=8)
        ax.set_ylim(70, 101)
        ax.set_title(f"pilote {p} dB", color=INK, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% du WMMSE en CSI parfait (20 dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Canaux de lancer de rayons jamais vus à l'entraînement (modèles entraînés sur 3GPP)",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "fig_raytracing")


def wmmse_iters():
    """(phase 13) CSI parfait : debit de WMMSE selon le nombre d'iterations, rapporte a notre precodeur."""
    d = json.load(open(os.path.join(RESULTS, "phase13", "checks.json")))
    its = d["iters"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for (name, lab), c, mk in zip([("stecath", "St-Catherine (labo)"), ("decarie", "Décarie (labo)"),
                                   ("ericsson", "Ericsson (labo)"), ("sionna_umi", "Sionna UMi (3GPP)")],
                                  [BLUE, ORANGE, AQUA, "#e87ba4"], ["o", "s", "^", "D"]):
        r = d["wmmse_iters"][name]
        ref = r["notre precodeur"][2]
        ax.plot(its, [100 * r[f"WMMSE {i} it."][2] / ref for i in its], color=c, marker=mk, ms=4, label=lab)
    ax.axhline(100, color=INK, lw=1.2, ls="--")
    ax.text(1.05, 100.4, "notre précodeur : une passe, sans inversion", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(its)
    ax.set_xticklabels([str(i) for i in its])
    ax.set_xlabel("itérations de WMMSE")
    ax.set_ylabel("débit WMMSE / notre précodeur (%)")
    ax.set_title("CSI parfait, puissances réelles, 20 dB", color=INK, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_wmmse_iters")


def interp():
    """(phase 13) CSI parfait : precodeur calcule sur 1 SC sur s puis interpole ; perte selon s."""
    d = json.load(open(os.path.join(RESULTS, "phase13", "interp.json")))
    steps = d["steps"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for (name, lab), c, mk in zip([("sionna_umi", "Sionna UMi (3GPP, ~400 ns)"), ("stecath", "St-Catherine (labo, 51 ns)"),
                                   ("decarie", "Décarie (labo, 2 ns)"), ("ericsson", "Ericsson (labo, 1 ns)")],
                                  ["#e87ba4", BLUE, ORANGE, AQUA], ["D", "o", "s", "^"]):
        for key, ls in (("notre precodeur", "-"), ("WMMSE 100 it.", ":")):
            r = d["rates"][name][key]
            ax.plot(steps, [100 * x / r[0] for x in r], color=c, ls=ls, marker=mk if ls == "-" else None, ms=4,
                    label=lab if ls == "-" else None)
    ax.plot([], [], color=MUTED, ls=":", label="WMMSE 100 it. (même interpolation)")
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(x) for x in steps])
    ax.set_xlabel("précodeur calculé sur 1 sous-porteuse sur s (interpolation linéaire)")
    ax.set_ylabel("débit, % du calcul sur chaque SC")
    ax.set_title("CSI parfait : la sélectivité fixe le coût du multiporteuse", color=INK, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_interp")


if __name__ == "__main__":
    style(plt)
    pilot_sweep()
    stecath_energy()
    raytracing()
    wmmse_iters()
    interp()
    print("->", os.path.join(RESULTS, "phase12"))
