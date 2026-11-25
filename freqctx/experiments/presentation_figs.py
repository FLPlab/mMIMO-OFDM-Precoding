"""Figures de presentation (pour F. Leduc-Primeau), conventions du memoire :
  figA_sumrate_snr   : debit somme vs SNR, double panneau CSI parfait | depuis les pilotes (Sionna UMi,
                       M=8, K=4, 612 SC, puissances reelles).
  figB_pareto_energy : debit vs energie par inference (St-Catherine, M=64, K=4, CSI parfait), modele
                       d'energie corrige (v5) applique a TOUTES les methodes, RZF et WMMSE compris.
                       Purement descriptive, comme dans le memoire.
  figC_wmmse_iters   : WMMSE selon le nombre d'iterations, rapporte a notre precodeur (CSI parfait).
  figD_interp        : precodeur calcule sur 1 SC sur s puis interpole (CSI parfait).
  figE_raytracing    : canaux de lancer de rayons jamais vus (labo + DeepMIMO), depuis les pilotes.
Aucun calcul : tout est relu depuis results/. Sortie : results/presentation/.

    python -m experiments.presentation_figs
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import experiments.phase12_mask as pm  # noqa: E402
from experiments import phase12_figs  # noqa: E402
from experiments.phase12_mask import AQUA, BLUE, GRID, INK, MUTED, ORANGE, save, style  # noqa: E402
from utils import RESULTS  # noqa: E402

PRES = os.path.join(RESULTS, "presentation")
PINK = "#e87ba4"


def load(*p):
    return json.load(open(os.path.join(RESULTS, *p)))


def figA():
    """Debit vs SNR (0 a 30 dB), Sionna UMi, puissances reelles, 2 x 2 :
    ligne 1 : M=8, K=4 (CSI parfait | pilotes 10 dB) ; ligne 2 : massif M=64, K=8 (CSI parfait | pilotes 10 dB).
    CSI parfait : un seul precodeur pour les deux tailles (phase13/train_massive.json), WMMSE / RZF sur les memes
    tirages (phase13/bench.json, massive_csi_parfait.json). Pilotes : phase13/figA_pilots.json, massive_pilots.json."""
    b, pp = load("phase13", "bench.json"), load("phase13", "figA_pilots.json")
    tm, mm = load("phase13", "train_massive.json"), load("phase13", "massive_csi_parfait.json")
    mp = load("phase13", "massive_pilots.json")
    snrs = b["snrs_db"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    refs = [("WMMSE 300 it.", "WMMSE 300 it. (convergé)", ORANGE, "-", "s"),
            ("WMMSE 30 it.", "WMMSE 30 it.", ORANGE, "--", "s"),
            ("WMMSE 10 it.", "WMMSE 10 it.", ORANGE, ":", "s"), ("RZF", "RZF", AQUA, "-.", "^")]
    pil = [("notre modele", "notre modèle (pilotes → précodeur)", BLUE, "-", "o"),
           ("Wiener formule fermee + WMMSE 100 it.", "Wiener (retards) + WMMSE", ORANGE, "--", "s"),
           ("LMMSE + WMMSE robuste 100 it.", "LMMSE + WMMSE robuste", PINK, ":", "D"),
           ("LMMSE + RZF", "LMMSE + RZF", AQUA, "-.", "^")]
    for ax, ours, ref, ttl in ((axes[0, 0], tm["M8K4"], b["sets"]["umi"], "(a) M=8, K=4 — CSI parfait"),
                               (axes[1, 0], tm["M64K8"], mm["rates"], "(c) massif M=64, K=8 — CSI parfait")):
        ax.plot(snrs, ours, color=BLUE, marker="o", ms=4, label="notre précodeur (une passe)")
        for key, lab, col, ls, mk in refs:
            ax.plot(snrs, ref[key], color=col, ls=ls, marker=mk, ms=4, label=lab)
        ax.set_title(ttl, color=INK, fontsize=10)
    for ax, src, ttl in ((axes[0, 1], pp, "(b) M=8, K=4 — depuis les pilotes (10 dB)"),
                         (axes[1, 1], mp, "(d) massif M=64, K=8 — depuis les pilotes (10 dB)")):
        for key, lab, col, ls, mk in pil:
            ax.plot(src["snrs_db"], src["rates"][key], color=col, ls=ls, marker=mk, ms=4, label=lab)
        ax.set_title(ttl, color=INK, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks(snrs)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("SNR (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("débit somme (bps/Hz)")
    fig.suptitle("Sionna UMi (3GPP), puissances réelles", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "figA_sumrate_snr")


V5CSV = "/export/tmp/sala/home_archive/ELE6310/ELE6310E/export_francois_v5_20261024/donnees/pareto_data_v5.csv"


def figB():
    """Pareto debit / energie, St-Catherine, CSI parfait, jeu de test v5 (200 000 canaux), Energy.py v5 pour
    tout. Transformers du labo (v5) : une courbe par taille, lambda non domines seulement. A droite, tableau
    des MAC par echantillon (torch_port/mac_table.py)."""
    import csv
    rows = [r for r in csv.DictReader(open(V5CSV)) if r["scenario"] == "stecath"]
    cl = load("stecath", "classical_energy.json")["methods"]
    ev_all = load("stecath", "energy_v5_ours.json")
    ev = ev_all["notre_precodeur"]
    tv = load("stecath", "v5_testset_ours.json")
    mt = load("stecath", "mac_table.json")
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0.06, 0.12, 0.56, 0.78])
    w = [v for k, v in cl.items() if k.startswith("WMMSE")]
    ax.plot([x["energy_uJ_32b"] for x in w], [x["sr"] for x in w], color=ORANGE, marker="s", ms=3, lw=1.3,
            label="WMMSE, 1 à 300 itérations (32 bits)")
    ax.plot([x["energy_uJ_16b"] for x in w], [x["sr"] for x in w], color=ORANGE, ls="--", marker="s", ms=3, lw=1.1,
            label="WMMSE (16 bits)")
    wv5 = sorted((int(r["lambda"]), float(r["energy"]), float(r["sr"])) for r in rows if r["config"] == "WMMSE")
    ax.plot([e for _, e, _ in wv5], [sr for _, _, sr in wv5], color=ORANGE, ls=":", marker="s", ms=3, lw=1,
            label="WMMSE, coût compté comme dans l'article SiPS")
    for x in w:
        if x["iters"] in (1, 10, 100, 300):
            ax.annotate(f"{x['iters']} it.", (x["energy_uJ_32b"], x["sr"]), fontsize=7, color=ORANGE,
                        xytext=(5, -10), textcoords="offset points")
    ax.plot([cl["RZF"]["energy_uJ_32b"], cl["RZF"]["energy_uJ_16b"]], [cl["RZF"]["sr"]] * 2, color=AQUA, ls="none",
            marker="^", ms=7, label="RZF (32 et 16 bits)")
    for cfg, mk, c in (("tiny", "v", "#8e8d85"), ("xlarge", "P", "#3b3a36")):
        pts = sorted((float(r["energy"]), float(r["sr"])) for r in rows if r["config"] == cfg)
        front, best = [], -1
        for e, sr in pts:
            if sr > best:
                front.append((e, sr))
                best = sr
        ax.plot([e for e, _ in front], [sr for _, sr in front], color=c, marker=mk, ms=5, lw=1.2,
                label=f"Transformer du labo, {cfg} ({mt[f'Transformer labo {cfg}']['params'] / 1e6:.1f} M param.)")
    e32, sr32 = ev["energy_uJ_32b"], tv["notre precodeur 32 bits"]["sr"]
    eq, srq = ev["energy_uJ_quant"], tv["notre precodeur quantifie"]["sr"]
    ax.plot([e32], [sr32], color=BLUE, marker="o", ms=8, mfc="white", mew=2, ls="none", label="notre précodeur, 32 bits")
    sizes = [("art_pair_h16_L2_lam5_qprod_norm", "2 couches"), ("art_pair_h16_L4_lam5_qprod_norm", "4 couches, FFN 16"),
             ("art_pair_h32_L4_lam5_qprod_norm", "4 couches"), ("art_pair_h32_lam5_qprod_norm", "8 couches")]
    es = [ev_all["tailles"][t]["energy_uJ_quant"] for t, _ in sizes]
    ss = [tv[t]["sr"] if t in tv else srq for t, _ in sizes]
    ax.plot(es, ss, color=BLUE, marker="o", ms=7, lw=1.5, label="notre précodeur quantifié (ART), 2 à 8 couches")
    for (t, lab), e_, s_ in zip(sizes, es, ss):
        ax.annotate(lab, (e_, s_), fontsize=7, color=BLUE, xytext=(-4, 7), textcoords="offset points", ha="right")
    ax.annotate("", xy=(eq * 1.15, srq), xytext=(e32 / 1.15, sr32), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xscale("log")
    ax.set_xlabel("énergie par inférence (µJ)")
    ax.set_ylabel("débit somme (bps/Hz)")
    ax.set_title("St-Catherine, M=64, K=4, CSI parfait", color=INK, fontsize=10)
    ax.legend(fontsize=6.8, loc="lower right")
    # tableau
    tx = fig.add_axes([0.64, 0.12, 0.35, 0.78])
    tx.axis("off")
    q = {(r["config"], r["lambda"]): r for r in rows}
    t1, x1 = q[("tiny", "1.0")], q[("xlarge", "0.1")]
    tab = [["RZF (32 bits)", "9 k", "—", f"{cl['RZF']['sr']:.1f}", f"{cl['RZF']['energy_uJ_32b']:.2f}"],
           ["WMMSE 10 it. (32 bits)", "102 k", "—", f"{cl['WMMSE 10 it.']['sr']:.1f}", f"{cl['WMMSE 10 it.']['energy_uJ_32b']:.2f}"],
           ["WMMSE 100 it. (32 bits)", "934 k", "—", f"{cl['WMMSE 100 it.']['sr']:.1f}", f"{cl['WMMSE 100 it.']['energy_uJ_32b']:.1f}"],
           ["Transf. labo tiny (λ=1)", f"{mt['Transformer labo tiny']['mac'] / 1e6:.1f} M", "482 k",
            f"{float(t1['sr']):.1f}", f"{float(t1['energy']):.2f}"],
           ["Transf. labo xlarge (λ=0,1)", f"{mt['Transformer labo xlarge']['mac'] / 1e6:.0f} M", "27 M",
            f"{float(x1['sr']):.1f}", f"{float(x1['energy']):.1f}"],
           ["Nous, 8 couches (ART)", f"{ev['mac'] / 1e3:.0f} k", "44 k", f"{srq:.1f}", f"{eq:.2f}"],
           ["Nous, 2 couches (ART)", f"{ev_all['tailles']['art_pair_h16_L2_lam5_qprod_norm']['mac'] / 1e3:.0f} k",
            "8 k", f"{tv['art_pair_h16_L2_lam5_qprod_norm']['sr']:.1f}",
            f"{ev_all['tailles']['art_pair_h16_L2_lam5_qprod_norm']['energy_uJ_quant']:.3f}"]]
    tbl = tx.table(cellText=tab, colLabels=["", "MAC", "param.", "débit", "µJ"], loc="center", cellLoc="center",
                   colWidths=[0.42, 0.14, 0.14, 0.13, 0.13])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r >= len(tab) - 1:
            cell.set_text_props(color=BLUE, fontweight="bold")
    tx.set_title("par échantillon (une inférence)", color=INK, fontsize=9)
    save(fig, "figB_pareto_energy")


def figF():
    """Barres, CSI parfait, % du WMMSE converge (300 it., CSI parfait) : six jeux de donnees, 20 et 30 dB."""
    d = load("phase13", "bench.json")
    snrs = d["snrs_db"]
    sets = [("umi", "UMi"), ("uma", "UMa"), ("rma", "RMa"), ("stecath", "St-Catherine"), ("decarie", "Décarie"),
            ("ericsson", "Ericsson")]
    series = [("notre precodeur", "notre précodeur", BLUE), ("WMMSE 30 it.", "WMMSE 30 it.", ORANGE),
              ("WMMSE 10 it.", "WMMSE 10 it.", "#f2a27e"), ("RZF", "RZF", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(sets))
    for ax, snr in zip(axes, (20, 30)):
        j = snrs.index(snr)
        for n, (key, lab, c) in enumerate(series):
            y = [100 * d["sets"][s][key][j] / d["sets"][s]["WMMSE 300 it."][j] for s, _ in sets]
            ax.bar(x + (n - 1.5) * 0.2, y, 0.18, color=c, label=lab)
        ax.axhline(100, color=INK, lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in sets], fontsize=8)
        ax.set_ylim(60, 102)
        ax.set_title(f"SNR {snr} dB", color=INK, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% du WMMSE convergé (CSI parfait)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("CSI parfait, puissances réelles (axe à partir de 60 %)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "figF_bars_csi_parfait")


if __name__ == "__main__":
    os.makedirs(PRES, exist_ok=True)
    pm.OUT = PRES                      # les fonctions de phase12_figs enregistrent ici
    style(plt)
    figA()
    figB()
    figF()
    for fn, name in ((phase12_figs.wmmse_iters, "fig_wmmse_iters"), (phase12_figs.interp, "fig_interp"),
                     (phase12_figs.raytracing, "fig_raytracing")):
        fn()
        new = {"fig_wmmse_iters": "figC_wmmse_iters", "fig_interp": "figD_interp", "fig_raytracing": "figE_raytracing"}[name]
        for ext in ("png", "pdf"):
            os.replace(os.path.join(PRES, f"{name}.{ext}"), os.path.join(PRES, f"{new}.{ext}"))
    print("->", PRES)
