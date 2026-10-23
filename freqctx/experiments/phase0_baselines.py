"""Phase 0b -- references classiques sur 20 MHz, evaluation APPARIEE : un
tirage de canal par lot, reutilise pour toutes les conditions de CSI, tous
les SNR et toutes les methodes (meme bruit pilote pour un SNR pilote donne).

Methodes (toutes par SC, appliquees au CSI connu de la BS, debit evalue sur
le canal vrai du slot de donnees) :
  MRT, ZF, RZF, WMMSE
  RZF+lissage(W) : RZF sur le CSI moyenne sur W SC voisines (baseline
                   classique du contexte frequentiel)
CSI "LS-lmmse" : LS + interpolation LMMSE de Sionna avec la covariance
frequentielle moyenne du scenario, estimee sur des tirages separes (graine
SEED+1) -- borne classique de ce que le traitement frequentiel peut recuperer.

    python -m experiments.phase0_baselines [--quick] [--scenarios umi uma]
"""
import argparse
import dataclasses

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, csi_at_bs, estimate_freq_cov, smooth_freq
from metrics import sum_rate
from precoders.classical import PRECODERS, rzf
from utils import BLUES, COLORS, plt, save

SEED = 2026
SNRS_DB = [0, 5, 10, 15, 20, 25]
WIDTHS = [1, 3, 5, 9, 13, 25, 49]
CSI_CONDITIONS = {
    "parfait":           CSIConfig("perfect", 0, 0),
    "LS-lin p10":        CSIConfig("ls_lin", 10, 0),
    "LS-lin p20":        CSIConfig("ls_lin", 20, 0),
    "LS-lmmse p10":      CSIConfig("ls_lmmse", 10, 0),
    "LS-lmmse p20":      CSIConfig("ls_lmmse", 20, 0),
    "parfait +1 slot":   CSIConfig("perfect", 0, 1),
    "parfait +2 slots":  CSIConfig("perfect", 0, 2),
}


def evaluate(cfg, num_batches, batch_size):
    gen = ChannelGenerator(cfg)
    sionna_config.seed = SEED + 1                    # covariance : tirages separes
    cov = estimate_freq_cov(gen)
    sionna_config.seed = SEED                        # graine d'evaluation
    sounders = {"lin": UplinkSounder(cfg, "lin"), "lmmse": UplinkSounder(cfg, "lmmse", cov)}
    acc = {c: {} for c in CSI_CONDITIONS}
    nmse = {c: [] for c in CSI_CONDITIONS}

    def add(cond, key, b, i, v):
        acc[cond].setdefault(key, np.zeros((num_batches, len(SNRS_DB))))[b, i] = v

    for b in range(num_batches):
        h = gen(batch_size)
        cache = {}
        for cond, csi in CSI_CONDITIONS.items():
            key = (csi.estimator, csi.pilot_snr_db)             # meme bruit pilote partage
            if key not in cache:
                cache[key] = csi_at_bs(h, dataclasses.replace(csi, delay_slots=0),
                                       sounders.get(csi.estimator[3:]))[0]
            h_bs, h_data = cache[key], h[:, csi.delay_slots]
            eps = float(tf.reduce_mean(tf.abs(h_bs - h_data) ** 2))
            nmse[cond].append(eps / float(tf.reduce_mean(tf.abs(h_data) ** 2)))
            smoothed = {W: smooth_freq(h_bs, W) for W in WIDTHS[1:]}
            for i, snr in enumerate(SNRS_DB):
                nv = 10.0 ** (-snr / 10)
                for name, fn in PRECODERS.items():
                    add(cond, name, b, i, float(sum_rate(h_data, fn(h_bs, nv), nv)))
                for W, hs in smoothed.items():
                    add(cond, f"RZF+lissage W={W}", b, i, float(sum_rate(h_data, rzf(hs, nv), nv)))
        print(f"  lot {b + 1}/{num_batches}")

    res = {}
    for cond in CSI_CONDITIONS:
        m = {k: v.mean(0) for k, v in acc[cond].items()}
        best_w = [max(WIDTHS, key=lambda W: m["RZF" if W == 1 else f"RZF+lissage W={W}"][i])
                  for i in range(len(SNRS_DB))]
        res[cond] = {
            "csi": dataclasses.asdict(CSI_CONDITIONS[cond]),
            "csi_nmse_db": (float(10 * np.log10(np.mean(nmse[cond])))
                            if np.mean(nmse[cond]) > 0 else None),
            "rate_bps_hz": {k: v.tolist() for k, v in m.items()},
            "rate_sem": {k: (v.std(0) / np.sqrt(num_batches)).tolist() for k, v in acc[cond].items()},
            "best_smoothing_width": best_w,
        }
    return res


def plot(all_res):
    scens = list(all_res)
    fig, axes = plt.subplots(len(scens), 3, figsize=(12, 3.6 * len(scens)), squeeze=False)
    i20 = SNRS_DB.index(20)
    for r, scen in enumerate(scens):
        res = all_res[scen]
        ax = axes[r, 0]
        for n in PRECODERS:
            ax.plot(SNRS_DB, res["parfait"]["rate_bps_hz"][n], "o-", color=COLORS[n], label=n)
        ax.set(title=f"{scen.upper()} : CSI parfait", xlabel="SNR [dB]", ylabel="Debit somme [bps/Hz]")
        ax.legend()

        ax = axes[r, 1]
        c = res["LS-lin p20"]
        ax.plot(SNRS_DB, res["parfait"]["rate_bps_hz"]["RZF"], "--", color="0.55", lw=1.2,
                label="RZF, CSI parfait")
        for n in ["ZF", "RZF", "WMMSE"]:
            ax.plot(SNRS_DB, c["rate_bps_hz"][n], "o-", color=COLORS[n], label=n)
        best = [c["rate_bps_hz"]["RZF" if W == 1 else f"RZF+lissage W={W}"][i]
                for i, W in enumerate(c["best_smoothing_width"])]
        ax.plot(SNRS_DB, best, "s-", color=COLORS["RZF+lissage"], label="RZF + meilleur lissage")
        ax.set(title=f"{scen.upper()} : CSI estime (LS, pilote 20 dB)", xlabel="SNR [dB]")
        ax.legend()

        ax = axes[r, 2]
        conds = ["parfait", "LS-lin p20", "LS-lin p10", "LS-lmmse p10", "parfait +2 slots"]
        for j, cond in enumerate(conds):
            rr = res[cond]["rate_bps_hz"]
            y = [rr["RZF" if W == 1 else f"RZF+lissage W={W}"][i20] for W in WIDTHS]
            ax.plot(WIDTHS, y, "o-", color=BLUES[1 + j], label=cond)
        ax.set(xscale="log", xticks=WIDTHS, xticklabels=WIDTHS,
               title=f"{scen.upper()} : RZF + lissage, SNR 20 dB",
               xlabel="Largeur de la fenetre W [SC]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=["umi", "uma"])
    args = ap.parse_args()
    nb, bs = (2, 8) if args.quick else (16, 32)
    sionna_config.seed = SEED
    all_res = {}
    for scen in args.scenarios:
        cfg = SystemConfig(scenario=scen)
        print(f"== {scen} ({nb} lots x {bs} tirages x {cfg.num_sc} SC)")
        all_res[scen] = {"config": dataclasses.asdict(cfg), **evaluate(cfg, nb, bs)}
        for cond, r in all_res[scen].items():
            if cond == "config":
                continue
            rr = r["rate_bps_hz"]
            i = SNRS_DB.index(20)
            print(f"  {cond:20s} NMSE {r['csi_nmse_db'] or float('-inf'):6.1f} dB | @20dB RZF {rr['RZF'][i]:.2f} "
                  f"WMMSE {rr['WMMSE'][i]:.2f} "
                  f"best W={r['best_smoothing_width'][i]}")
    payload = {"seed": SEED, "snrs_db": SNRS_DB, "widths": WIDTHS,
               "num_batches": nb, "batch_size": bs, "results": all_res}
    save("phase0", "baselines" + ("_quick" if args.quick else ""), payload, plot(
        {s: {k: v for k, v in r.items() if k != "config"} for s, r in all_res.items()}))


if __name__ == "__main__":
    main()
