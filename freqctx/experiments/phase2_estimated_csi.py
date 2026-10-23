"""Phase 2 -- le modele candidat (D32, 1 bloc x4, perte relative) entraine
sur le CSI ESTIME (sondage Sionna, LS + interpolation lineaire), toujours par
SC et sans vieillissement. Le debit est calcule sur le vrai canal.

Question : sans contexte frequentiel, l'apprentissage fait-il mieux que
RZF(LS), voire que RZF(LMMSE) ? Et faut-il l'entrainer sur du CSI estime,
ou un modele de la phase 1 (CSI parfait) suffit-il ?

References sur les MEMES tirages (graine 2026, meme bruit pilote) :
RZF/WMMSE sur LS, RZF sur LMMSE (covariance du scenario, graine 2027),
RZF sur le vrai canal, et les modeles de la phase 1 appliques au LS.

    python -m experiments.phase2_estimated_csi --pilot 10 --attn linear --tag a
    python -m experiments.phase2_estimated_csi --pilot 10 --merge
"""
import argparse
import dataclasses
import os

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import estimate_freq_cov
from models.per_sc import PerSCPrecoder
from precoders.classical import rzf
from train import csi_pool, eval_csi_pool, evaluate, references, run_variant
from utils import BLUES, COLORS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
W1 = "/export/tmp/sala/freqctx/weights/phase1"
W2 = "/export/tmp/sala/freqctx/weights/phase2"
MODEL = dict(d=32, layers=4, tied=True)
PHASE1 = {"linear": "D32_T4_linear_rel_s0", "signed": "D32_tied_T4"}   # poids CSI parfait


def rzf_fn(x, s):
    return rzf(x, 10.0 ** (-s[0] / 10.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=float, required=True)
    ap.add_argument("--attn", default="linear,signed")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    base = f"estimated_csi_p{int(args.pilot)}"
    if args.merge:
        return merge_parts("phase2", base, plot, sort_key=lambda r: r["name"])
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    cfg = SystemConfig()

    sionna_config.seed = 2027                                  # covariance : tirages separes
    cov = estimate_freq_cov(ChannelGenerator(cfg))
    ls = eval_csi_pool(cfg, CSIConfig("ls_lin", args.pilot, 0))
    lmmse = eval_csi_pool(cfg, CSIConfig("ls_lmmse", args.pilot, 0), cov)
    assert float(tf.reduce_max(tf.abs(ls[1] - lmmse[1]))) == 0.0, "tirages non apparies"
    pool = csi_pool(cfg, CSIConfig("ls_lin", args.pilot, 0), drops, seed=1)
    if args.quick:
        ls, lmmse = (ls[0][::50], ls[1][::50]), (lmmse[0][::50], lmmse[1][::50])

    ref = references(ls, SNRS_DB)                              # RZF, WMMSE sur LS
    refs = {"RZF (LS)": ref["RZF"], "WMMSE (LS)": ref["WMMSE"],
            "RZF (LMMSE)": evaluate(rzf_fn, lmmse, SNRS_DB),
            "RZF (CSI parfait)": evaluate(rzf_fn, ls[1], SNRS_DB)}
    print("references @20dB :", {k: round(v[4], 2) for k, v in refs.items()})

    attns = args.attn.split(",")
    phase1 = {}
    for attn in attns:                                         # phase 1 telle quelle sur LS
        m = PerSCPrecoder(cfg.num_bs_ant, attn=attn, **MODEL)
        m(ls[0][:2], tf.zeros([2]))
        m.load_weights(os.path.join(W1, PHASE1[attn] + ".weights.h5"))
        phase1[attn] = evaluate(m, ls, SNRS_DB)
        print(f"phase 1 {attn} sur LS @20dB : {phase1[attn][4]:.2f}")

    runs = []
    for attn in attns:
        run = run_variant(f"D32_T4_{attn}_p{int(args.pilot)}",
                          PerSCPrecoder(cfg.num_bs_ant, attn=attn, **MODEL), pool, ls, SNRS_DB,
                          ref, steps, seed=0, loss="rel_rzf",
                          weights_dir=None if args.quick else W2)
        run["pct_rzf_lmmse"] = [r / z * 100 for r, z in zip(run["rate_bps_hz"], refs["RZF (LMMSE)"])]
        runs.append({"attn": attn, **run})

    payload = {"config": dataclasses.asdict(cfg), "csi": {"estimator": "ls_lin",
               "pilot_snr_db": args.pilot}, "snrs_db": SNRS_DB, "model": MODEL,
               "steps": steps, "train_drops": drops, "references": refs,
               "phase1_on_ls": phase1, "phase1_weights": PHASE1, "runs": runs}
    name = base + ("_quick" if args.quick else "") + (f"_part_{args.tag}" if args.tag else "")
    save("phase2", name, payload, plot(payload))


def plot(p):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = {"RZF (CSI parfait)": ("--", "0.55"), "RZF (LS)": ("o-", COLORS["RZF"]),
              "WMMSE (LS)": ("o-", COLORS["WMMSE"]), "RZF (LMMSE)": ("s-", COLORS["RZF+lissage"])}
    for k, (ls_, c) in styles.items():
        ax.plot(p["snrs_db"], p["references"][k], ls_, color=c, lw=1.4, label=k)
    for j, (attn, r) in enumerate(p["phase1_on_ls"].items()):
        ax.plot(p["snrs_db"], r, ":", color=BLUES[2 + 2 * j], lw=1.4,
                label=f"phase 1 ({attn}, entraine CSI parfait)")
    for j, r in enumerate(p["runs"]):
        ax.plot(p["snrs_db"], r["rate_bps_hz"], "D-", color=BLUES[3 + 2 * j], lw=1.8,
                label=f"phase 2 ({r['attn']}, entraine CSI estime)")
    ax.set(xlabel="SNR de donnees [dB]", ylabel="Debit somme [bps/Hz]",
           title=f"CSI estime, pilote {p['csi']['pilot_snr_db']:.0f} dB (par SC)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
