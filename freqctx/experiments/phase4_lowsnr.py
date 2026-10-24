"""Phase 4d -- diagnostic du faible SNR : en CSI parfait, tous nos precodeurs
plafonnent vers 102,5-103 % de RZF a 0 dB (WMMSE : 105,4 %), quelle que soit
leur taille. Hypothese : un seul modele pour 0-25 dB fait un compromis.
Test (un seul changement) : le precodeur du candidat (D64, 1 bloc lineaire x4,
perte relative) entraine sur 0-25 dB, puis sur 0-10 dB seulement.

Donnees et references de la phase 1 (UMi, CSI parfait, graine 2026 ;
references relues dans results/phase1/size_sweep.json, verifiees).

    python -m experiments.phase4_lowsnr --snr_max 10 --tag a
    python -m experiments.phase4_lowsnr --merge
"""
import argparse
import json
import os

from config import SystemConfig
from models.per_sc import PerSCPrecoder
from precoders.classical import rzf
from train import channel_pool, eval_pool, evaluate, run_variant
from utils import COLORS, RESULTS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
W4 = "/export/tmp/sala/freqctx/weights/phase4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_max", type=float, default=25.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge_parts("phase4", "lowsnr", plot, sort_key=lambda r: r["snr_max"])
    ref = json.load(open(os.path.join(RESULTS, "phase1", "size_sweep.json")))["references"]
    cfg = SystemConfig()
    pool, h_eval = channel_pool(cfg, 1024, seed=1), eval_pool(cfg)
    r20 = evaluate(lambda x, s: rzf(x, 10.0 ** (-s[0] / 10.0)), h_eval, [20])[0]
    assert abs(r20 - ref["RZF"][4]) < 1e-3, (r20, ref["RZF"][4])
    run = run_variant(f"D64_T4_linear_parfait_snr0-{int(args.snr_max)}",
                      PerSCPrecoder(cfg.num_bs_ant, d=64, layers=4, tied=True, attn="linear"),
                      pool, h_eval, SNRS_DB, ref, 40000, seed=0, loss="rel_rzf",
                      snr_range=(0.0, args.snr_max), weights_dir=W4)
    run["snr_max"] = args.snr_max
    run["pct_wmmse"] = [r / w * 100 for r, w in zip(run["rate_bps_hz"], ref["WMMSE"])]
    print("% RZF", [round(x, 1) for x in run["pct_rzf"]], "| % WMMSE", [round(x, 1) for x in run["pct_wmmse"]])
    save("phase4", "lowsnr" + (f"_part_{args.tag}" if args.tag else ""),
         {"snrs_db": SNRS_DB, "references": ref, "runs": [run]}, plot({"snrs_db": SNRS_DB, "references": ref, "runs": [run]}))


def plot(p):
    fig, ax = plt.subplots(figsize=(6, 4))
    rz = p["references"]["RZF"]
    ax.plot(p["snrs_db"], [100 * w / r for w, r in zip(p["references"]["WMMSE"], rz)], "--",
            color=COLORS["WMMSE"], label="WMMSE")
    ax.axhline(100, color=COLORS["RZF"], ls="--", lw=1, label="RZF")
    for r, c in zip(p["runs"], ["#2a78d6", "#1baf7a"]):
        ax.plot(p["snrs_db"], r["pct_rzf"], "o-", color=c, label=f"entraine 0-{r['snr_max']:.0f} dB")
    ax.set(xlabel="SNR [dB]", ylabel="% du debit RZF", title="Faible SNR, CSI parfait (D64, 1 bloc x4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
