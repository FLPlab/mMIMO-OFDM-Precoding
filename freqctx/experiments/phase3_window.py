"""Phase 3 -- contexte frequentiel : le token de chaque utilisateur recoit
le CSI estime (LS) des W SC centrees sur la SC a precoder, concatene. Seul
changement par rapport a la phase 2 (meme modele : D32, 1 bloc x4,
attention lineaire, perte relative a RZF(LS)).

Contrepartie classique, a W egal : RZF sur le LS moyenne sur W SC.
Cible : battre RZF(LMMSE), qui connait la covariance du canal.

    python -m experiments.phase3_window --pilot 10 --windows 1,9 --tag a
    python -m experiments.phase3_window --pilot 10 --merge
"""
import argparse
import dataclasses

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import estimate_freq_cov, smooth_freq
from models.per_sc import PerSCPrecoder
from precoders.classical import rzf
from train import WindowedPool, csi_pool, eval_csi_pool, evaluate, references, run_variant
from utils import BLUES, COLORS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
W3 = "/export/tmp/sala/freqctx/weights/phase3"
MODEL = dict(d=32, layers=4, tied=True, attn="linear")


def rzf_fn(x, s):
    return rzf(x, 10.0 ** (-s[0] / 10.0))


def flat(h, cfg):
    return tf.reshape(h, [-1, cfg.num_users, cfg.num_bs_ant])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=float, default=10.0)
    ap.add_argument("--windows", default="1,3,5,9,13")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    base = f"window_p{int(args.pilot)}"
    if args.merge:
        return merge_parts("phase3", base, plot, sort_key=lambda r: r["window"])
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    cfg = SystemConfig()
    csi = CSIConfig("ls_lin", args.pilot, 0)
    windows = [int(w) for w in args.windows.split(",")]

    sionna_config.seed = 2027
    cov = estimate_freq_cov(ChannelGenerator(cfg))
    ls = eval_csi_pool(cfg, csi, flat=False)                         # [512, F, K, M]
    lmmse = eval_csi_pool(cfg, CSIConfig("ls_lmmse", args.pilot, 0), cov, flat=False)
    assert float(tf.reduce_max(tf.abs(ls[1] - lmmse[1]))) == 0.0, "tirages non apparies"
    train_in, train_true = csi_pool(cfg, csi, drops, seed=1, flat=False)
    if args.quick:
        ls, lmmse = (ls[0][::50], ls[1][::50]), (lmmse[0][::50], lmmse[1][::50])

    ls_flat = (flat(ls[0], cfg), flat(ls[1], cfg))
    ref = references(ls_flat, SNRS_DB)
    refs = {"RZF (LS)": ref["RZF"], "WMMSE (LS)": ref["WMMSE"],
            "RZF (LMMSE)": evaluate(rzf_fn, (flat(lmmse[0], cfg), ls_flat[1]), SNRS_DB),
            "RZF (CSI parfait)": evaluate(rzf_fn, ls_flat[1], SNRS_DB)}
    for W in windows:
        if W > 1:
            refs[f"RZF + lissage W={W}"] = evaluate(
                rzf_fn, (flat(smooth_freq(ls[0], W), cfg), ls_flat[1]), SNRS_DB)
    print("references @20dB :", {k: round(v[4], 2) for k, v in refs.items()})

    runs = []
    for W in windows:
        run = run_variant(f"D32_T4_linear_W{W}_p{int(args.pilot)}",
                          PerSCPrecoder(cfg.num_bs_ant, window=W, **MODEL),
                          WindowedPool(train_in, train_true, W), WindowedPool(ls[0], ls[1], W),
                          SNRS_DB, ref, steps, seed=0, loss="rel_rzf",
                          weights_dir=None if args.quick else W3)
        run["pct_rzf_lmmse"] = [r / z * 100 for r, z in zip(run["rate_bps_hz"], refs["RZF (LMMSE)"])]
        runs.append({"window": W, **run})

    payload = {"config": dataclasses.asdict(cfg), "csi": dataclasses.asdict(csi),
               "snrs_db": SNRS_DB, "model": MODEL, "steps": steps, "train_drops": drops,
               "references": refs, "runs": runs}
    name = base + ("_quick" if args.quick else "") + (f"_part_{args.tag}" if args.tag else "")
    save("phase3", name, payload, plot(payload))


def plot(p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    i20 = p["snrs_db"].index(20)
    ws = [r["window"] for r in p["runs"]]
    ax1.plot(ws, [r["rate_bps_hz"][i20] for r in p["runs"]], "D-", color=BLUES[4],
             label="transformer, fenetre W (appris)")
    smooth = [p["references"]["RZF (LS)"][i20] if w == 1 else
              p["references"].get(f"RZF + lissage W={w}", [None] * 6)[i20] for w in ws]
    ax1.plot(ws, smooth, "o-", color=COLORS["RZF+lissage"], label="RZF + moyenne sur W (classique)")
    for k, c in (("RZF (LMMSE)", COLORS["WMMSE"]), ("RZF (CSI parfait)", "0.5")):
        ax1.axhline(p["references"][k][i20], color=c, ls="--", lw=1.2, label=k)
    ax1.set(xlabel="Largeur de fenetre W [SC]", ylabel="Debit somme a 20 dB [bps/Hz]",
            title=f"Contexte frequentiel, pilote {p['csi']['pilot_snr_db']:.0f} dB")
    ax1.legend(fontsize=7)
    for k, (st, c) in {"RZF (CSI parfait)": ("--", "0.5"), "RZF (LMMSE)": ("s-", COLORS["WMMSE"]),
                       "RZF (LS)": ("o-", COLORS["RZF"])}.items():
        ax2.plot(p["snrs_db"], p["references"][k], st, color=c, lw=1.3, label=k)
    best = max(p["runs"], key=lambda r: r["rate_bps_hz"][i20])
    ax2.plot(p["snrs_db"], best["rate_bps_hz"], "D-", color=BLUES[4], lw=1.8,
             label=f"transformer W={best['window']}")
    ax2.set(xlabel="SNR de donnees [dB]", ylabel="Debit somme [bps/Hz]", title="Meilleure fenetre")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
