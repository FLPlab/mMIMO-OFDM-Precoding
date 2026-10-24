"""Phase 3 -- scenarios melanges (UMi + UMa + RMa, RMa bien plus plat en
frequence). Une covariance fixe pour LMMSE perd-elle, et le modele
(fenetre concatenee, filtre appris) fait-il mieux ?

LMMSE en deux versions, memes tirages et meme bruit pilote :
    "melange" : covariance moyenne des trois scenarios (realiste)
    "oracle"  : covariance du scenario du tirage (connait le scenario)
Modeles : D64, W=13, attention lineaire, (a) entraine sur le melange,
(b) entraine sur UMi seul (phase 3, diagnostic) applique tel quel.

    python -m experiments.phase3_mixed [--quick]
    python -m experiments.phase3_mixed --train rma --tag specialiste_rma   # entraine sur RMa seul
    python -m experiments.phase3_mixed --window 49 --tag W49              # fenetre plus large
L'evaluation porte toujours sur les trois scenarios ; meme nombre total de
tirages d'entrainement (1056) quel que soit --train.
"""
import argparse
import dataclasses
import os

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import estimate_freq_cov
from models.per_sc import PerSCPrecoder
from precoders.classical import rzf, wmmse
from train import WindowedPool, csi_pool, evaluate, run_variant
from utils import BLUES, COLORS, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS = ["umi", "uma", "rma"]
PILOT = 10.0
MODEL = dict(d=64, layers=4, tied=True, attn="linear")
W3 = "/export/tmp/sala/freqctx/weights/phase3"
UMI_ONLY = "D64_T4_linear_W13_ls_p10"          # entraine sur UMi seul (diagnostic)


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train", default="umi,uma,rma", help="scenarios d'entrainement")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--tag", default="", help="suffixe du fichier de resultats")
    args = ap.parse_args()
    W, train_scen = args.window, args.train.split(",")
    model_kw = {**MODEL, "window": W}
    steps, total_train, eval_drops = (2000, 192, 32) if args.quick else (40000, 1056, 192)
    train_drops = total_train // len(train_scen) // 32 * 32
    nv = lambda s: 10.0 ** (-s[0] / 10.0)

    covs = {}
    for i, scen in enumerate(SCENARIOS):                       # tirages separes
        sionna_config.seed = 2027 + 10 * i
        covs[scen] = estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen)))
    cov_mix = tf.add_n(list(covs.values())) / len(covs)

    ev, tr_in, tr_true = {}, [], []
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        seed = 2026 + 10 * i
        ls = csi_pool(cfg, CSIConfig("ls_lin", PILOT, 0), eval_drops, seed, flat=False)
        lo = csi_pool(cfg, CSIConfig("ls_lmmse", PILOT, 0), eval_drops, seed, cov_freq=covs[scen],
                      flat=False)
        lm = csi_pool(cfg, CSIConfig("ls_lmmse", PILOT, 0), eval_drops, seed, cov_freq=cov_mix,
                      flat=False)
        assert all(float(tf.reduce_max(tf.abs(x[1] - ls[1]))) == 0.0 for x in (lo, lm))
        ev[scen] = {"ls": ls, "lmmse_oracle": lo, "lmmse_melange": lm}
        if scen in train_scen:
            a, b = csi_pool(cfg, CSIConfig("ls_lin", PILOT, 0), train_drops, seed=1 + i, flat=False)
            tr_in.append(a)
            tr_true.append(b)
    train_pool = WindowedPool(tf.concat(tr_in, 0), tf.concat(tr_true, 0), W)

    refs = {}
    for scen, e in ev.items():
        true = flat(e["ls"][1])
        r = {"RZF (CSI parfait)": evaluate(lambda x, s: rzf(x, nv(s)), true, SNRS_DB),
             "WMMSE (CSI parfait)": evaluate(lambda x, s: wmmse(x, nv(s)), true, SNRS_DB, 16384),
             "RZF (LS)": evaluate(lambda x, s: rzf(x, nv(s)), (flat(e["ls"][0]), true), SNRS_DB)}
        for k in ("oracle", "melange"):
            d = (flat(e[f"lmmse_{k}"][0]), true)
            r[f"RZF (LMMSE {k})"] = evaluate(lambda x, s: rzf(x, nv(s)), d, SNRS_DB)
            r[f"WMMSE (LMMSE {k})"] = evaluate(lambda x, s: wmmse(x, nv(s)), d, SNRS_DB, 16384)
        refs[scen] = r
        print(scen, "@20dB", {k: round(v[4], 2) for k, v in r.items()})

    cfg = SystemConfig()
    pools = {scen: WindowedPool(e["ls"][0], e["ls"][1], W) for scen, e in ev.items()}
    rzf_ls_all = {"RZF": [float(np.mean([refs[s]["RZF (LS)"][i] for s in SCENARIOS]))
                          for i in range(len(SNRS_DB))]}
    label = "melange" if len(train_scen) == len(SCENARIOS) else "+".join(train_scen)
    model = PerSCPrecoder(cfg.num_bs_ant, **model_kw)
    run = run_variant(f"D64_T4_linear_W{W}_{label}_p10", model, train_pool, pools[train_scen[0]],
                      SNRS_DB, {"RZF": refs[train_scen[0]]["RZF (LS)"]}, steps, seed=0,
                      loss="rel_rzf", weights_dir=None if args.quick else W3)
    models = {f"modele W={W} (entraine {label})": {s: evaluate(model, p, SNRS_DB)
                                                     for s, p in pools.items()}}
    path = os.path.join(W3, UMI_ONLY + ".weights.h5")
    if W == 13 and not args.tag and os.path.exists(path):
        m_umi = PerSCPrecoder(cfg.num_bs_ant, **model_kw)
        m_umi(next(pools["umi"].chunks(1))[0][:2], tf.zeros([2]))
        m_umi.load_weights(path)
        models["modele (entraine UMi seul)"] = {s: evaluate(m_umi, p, SNRS_DB)
                                                for s, p in pools.items()}
    for name, per in models.items():
        print(name, "@20dB", {s: round(v[4], 2) for s, v in per.items()})

    payload = {"scenarios": SCENARIOS, "train_scenarios": train_scen, "pilot_snr_db": PILOT,
               "window": W, "model": model_kw,
               "snrs_db": SNRS_DB, "steps": steps, "train_drops_per_scenario": train_drops,
               "eval_drops_per_scenario": eval_drops, "references": refs, "models": models,
               "training_history": run["history"]}
    name = "mixed_p10" + (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")
    save("phase3", name, payload, plot(payload))


def plot(p):
    i20 = p["snrs_db"].index(20)
    keys = ["RZF (LS)", "RZF (LMMSE melange)", "RZF (LMMSE oracle)", "WMMSE (LMMSE melange)",
            "WMMSE (LMMSE oracle)"]
    names = keys + list(p["models"])
    colors = ["#9ec5f4", "#6da7ec", "#2a78d6", "#f2a27e", "#eb6834", "#1baf7a", "#008300"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(p["scenarios"]))
    width = 0.8 / len(names)
    for j, n in enumerate(names):
        vals = [(p["references"][s][n] if n in keys else p["models"][n][s])[i20] for s in p["scenarios"]]
        ax.bar(x + (j - len(names) / 2 + 0.5) * width, vals, width * 0.92, color=colors[j % 7], label=n)
    for k, s in enumerate(p["scenarios"]):
        ax.plot([k - 0.42, k + 0.42], [p["references"][s]["RZF (CSI parfait)"][i20]] * 2, "--",
                color="0.4", lw=1)
    ax.set(xticks=x, xticklabels=[s.upper() for s in p["scenarios"]],
           ylabel="Debit somme a 20 dB [bps/Hz]",
           title=f"Scenarios melanges, pilote {p['pilot_snr_db']:.0f} dB (tirets : RZF CSI parfait)")
    lo = min(min((p["references"][s][n] if n in keys else p["models"][n][s])[i20] for n in names)
             for s in p["scenarios"])
    ax.set_ylim(lo - 1.5, None)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
