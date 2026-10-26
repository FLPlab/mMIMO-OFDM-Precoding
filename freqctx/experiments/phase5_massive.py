"""Phase 5 -- MIMO massif : M=64 antennes (1 rangee de 32 colonnes, double
polarisation), K=8 utilisateurs, 50 RB = 600 SC (divisible par 8 pour le
peigne de pilotes : 1 SC sur 8 par utilisateur), pilote 10 dB, melange
UMi + UMa + RMa. Le modele est celui retenu en M8K4, sans modification :
retards sur grille 2x + filtre a bande + precodeur D64 (1 bloc lineaire x4).

Les donnees d'evaluation sont generees UNE fois (role refs), sauvegardees,
puis relues par le role model : memes tirages et meme bruit pilote.

    python -m experiments.phase5_massive --role refs
    python -m experiments.phase5_massive --role model [--train_drops 352 --steps 40000]
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, estimate_freq_cov, genie_lmmse
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import rzf, wmmse
from train import DropPool, csi_pool, evaluate, rate_per_sample, run_variant
from utils import RESULTS, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, BATCH = ["umi", "uma", "rma"], 10.0, 16
DATA = "/export/tmp/sala/freqctx/data"
W5 = "/export/tmp/sala/freqctx/weights/phase5"


def cfg_of(scen="umi"):
    return SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario=scen, num_slots=1)


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def rates(precode, h_in, h_true, err=None, chunk=4096):
    out = []
    for snr in SNRS_DB:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h_true[i:i + chunk],
                             precode(h_in[i:i + chunk], nv, None if err is None else err[i:i + chunk]), nv)
             for i in range(0, h_in.shape[0], chunk)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def refs(eval_drops):
    os.makedirs(DATA, exist_ok=True)
    covs = {}
    for i, scen in enumerate(SCENARIOS):
        sionna_config.seed = 2027 + 10 * i
        covs[scen] = estimate_freq_cov(ChannelGenerator(cfg_of(scen)), num_batches=16, batch_size=16)
    cov_mix = tf.add_n(list(covs.values())) / len(covs)
    rz = lambda x, nv, e: rzf(x, nv)
    wm = lambda x, nv, e: wmmse(x, nv, err_var=e)
    res = {}
    for i, scen in enumerate(SCENARIOS):
        t0 = time.time()
        c = cfg_of(scen)
        sionna_config.seed = 2026 + 10 * i
        gen, lin, lm = ChannelGenerator(c), UplinkSounder(c, "lin"), UplinkSounder(c, "lmmse", cov_mix)
        d = {k: [] for k in ("ls", "true", "lmmse", "err", "genie")}
        for _ in range(eval_drops // BATCH):
            h = gen(BATCH)[:, 0]
            ls = lin(h, PILOT)
            a, b = lm(h, PILOT, return_err_var=True)
            d["ls"].append(ls), d["true"].append(h), d["lmmse"].append(a), d["err"].append(b)
            d["genie"].append(genie_lmmse(ls, h, gen.last_cir, gen.frequencies, PILOT))
        d = {k: tf.concat(v, 0) for k, v in d.items()}
        np.savez(os.path.join(DATA, f"massive_eval_{scen}.npz"), ls=d["ls"].numpy(), true=d["true"].numpy())
        t = flat(d["true"])
        res[scen] = {"RZF (LS)": rates(rz, flat(d["ls"]), t),
                     "WMMSE (LMMSE melange)": rates(wm, flat(d["lmmse"]), t),
                     "WMMSE robuste (LMMSE melange)": rates(wm, flat(d["lmmse"]), t, flat(d["err"])),
                     "WMMSE (genie par lien)": rates(wm, flat(d["genie"]), t),
                     "RZF (CSI parfait)": rates(rz, t, t),
                     "WMMSE (CSI parfait)": rates(wm, t, t)}
        print(scen, "@20dB", {k: round(v[4], 2) for k, v in res[scen].items()},
              f"({time.time() - t0:.0f} s)", flush=True)
    save("phase5", "massive_refs", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT, "M": 64, "K": 8,
                                    "num_sc": 600, "eval_drops_per_scenario": eval_drops,
                                    "references": res})


def model(train_drops, steps, quick, precoder="persc", p_layers=8, p_untied=False):
    c0 = cfg_of()
    tr_in, tr_true = [], []
    for i, scen in enumerate(SCENARIOS):
        a, b = csi_pool(cfg_of(scen), CSIConfig("ls_lin", PILOT, 0), train_drops, seed=1 + i,
                        batch=BATCH, flat=False)
        tr_in.append(a), tr_true.append(b)
    pool = DropPool(tf.concat(tr_in, 0), tf.concat(tr_true, 0))
    if precoder == "persc":                                   # precodeur M8K4 transpose (phase 5a)
        pkw = dict(d=64, layers=4, tied=True, attn="linear")
        ptag = "_D64"
    else:                                                     # precodeur par paire (phase 6)
        pkw = dict(precoder_arch="pair", d=16, hidden=64, layers=p_layers, tied=not p_untied,
                   mix={"gram": "gram", "trig": "triangular_g"}[precoder])
        ptag = f"_{precoder}L{p_layers}" + ("_untied" if p_untied else "")
    net = DelayDomainPrecoder(c0.num_sc, c0.num_users, c0.num_bs_ant, banded=True, oversample=2, **pkw)
    name = f"massive_delay_bande_os2{ptag}_n{train_drops}_s{steps // 1000}k"
    val = DropPool(tr_in[0][:16], tr_true[0][:16])            # suivi pendant l'entrainement
    run = run_variant(name, net, pool, val, SNRS_DB, {"RZF": [1.0] * len(SNRS_DB)}, steps, seed=0,
                      loss="rel_rzf", weights_dir=None if quick else W5)
    out = {}
    for scen in SCENARIOS:
        f = os.path.join(DATA, f"massive_eval_{scen}.npz")
        while not os.path.exists(f):                          # attendre le role refs
            time.sleep(60)
        z = np.load(f)
        ev = DropPool(tf.constant(z["ls"]), tf.constant(z["true"]))
        out[scen] = evaluate(net, ev, SNRS_DB)
        print(scen, "@20dB modele", round(out[scen][4], 2), flush=True)
    save("phase5", "massive_model" + ("" if precoder == "persc" else ptag) + ("_quick" if quick else ""),
         {"snrs_db": SNRS_DB, "model": name, "params": run["params"], "macs_per_sc": run["macs_per_sc"],
          "train_drops_per_scenario": train_drops, "steps": steps, "rates": out,
          "training_history": run["history"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["refs", "model"], required=True)
    ap.add_argument("--eval_drops", type=int, default=96)
    ap.add_argument("--train_drops", type=int, default=352)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--precoder", default="persc", choices=["persc", "gram", "trig"])
    ap.add_argument("--p_layers", type=int, default=8)
    ap.add_argument("--p_untied", action="store_true")
    args = ap.parse_args()
    if args.role == "refs":
        refs(16 if args.quick else args.eval_drops)
    else:
        model(32 if args.quick else args.train_drops, 300 if args.quick else args.steps, args.quick,
              args.precoder, args.p_layers, args.p_untied)


if __name__ == "__main__":
    main()
