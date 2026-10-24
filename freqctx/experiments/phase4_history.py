"""Phase 4f -- utiliser les sondages PRECEDENTS. Chaque tirage : Q=4 sondages
passes + le sondage courant, espaces de 2 ms (4 slots, periodicite SRS
realiste), memes trajets (retards, puissances), phases qui evoluent (3-30 km/h).
Donnees a l'instant du sondage courant (pas de vieillissement).

Historique = profil des trajets : puissance moyenne par retard (IDFT du LS de
chaque sondage passe, moyenne sur antennes et sondages), [K, N] par tirage.

Nouvelles donnees -> tout est refait sur les memes tirages :
    --role refs     : WMMSE / WMMSE robuste sur LMMSE (covariance melange),
                      RZF / WMMSE sur LMMSE construit a partir de l'historique
                      (contrepartie classique equitable), borne genie
    --role model    : modele retards + precodeur D64 SANS historique (apparie)
    --role history  : le meme AVEC l'historique (seul changement)

    python -m experiments.phase4_history --role refs|model|history
    python -m experiments.phase4_history --merge
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, estimate_freq_cov, genie_lmmse, pdp_lmmse
from models.delay_domain import DelayDomainPrecoder, DelayEstimator
from precoders.classical import rzf, wmmse
from train import DropPool, evaluate, rate_per_sample, run_variant
from utils import RESULTS, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, Q, STEP, BATCH = ["umi", "uma", "rma"], 10.0, 4, 4, 32
W4 = "/export/tmp/sala/freqctx/weights/phase4"


def make_data(scen, num_drops, seed, cov=None, want_cir=False):
    """-> ls_cur, true_cur [D, F, K, M], pdp [D, K, N] (+ lmmse, err_var, cir)."""
    cfg = SystemConfig(scenario=scen, num_slots=Q + 1, slot_step=STEP)
    sionna_config.seed = seed
    gen = ChannelGenerator(cfg)
    lin = UplinkSounder(cfg, "lin")
    lm = UplinkSounder(cfg, "lmmse", cov) if cov is not None else None
    est = DelayEstimator(cfg.num_sc, cfg.num_users, cfg.num_bs_ant)   # pour l'IDFT seulement
    out = {k: [] for k in ("ls", "true", "pdp", "lmmse", "err", "genie")}
    for _ in range(num_drops // BATCH):
        h = gen(BATCH)                                               # [B, Q+1, F, K, M]
        hist = [lin(h[:, q], PILOT) for q in range(Q)]               # sondages passes (bruit independant)
        out["pdp"].append(tf.add_n([tf.reduce_mean(tf.abs(est.delays(x)) ** 2, -1) for x in hist]) / Q)
        cur = lin(h[:, Q], PILOT)
        out["ls"].append(cur)
        out["true"].append(h[:, Q])
        if lm is not None:
            a, b = lm(h[:, Q], PILOT, return_err_var=True)
            out["lmmse"].append(a)
            out["err"].append(b)
        if want_cir:
            out["genie"].append(genie_lmmse(cur, h[:, Q], gen.last_cir, gen.frequencies, PILOT, t_idx=Q))
    return {k: tf.concat(v, 0) for k, v in out.items() if v}


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def eval_wmmse(h_hat, h_true, err=None, chunk=16384, precoder="wmmse"):
    out = []
    for snr in SNRS_DB:
        nv = 10.0 ** (-snr / 10.0)
        r = []
        for i in range(0, h_hat.shape[0], chunk):
            x = h_hat[i:i + chunk]
            w = rzf(x, nv) if precoder == "rzf" else wmmse(
                x, nv, err_var=None if err is None else err[i:i + chunk])
            r.append(rate_per_sample(h_true[i:i + chunk], w, nv))
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["refs", "model", "history"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge()
    if args.role == "refs":
        covs = {}
        for i, scen in enumerate(SCENARIOS):
            sionna_config.seed = 2027 + 10 * i
            covs[scen] = estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen)))
        cov_mix = tf.add_n(list(covs.values())) / len(covs)
        res = {}
        for i, scen in enumerate(SCENARIOS):
            d = make_data(scen, 192, 3026 + 10 * i, cov=cov_mix, want_cir=True)
            t = flat(d["true"])
            hist_est = pdp_lmmse(d["ls"], d["pdp"], PILOT)
            res[scen] = {
                "RZF (LS)": eval_wmmse(flat(d["ls"]), t, precoder="rzf"),
                "WMMSE (LMMSE melange)": eval_wmmse(flat(d["lmmse"]), t),
                "WMMSE robuste (LMMSE melange)": eval_wmmse(flat(d["lmmse"]), t, flat(d["err"])),
                "RZF (LMMSE historique)": eval_wmmse(flat(hist_est), t, precoder="rzf"),
                "WMMSE (LMMSE historique)": eval_wmmse(flat(hist_est), t),
                "WMMSE (genie par lien)": eval_wmmse(flat(d["genie"]), t)}
            print(scen, "@20dB", {k: round(v[4], 2) for k, v in res[scen].items()})
        return save("phase4", "history_part_refs", {"references": res, "runs": []})

    use_hist = args.role == "history"
    tr = [make_data(s, 352, 1 + i) for i, s in enumerate(SCENARIOS)]
    ev = {s: make_data(s, 192, 3026 + 10 * i) for i, s in enumerate(SCENARIOS)}
    aux = lambda d: d["pdp"] if use_hist else None
    pool = DropPool(tf.concat([d["ls"] for d in tr], 0), tf.concat([d["true"] for d in tr], 0),
                    tf.concat([d["pdp"] for d in tr], 0) if use_hist else None)
    cfg = SystemConfig()
    model = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, history=use_hist,
                                d=64, layers=4, tied=True, attn="linear")
    name = "delay_D64" + ("_historique" if use_hist else "_sans_historique")
    eu = ev["umi"]
    rz = eval_wmmse(flat(eu["ls"]), flat(eu["true"]), precoder="rzf")
    run = run_variant(name, model, pool, DropPool(eu["ls"], eu["true"], aux(eu)), SNRS_DB,
                      {"RZF": rz}, 40000, seed=0, loss="rel_rzf", weights_dir=W4)
    run["rates"] = {s: evaluate(model, DropPool(d["ls"], d["true"], aux(d)), SNRS_DB)
                    for s, d in ev.items()}
    run["history"] = use_hist
    print(name, "@20dB", {s: round(v[4], 2) for s, v in run["rates"].items()})
    save("phase4", f"history_part_{args.role}", {"references": {}, "runs": [run]})


def merge():
    parts = {os.path.basename(f): json.load(open(f))
             for f in sorted(glob.glob(os.path.join(RESULTS, "phase4", "history_part_*.json")))}
    refs = next(p["references"] for p in parts.values() if p["references"])
    runs = [r for p in parts.values() for r in p["runs"]]
    for s in SCENARIOS:
        line = {k: round(v[4], 2) for k, v in refs[s].items()}
        line.update({r["name"]: round(r["rates"][s][4], 2) for r in runs})
        print(s, "@20dB", line)
    save("phase4", "history", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT, "Q": Q,
                               "srs_period_slots": STEP, "references": refs,
                               "runs": [{k: v for k, v in r.items() if k != "history_log"} for r in runs],
                               "parts": list(parts)})


if __name__ == "__main__":
    main()
