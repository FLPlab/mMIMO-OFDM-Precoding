"""Phase 13 -- massif (M=64, K=8, 600 SC) DEPUIS LES PILOTES, puissances reelles (panneau (d) de la figure A).
Modele : encodeur frequentiel a convolution (domaine des retards, grille 2x, filtre a bande) + precodeur par
paires (entree normalisee, FFN 32), appris de bout en bout ; precodeur initialise par celui entraine en CSI
parfait a puissances reelles (phase13_train_massive). Entrainement : UMi + UMa + RMa, puissances reelles,
controle de puissance montant, SNR pilote U(0, 20) dB. Evaluation : Sionna UMi, 64 tirages (memes que
phase13_massive), pilote 10 dB, SNR 0 a 30 dB : notre modele, Wiener (formule fermee) + WMMSE 100 it.,
LMMSE (covariance moyenne des 3 scenarios, massif) + WMMSE robuste 100 it., LMMSE + RZF.

    python -u -m experiments.phase13_massive_pilots
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder, estimate_freq_cov
from experiments.phase7_context import sound_power_controlled
from experiments.phase13_bench import SNRS
from models.delay_domain import DelayDomainPrecoder, WienerDelayEstimator
from precoders.classical import rzf, wmmse
from train import DropPool, rate_per_sample, run_variant
from utils import RESULTS, git_commit

KW = dict(num_bs_ant=64, num_users=8, num_rb=50)
W13 = "/export/tmp/sala/freqctx/weights/phase13"
TRAIN_DROPS, STEPS, B = 256, 30000, 8


def cfg(scen):
    return SystemConfig(scenario=scen, num_slots=1, relative_power=True, **KW)


def rates(fn, h_in, h, err=None, chunk=2048):
    """fn(entree [n, K, M], nv, err) -> W ; h_in, h [N, F, K, M]."""
    hi, ht = tf.reshape(h_in, [-1, 8, 64]), tf.reshape(h, [-1, 8, 64])
    er = None if err is None else tf.reshape(err, [-1, 8, 64])
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(ht[i:i + chunk], fn(hi[i:i + chunk], nv, None if er is None else er[i:i + chunk]), nv)
             for i in range(0, hi.shape[0], chunk)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    tr_in, tr_true = [], []
    for i, scen in enumerate(["umi", "uma", "rma"]):
        sionna_config.seed = 1 + i
        gen, lin = ChannelGenerator(cfg(scen)), UplinkSounder(cfg(scen), "lin")
        for _ in range(TRAIN_DROPS // B):
            h = gen(B)[:, 0]
            tr_in.append(sound_power_controlled(lin, h, float(rng.uniform(0, 20)))), tr_true.append(h)
    pool = DropPool(tf.concat(tr_in, 0), tf.concat(tr_true, 0))
    print(f"donnees : {len(tr_in) * B} tirages ({time.time() - t0:.0f} s)", flush=True)
    c0 = cfg("umi")
    net = DelayDomainPrecoder(c0.num_sc, 8, 64, banded=True, oversample=2, est_mix="conv", precoder_arch="pair",
                              d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(tr_in[0][:1], tf.zeros([1]))
    net.precoder.load_weights(os.path.join(W13, "precoder_relpow_mix.weights.h5"))
    run = run_variant("massive_pilots_conv_relpow", net, pool, DropPool(tr_in[0][:2], tr_true[0][:2]), SNRS,
                      {"RZF": [1.0] * len(SNRS)}, STEPS, seed=0, loss="rel_rzf", weights_dir=W13)
    # evaluation
    covs = []
    for i, scen in enumerate(["umi", "uma", "rma"]):
        sionna_config.seed = 2027 + 10 * i
        covs.append(estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen, **KW))))
    cov = tf.add_n(covs) / 3
    sionna_config.seed = 4026
    gen = ChannelGenerator(c0)
    h = tf.concat([gen(B)[:, 0] for _ in range(8)], 0)                       # 64 tirages (phase13_massive)
    lin, lmm = UplinkSounder(c0, "lin"), UplinkSounder(c0, "lmmse", cov)
    ls = tf.concat([sound_power_controlled(lin, h[j:j + B], 10.0) for j in range(0, 64, B)], 0)
    pr = [sound_power_controlled(lmm, h[j:j + B], 10.0, return_err_var=True) for j in range(0, 64, B)]
    lm, ev = tf.concat([a for a, _ in pr], 0), tf.concat([b for _, b in pr], 0)
    west = WienerDelayEstimator(c0.num_sc, 8, 64, kappa=4.0)
    wn = tf.concat([west(ls[j:j + B]) for j in range(0, 64, B)], 0)
    r = {"notre modele": [], "Wiener formule fermee + WMMSE 100 it.": rates(lambda x, nv, e: wmmse(x, nv, num_iter=100), wn, h),
         "LMMSE + WMMSE robuste 100 it.": rates(lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev),
         "LMMSE + RZF": rates(lambda x, nv, e: rzf(x, nv), lm, h)}
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        rr = [rate_per_sample(tf.reshape(h[j:j + B], [-1, 8, 64]),
                              tf.reshape(net(ls[j:j + B], tf.fill([B], float(snr))), [-1, 64, 8]), nv) for j in range(0, 64, B)]
        r["notre modele"].append(float(tf.reduce_mean(tf.concat(rr, 0))))
    print({k: [round(v, 2) for v in x] for k, x in r.items()}, flush=True)
    json.dump({"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS,
               "pilot_db": 10, "params": run["params"], "rates": r, "history": run["history"]},
              open(os.path.join(RESULTS, "phase13", "massive_pilots.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
