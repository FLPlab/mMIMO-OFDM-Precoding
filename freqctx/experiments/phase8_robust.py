"""Phase 8 -- robustesse du modele principal (M8K4, entraine a pilote 10 dB,
donnees au slot du sondage), SANS reentrainer.

1. SNR pilote different de l'entrainement : 0, 5, 10, 15, 20 dB.
2. Vieillissement : donnees 0, 2,5 ou 5 ms apres le sondage (slot_step = 5
   slots de 0,5 ms), utilisateurs a 3-30 km/h, pilote 10 dB.
Modeles : principal (par SC) et interpolation 1 SC sur 3. References au meme
pilote et au meme instant : WMMSE et WMMSE robuste sur LMMSE (covariance du
melange, variance d'erreur annoncee par Sionna, qui ignore le vieillissement,
comme une BS reelle).

    python -u -m experiments.phase8_robust [--models "modele (pilote aleatoire)" --out robustness_pilot_rand]
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import PRECODER, SCENARIOS, cov_mixture, flat
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

MODELS = {"modele (par SC)": ("/export/tmp/sala/freqctx/weights/phase4",
                              "delay_bande_os2_trigL8_untied_n1056_s80k_est32_D64_T4_linear_melange_p10", None),
          "modele (1 SC sur 3)": ("/export/tmp/sala/freqctx/weights/phase7", "interp_s3", 3),
          "modele (pilote aleatoire)": ("/export/tmp/sala/freqctx/weights/phase7", "pilot_rand", None),
          "base figee (pilote aleatoire, FFN 32)": ("/export/tmp/sala/freqctx/weights/phase7", "pilot_rand_ph32",
                                                     None, {"hidden": 32})}
PILOTS, DELAYS_MS, SNRS = [0, 5, 10, 15, 20], [0.0, 2.5, 5.0], [0, 10, 20, 25]
EVAL_DROPS, BATCH, CHUNK = 192, 16, 16384


def load(path, name, step, override=None):
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", anchor_step=step, **dict(PRECODER, **(override or {})))
    net(tf.zeros([1, cfg.num_sc, cfg.num_users, cfg.num_bs_ant], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(path, f"{name}.weights.h5"))
    return net


def model_rates(net, ls, h):
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(flat(h[i:i + BATCH]),
                             tf.reshape(net(ls[i:i + BATCH], tf.fill([BATCH], float(snr))), [-1, 8, 4]), nv)
             for i in range(0, ls.shape[0], BATCH)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def wmmse_rates(hh, h, err=None):
    hh, h = flat(hh), flat(h)
    err = None if err is None else flat(err)
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h[i:i + CHUNK], wmmse(hh[i:i + CHUNK], nv,
                                                   err_var=None if err is None else err[i:i + CHUNK]), nv)
             for i in range(0, hh.shape[0], CHUNK)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def evaluate(nets, ls, lm, err, h):
    res = {k: model_rates(n, ls, h) for k, n in nets.items()}
    res["WMMSE (LMMSE melange)"] = wmmse_rates(lm, h)
    res["WMMSE robuste (LMMSE melange)"] = wmmse_rates(lm, h, err)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="modele (par SC),modele (1 SC sur 3)")
    ap.add_argument("--out", default="robustness")
    a = ap.parse_args()
    nets = {k: load(*MODELS[k]) for k in a.models.split(",")}
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "snrs_db": SNRS, "pilot_sweep": {}, "aging": {}}
    for i, scen in enumerate(SCENARIOS):
        # 1. SNR pilote (donnees au slot du sondage)
        cfg = SystemConfig(scenario=scen, num_slots=1)
        sionna_config.seed = 2026 + 10 * i
        gen = ChannelGenerator(cfg)
        h = tf.concat([gen(BATCH)[:, 0] for _ in range(EVAL_DROPS // BATCH)], 0)
        lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
        out["pilot_sweep"][scen] = {}
        for p in PILOTS:
            ls = tf.concat([lin(h[j:j + BATCH], float(p)) for j in range(0, EVAL_DROPS, BATCH)], 0)
            pairs = [lmm(h[j:j + BATCH], float(p), return_err_var=True) for j in range(0, EVAL_DROPS, BATCH)]
            lm, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
            out["pilot_sweep"][scen][p] = r = evaluate(nets, ls, lm, ev, h)
            print(f"{scen} pilote {p} dB @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
        # 2. Vieillissement (pilote 10 dB)
        cfg = SystemConfig(scenario=scen, num_slots=3, slot_step=5)
        sionna_config.seed = 3026 + 10 * i
        gen = ChannelGenerator(cfg)
        hs = tf.concat([gen(BATCH) for _ in range(EVAL_DROPS // BATCH)], 0)       # [N, 3, F, K, M]
        lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
        ls = tf.concat([lin(hs[j:j + BATCH, 0], 10.0) for j in range(0, EVAL_DROPS, BATCH)], 0)
        pairs = [lmm(hs[j:j + BATCH, 0], 10.0, return_err_var=True) for j in range(0, EVAL_DROPS, BATCH)]
        lm, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
        out["aging"][scen] = {}
        for s, dms in enumerate(DELAYS_MS):
            out["aging"][scen][dms] = r = evaluate(nets, ls, lm, ev, hs[:, s])
            print(f"{scen} retard {dms} ms @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()),
                  flush=True)
    os.makedirs(os.path.join(RESULTS, "phase8"), exist_ok=True)
    path = os.path.join(RESULTS, "phase8", f"{a.out}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
