"""Phase 13 -- panneau « depuis les pilotes » de la figure A avec plus de points de SNR (0 a 30 dB, pas de 5) :
Sionna UMi, puissances reelles, pilote 10 dB, tirages de la phase 11. Notre modele (encodeur convolutif),
Wiener (formule fermee) + WMMSE 100 it., LMMSE (melange) + WMMSE robuste 100 it., LMMSE + RZF.

    python -u -m experiments.phase13_figA_pilots
"""
import json
import os

import tensorflow as tf
from sionna.phy.config import config as sionna_config

import experiments.phase11_power as p11
from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase7_context import cov_mixture
from experiments.phase12_mix import load as load_mix
from experiments.phase12_robust import KAPPA, sound
from experiments.phase12_wiener import wiener
from precoders.classical import rzf, wmmse
from utils import RESULTS

p11.SNRS = [0, 5, 10, 15, 20, 25, 30]


def main():
    net = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv")
    cov = cov_mixture()
    cfg = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen = ChannelGenerator(cfg)
    h = tf.concat([gen(p11.BATCH)[:, 0] for _ in range(p11.DROPS // p11.BATCH)], 0)
    ls, lm, ev = sound(cfg, cov, h, 10.0)
    r = {"notre modele": p11.model_rates(net, ls, h),
         "Wiener formule fermee + WMMSE 100 it.": p11.classical_rates(
             lambda x, nv, e: wmmse(x, nv, num_iter=100), wiener(ls, KAPPA), h),
         "LMMSE + WMMSE robuste 100 it.": p11.classical_rates(
             lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev),
         "LMMSE + RZF": p11.classical_rates(lambda x, nv, e: rzf(x, nv), lm, h)}
    print({k: [round(v, 2) for v in x] for k, x in r.items()}, flush=True)
    json.dump({"snrs_db": p11.SNRS, "pilot_db": 10, "rates": r},
              open(os.path.join(RESULTS, "phase13", "figA_pilots.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
