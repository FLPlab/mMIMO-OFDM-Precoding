"""Phase 12 -- encodeur sans softmax (plus simple en HLS) : convolution sur les retards + moyenne globale,
ou attention lineaire, a la place des 2 couches d'attention softmax. Un seul changement par rapport a la
base v2 (`phase7_context --variant pilot_rand --p_hidden 32 --inp norm --relpow --est_mix conv|linear`).
Evaluation dans le cadre de la phase 11 (puissances reelles, controle de puissance montant), memes
tirages que phase12_robust, pilote 0 et 10 dB ; references : base v2, Wiener formule fermee + WMMSE.

    python -u -m experiments.phase12_mix
"""
import json
import os
import time

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase7_context import PRECODER, SCENARIOS, W7, cov_mixture
from experiments.phase11_power import BATCH, DROPS, SNRS, classical_rates, model_rates
from experiments.phase12_robust import KAPPA, sound
from experiments.phase12_wiener import wiener
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import wmmse
from utils import RESULTS, git_commit

MODELS = {"base v2 (softmax)": ("pilot_rand_ph32_norm_relpow", "softmax"),
          "convolution + moyenne": ("pilot_rand_ph32_emconv_norm_relpow", "conv"),
          "attention lineaire": ("pilot_rand_ph32_emlinear_norm_relpow", "linear")}
PILOTS = [0, 10]


def load(name, mix):
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", est_mix=mix, **dict(PRECODER, hidden=32, inp="norm"))
    net(tf.zeros([1, cfg.num_sc, cfg.num_users, cfg.num_bs_ant], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W7, f"{name}.weights.h5"))
    return net


def main():
    nets = {k: load(*v) for k, v in MODELS.items()}
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "snrs_db": SNRS, "params": {k: int(n.count_params()) for k, n in nets.items()},
           "macs_per_sc": {k: int(n.macs_per_sc(4)) for k, n in nets.items()}, "rates": {}}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen, num_slots=1, relative_power=True)
        sionna_config.seed = 4026 + 10 * i                     # tirages de la phase 11
        gen = ChannelGenerator(cfg)
        h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
        out["rates"][scen] = {}
        for p in PILOTS:
            ls, _, _ = sound(cfg, cov, h, float(p))
            r = {k: model_rates(n, ls, h) for k, n in nets.items()}
            r["Wiener formule fermee + WMMSE 100 it."] = classical_rates(
                lambda x, nv, e: wmmse(x, nv, num_iter=100), wiener(ls, KAPPA), h)
            out["rates"][scen][p] = r
            print(f"{scen} pilote {p} dB @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase12"), exist_ok=True)
    path = os.path.join(RESULTS, "phase12", "encoder_mix.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
