"""Phase 11 -- puissances inegales entre utilisateurs (affaiblissement de parcours + masque 3GPP de
Sionna gardes ; ecart median ~30 dB entre le plus fort et le plus faible utilisateur d'un tirage).
Sondage avec controle de puissance montant complet (pilotes au meme SNR, la BS remet chaque
estimation a l'echelle) ; debit sur le vrai canal, puissances reelles. M8K4, pilote 10 dB, melange
UMi + UMa + RMa, 192 tirages / scenario. Evaluation seule.

Modeles (base figee, FFN 32, pilote aleatoire) : entree d'origine ou normalisee par utilisateur,
entraines a puissances egales ou reelles. References : RZF et WMMSE robuste (30 et 100 iterations)
sur LMMSE (covariance du melange, appliquee aux canaux remis a puissance egale : exacte ici).

    python -u -m experiments.phase11_power
"""
import json
import os
import time

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import PRECODER, SCENARIOS, W7, cov_mixture, flat, sound_power_controlled
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

MODELS = {"origine, puissances egales": ("pilot_rand_ph32", "raw"),
          "normalisee, puissances egales": ("pilot_rand_ph32_norm", "norm"),
          "origine, puissances reelles": ("pilot_rand_ph32_relpow", "raw"),
          "normalisee, puissances reelles": ("pilot_rand_ph32_norm_relpow", "norm")}
SNRS, PILOT, DROPS, BATCH, CHUNK = [0, 10, 20, 25], 10.0, 192, 16, 8192


def load(name, inp):
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", **dict(PRECODER, hidden=32, inp=inp))
    net(tf.zeros([1, cfg.num_sc, cfg.num_users, cfg.num_bs_ant], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W7, f"{name}.weights.h5"))
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


def classical_rates(fn, hh, h, err=None):
    hh, h = flat(hh), flat(h)
    err = None if err is None else flat(err)
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h[i:i + CHUNK], fn(hh[i:i + CHUNK], nv, None if err is None else err[i:i + CHUNK]), nv)
             for i in range(0, hh.shape[0], CHUNK)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def main():
    nets = {k: load(*v) for k, v in MODELS.items() if os.path.exists(os.path.join(W7, f"{v[0]}.weights.h5"))}
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "snrs_db": SNRS, "pilot_snr_db": PILOT, "rates": {}}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen, num_slots=1, relative_power=True)
        sionna_config.seed = 4026 + 10 * i
        gen = ChannelGenerator(cfg)
        h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
        lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
        ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], PILOT) for j in range(0, DROPS, BATCH)], 0)
        pairs = [sound_power_controlled(lmm, h[j:j + BATCH], PILOT, return_err_var=True) for j in range(0, DROPS, BATCH)]
        lm, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
        r = {k: model_rates(n, ls, h) for k, n in nets.items()}
        r["RZF (LMMSE)"] = classical_rates(lambda x, nv, e: rzf(x, nv), lm, h)
        r["WMMSE robuste 30 it. (LMMSE)"] = classical_rates(lambda x, nv, e: wmmse(x, nv, err_var=e), lm, h, ev)
        r["WMMSE robuste 100 it. (LMMSE)"] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev)
        out["rates"][scen] = r
        print(f"{scen} @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase11"), exist_ok=True)
    path = os.path.join(RESULTS, "phase11", "power_spread.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
