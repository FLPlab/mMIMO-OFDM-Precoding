"""Phase 4g -- attribution : d'ou vient l'ecart restant au genie ? On croise
l'estimation (notre module / genie / canal parfait) et le precodeur (notre
precodeur / WMMSE / RZF), avec le modele retenu (retards + filtre a bande +
precodeur D64), recharge depuis ses poids.

Appris ensemble sur le debit, le module peut sortir le canal a une echelle
arbitraire (le precodeur normalise sa puissance), et le precodeur s'est adapte
a cette echelle. On compare donc aussi a echelle corrigee :
  - notre estimation remise a l'echelle, par (tirage, utilisateur), pour que
    son energie egale celle du LS moins le bruit (realiste, sans vrai canal) ;
  - le genie / le vrai canal donnes au precodeur a l'echelle de NOTRE module. Memes tirages que le melange
(pilote 10 dB). Sans entrainement.

    python -m experiments.phase4_attribution
"""
import json
import os

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, csi_at_bs, genie_lmmse
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import rzf, wmmse
from train import DropPool, evaluate, rate_per_sample
from utils import RESULTS, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, EVAL_DROPS, BATCH = ["umi", "uma", "rma"], 10.0, 192, 32
WEIGHTS = "/export/tmp/sala/freqctx/weights/phase4/delay_bande_est32_D64_T4_linear_melange_p10.weights.h5"


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def rates(precode, h_in, h_true, chunk=16384):
    """precode(x, nv, snr_db) -> W ; debit sur h_true."""
    out = []
    for snr in SNRS_DB:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h_true[i:i + chunk], precode(h_in[i:i + chunk], nv, snr), nv)
             for i in range(0, h_in.shape[0], chunk)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def main():
    cfg = SystemConfig()
    model = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, d=64,
                                layers=4, tied=True, attn="linear")
    ours_p = lambda x, nv, snr: model.precoder(x, tf.fill([x.shape[0]], float(snr)))
    wm = lambda x, nv, snr: wmmse(x, nv)
    rz = lambda x, nv, snr: rzf(x, nv)
    res, built = {}, False
    for i, scen in enumerate(SCENARIOS):
        c = SystemConfig(scenario=scen)
        sionna_config.seed = 2026 + 10 * i                      # memes tirages que csi_pool
        gen, sounder = ChannelGenerator(c), UplinkSounder(c, "lin")
        ls, true, gen_est = [], [], []
        for _ in range(EVAL_DROPS // BATCH):
            a, b = csi_at_bs(gen(BATCH), CSIConfig("ls_lin", PILOT, 0), sounder)
            ls.append(a), true.append(b)
            gen_est.append(genie_lmmse(a, b, gen.last_cir, gen.frequencies, PILOT))
        ls, true, gen_est = (tf.concat(x, 0) for x in (ls, true, gen_est))
        if not built:
            model(ls[:1], tf.zeros([1]))
            model.load_weights(WEIGHTS)
            built = True
        ours_est = tf.concat([model.estimator(ls[j:j + 16]) for j in range(0, ls.shape[0], 16)], 0)
        en = lambda x: tf.reduce_mean(tf.abs(x) ** 2, axis=[1, 3], keepdims=True)   # par (tirage, utilisateur)
        noise = 10.0 ** (-PILOT / 10.0) / cfg.num_users
        ours_n = ours_est * tf.cast(tf.sqrt(tf.nn.relu(en(ls) - noise) / en(ours_est)), tf.complex64)
        to_ours = lambda x: x * tf.cast(tf.sqrt(en(ours_est) / en(x)), tf.complex64)
        print(scen, "echelle moyenne de notre module :", round(float(tf.reduce_mean(tf.sqrt(en(ours_est) / en(true)))), 2))
        nmse = lambda e: float(10 * np.log10(tf.reduce_mean(tf.abs(e - true) ** 2)
                                             / tf.reduce_mean(tf.abs(true) ** 2)))
        t = flat(true)
        r = {"nmse_db": {"LS": nmse(ls), "notre module": nmse(ours_est),
                         "notre module, echelle corrigee": nmse(ours_n), "genie": nmse(gen_est)},
             "modele complet": evaluate(model, DropPool(ls, true), SNRS_DB),
             "notre estimation + WMMSE": rates(wm, flat(ours_est), t),
             "notre estimation (echelle corrigee) + WMMSE": rates(wm, flat(ours_n), t),
             "notre estimation (echelle corrigee) + RZF": rates(rz, flat(ours_n), t),
             "genie + notre precodeur": rates(ours_p, flat(gen_est), t),
             "genie (a l'echelle du module) + notre precodeur": rates(ours_p, flat(to_ours(gen_est)), t),
             "canal parfait (a l'echelle du module) + notre precodeur": rates(ours_p, flat(to_ours(true)), t),
             "genie + WMMSE": rates(wm, flat(gen_est), t),
             "canal parfait + notre precodeur": rates(ours_p, t, t),
             "canal parfait + WMMSE": rates(wm, t, t)}
        res[scen] = r
        print(scen, "NMSE", {k: round(v, 1) for k, v in r["nmse_db"].items()}, "| @20dB",
              {k: round(v[4], 2) for k, v in r.items() if k != "nmse_db"})
    save("phase4", "attribution_p10", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT, "results": res})


if __name__ == "__main__":
    main()
