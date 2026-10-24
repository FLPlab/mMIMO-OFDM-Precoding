"""Phase 4e -- reference ROBUSTE : WMMSE a erreur moyenne sur l'estimation
LMMSE, avec la covariance d'erreur (diagonale) de l'estimateur Sionna.
Deux versions de la variance d'erreur :
    "Sionna"  : celle annoncee par LMMSEInterpolator (ce qu'une BS aurait)
    "calibree": la meme, remise a l'echelle pour egaler l'erreur reelle moyenne
                du scenario (version la plus forte, favorable au classique)
Deux covariances de canal pour LMMSE : melange (realiste) et oracle.
Memes tirages et meme bruit pilote que le melange (pilote 10 dB). Sans entrainement.
Comparaison : notre modele (results/phase4/delay_model_D64.json).

    python -m experiments.phase4_robust_ref
"""
import json
import os

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder, estimate_freq_cov
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, EVAL_DROPS, BATCH = ["umi", "uma", "rma"], 10.0, 192, 32


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def rates(h_hat, h_true, err=None, chunk=16384):
    out = []
    for snr in SNRS_DB:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h_true[i:i + chunk],
                             wmmse(h_hat[i:i + chunk], nv,
                                   err_var=None if err is None else err[i:i + chunk]), nv)
             for i in range(0, h_hat.shape[0], chunk)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def main():
    ours = json.load(open(os.path.join(RESULTS, "phase4", "delay_model_D64.json")))
    mixed = json.load(open(os.path.join(RESULTS, "phase3", "mixed_p10.json")))["references"]
    covs = {}
    for i, scen in enumerate(SCENARIOS):                      # comme phase3_mixed
        sionna_config.seed = 2027 + 10 * i
        covs[scen] = estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen)))
    cov_mix = tf.add_n(list(covs.values())) / len(covs)

    res = {}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        res[scen] = {}
        for label, cov in (("melange", cov_mix), ("oracle", covs[scen])):
            sionna_config.seed = 2026 + 10 * i                 # memes tirages que csi_pool
            gen, sounder = ChannelGenerator(cfg), UplinkSounder(cfg, "lmmse", cov)
            hh, ev, ht = [], [], []
            for _ in range(EVAL_DROPS // BATCH):
                h = gen(BATCH)
                a, b = sounder(h[:, 0], PILOT, return_err_var=True)
                hh.append(a), ev.append(b), ht.append(h[:, 0])
            hh, ev, ht = (flat(tf.concat(x, 0)) for x in (hh, ev, ht))
            real_err = float(tf.reduce_mean(tf.abs(hh - ht) ** 2))
            scale = real_err / float(tf.reduce_mean(ev))
            r = {"WMMSE": rates(hh, ht), "WMMSE robuste (Sionna)": rates(hh, ht, ev),
                 "WMMSE robuste (calibree)": rates(hh, ht, ev * scale),
                 "err_var_sionna": float(tf.reduce_mean(ev)), "err_reelle": real_err}
            stored = mixed[scen][f"WMMSE (LMMSE {label})"][4]
            assert abs(r["WMMSE"][4] - stored) < 1e-2, (scen, label, r["WMMSE"][4], stored)
            res[scen][label] = r
            print(f"{scen} {label} @20dB : WMMSE {r['WMMSE'][4]:.2f} | robuste Sionna "
                  f"{r['WMMSE robuste (Sionna)'][4]:.2f} | robuste calibree "
                  f"{r['WMMSE robuste (calibree)'][4]:.2f} | notre modele {ours['rates'][scen][4]:.2f}"
                  f" | erreur annoncee/reelle {r['err_var_sionna']:.4f}/{real_err:.4f}")
    save("phase4", "robust_ref_p10", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT, "results": res,
                                      "notre_modele": ours["rates"]})


if __name__ == "__main__":
    main()
