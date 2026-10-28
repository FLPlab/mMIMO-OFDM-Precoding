"""Phase 7 -- verification classique (CSI parfait, RZF) avant de reentrainer :
les coefficients C (W = h^H C) peuvent-ils etre calcules une fois par groupe
de SC et interpoles, si l'on fixe la "jauge" du canal par SC ?

Jauge : chaque vecteur h_k(f) est normalise (norme 1) et sa phase est fixee
par l'antenne 0 (h_k0(f) reel positif). C = (G + aI)^-1 calcule aux centres
des groupes (G de la jauge), tenu ou interpole lineairement, W(f) = h(f)^H C.
Compare a W calcule a chaque SC, et a W tenu ou interpole sur le groupe
(groupes de 2 SC a 4 RB).

    python -u -m experiments.phase7_gauge_check
"""
import json
import os

import numpy as np
import tensorflow as tf

from config import SystemConfig
from experiments.phase7_analysis import group_center, interp_centers
from experiments.phase7_context import SCENARIOS
from precoders.classical import normalize_power, rzf
from train import channel_pool, rate_per_sample
from utils import RESULTS

GROUPS = {"2 SC": 2, "3 SC": 3, "4 SC": 4, "6 SC": 6, "1 RB": 12, "2 RB": 24, "4 RB": 48}
SNRS = [10, 20, 25]


def gauge(h):
    """h [B, F, K, M] : norme 1 et phase de l'antenne 0 nulle, par (tirage, SC, utilisateur)."""
    h = h / tf.cast(tf.norm(h, axis=-1, keepdims=True), h.dtype)
    ph = h[..., :1] / tf.cast(tf.abs(h[..., :1]) + 1e-12, h.dtype)
    return h * tf.math.conj(ph)


def c_rzf(h, nv):
    K = h.shape[-2]
    g = tf.matmul(h, h, adjoint_b=True)
    return tf.linalg.inv(g + tf.cast(K * nv, h.dtype) * tf.eye(K, dtype=h.dtype))


def rate(h, w, nv):
    K, M = h.shape[-2], h.shape[-1]
    return float(tf.reduce_mean(rate_per_sample(tf.reshape(h, [-1, K, M]), tf.reshape(w, [-1, M, K]), nv)))


def main():
    out = {}
    for i, scen in enumerate(SCENARIOS):
        h = tf.reshape(channel_pool(SystemConfig(scenario=scen), 64, 2026 + 10 * i, batch=16), [64, 612, 4, 8])
        out[scen] = {}
        for snr in SNRS:
            nv = 10.0 ** (-snr / 10.0)
            res = {}
            for name, x in (("brut", h), ("jauge fixee", gauge(h))):
                c = c_rzf(x, nv)
                w_sc = normalize_power(tf.matmul(x, c, adjoint_a=True))
                res[name] = {"par SC": rate(h, w_sc, nv)}
                for g, n in GROUPS.items():
                    res[name][g] = {"W tenu": rate(h, group_center(w_sc, n), nv),
                                    "W interpole": rate(h, normalize_power(interp_centers(w_sc, n)), nv),
                                    "C tenu": rate(h, normalize_power(tf.matmul(x, group_center(c, n),
                                                                                adjoint_a=True)), nv),
                                    "C interpole": rate(h, normalize_power(tf.matmul(x, interp_centers(c, n),
                                                                                     adjoint_a=True)), nv)}
            out[scen][snr] = res
            print(scen, snr, json.dumps(res), flush=True)
    path = os.path.join(RESULTS, "phase7", "gauge_check.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
