"""Phase 13 -- CSI parfait : ou la selectivite en frequence compte-t-elle ? Le precodeur (le notre, ou WMMSE
100 it.) n'est calcule que sur 1 SC sur s (ancres), W est interpole lineairement entre ancres puis
renormalise. Canal 3GPP selectif (Sionna UMi, ~400 ns) contre sites du labo presque plats (1-51 ns).
Puissances reelles, debit a 20 dB, une graine (exploration). Notre precodeur n'est PAS reentraine pour
l'interpolation (celui de la base a encodeur convolutif).

    python -u -m experiments.phase13_interp
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase11_power import BATCH
from experiments.phase12_mix import load as load_mix
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

STEPS, SNR = [1, 3, 6, 12, 24, 48], 20.0


def interp_rate(h, fn, s):
    """h [B, F, K, M] ; fn(h_flat, nv) -> W ; precodeur aux ancres, W interpole, debit sur toutes les SC."""
    B, F, K, M = h.shape
    anchors = np.unique(np.append(np.arange(0, F, s), F - 1))
    nv = 10 ** (-SNR / 10)
    out = []
    for b in range(0, B, BATCH):
        hb = h[b:b + BATCH]
        ha = tf.gather(hb, anchors, axis=1)
        wa = tf.reshape(fn(tf.reshape(ha, [-1, K, M]), nv), [hb.shape[0], len(anchors), M, K])
        j = np.clip(np.searchsorted(anchors, np.arange(F), side="right") - 1, 0, len(anchors) - 2)
        t = ((np.arange(F) - anchors[j]) / (anchors[j + 1] - anchors[j])).astype(np.float32)[None, :, None, None]
        t = tf.cast(t, wa.dtype)
        w = (1 - t) * tf.gather(wa, j, axis=1) + t * tf.gather(wa, j + 1, axis=1)
        w = normalize_power(tf.reshape(w, [-1, M, K]))
        out.append(rate_per_sample(tf.reshape(hb, [-1, K, M]), w, nv))
    return float(tf.reduce_mean(tf.concat(out, 0)))


def main():
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    prec = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv").precoder
    sets = {s: drops(read_cells(s, "psi2az"), freqs, equal=False, seed=13) for s in ("stecath", "decarie", "ericsson")}
    sionna_config.seed = 4026
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    sets["sionna_umi"] = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snr_db": SNR,
           "steps": STEPS, "rates": {}}
    for name, h in sets.items():
        r = {"notre precodeur": [interp_rate(h, lambda x, nv: prec(x, tf.fill([x.shape[0]], SNR)), s) for s in STEPS],
             "WMMSE 100 it.": [interp_rate(h, lambda x, nv: wmmse(x, nv, num_iter=100), s) for s in STEPS]}
        out["rates"][name] = r
        print(name, {k: [round(x, 2) for x in v] for k, v in r.items()}, flush=True)
    path = os.path.join(RESULTS, "phase13", "interp.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
