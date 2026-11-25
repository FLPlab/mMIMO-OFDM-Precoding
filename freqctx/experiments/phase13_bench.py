"""Phase 13 -- banc en CSI PARFAIT, plusieurs jeux de donnees et SNR de 0 a 30 dB (exploration, une graine) :
notre precodeur (base a encodeur convolutif, precodeur seul, non reentraine) contre WMMSE a 10 / 30 / 100 /
300 iterations et RZF. Puissances reelles. Jeux : Sionna UMi / UMa / RMa (3GPP, graines de la phase 11) et
sites du labo St-Catherine / Decarie / Ericsson (convention psi2az). Aucune fonction Sionna dans le calcul
des debits.

    python -u -m experiments.phase13_bench --set umi|uma|rma|stecath|decarie|ericsson
    python -u -m experiments.phase13_bench --merge
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase11_power import BATCH, CHUNK
from experiments.phase12_mix import load as load_mix
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

SNRS = [0, 5, 10, 15, 20, 25, 30]
SETS = ["umi", "uma", "rma", "stecath", "decarie", "ericsson"]
OUT = os.path.join(RESULTS, "phase13")


def rates(fn, h):
    h = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h[i:i + CHUNK], fn(h[i:i + CHUNK], nv, snr), nv) for i in range(0, h.shape[0], CHUNK)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def channels(name):
    if name in ("umi", "uma", "rma"):
        sionna_config.seed = 4026 + 10 * ["umi", "uma", "rma"].index(name)       # tirages de la phase 11
        gen = ChannelGenerator(SystemConfig(scenario=name, num_slots=1, relative_power=True))
        return tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    cfg = SystemConfig()
    return drops(read_cells(name, "psi2az"), subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy(),
                 equal=False, seed=13)


def run(name):
    prec = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv").precoder
    h = channels(name)
    r = {"notre precodeur": rates(lambda x, nv, s: prec(x, tf.fill([x.shape[0]], float(s))), h),
         "RZF": rates(lambda x, nv, s: rzf(x, nv), h)}
    for it in (10, 30, 100, 300):
        r[f"WMMSE {it} it."] = rates(lambda x, nv, s: wmmse(x, nv, num_iter=it), h)
    print(name, {k: [round(v, 2) for v in x] for k, x in r.items()}, flush=True)
    json.dump({"snrs_db": SNRS, "rates": r}, open(os.path.join(OUT, f"bench_{name}.json"), "w"), indent=1)


def merge():
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS,
           "sets": {s: json.load(open(os.path.join(OUT, f"bench_{s}.json")))["rates"] for s in SETS}}
    json.dump(out, open(os.path.join(OUT, "bench.json"), "w"), indent=1)
    for s in SETS:
        os.remove(os.path.join(OUT, f"bench_{s}.json"))
    print("->", os.path.join(OUT, "bench.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a.set)
