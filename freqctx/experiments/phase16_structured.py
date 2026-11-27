"""Phase 16 -- entrainement de la variante STRUCTUREE (models/structured_precoder.py, forme de
Bjornson 2014, Newton-Schulz deroule a T pas) -- prix de la liberte quand la version structuree
est elle-meme ENTRAINEE (demande du 26 nov. 2026). Comparaison : reseau LIBRE (models/gram_precoder.py,
weights/phase13/precoder_relpow_mix.weights.h5).

Memes donnees (3GPP M8K4+M64K8, puissances reelles, import SPLIT/pool depuis
experiments.phase13_train_massive SANS MODIFIER), meme perte (-moyenne(debit/debit RZF)), meme SNR
d'entrainement U(0, 30) dB, meme budget (60 000 pas, batch 2048) que experiments/phase13_train_massive.py.

    python -u -m experiments.phase16_structured --ns T [--steps 60000] [--lam_max X]
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_train_massive import SPLIT, pool
from models.structured_precoder import StructuredPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W16 = "/export/tmp/sala/freqctx/weights/phase16"
OUT_DIR = os.path.join(RESULTS, "phase16")
STEPS, BATCH = 60000, 2048


def main(ns_steps, steps=STEPS, lam_max=None):
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, f"donnees {time.time() - t0:.0f} s", flush=True)

    net = StructuredPrecoder(64, ns_steps=ns_steps, d=16, hidden=32, layers=2, lam_max=lam_max)
    for p in pools.values():
        net(p[:2], tf.zeros([2]))
    print(f"T={ns_steps}  lam_max={lam_max}  parametres {net.count_params()}", flush=True)

    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps))

    def make_step(p):
        n = p.shape[0]

        @tf.function
        def step():
            h = tf.gather(p, tf.random.uniform([BATCH], 0, n, dtype=tf.int32))
            snr = tf.random.uniform([BATCH], 0.0, 30.0)
            nv = 10.0 ** (-snr / 10.0)
            ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
            with tf.GradientTape() as tape:
                w = net(h, snr)
                rate = rate_per_sample(h, w, nv)
                loss = -tf.reduce_mean(rate / tf.stop_gradient(ref))
            g = tape.gradient(loss, net.trainable_variables)
            g, gnorm = tf.clip_by_global_norm(g, 1.0)
            opt.apply_gradients(zip(g, net.trainable_variables))
            finite = tf.reduce_all(tf.math.is_finite(rate)) and tf.math.is_finite(loss)
            return loss, gnorm, finite
        return step

    keys = list(pools)
    stepfns = {k: make_step(pools[k]) for k in keys}
    hist = []
    for s in range(1, steps + 1):
        loss, gnorm, finite = stepfns[keys[s % 2]]()
        if not bool(finite):
            print(f"!! pas fini (NaN/Inf) au pas {s}, config {keys[s % 2]} -- ARRET", flush=True)
            break
        if s % 5000 < 2 or s == steps:
            hist.append({"step": s, "obj": -float(loss), "gnorm": float(gnorm), "config": str(keys[s % 2]),
                         "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)

    os.makedirs(W16, exist_ok=True)
    wpath = os.path.join(W16, f"structured_T{ns_steps}.weights.h5")
    net.save_weights(wpath)

    sionna_config.seed = 4026                                           # memes tirages que phase13_train_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "steps": steps,
                    "ns_steps": ns_steps, "lam_max": lam_max},
           "snrs_db": SNRS, "history": hist, "params": int(net.count_params()),
           "macs_M8K4": int(net.macs_per_sc(4, 8)),
           "M64K8": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in ("M64K8", "M8K4")}, flush=True)
    json.dump(out, open(os.path.join(OUT_DIR, f"train_T{ns_steps}.json"), "w"), indent=1)
    print(f"-> poids {wpath}", flush=True)
    print(f"-> termine en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lam_max", type=float, default=None)
    a = ap.parse_args()
    main(a.ns, steps=a.steps, lam_max=a.lam_max)
