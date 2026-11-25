"""Phase 14 -- l'ecart au fort SNR sur les sites du labo vient-il de la capacite du precodeur ou de l'entrainement ?
Un seul changement par rapport au precodeur de reference (phase13_train_massive : 8 couches, M8K4 + M64K8,
UMi + UMa + RMa, puissances reelles, SNR U(0, 30) dB, 60 000 pas) :
  --layers 12            plus de couches ;
  --snr_lo 10 --snr_hi 35  entrainement deplace vers le fort SNR.
Evaluation zero-shot (CSI parfait) sur les 3 modeles 3GPP et les 3 sites du labo, memes tirages que phase13_bench.

    python -u -m experiments.phase14_capacity --layers 12
    python -u -m experiments.phase14_capacity --snr_lo 10 --snr_hi 35
"""
import argparse
import json
import os
import time

import keras
import tensorflow as tf

from experiments.phase13_bench import SETS, SNRS, channels, rates
from experiments.phase13_train_massive import SPLIT, pool
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W14 = "/export/tmp/sala/freqctx/weights/phase14"
STEPS, BATCH = 60000, 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--snr_lo", type=float, default=0.0)
    ap.add_argument("--snr_hi", type=float, default=30.0)
    a = ap.parse_args()
    tag = f"L{a.layers}_snr{a.snr_lo:g}-{a.snr_hi:g}"
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    net = GramPrecoder(64, d=16, hidden=32, layers=a.layers, mix="triangular_g", tied=False, inp="norm")
    for p in pools.values():
        net(p[:2], tf.zeros([2]))
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, STEPS))

    def make_step(p):
        n = p.shape[0]

        @tf.function
        def step():
            h = tf.gather(p, tf.random.uniform([BATCH], 0, n, dtype=tf.int32))
            snr = tf.random.uniform([BATCH], a.snr_lo, a.snr_hi)
            nv = 10.0 ** (-snr / 10.0)
            ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
            g, _ = tf.clip_by_global_norm(tape.gradient(loss, net.trainable_variables), 1.0)
            opt.apply_gradients(zip(g, net.trainable_variables))
            return loss
        return step

    keys = list(pools)
    steps = {k: make_step(pools[k]) for k in keys}
    for s in range(1, STEPS + 1):
        loss = steps[keys[s % 2]]()
        if s % 10000 == 0:
            print(f"   pas {s} objectif {-float(loss):.3f} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(W14, exist_ok=True)
    net.save_weights(os.path.join(W14, f"capacity_{tag}.weights.h5"))
    ref = json.load(open(os.path.join(RESULTS, "phase13", "bench.json")))["sets"]
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a),
                    "params": int(net.count_params())}, "snrs_db": SNRS, "rates": {}}
    for name in SETS:
        out["rates"][name] = r = rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), channels(name))
        print(f"{name} : " + " ".join(f"{v:.2f}" for v in r) + "  | WMMSE300 : "
              + " ".join(f"{v:.2f}" for v in ref[name]["WMMSE 300 it."]), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase14", f"capacity_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
