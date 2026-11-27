"""Phase 18 -- precodeur par paires RETRECI, entraine DE ZERO (copie minimale de
experiments/phase13_train_massive.py ; ne modifie pas l'original). Meme architecture
(attention triangulaire, G reinjecte, blocs distincts, entree normalisee "norm"), meme
donnees (melange M64K8 + M8K4, UMi+UMa+RMa Sionna, puissances reelles), meme perte
(-moyenne(debit / debit RZF)), meme SNR d'entrainement U(0, 30) dB, meme budget
(60 000 pas, batch 2048). Seuls d, hidden, layers changent (--d --hidden --layers).

Objectif (26 nov. 2026) : reduire le cout du precodeur libre sans imposer de forme.
  E1 : d=8, hidden=16, layers=8  (~190 k MAC/SC en M8K4)
  E2 : d=8, hidden=16, layers=12 (~290 k MAC/SC en M8K4)

    python -u -m experiments.phase18_narrow --d 8 --hidden 16 --layers 8
"""
import argparse
import json
import os
import time

import keras
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_train_massive import SPLIT, pool
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W18 = "/export/tmp/sala/freqctx/weights/phase18"
STEPS, BATCH = 60000, 2048


def main(d, hidden, layers, steps=STEPS, tag=None):
    tag = tag or f"narrow_d{d}_h{hidden}_L{layers}"
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, f"donnees {time.time() - t0:.0f} s", flush=True)
    net = GramPrecoder(64, d=d, hidden=hidden, layers=layers, mix="triangular_g", tied=False, inp="norm")
    for p in pools.values():
        net(p[:2], tf.zeros([2]))
    print(f"MAC/SC M8K4 {net.macs_per_sc(4, 8)}  M64K8 {net.macs_per_sc(8, 64)}  "
          f"params {int(net.count_params())}", flush=True)
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
                loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
            g, _ = tf.clip_by_global_norm(tape.gradient(loss, net.trainable_variables), 1.0)
            opt.apply_gradients(zip(g, net.trainable_variables))
            return loss
        return step

    keys = list(pools)
    steps_fn = {k: make_step(pools[k]) for k in keys}
    hist = []
    for s in range(1, steps + 1):
        loss = steps_fn[keys[s % 2]]()
        if s % 5000 < 2 or s == steps:
            hist.append({"step": s, "obj": -float(loss), "config": str(keys[s % 2]), "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    os.makedirs(W18, exist_ok=True)
    wpath = os.path.join(W18, f"{tag}.weights.h5")
    net.save_weights(wpath)
    sionna_config.seed = 4026                                           # memes tirages que phase13_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026                                           # memes tirages que phase13_bench (umi)
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                     "d": d, "hidden": hidden, "layers": layers, "steps": steps}, "snrs_db": SNRS,
           "history": hist, "params": int(net.count_params()),
           "macs_per_sc_M8K4": int(net.macs_per_sc(4, 8)), "macs_per_sc_M64K8": int(net.macs_per_sc(8, 64)),
           "M64K8": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in ("M64K8", "M8K4")}, flush=True)
    os.makedirs(os.path.join(RESULTS, "phase18"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase18", f"{tag}_train.json"), "w"), indent=1)
    print(f"-> poids {wpath}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", type=str, default=None)
    a = ap.parse_args()
    main(a.d, a.hidden, a.layers, steps=a.steps, tag=a.tag)
