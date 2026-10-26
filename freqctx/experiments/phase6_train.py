"""Phase 6 -- entraine UNE architecture de precodeur sur le melange M64K8 + M8K4
(banc experiments/phase6_bench.py), puis l'evalue sur toutes les configurations.

Un pas = un lot d'une seule configuration, en alternance (les formes de
tenseur different). Perte : -moyenne(debit / debit RZF) (chaque SNR et
chaque configuration pesent pareil). Adam, decroissance cosinus.

    python -u -m experiments.phase6_train --arch gram --layers 8 [--tag ...]
Architectures :
    gram        : etat par paire, bloc G.E (forme "deroulee"), W = h^H C
    tri         : etat par paire, attention triangulaire apprise, W = h^H C
    trig        : tri + G reinjecte a chaque couche
    axial       : un token par (antenne, utilisateur), Gram appris par couche, W direct
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf

from experiments.phase6_bench import SNRS, evaluate_all, train_pools
from models.axial_precoder import AxialPrecoder
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W6 = "/export/tmp/sala/freqctx/weights/phase6"


def build(a):
    if a.arch == "axial":
        return AxialPrecoder(d=a.d, hidden=a.hidden, heads=a.heads, layers=a.layers, tied=not a.untied)
    mix = {"gram": "gram", "tri": "triangular", "trig": "triangular_g"}[a.arch]
    return GramPrecoder(64, d=a.d, hidden=a.hidden, layers=a.layers, mix=mix, tied=not a.untied,
                        gnorm=a.gnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["gram", "tri", "trig", "axial"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--gnorm", action="store_true")
    ap.add_argument("--train", default="mix", choices=["mix", "massive"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = a.tag or (f"{a.arch}_L{a.layers}_d{a.d}_h{a.hidden}" + ("_untied" if a.untied else "")
                     + ("_gnorm" if a.gnorm else "") + ("" if a.train == "mix" else "_trainmassive")
                     + (f"_s{a.seed}" if a.seed else ""))
    print("==", name, flush=True)
    keras.utils.set_random_seed(a.seed)
    pools = train_pools()
    if a.train == "massive":
        pools = {(64, 8): pools[(64, 8)]}
    keys = list(pools)
    net = build(a)
    for (M, K) in keys:
        net(pools[(M, K)][:2], tf.zeros([2]))
    params = int(net.count_params())
    sched = keras.optimizers.schedules.CosineDecay(a.lr, a.steps)
    opt = keras.optimizers.Adam(sched)

    def make_step(pool):
        n = pool.shape[0]

        @tf.function
        def step():
            idx = tf.random.uniform([a.batch], 0, n, dtype=tf.int32)
            h = tf.gather(pool, idx)
            snr = tf.random.uniform([a.batch], 0.0, 30.0)
            nv = 10.0 ** (-snr / 10.0)
            r_ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
            with tf.GradientTape() as tape:
                r = rate_per_sample(h, net(h, snr), nv)
                loss = -tf.reduce_mean(r / tf.stop_gradient(r_ref))
            g = tape.gradient(loss, net.trainable_variables)
            g, _ = tf.clip_by_global_norm(g, 1.0)
            opt.apply_gradients(zip(g, net.trainable_variables))
            return loss
        return step

    steps = {k: make_step(pools[k]) for k in keys}
    hist, t0, acc = [], time.time(), {k: [] for k in keys}
    for s in range(1, a.steps + 1):
        k = keys[s % len(keys)]
        loss = steps[k]()
        if s % 100 < len(keys):                                 # releve (synchronise) ~1 pas sur 100
            acc[k].append(float(loss))
        if s % 5000 == 0:
            entry = {"step": s, "time_s": round(time.time() - t0, 1)}
            for kk in keys:
                vals = acc[kk]
                entry[f"obj_M{kk[0]}K{kk[1]}"] = -float(np.mean(vals)) if vals else None
            acc = {kk: [] for kk in keys}
            hist.append(entry)
            print("   ", entry, flush=True)
            if any(e is not None and not np.isfinite(e) for kk, e in entry.items() if kk.startswith("obj")):
                print("   divergence : arret", flush=True)
                break
    os.makedirs(W6, exist_ok=True)
    net.save_weights(os.path.join(W6, f"{name}.weights.h5"))
    res = evaluate_all(net)
    macs = {f"M{M}K{K}": int(net.macs_per_sc(K, M)) for (M, K) in [(64, 8), (8, 4)]}
    os.makedirs(os.path.join(RESULTS, "phase6"), exist_ok=True)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "name": name, "args": vars(a), "params": params, "macs_per_sc": macs,
           "snrs_db": SNRS, "training_history": hist, "eval": res}
    path = os.path.join(RESULTS, "phase6", f"{name}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, "| params", params, "| MAC/SC", macs, flush=True)


if __name__ == "__main__":
    main()
