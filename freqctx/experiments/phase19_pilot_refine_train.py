"""Phase 19 -- 2e passe ADAPTEE a la chaine v2 (encodeur + precodeur, FIGES).

Meme architecture que la passe phase 15 (RefineNet : 4 blocs triangulaires, d 16, FFN 32, lecture
initialisee a zero -> depart = v2 exactement), mais la base est le precodeur v2 et l'entree est la SORTIE
DE L'ENCODEUR h^ (G et E calcules sur h^), le debit est calcule sur le VRAI canal.
Donnees = celles de l'entrainement v2 (phase7_context.data_random_pilot(relative_power=True) : UMi + UMa +
RMa, 1 056 tirages / scenario, SNR pilote U(0, 20) par lot de 8 tirages, controle de puissance montant).
L'encodeur etant fige, h^ est calcule une fois pour tout le jeu ; on tire ensuite des lots de 2 048 SC.
Perte : -moyenne(debit / debit RZF sur le LS), comme v2 (rel_rzf). SNR de donnees U(0, 30) comme la passe
phase 15 (la passe ne sert qu'au fort SNR ; v2 a ete entraine sur U(0, 25)).

    python -u -m experiments.phase19_pilot_refine_train [--steps 50000] [--quick]
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf

from experiments.phase7_context import data_random_pilot
from experiments.phase15_refine import check_start_equals_base, own_vars
from experiments.phase19_pilot_refine import encode, load_v2, refine_on
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W19 = "/export/tmp/sala/freqctx/weights/phase19"
BATCH = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--snr_hi", type=float, default=30.0)
    a = ap.parse_args()
    if a.quick:
        import experiments.phase7_context as p7
        p7.TRAIN_DROPS, a.steps = 32, 300
    keras.utils.set_random_seed(0)
    t0 = time.time()
    ls, h = data_random_pilot(relative_power=True)                       # [D, F, K, M], memes donnees que v2
    net = load_v2()
    prec = net.precoder
    prec.trainable = False
    hh = encode(net, ls)
    K, M = h.shape[-2], h.shape[-1]
    ls, h, hh = (tf.reshape(x, [-1, K, M]) for x in (ls, h, hh))
    n = int(h.shape[0])
    print(f"donnees : {ls.shape[0] // 612} tirages, {n} SC ({time.time() - t0:.0f} s)", flush=True)

    r = refine_on(prec)                                                  # lecture nulle -> depart = v2
    check_start_equals_base(r, hh[:256])
    tv = own_vars(r)
    print(f"variables entrainees (passe seule) : {sum(int(tf.size(v)) for v in tv)}", flush=True)
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, a.steps))
    w_before = [w.numpy().copy() for w in prec.weights]

    @tf.function
    def step():
        idx = tf.random.uniform([BATCH], 0, n, dtype=tf.int32)
        x, y, l = tf.gather(hh, idx), tf.gather(h, idx), tf.gather(ls, idx)
        snr = tf.random.uniform([BATCH], 0.0, a.snr_hi)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(y, rzf(l, tf.reshape(nv, [-1, 1, 1])), nv)
        base = rate_per_sample(y, prec(x, snr), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(y, r(x, snr), nv) / tf.stop_gradient(ref))
        g = tape.gradient(loss, tv)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, tv))
        return loss, -tf.reduce_mean(base / ref)

    hist = []
    for s in range(1, a.steps + 1):
        loss, lb = step()
        if s % 2500 == 0 or s == a.steps or s == 1:
            hist.append({"step": s, "obj": -float(loss), "obj_v2": -float(lb), "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    assert all(np.array_equal(u, v.numpy()) for u, v in zip(w_before, prec.weights)), "base v2 modifiee"
    os.makedirs(W19, exist_ok=True)
    wpath = os.path.join(W19, "refine_on_v2" + ("_quick" if a.quick else "") + ".weights.h5")
    r.save_weights(wpath)
    os.makedirs(os.path.join(RESULTS, "phase19"), exist_ok=True)
    json.dump({"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "steps": a.steps,
                        "batch": BATCH, "snr_train": [0.0, a.snr_hi], "pilot_train": [0.0, 20.0]},
               "history": hist},
              open(os.path.join(RESULTS, "phase19", "refine_on_v2_train" + ("_quick" if a.quick else "") + ".json"), "w"),
              indent=1)
    print(f"-> poids {wpath} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
