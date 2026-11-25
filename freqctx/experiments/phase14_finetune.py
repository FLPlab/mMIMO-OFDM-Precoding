"""Phase 14 -- fine-tuning avec peu de donnees sur un site du labo (demande de F. Leduc-Primeau, 2026-11-25).

Precodeur de depart : precodeur par paires entraine sur les canaux 3GPP (M8K4 + M64K8, puissances reelles,
phase13_train_massive), CSI parfait, M=8, K=4. Site du labo (MIMO GenEE en OFDM, convention psi2az).
Les cellules du site sont separees en deux moities disjointes : adaptation / test.
Adaptation : N echantillons (N = 40, 400, 4 000 ; un echantillon = 4 utilisateurs sur une sous-porteuse), utilisateurs
permutes au hasard a chaque pas, SNR U(0, 30) dB, perte -debit / debit RZF. Trois methodes :
  complet : tous les poids (pas 1e-4) ;
  lora    : W + A B sur chaque couche dense, rang 4 (A, B appris, W fige ; pas 1e-3) ;
  svd     : W = U diag(s) V^T, seules les valeurs singulieres s apprises (« vecteurs propres » ; pas 1e-3).
Test : 64 tirages de 4 cellules de la moitie test, SNR 0 a 30 dB, contre WMMSE 30 et 300 it. et le depart (zero-shot).

    python -u -m experiments.phase14_finetune --site stecath|decarie|ericsson
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_deepmimo import channel
from experiments.phase13_lab import read_cells
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

BASE = "/export/tmp/sala/freqctx/weights/phase13/precoder_relpow_mix.weights.h5"
NS, TEST_DROPS = [40, 400, 4000], 64
STEPS = {40: 500, 400: 1000, 4000: 2000}


def load_base():
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(BASE)
    return net


def site_drops(cells, n, freqs, seed, per_sc):
    """per_sc : n echantillons (1 SC au hasard chacun) [n, K, M] ; sinon n tirages complets [n, F, K, M]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = np.stack([channel(cells[i], freqs, rng) for i in rng.choice(len(cells), 4, replace=False)], 1)   # [F, K, M]
        h /= np.sqrt(np.mean(np.abs(h) ** 2))
        out.append(h[rng.integers(h.shape[0])] if per_sc else h)
    return tf.constant(np.stack(out).astype(np.complex64))


def peft(net, method, rank=4):
    """Remplace l'appel de chaque couche dense ; renvoie les variables a apprendre."""
    if method == "complet":
        return net.trainable_variables
    train_vars = []
    for layer in [l for l in net._flatten_layers() if isinstance(l, keras.layers.Dense)]:
        w0 = tf.constant(layer.kernel.numpy())
        b = layer.bias
        act = layer.activation
        if method == "lora":
            a = tf.Variable(tf.random.normal([w0.shape[0], rank], stddev=0.01))
            bb = tf.Variable(tf.zeros([rank, w0.shape[1]]))
            train_vars += [a, bb]
            kern = lambda w0=w0, a=a, bb=bb: w0 + a @ bb
        else:                                                                   # svd : valeurs singulieres seules
            s, u, v = tf.linalg.svd(w0)
            sv = tf.Variable(s)
            train_vars.append(sv)
            kern = lambda u=u, v=v, sv=sv: (u * sv) @ tf.transpose(v)

        def call(x, *args, kern=kern, b=b, act=act, **kw):
            y = tf.tensordot(x, kern(), axes=[[x.shape.rank - 1], [0]])
            return act(y + b if b is not None else y)
        layer.call = call
    return train_vars


def finetune(net, method, data, steps, lr):
    variables = peft(net, method)
    opt = keras.optimizers.Adam(lr)
    n = data.shape[0]
    bs = min(n, 256)

    @tf.function
    def step():
        h = tf.gather(data, tf.random.uniform([bs], 0, n, dtype=tf.int32))
        perm = tf.argsort(tf.random.uniform([bs, 4]), axis=-1)                    # permutation des utilisateurs
        h = tf.gather(h, perm, batch_dims=1)
        snr = tf.random.uniform([bs], 0.0, 30.0)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
        g = tape.gradient(loss, variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, variables))
        return loss

    for _ in range(steps):
        step()
    return sum(int(np.prod(v.shape)) for v in variables)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["stecath", "decarie", "ericsson"])
    a = ap.parse_args()
    keras.utils.set_random_seed(0)
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    cells = read_cells(a.site, "psi2az")
    adapt, test = cells[0::2], cells[1::2]                                      # moities disjointes
    h_test = site_drops(test, TEST_DROPS, freqs, seed=77, per_sc=False)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "site": a.site,
                    "cells_adapt": len(adapt), "cells_test": len(test)}, "snrs_db": SNRS, "rates": {}, "params": {}}
    t0 = time.time()
    out["rates"]["WMMSE 300 it."] = rates(lambda x, nv, s: wmmse(x, nv, num_iter=300), h_test)
    out["rates"]["WMMSE 30 it."] = rates(lambda x, nv, s: wmmse(x, nv, num_iter=30), h_test)
    base = load_base()
    out["rates"]["zero-shot"] = rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), h_test)
    print(f"{a.site} references ({time.time() - t0:.0f} s) : WMMSE300 {out['rates']['WMMSE 300 it.'][4]:.2f} / "
          f"zero-shot {out['rates']['zero-shot'][4]:.2f} a 20 dB, "
          f"{out['rates']['WMMSE 300 it.'][6]:.2f} / {out['rates']['zero-shot'][6]:.2f} a 30 dB", flush=True)
    for n in NS:
        data = site_drops(adapt, n, freqs, seed=100 + n, per_sc=True)
        for method, lr in (("complet", 1e-4), ("lora", 1e-3), ("svd", 1e-3)):
            net = load_base()
            npar = finetune(net, method, data, STEPS[n], lr)
            key = f"{method} N={n}"
            out["rates"][key] = rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h_test)
            out["params"][key] = npar
            print(f"{a.site} {key} ({npar} param. appris) : 20 dB {out['rates'][key][4]:.2f}, 30 dB {out['rates'][key][6]:.2f}",
                  flush=True)
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase14", f"finetune_{a.site}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
