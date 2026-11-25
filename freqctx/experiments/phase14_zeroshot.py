"""Phase 14 -- zero-shot (demande de F. Leduc-Primeau, 2026-11-25) : precodeur par paires (CSI parfait, M=8, K=4,
puissances reelles, entree normalisee, FFN 32, 8 couches) entraine sur UMa SEUL ou sur UMi + UMa + RMa (Sionna),
puis evalue SANS reentrainement sur les 3 modeles 3GPP et les 3 sites du labo (MIMO GenEE en OFDM), SNR 0 a
30 dB, memes tirages que phase13_bench (debits rapportes au WMMSE 300 it. de results/phase13/bench.json).
Le precodeur deja entraine sur M8K4 + M64K8 (3 modeles 3GPP, phase13_train_massive) est evalue aussi.

    python -u -m experiments.phase14_zeroshot --train uma|3gpp|existant
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
from experiments.phase13_bench import SETS, SNRS, channels, rates
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W14 = "/export/tmp/sala/freqctx/weights/phase14"
STEPS, BATCH, DROPS = 40000, 2048, 512


def pool(scens):
    hs = []
    for i, scen in enumerate(scens):
        sionna_config.seed = 1 + i
        gen = ChannelGenerator(SystemConfig(scenario=scen, num_slots=1, relative_power=True))
        for _ in range(DROPS // 8):
            hs.append(tf.reshape(gen(8)[:, 0, ::3], [-1, 4, 8]))           # 1 SC sur 3
    return tf.concat(hs, 0)


def train(scens):
    keras.utils.set_random_seed(0)
    p = pool(scens)
    net = GramPrecoder(8, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(p[:2], tf.zeros([2]))
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, STEPS))

    @tf.function
    def step():
        h = tf.gather(p, tf.random.uniform([BATCH], 0, p.shape[0], dtype=tf.int32))
        snr = tf.random.uniform([BATCH], 0.0, 30.0)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
        g, _ = tf.clip_by_global_norm(tape.gradient(loss, net.trainable_variables), 1.0)
        opt.apply_gradients(zip(g, net.trainable_variables))
        return loss

    t0 = time.time()
    for s in range(1, STEPS + 1):
        loss = step()
        if s % 10000 == 0:
            print(f"   pas {s} objectif {-float(loss):.3f} ({time.time() - t0:.0f} s)", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, choices=["uma", "3gpp", "existant"])
    a = ap.parse_args()
    if a.train == "existant":
        net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
        net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
        net.load_weights("/export/tmp/sala/freqctx/weights/phase13/precoder_relpow_mix.weights.h5")
    else:
        net = train(["uma"] if a.train == "uma" else ["umi", "uma", "rma"])
        os.makedirs(W14, exist_ok=True)
        net.save_weights(os.path.join(W14, f"zeroshot_{a.train}.weights.h5"))
    ref = json.load(open(os.path.join(RESULTS, "phase13", "bench.json")))["sets"]
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "train": a.train},
           "snrs_db": SNRS, "rates": {}, "pct_wmmse300": {}}
    for name in SETS:
        r = rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), channels(name))
        out["rates"][name] = r
        out["pct_wmmse300"][name] = [100 * v / w for v, w in zip(r, ref[name]["WMMSE 300 it."])]
        print(name, " ".join(f"{p:.1f}" for p in out["pct_wmmse300"][name]), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase14", f"zeroshot_{a.train}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
