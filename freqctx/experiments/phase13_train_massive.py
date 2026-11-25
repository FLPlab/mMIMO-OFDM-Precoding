"""Phase 13 -- precodeur par paires (CSI parfait) entraine a PUISSANCES REELLES sur le melange M64K8 + M8K4
(UMi + UMa + RMa, Sionna), entree normalisee par utilisateur. Seul changement par rapport au banc de la phase 6
(qui etait a puissances egales) : les donnees. Meme architecture que la base (attention triangulaire, G
reinjecte, 8 couches distinctes, etat 16, FFN 32). Perte : -moyenne(debit / debit RZF), SNR U(0, 30) dB.
Evaluation : tirages de phase13_massive (M64K8) et de phase13_bench (M8K4 UMi), WMMSE de reference.

    python -u -m experiments.phase13_train_massive
"""
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
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W13 = "/export/tmp/sala/freqctx/weights/phase13"
STEPS, BATCH = 60000, 2048
SPLIT = {(64, 8): (dict(num_bs_ant=64, num_users=8, num_rb=50), 128), (8, 4): (dict(), 512)}   # tirages / scenario


def pool(M, K, seed0):
    kw, drops = SPLIT[(M, K)]
    hs = []
    for i, scen in enumerate(["umi", "uma", "rma"]):
        sionna_config.seed = seed0 + i
        gen = ChannelGenerator(SystemConfig(scenario=scen, num_slots=1, relative_power=True, **kw))
        for _ in range(drops // 8):
            h = gen(8)[:, 0]                                            # [8, F, K, M]
            hs.append(tf.reshape(h[:, ::3], [-1, K, M]))                # 1 SC sur 3 (SC voisines tres correlees)
    return tf.concat(hs, 0)


def main():
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, f"donnees {time.time() - t0:.0f} s", flush=True)
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    for p in pools.values():
        net(p[:2], tf.zeros([2]))
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, STEPS))

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
    steps = {k: make_step(pools[k]) for k in keys}
    hist = []
    for s in range(1, STEPS + 1):
        loss = steps[keys[s % 2]]()
        if s % 5000 < 2:
            hist.append({"step": s, "obj": -float(loss), "config": str(keys[s % 2]), "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    os.makedirs(W13, exist_ok=True)
    net.save_weights(os.path.join(W13, "precoder_relpow_mix.weights.h5"))
    sionna_config.seed = 4026                                           # memes tirages que phase13_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026                                           # memes tirages que phase13_bench (umi)
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS,
           "history": hist, "params": int(net.count_params()),
           "M64K8": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in ("M64K8", "M8K4")}, flush=True)
    json.dump(out, open(os.path.join(RESULTS, "phase13", "train_massive.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
