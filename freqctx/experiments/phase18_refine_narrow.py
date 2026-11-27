"""Phase 18 -- entrainement de la PASSE 2 ETROITE (RefineNet, models/refine.py, d=8, hidden=16,
layers=4, ~96 k MAC) sur un tronc RETRECI FIGE (E1 = GramPrecoder d8/h16/L8, ~190 k MAC, ou
E2 = GramPrecoder d8/h16/L12, ~290 k MAC, entraines par experiments/phase18_narrow.py), demande
F. Leduc-Primeau 26 nov. 2026 -- combler l'ecart au fort SNR des troncs etroits.

Copie minimale de experiments/phase15_refine.py (architecture, verification de depart, donnees,
perte, SNR d'entrainement U(0, 30) dB, budget 60 000 pas / batch 2048 IDENTIQUES, import SANS
MODIFIER de check_start_equals_base/own_vars) ; SEULES differences :
  - la base figee est le tronc etroit (E1 ou E2, --trunk L8|L12) au lieu du tronc phase13 large ;
  - la passe 2 est elle-meme ETROITE : RefineNet(d=8, hidden=16, layers=4) au lieu de (16, 32, 4).

    python -u -m experiments.phase18_refine_narrow --trunk L8
    python -u -m experiments.phase18_refine_narrow --trunk L12
"""
import argparse
import os
import time

import json
import keras
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_train_massive import SPLIT, pool
from experiments.phase15_refine import check_start_equals_base, own_vars
from models.gram_precoder import GramPrecoder
from models.refine import RefineNet
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W18 = "/export/tmp/sala/freqctx/weights/phase18"
STEPS, BATCH = 60000, 2048
TRUNKS = {"L8": dict(d=8, hidden=16, layers=8, tag="narrow_d8_h16_L8"),
          "L12": dict(d=8, hidden=16, layers=12, tag="narrow_d8_h16_L12")}


def load_trunk(trunk):
    kw = TRUNKS[trunk]
    net = GramPrecoder(64, d=kw["d"], hidden=kw["hidden"], layers=kw["layers"],
                        mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W18, f"{kw['tag']}.weights.h5"))
    return net


def main(trunk, steps=STEPS):
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, f"donnees {time.time() - t0:.0f} s", flush=True)

    base = load_trunk(trunk)
    base.trainable = False
    net = RefineNet(base, d=8, hidden=16, layers=4)
    for p in pools.values():
        net(p[:2], tf.zeros([2]))

    check_start_equals_base(net, pools[(8, 4)][:256])

    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps))
    tv = own_vars(net)
    print(f"variables entrainees (passe 2 etroite seule) : {sum(int(tf.size(v)) for v in tv)}", flush=True)

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
            g = tape.gradient(loss, tv)
            g, _ = tf.clip_by_global_norm(g, 1.0)
            opt.apply_gradients(zip(g, tv))
            return loss
        return step

    keys = list(pools)
    stepfns = {k: make_step(pools[k]) for k in keys}
    hist = []
    for s in range(1, steps + 1):
        loss = stepfns[keys[s % 2]]()
        if s % 5000 < 2 or s == steps:
            hist.append({"step": s, "obj": -float(loss), "config": str(keys[s % 2]), "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)

    os.makedirs(W18, exist_ok=True)
    wpath = os.path.join(W18, f"refine_narrow_on_{trunk}.weights.h5")
    net.save_weights(wpath)

    sionna_config.seed = 4026                                           # memes tirages que phase13_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026                                           # memes tirages que phase13_bench (umi)
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "steps": steps,
                     "trunk": trunk, "trunk_tag": TRUNKS[trunk]["tag"]},
           "snrs_db": SNRS, "history": hist, "params_passe2": sum(int(tf.size(v)) for v in tv),
           f"M64K8_{trunk}+passe2_etroite": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           f"M8K4_{trunk}+passe2_etroite": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4),
           f"M64K8_{trunk}": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), hm),
           f"M8K4_{trunk}": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in
           (f"M64K8_{trunk}+passe2_etroite", f"M8K4_{trunk}+passe2_etroite", f"M64K8_{trunk}", f"M8K4_{trunk}")},
          flush=True)
    os.makedirs(os.path.join(RESULTS, "phase18"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase18", f"refine_narrow_on_{trunk}_train.json"), "w"), indent=1)
    print(f"-> poids {wpath}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", type=str, required=True, choices=list(TRUNKS))
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    main(a.trunk, steps=a.steps)
