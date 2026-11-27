"""Phase 15 -- variante de experiments/phase15_refine.py (copie minimale, l'original n'est PAS modifie).
SEUL changement : SNR d'entrainement de la passe 2 U(0, 40) dB au lieu de U(0, 30) dB. Meme architecture
(RefineNet, d=16, hidden=32, layers=4), meme passe 1 figee (GramPrecoder phase13), memes donnees
(3GPP M8K4+M64K8, phase13_train_massive.pool/SPLIT, SANS MODIFIER), meme perte (-moyenne(debit/debit RZF)),
meme budget (60 000 pas, batch 2048). Hypothese : entrainer seulement la passe 2 jusqu'a 40 dB (passe 1
figee, depart = base) gagne au fort SNR sans le compromis observe en entrainant tout le precodeur jusqu'a
40 dB (resultats/phase15/snr40.json, ECARTE).

Verification avant le vrai entrainement : au pas 0, la lecture de la passe 2 est initialisee a
zero -> DELTA_C = 0 -> C2 = C1 -> meme W, meme debit que la base seule. check_start_equals_base()
l'assert sur un lot M8K4.

    python -u -m experiments.phase15_refine40 [--steps 60000]
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
from experiments.phase14_finetune import load_base
from experiments.phase15_refine import check_start_equals_base, own_vars
from models.refine import RefineNet
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W15 = "/export/tmp/sala/freqctx/weights/phase15"
STEPS, BATCH = 60000, 2048
SNR_MAX_TRAIN = 40.0                                     # seul changement par rapport a phase15_refine.py (30.0)


def main(steps=STEPS):
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools = {k: pool(*k, seed0=1) for k in SPLIT}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, f"donnees {time.time() - t0:.0f} s", flush=True)

    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    for p in pools.values():
        net(p[:2], tf.zeros([2]))

    check_start_equals_base(net, pools[(8, 4)][:256])

    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps))
    tv = own_vars(net)
    print(f"variables entrainees (passe 2 seule) : {sum(int(tf.size(v)) for v in tv)}", flush=True)

    def make_step(p):
        n = p.shape[0]

        @tf.function
        def step():
            h = tf.gather(p, tf.random.uniform([BATCH], 0, n, dtype=tf.int32))
            snr = tf.random.uniform([BATCH], 0.0, SNR_MAX_TRAIN)
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

    os.makedirs(W15, exist_ok=True)
    wpath = os.path.join(W15, "refine_L4_snr0-40.weights.h5")
    net.save_weights(wpath)

    sionna_config.seed = 4026                                           # memes tirages que phase13_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026                                           # memes tirages que phase13_bench (umi)
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "steps": steps,
                     "snr_max_train": SNR_MAX_TRAIN},
           "snrs_db": SNRS, "history": hist, "params_passe2": sum(int(tf.size(v)) for v in tv),
           "M64K8_base+passe2": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4_base+passe2": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4),
           "M64K8_base": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4_base": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in ("M64K8_base+passe2", "M8K4_base+passe2", "M64K8_base", "M8K4_base")},
          flush=True)
    os.makedirs(os.path.join(RESULTS, "phase15"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase15", "refine40_train.json"), "w"), indent=1)
    print(f"-> poids {wpath}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    main(steps=a.steps)
