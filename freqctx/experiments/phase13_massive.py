"""Phase 13 -- massif (M=64, K=8, 600 SC), CSI PARFAIT, puissances reelles, Sionna UMi, SNR 0 a 30 dB.
Precodeurs par paires sans reentrainement : (a) celui de la base a encodeur convolutif (entree normalisee,
entraine en M8K4 seulement) ; (b) celui du banc de la phase 6 (entrainé sur M64K8 + M8K4, CSI parfait,
puissances egales, entree brute). References : WMMSE 10 / 30 / 100 / 300 it. et RZF. 64 tirages (exploration).

    python -u -m experiments.phase13_massive
"""
import json
import os

import keras
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase12_mix import load as load_mix
from experiments.phase13_bench import SNRS, rates
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from utils import RESULTS

DROPS, BATCH = 64, 8


def main():
    sionna_config.seed = 4026
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)        # [64, 600, 8, 64]
    p_conv = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv").precoder
    p6 = GramPrecoder(64, d=16, hidden=64, layers=8, mix="triangular_g", tied=False)
    p6(tf.zeros([1, 8, 64], tf.complex64), tf.zeros([1]))
    p6.load_weights("/export/tmp/sala/freqctx/weights/phase6/trig_L8_d16_h64_untied.weights.h5")
    r = {"notre precodeur (entraine M8K4, entree normalisee)":
         rates(lambda x, nv, s: p_conv(x, tf.fill([x.shape[0]], float(s))), h),
         "precodeur phase 6 (entraine M64K8 + M8K4, puissances egales)":
         rates(lambda x, nv, s: p6(x, tf.fill([x.shape[0]], float(s))), h),
         "RZF": rates(lambda x, nv, s: rzf(x, nv), h)}
    for it in (10, 30, 100, 300):
        r[f"WMMSE {it} it."] = rates(lambda x, nv, s: wmmse(x, nv, num_iter=it), h)
    print({k: [round(v, 2) for v in x] for k, x in r.items()}, flush=True)
    json.dump({"snrs_db": SNRS, "rates": r}, open(os.path.join(RESULTS, "phase13", "massive_csi_parfait.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
