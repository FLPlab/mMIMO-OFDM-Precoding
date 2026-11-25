"""Phase 14 -- CSI imparfait (demande de F. Leduc-Primeau, 2026-11-25) : entrainer le precodeur avec un modele
d'erreur SYNTHETIQUE simple, le deployer dans Sionna avec un VRAI estimateur.

Entrainement : precodeur par paires (M=8, K=4, puissances reelles, entree normalisee), initialise par le precodeur
CSI parfait (phase13_train_massive), puis 20 000 pas avec h_est = h + e, e ~ CN(0, s2 * puissance de
l'utilisateur), s2 tire en dB uniformement dans [-25, -5] (erreur relative, comme apres controle de puissance
montant) ; debit calcule sur le vrai canal h. 3 modeles 3GPP melanges.
Deploiement : Sionna UMi, 192 tirages (phase 11), pilotes 0 / 10 / 20 dB, estimation LS + interpolation LMMSE
(covariance du melange) ou LS + interpolation lineaire (Sionna). Precodeurs sur la meme estimation : precodeur
entraine en CSI parfait, precodeur entraine avec l'erreur synthetique, WMMSE 100 it. (robuste pour LMMSE), RZF.
Mesures : debit somme et BER non code QPSK / 16-QAM (Gray), calcule a partir du SINR de chaque flux
(approximation gaussienne de l'interference).

    python -u -m experiments.phase14_csi
"""
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from scipy.special import erfc
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase14_zeroshot import pool
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

W14 = "/export/tmp/sala/freqctx/weights/phase14"
SNRS, PILOTS, STEPS, BATCH, DROPS, B = [0, 5, 10, 15, 20, 25, 30], [0, 10, 20], 20000, 2048, 192, 16


def load_perfect():
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights("/export/tmp/sala/freqctx/weights/phase13/precoder_relpow_mix.weights.h5")
    return net


def train_synthetic():
    keras.utils.set_random_seed(0)
    p = pool(["umi", "uma", "rma"])
    net = load_perfect()
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(3e-4, STEPS))

    @tf.function
    def step():
        h = tf.gather(p, tf.random.uniform([BATCH], 0, p.shape[0], dtype=tf.int32))
        pw = tf.reduce_mean(tf.abs(h) ** 2, axis=-1, keepdims=True)                     # [n, K, 1]
        s2 = 10.0 ** (tf.random.uniform([BATCH, 1, 1], -25.0, -5.0) / 10.0)
        sd = tf.cast(tf.sqrt(s2 * pw / 2), tf.complex64)
        e = tf.complex(tf.random.normal(tf.shape(h, out_type=tf.int32)), tf.random.normal(tf.shape(h, out_type=tf.int32))) * sd
        snr = tf.random.uniform([BATCH], 0.0, 30.0)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, net(h + e, snr), nv) / tf.stop_gradient(ref))
        g, _ = tf.clip_by_global_norm(tape.gradient(loss, net.trainable_variables), 1.0)
        opt.apply_gradients(zip(g, net.trainable_variables))
        return loss

    t0 = time.time()
    for s in range(1, STEPS + 1):
        loss = step()
        if s % 5000 == 0:
            print(f"   pas {s} objectif {-float(loss):.3f} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(W14, exist_ok=True)
    net.save_weights(os.path.join(W14, "precoder_synth_csi.weights.h5"))
    return net


def metrics(h, w, nv):
    """Debit moyen et BER moyen (QPSK, 16-QAM) ; h [n, K, M], w [n, M, K]."""
    hw = tf.matmul(h, w)
    sig = tf.abs(tf.linalg.diag_part(hw)) ** 2
    sinr = (sig / (tf.reduce_sum(tf.abs(hw) ** 2, -1) - sig + nv)).numpy()
    q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    return float(np.log2(1 + sinr).sum(-1).mean()), float(q(np.sqrt(sinr)).mean()), float((0.75 * q(np.sqrt(sinr / 5))).mean())


def evaluate(fns, h_est, h, err=None, chunk=8192):
    hf, he = tf.reshape(h, [-1, 4, 8]), tf.reshape(h_est, [-1, 4, 8])
    ef = None if err is None else tf.reshape(err, [-1, 4, 8])
    out = {}
    for name, fn in fns.items():
        res = {"rate": [], "ber_qpsk": [], "ber_16qam": []}
        for snr in SNRS:
            nv = 10.0 ** (-snr / 10.0)
            m = [metrics(hf[i:i + chunk], fn(he[i:i + chunk], nv, snr, None if ef is None else ef[i:i + chunk]), nv)
                 for i in range(0, hf.shape[0], chunk)]
            for k, j in (("rate", 0), ("ber_qpsk", 1), ("ber_16qam", 2)):
                res[k].append(float(np.mean([x[j] for x in m])))
        out[name] = res
    return out


def main():
    net_s = train_synthetic()
    net_p = load_perfect()
    cfg = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen = ChannelGenerator(cfg)
    h = tf.concat([gen(B)[:, 0] for _ in range(DROPS // B)], 0)
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS,
           "csi_parfait": {}, "estime": {}}
    perfect = {"notre precodeur (CSI parfait)": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
               "WMMSE 100 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=100), "RZF": lambda x, nv, s, e: rzf(x, nv)}
    out["csi_parfait"] = evaluate(perfect, h, h)
    for est in ("lmmse", "lin"):
        snd = UplinkSounder(cfg, est, cov if est == "lmmse" else None)
        for p in PILOTS:
            pr = [sound_power_controlled(snd, h[j:j + B], float(p), return_err_var=True) for j in range(0, DROPS, B)]
            he, ev = tf.concat([a for a, _ in pr], 0), tf.concat([b for _, b in pr], 0)
            fns = {"precodeur entraine CSI parfait": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                   "precodeur entraine erreur synthetique": lambda x, nv, s, e: net_s(x, tf.fill([x.shape[0]], float(s))),
                   "WMMSE 100 it." + (" robuste" if est == "lmmse" else ""):
                       (lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e)) if est == "lmmse"
                       else (lambda x, nv, s, e: wmmse(x, nv, num_iter=100)),
                   "RZF": lambda x, nv, s, e: rzf(x, nv)}
            r = evaluate(fns, he, h, ev if est == "lmmse" else None)
            out["estime"][f"{est}_pilote{p}"] = r
            print(f"{est} pilote {p} dB @20dB : " + " | ".join(f"{k} {v['rate'][4]:.2f} (BER16 {v['ber_16qam'][4]:.3f})"
                                                           for k, v in r.items()), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase14", "csi_imparfait.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
