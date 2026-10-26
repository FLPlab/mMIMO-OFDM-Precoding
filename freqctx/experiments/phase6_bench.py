"""Phase 6 -- banc commun pour chercher l'architecture de precodeur la plus
generale (CSI parfait ; le precodeur seul). UN modele, entraine une fois,
evalue sur plusieurs configurations.

Configurations d'ENTRAINEMENT (les deux cibles de l'article) :
    M64K8 (massif, 600 SC) et M8K4 (612 SC), UMi + UMa + RMa ;
    la moitie des echantillons avec utilisateurs correles par paires
    (rho ~ U(0, 0.95)), SNR ~ U(0, 30) dB.
Configurations d'EVALUATION :
    vues      : M64K8, M8K4 ; et leurs versions correlees (rho = 0.8, 0.95) ;
    non vues  : M32K8, M16K8, M8K8 (sous-ensembles d'antennes du massif),
                M64K4, M32K4, M16K4 (sous-ensembles d'utilisateurs et d'antennes).
Metriques par (configuration, SNR) : debit moyen, % RZF, % WMMSE (WMMSE sur
un sous-ensemble), 5e centile du rapport modele / RZF par echantillon.

    python -m experiments.phase6_bench --make_data     # une fois (puis rsync)
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from config import SystemConfig
from experiments.phase5_gram_stress import correlate
from experiments.phase5_gram_hard import correlate_random
from experiments.phase5_massive import DATA, SCENARIOS
from precoders.classical import rzf, wmmse
from train import channel_pool, rate_per_sample

P6 = "/export/tmp/sala/freqctx/data/phase6"
SNRS = [0, 10, 20, 30]
TRAIN_DROPS = 352
CHUNK, WM_STEP = 8192, 25


def cfg(M, K, scen):
    return SystemConfig(num_bs_ant=M, num_users=K, num_rb=50 if M == 64 else 51, scenario=scen, num_slots=1)


def make_data():
    os.makedirs(P6, exist_ok=True)
    for (M, K) in [(64, 8), (8, 4)]:
        tr = tf.concat([channel_pool(cfg(M, K, s), TRAIN_DROPS, seed=1 + i, batch=16)
                        for i, s in enumerate(SCENARIOS)], 0)
        np.save(os.path.join(P6, f"train_M{M}K{K}.npy"), tr.numpy())
        print("train", M, K, tr.shape, flush=True)
    ev = tf.concat([channel_pool(cfg(8, 4, s), 64, seed=2026 + 10 * i, batch=16)
                    for i, s in enumerate(SCENARIOS)], 0)
    np.save(os.path.join(P6, "eval_M8K4.npy"), ev.numpy())
    print("eval M8K4", ev.shape, flush=True)


def train_pools(seed=7):
    """{(M, K): h [N, K, M]} avec la moitie des echantillons correles."""
    return {(M, K): correlate_random(tf.constant(np.load(os.path.join(P6, f"train_M{M}K{K}.npy"))), seed)
            for (M, K) in [(64, 8), (8, 4)]}


def eval_sets():
    """{nom: (h [N, K, M], vue a l'entrainement ?)}."""
    hm = tf.concat([tf.reshape(tf.constant(np.load(os.path.join(DATA, f"massive_eval_{s}.npz"))["true"]),
                               [-1, 8, 64]) for s in SCENARIOS], 0)[::6]          # 28 800 SC
    h4 = tf.constant(np.load(os.path.join(P6, "eval_M8K4.npy")))[::4]           # ~29 000 SC
    return {
        "M64K8": (hm, True), "M64K8_rho0.8": (correlate(hm, 0.8), True),
        "M64K8_rho0.95": (correlate(hm, 0.95), True),
        "M8K4": (h4, True), "M8K4_rho0.8": (correlate(h4, 0.8), True),
        "M32K8": (hm[..., :32], False), "M16K8": (hm[..., :16], False), "M8K8": (hm[..., :8], False),
        "M64K4": (hm[:, :4], False), "M32K4": (hm[:, :4, :32], False), "M16K4": (hm[:, :4, :16], False),
    }


def per_sample(fn, h, snr):
    nv = 10.0 ** (-snr / 10.0)
    return tf.concat([rate_per_sample(h[i:i + CHUNK], fn(h[i:i + CHUNK], nv), nv)
                      for i in range(0, h.shape[0], CHUNK)], 0).numpy()


_REF_CACHE = {}


def refs(name, h, snr):
    """RZF par echantillon et WMMSE sur un sous-ensemble (mis en cache par configuration)."""
    key = (name, snr)
    if key not in _REF_CACHE:
        r = per_sample(lambda x, nv: rzf(x, nv), h, snr)
        w = per_sample(lambda x, nv: wmmse(x, nv), h[::WM_STEP], snr)
        _REF_CACHE[key] = (r, w)
    return _REF_CACHE[key]


def evaluate_all(net, sets=None):
    sets = sets or eval_sets()
    out = {}
    for name, (h, seen) in sets.items():
        out[name] = {"seen": seen, "M": int(h.shape[2]), "K": int(h.shape[1])}
        for snr in SNRS:
            m = per_sample(lambda x, nv: net(x, tf.fill([x.shape[0]], float(snr))), h, snr)
            r, w = refs(name, h, snr)
            ratio = m / r
            out[name][snr] = {"model": float(m.mean()), "RZF": float(r.mean()),
                              "WMMSE_sub": float(w.mean()), "model_sub": float(m[::WM_STEP].mean()),
                              "pct_rzf": float(100 * m.mean() / r.mean()),
                              "pct_wmmse": float(100 * m[::WM_STEP].mean() / w.mean()),
                              "ratio_p5": float(np.percentile(ratio, 5))}
        print(f"  {name:15s} " + " | ".join(f"{s}dB {out[name][s]['pct_rzf']:5.1f}%RZF "
                                            f"{out[name][s]['pct_wmmse']:5.1f}%W" for s in SNRS), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_data", action="store_true")
    if ap.parse_args().make_data:
        make_data()
