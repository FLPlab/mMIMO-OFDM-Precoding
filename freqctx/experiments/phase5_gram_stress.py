"""Phase 5c -- le precodeur a attention de Gram (4 etapes deroulees) est-il
toujours au niveau de RZF/WMMSE, ou seulement en moyenne ? Evaluation SEULE
des poids entraines (phase5_precoder --variant gram), CSI parfait, massif.

Argument : l'inverse exact (G + aI)^-1 est un polynome de degre K-1 en G
(Cayley-Hamilton) ; 4 produits par G n'atteignent que le degre 4 (< 7 pour
K = 8). Ca suffit si G est bien conditionne (massif : canaux presque
orthogonaux), ca devrait casser quand G est mal conditionne. Deux tests :
  1. par echantillon (SC) : rapport modele / RZF en fonction du
     conditionnement de G + aI, sur les donnees d'evaluation ;
  2. stress : (a) utilisateurs correles par paires
     h_k <- rho h_{k-1} + sqrt(1 - rho^2) h_k (k impair), (b) moins
     d'antennes (M = 32, 16, 8 premieres antennes, K = 8), (c) SNR 30 dB
     (hors plage d'entrainement 0-25 dB). (b) et (c) sont hors distribution
     d'entrainement : un echec y mesure la generalisation autant que la forme.
WMMSE est calcule sur un sous-ensemble pour verifier que RZF reste une
reference proche de l'optimum dans les cas de stress.

    python -m experiments.phase5_gram_stress
"""
import os

import numpy as np
import tensorflow as tf

from experiments.phase5_massive import DATA, SCENARIOS
from experiments.phase5_precoder import VARIANTS, W5
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import save

SNRS = [0, 10, 20, 25, 30]
RHOS = [0.0, 0.5, 0.8, 0.95]
ANTS = [64, 32, 16, 8]
CHUNK, WM_STEP = 8192, 40


def load_model(M):
    net = GramPrecoder(M, layers=4, **VARIANTS["gram"])
    net(tf.zeros([1, 8, M], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W5, "massive_precodeur_parfait_gram.weights.h5"))
    return net


def per_sample(fn, h, snr):
    nv = 10.0 ** (-snr / 10.0)
    return tf.concat([rate_per_sample(h[i:i + CHUNK], fn(h[i:i + CHUNK], nv, snr), nv)
                      for i in range(0, h.shape[0], CHUNK)], 0).numpy()


def cond(h, snr):
    K = h.shape[-2]
    out = []
    for i in range(0, h.shape[0], CHUNK):                 # par paquets (memoire GPU 16 Go)
        g = tf.matmul(h[i:i + CHUNK], h[i:i + CHUNK], adjoint_b=True) \
            + K * 10.0 ** (-snr / 10.0) * tf.eye(K, dtype=h.dtype)
        s = np.real(tf.linalg.eigvalsh(g).numpy())
        out.append(s[:, -1] / s[:, 0])
    return np.concatenate(out)


def correlate(h, rho):
    if rho == 0.0:
        return h
    idx = np.arange(h.shape[1])
    prev = tf.gather(h, np.where(idx % 2 == 1, idx - 1, idx), axis=1)
    mix = np.where(idx % 2 == 1, rho, 0.0).astype(np.float32)[None, :, None]
    return tf.cast(np.sqrt(1 - mix ** 2), h.dtype) * h + tf.cast(mix, h.dtype) * prev


def compare(net, h, snr):
    """Debits moyens + distribution du rapport modele / RZF par echantillon."""
    m = per_sample(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h, snr)
    r = per_sample(lambda x, nv, s: rzf(x, nv), h, snr)
    hs = h[::WM_STEP]
    wm = per_sample(lambda x, nv, s: wmmse(x, nv), hs, snr).mean()
    ratio = m / r
    return {"model": float(m.mean()), "RZF": float(r.mean()), "WMMSE_subset": float(wm),
            "RZF_subset": float(r[::WM_STEP].mean()), "model_subset": float(m[::WM_STEP].mean()),
            "ratio_mean": float(ratio.mean()), "ratio_p1": float(np.percentile(ratio, 1)),
            "ratio_p5": float(np.percentile(ratio, 5)), "frac_below_95": float((ratio < 0.95).mean()),
            "_ratio": ratio}


def main():
    h = tf.concat([tf.reshape(tf.constant(np.load(os.path.join(DATA, f"massive_eval_{s}.npz"))["true"]),
                              [-1, 8, 64]) for s in SCENARIOS], 0)
    h = h[::3]                                            # ~1/3 des SC : assez pour les percentiles
    print("echantillons", h.shape[0], flush=True)
    net = load_model(64)
    out = {"snrs_db": SNRS, "num_samples": int(h.shape[0])}

    # 1. par echantillon, en fonction du conditionnement
    t1 = {}
    for snr in SNRS:
        c = compare(net, h, snr)
        ratio = c.pop("_ratio")
        k = cond(h, snr)
        edges = np.percentile(k, [0, 50, 90, 99, 100])
        c["by_cond"] = [{"cond_range": [float(edges[i]), float(edges[i + 1])],
                         "ratio_mean": float(ratio[(k >= edges[i]) & (k <= edges[i + 1])].mean()),
                         "ratio_p5": float(np.percentile(ratio[(k >= edges[i]) & (k <= edges[i + 1])], 5))}
                        for i in range(4)]
        t1[snr] = c
        print(f"[1] {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} | ratio moy {c['ratio_mean']:.3f} "
              f"p1 {c['ratio_p1']:.3f} <95% {100 * c['frac_below_95']:.1f}% | par cond (0-50,50-90,90-99,"
              f"99-100%) {[round(b['ratio_mean'], 3) for b in c['by_cond']]}", flush=True)
    out["per_sample"] = t1

    # 2a. utilisateurs correles
    t2 = {}
    for rho in RHOS:
        hr = correlate(h, rho)
        t2[rho] = {}
        for snr in (10, 20, 30):
            c = compare(net, hr, snr)
            c.pop("_ratio")
            c["cond_median"] = float(np.median(cond(hr, snr)))
            t2[rho][snr] = c
            print(f"[2a] rho {rho} {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} "
                  f"WMMSE(sous-ens.) {c['WMMSE_subset']:.2f} vs RZF {c['RZF_subset']:.2f} | ratio p5 "
                  f"{c['ratio_p5']:.3f} | cond med {c['cond_median']:.1f}", flush=True)
    out["correlated_users"] = t2

    # 2b. moins d'antennes (K = 8 fixe)
    t3 = {}
    for M in ANTS:
        hm = h[..., :M]                                   # memes antennes, gain par antenne inchange
        nm = load_model(M)
        t3[M] = {}
        for snr in (10, 20, 30):
            c = compare(nm, hm, snr)
            c.pop("_ratio")
            c["cond_median"] = float(np.median(cond(hm, snr)))
            t3[M][snr] = c
            print(f"[2b] M {M} {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} "
                  f"WMMSE(sous-ens.) {c['WMMSE_subset']:.2f} vs RZF {c['RZF_subset']:.2f} | ratio p5 "
                  f"{c['ratio_p5']:.3f} | cond med {c['cond_median']:.1f}", flush=True)
    out["fewer_antennas"] = t3
    save("phase5", "gram_stress", out)


if __name__ == "__main__":
    main()
