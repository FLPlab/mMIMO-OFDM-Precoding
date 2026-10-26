"""Phase 5d -- le decrochage du precodeur de Gram sur G mal conditionne
(phase5_gram_stress) vient-il de la FORME (4 produits par G) ou de la
GENERALISATION (jamais vu de cas difficiles) ? Reentrainement sur un melange
avec cas difficiles, 4 contre 8 etapes, tout le reste identique.
--mix triangular : bloc G.E remplace par une attention triangulaire apprise
entre paires (models/gram_precoder.py, TriangularBlock).

Melange d'entrainement (CSI parfait, massif M=64 K=8, UMi+UMa+RMa) : la
moitie des echantillons inchanges, l'autre moitie avec utilisateurs correles
par paires (rho ~ U(0, 0.95) par echantillon, meme operation que le stress).
SNR d'entrainement 0-30 dB (au lieu de 0-25) pour couvrir le haut SNR teste.
Evaluation : standard, rho = 0.5/0.8/0.95, et M = 32/16/8 antennes (jamais
vu a l'entrainement : test de generalisation).

    python -m experiments.phase5_gram_hard --layers 4|8 [--mix triangular]
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from config import CSIConfig
from experiments.phase5_gram_stress import compare, cond, correlate
from experiments.phase5_massive import DATA, PILOT, SCENARIOS, cfg_of
from experiments.phase5_precoder import VARIANTS, W5
from models.gram_precoder import GramPrecoder
from train import csi_pool, run_variant
from utils import save

SNRS = [0, 10, 20, 25, 30]


def correlate_random(h, seed):
    """Moitie inchangee, moitie correlee par paires avec rho ~ U(0, 0.95)."""
    rng = np.random.default_rng(seed)
    n, K = h.shape[0], h.shape[1]
    rho = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 0.95, n)).astype(np.float32)
    idx = np.arange(K)
    prev = tf.gather(h, np.where(idx % 2 == 1, idx - 1, idx), axis=1)
    mix = rho[:, None, None] * (idx % 2 == 1).astype(np.float32)[None, :, None]
    return tf.cast(np.sqrt(1 - mix ** 2), h.dtype) * h + tf.cast(mix, h.dtype) * prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--mix", choices=["gram", "triangular"], default="gram")
    ap.add_argument("--train_drops", type=int, default=352)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--load", action="store_true", help="evaluation seule des poids sauves")
    args = ap.parse_args()
    if not args.load:
        pool = tf.concat([csi_pool(cfg_of(s), CSIConfig("perfect", PILOT, 0), args.train_drops, seed=1 + i,
                                   batch=16)[1] for i, s in enumerate(SCENARIOS)], 0)
        pool = correlate_random(pool, seed=7)
    h = tf.concat([tf.reshape(tf.constant(np.load(os.path.join(DATA, f"massive_eval_{s}.npz"))["true"]),
                              [-1, 8, 64]) for s in SCENARIOS], 0)[::3]
    tag = "" if args.mix == "gram" else "_tri"
    name = f"massive_gram_dur{tag}_L{args.layers}"
    net = GramPrecoder(64, layers=args.layers, mix=args.mix, **VARIANTS["gram"])
    from precoders.classical import rzf
    from train import evaluate
    ref = {"RZF": evaluate(lambda x, s: rzf(x, 10.0 ** (-s[0] / 10.0)), h[::10], SNRS)}
    w_path = os.path.join(W5, f"{name}.weights.h5")
    if args.load:                                         # historique d'entrainement non conserve
        net(h[:2], tf.zeros([2]))
        net.load_weights(w_path)
        run = {"params": int(net.count_params()), "macs_per_sc": net.macs_per_sc(8), "history": None}
    else:
        run = run_variant(name, net, pool, h[::10], SNRS, ref, args.steps, seed=0, loss="rel_rzf",
                          weights_dir=W5, snr_range=(0.0, 30.0))
    out = {"layers": args.layers, "mix": args.mix, "params": run["params"], "macs_per_sc": run["macs_per_sc"],
           "training_history": run["history"], "snrs_db": SNRS}

    std = {}
    for snr in SNRS:
        c = compare(net, h, snr)
        ratio, k = c.pop("_ratio"), cond(h, snr)
        top = k >= np.percentile(k, 99)
        c["ratio_worst_cond_1pct"] = float(ratio[top].mean())
        std[snr] = c
        print(f"[std] {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} | ratio moy {c['ratio_mean']:.3f} "
              f"p1 {c['ratio_p1']:.3f} | 1% pire cond {c['ratio_worst_cond_1pct']:.3f}", flush=True)
    out["standard"] = std

    cor = {}
    for rho in (0.5, 0.8, 0.95):
        hr = correlate(h, rho)
        cor[rho] = {}
        for snr in (10, 20, 30):
            c = compare(net, hr, snr)
            c.pop("_ratio")
            cor[rho][snr] = c
            print(f"[rho {rho}] {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} "
                  f"WMMSE(s.-e.) {c['WMMSE_subset']:.2f} | ratio moy {c['ratio_mean']:.3f} "
                  f"p5 {c['ratio_p5']:.3f}", flush=True)
    out["correlated_users"] = cor

    ant = {}
    for M in (32, 16, 8):
        nm = GramPrecoder(M, layers=args.layers, mix=args.mix, **VARIANTS["gram"])
        nm(tf.zeros([1, 8, M], tf.complex64), tf.zeros([1]))
        nm.load_weights(w_path)
        ant[M] = {}
        for snr in (10, 20, 30):
            c = compare(nm, h[..., :M], snr)
            c.pop("_ratio")
            ant[M][snr] = c
            print(f"[M {M}] {snr} dB  modele {c['model']:.2f} RZF {c['RZF']:.2f} "
                  f"WMMSE(s.-e.) {c['WMMSE_subset']:.2f} | ratio moy {c['ratio_mean']:.3f} "
                  f"p5 {c['ratio_p5']:.3f}", flush=True)
    out["fewer_antennas"] = ant
    save("phase5", f"gram_dur{tag}_L{args.layers}", out)


if __name__ == "__main__":
    main()
