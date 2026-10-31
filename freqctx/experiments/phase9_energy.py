"""Phase 9 -- energie estimee par sous-porteuse et par mise a jour du precodeur,
avec le modele d'energie du laboratoire (le meme que le memoire :
mimo_precoding/precoders/rb_grouping.py, energy_constants / compute_energy_uJ,
recopie ici sans modification) : energie des MAC, des acces aux poids et aux
activations, fonction de la precision Q (bits).

Conventions (comptes ouverts, a affiner sur FPGA) :
  - MAC reelles ; nos IDFT/DFT comptees en vraies FFT ;
  - classique : LMMSE = une matrice F x N par (utilisateur, antenne) appliquee
    aux pilotes (4 N K M MAC/SC, matrice stockee = "poids") ; RZF = G, inversion
    K x K, W = h^H C ; WMMSE (notre implementation, 30 iterations) =
    4 (K^2 M + 2 M^2 K + M^3 / 3) MAC par iteration (systeme M x M) ;
  - activations : sorties des couches (ordre de grandeur) ;
  - precision : notre modele en INT8 (Q=8, a valider par quantification),
    classique en FP32 (Q=32, favorable au classique : notre WMMSE utilise en
    realite des inversions en double precision).

    python -m experiments.phase9_energy
"""
import json
import math
import os

import tensorflow as tf

from models.delay_domain import DelayDomainPrecoder
from utils import RESULTS


def energy_constants(Q):                                  # modele du laboratoire (memoire)
    Q = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    return EMAC, 2.0 * EMAC, EMAC


def energy_uJ(flops, weights, acts, Q_W=32, Q_A=32):      # modele du laboratoire (memoire)
    macs = flops / 2.0
    EMAC, EM, EL = energy_constants(Q_W)
    _, EM_A, EL_A = energy_constants(Q_A)
    EC = EMAC * (macs + 3.0 * acts)
    EW = EM * weights + EL * (macs / math.sqrt(64.0 * (Q_W / 16.0)))
    EA = 2.0 * EM_A * acts + EL_A * (macs / math.sqrt(64.0 * (Q_A / 16.0)))
    return (EC + EW + EA) / 1e9


CONFIGS = {"M8K4": dict(F=612, K=4, M=8), "M64K8": dict(F=600, K=8, M=64)}


def classical(F, K, M, precoder):
    N = F // K
    lmmse = 4 * N * K * M
    if precoder == "RZF":
        prec = 4 * K * K * M + 4 * K ** 3 + 4 * M * K * K
    else:
        prec = 30 * 4 * (K * K * M + 2 * M * M * K + M ** 3 / 3)
    acts = 2 * K * M + 2 * (K * M + 2 * K * K + M * K) * (1 if precoder == "RZF" else 30)
    return {"mac": lmmse + prec, "weights": 2 * F * N, "acts": acts}


def ours(F, K, M, d=16, hidden=64, layers=8, sym=False, step=None):
    net = DelayDomainPrecoder(F, K, M, banded=True, oversample=2, precoder_arch="pair", d=d, hidden=hidden,
                              layers=layers, mix="triangular_g", tied=False, sym=sym, anchor_step=step)
    net(tf.zeros([1, F, K, M], tf.complex64), tf.zeros([1]))
    mac = net.macs_per_sc(K, fft=True)
    T, de = net.estimator.T, net.est_d
    enc_acts = K * T * (de + 2 * (8 * de + 2 * T) + 6) + 2 * F * K * M          # par bande
    pairs = K * (K + 1) // 2 if sym else K * K
    prec_acts = K * K * (5 + 2 * d) + layers * (K * K * (2 * d + 3 + 6 * d + 2 * d) + pairs * hidden) \
        + 2 * K * K + 2 * M * K                                                   # par SC calculee
    frac = net.anchors.shape[0] / F if step else 1.0
    acts = enc_acts / F + prec_acts * frac + (2 * M * K if step else 0)
    return {"mac": mac, "weights": int(net.count_params()), "acts": acts}


def main():
    out = {}
    for cname, c in CONFIGS.items():
        F, K, M = c["F"], c["K"], c["M"]
        rows = {"LMMSE + RZF (FP32)": (classical(F, K, M, "RZF"), 32),
                "LMMSE + WMMSE (FP32)": (classical(F, K, M, "WMMSE"), 32),
                "notre modele, par SC (INT8)": (ours(F, K, M), 8),
                "notre modele, symetrique (INT8)": (ours(F, K, M, sym=True), 8),
                "notre modele, 1 SC sur 3 (INT8)": (ours(F, K, M, step=3), 8),
                "notre modele, symetrique + 1 SC sur 3 (INT8)": (ours(F, K, M, sym=True, step=3), 8),
                "notre modele, par SC (FP32)": (ours(F, K, M), 32)}
        out[cname] = {}
        for name, (r, q) in rows.items():
            # par bande : poids charges une fois, MAC et activations sur les F SC
            e = energy_uJ(2 * r["mac"] * F, r["weights"], r["acts"] * F, q, q) / F * 1e3   # nJ par SC
            out[cname][name] = {"mac_per_sc": int(r["mac"]), "weights": int(r["weights"]),
                                "acts_per_sc": int(r["acts"]), "bits": q, "energy_nJ_per_sc": e}
            print(f"{cname:6s} {name:46s} MAC/SC {r['mac'] / 1e3:9.0f} k | poids {r['weights'] / 1e3:7.0f} k "
                  f"| {q:2d} bits | {e:8.2f} nJ/SC", flush=True)
    os.makedirs(os.path.join(RESULTS, "phase9"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase9", "energy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
