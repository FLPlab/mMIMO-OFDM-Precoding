"""Phase 16 -- deux verifications du precodeur STRUCTURE T = 20 (models/structured_precoder.py) :

  --part seed  : 2e graine (weights/phase16/structured_T20_s1.weights.h5, entrainee par
                 experiments.phase16_structured --ns 20 --seed 1) evaluee EXACTEMENT comme
                 experiments.phase16_eval (memes fonctions run_parfait / run_estime importees), a cote de
                 la graine 0 (structured_T20.weights.h5).
  --part fixed : VIRGULE FIXE emulee (CPU) de la partie Newton-Schulz + C + W, le petit transformer
                 restant flottant (puis, option, transformer en entier 8 bits). Conventions de
                 torch_port/fixedpoint_classical.py : arrondi au plus proche + saturation sur b bits
                 (signe compris, Re/Im separees), exposant de bloc choisi apres calcul (accumulateurs
                 exacts, arrondi a l'ecriture), exposant par LIGNE pour H (par utilisateur), par COLONNE
                 pour les matrices K x K et W, par element pour les scalaires par utilisateur (lambda_k,
                 sqrt(p_k)) ; reciproque scalaire (X0 = A^H / (||A||_1 ||A||_inf)) par table 256 entrees
                 + Newton (1 tour si b <= 16, 2 sinon) ; sortie F = W rsqrt(||W||^2) a exposant commun.
                 Etapes : in (H), gram (G = H H^H), scal (lambda, sqrt p, sigma^2, normes, reciproque),
                 inv (A, X0, iterees NS : A X, 2I - A X, X(2I - A X)), out (C, W, F).
                 FIDELE au modele entraine : NS sur A physique (pas d'equilibrage de Jacobi, qui changerait
                 la fonction apprise a T fini). Le transformer voit G QUANTIFIE quand l'etape gram l'est.
                 Divergence : residu ||I - A X_T||_F (A, X_T quantifiees) > 1, ou precodeur non fini
                 (compte comme debit nul).

    CUDA_VISIBLE_DEVICES="" python -u -m experiments.phase16_check --part fixed
    python -u -m experiments.phase16_check --part seed
Resultats fusionnes dans results/phase16/structured_check.json.
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from experiments.phase15_diag import get_datasets, subsample
from models.structured_precoder import StructuredPrecoder
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase16")
OUT_PATH = os.path.join(OUT_DIR, "structured_check.json")
W16 = "/export/tmp/sala/freqctx/weights/phase16"
N_EVAL = 16384
STAGES = ("in", "gram", "scal", "inv", "out")
BITS = list(range(8, 21))
SNRS_FX = [20, 30, 40]
SITES_FX = ["umi", "ericsson"]
TBL = 8
C128, F64 = tf.complex128, tf.float64


def load(t_steps=20, suffix=""):
    net = StructuredPrecoder(8, ns_steps=t_steps, d=16, hidden=32, layers=2)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W16, f"structured_T{t_steps}{suffix}.weights.h5"))
    return net


def merge_save(key, val):
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}
    out[key] = val
    out.setdefault("meta", {})[key] = {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(OUT_PATH, "w"), indent=1)


# ---------------------------------------------------------------------- 1. 2e graine


def part_seed():
    from experiments.phase16_eval import run_estime, run_parfait
    nets = {"T20_s0": load(20, ""), "T20_s1": load(20, "_s1")}
    fns = {k: (lambda h, snr, n=n: n(h, snr)) for k, n in nets.items()}
    res = {"csi_parfait": run_parfait(fns), "csi_estime_pilote20": run_estime(fns)}
    merge_save("seed", res)


# ---------------------------------------------------------------------- 2. virgule fixe


def blockq(x, b, axis):
    """Arrondi + saturation b bits, exposant partage sur axis (None = par element)."""
    if x.dtype.is_complex:
        a = tf.maximum(tf.abs(tf.math.real(x)), tf.abs(tf.math.imag(x)))
    else:
        a = tf.abs(x)
    if axis is not None:
        a = tf.reduce_max(a, axis=axis, keepdims=True)
    s = tf.pow(tf.constant(2.0, F64), tf.math.ceil(tf.math.log(tf.maximum(a, 1e-300)) / np.log(2.0)) - (b - 1))
    lo, hi = -2.0 ** (b - 1), 2.0 ** (b - 1) - 1

    def f(v):
        return tf.clip_by_value(tf.round(v / s), lo, hi) * s
    return tf.complex(f(tf.math.real(x)), f(tf.math.imag(x))) if x.dtype.is_complex else f(x)


def qf(x, b, frac):
    s = 2.0 ** frac
    return tf.clip_by_value(tf.round(x * s), -2.0 ** (b - 1), 2.0 ** (b - 1) - 1) / s


class FX:
    def __init__(self, b=None, stages=STAGES):
        self.b, self.stages = b, set(stages)
        self.newton = 1 if (b or 0) <= 16 else 2

    def on(self, st):
        return self.b is not None and st in self.stages

    def q(self, x, st, axis=None):
        return blockq(x, self.b, axis) if self.on(st) else x

    def recip(self, x, st):
        if not self.on(st):
            return 1.0 / x
        b = self.b
        x = tf.maximum(x, 1e-300)
        e = tf.floor(tf.math.log(x) / np.log(2.0))
        m = x / tf.pow(tf.constant(2.0, F64), e)
        idx = tf.clip_by_value(tf.floor((m - 1) * 2 ** TBL), 0, 2 ** TBL - 1)
        y = qf(1.0 / (1 + (idx + 0.5) / 2 ** TBL), b, b - 2)
        for _ in range(self.newton):
            y = qf(y * qf(2 - qf(m * y, b, b - 2), b, b - 2), b, b - 2)
        return y / tf.pow(tf.constant(2.0, F64), e)

    def rsqrt(self, x, st):
        if not self.on(st):
            return tf.math.rsqrt(x)
        b = self.b
        x = tf.maximum(x, 1e-300)
        e = tf.floor(tf.math.log(x) / np.log(2.0) / 2)
        m = x / tf.pow(tf.constant(2.0, F64), 2 * e)
        idx = tf.clip_by_value(tf.floor((m - 1) / 3 * 2 ** TBL), 0, 2 ** TBL - 1)
        y = qf((1 + 3 * (idx + 0.5) / 2 ** TBL) ** -0.5, b, b - 3)
        for _ in range(self.newton):
            y = qf(y * qf(1.5 - qf(0.5 * m * qf(y * y, b, b - 3), b, b - 3), b, b - 3), b, b - 3)
        return y / tf.pow(tf.constant(2.0, F64), e)


def structured_fx(net, h, snr, fx):
    """h [n,K,M] complex64 -> (F [n,M,K] complex64, residu NS [n]). Meme algebre que
    StructuredPrecoder.coeffs/call, en complex128 + quantification."""
    n, K = h.shape[0], h.shape[1]
    hq = fx.q(tf.cast(h, C128), "in", 2)
    G = fx.q(tf.matmul(hq, hq, adjoint_b=True), "gram", 1)
    lam, p = net.lambda_p(tf.cast(G, tf.complex64), tf.fill([n], float(snr)))       # transformer flottant (ou int8)
    lam = fx.q(tf.cast(lam, F64), "scal")
    sp = fx.q(tf.sqrt(tf.cast(p, F64)), "scal")
    nv = fx.q(tf.constant(10.0 ** (-snr / 10.0), F64), "scal")
    eye = tf.eye(K, dtype=C128)[None]
    A = fx.q(tf.cast(lam, C128)[:, :, None] * G + tf.cast(nv, C128) * eye, "inv", 1)
    absA = tf.abs(A)
    n1 = fx.q(tf.reduce_max(tf.reduce_sum(absA, -2), -1), "scal")
    ninf = fx.q(tf.reduce_max(tf.reduce_sum(absA, -1), -1), "scal")
    scale = fx.q(fx.recip(fx.q(tf.maximum(n1 * ninf, 1e-12), "scal"), "scal"), "scal")
    X = fx.q(tf.linalg.adjoint(A) * tf.cast(scale, C128)[:, None, None], "inv", 1)
    for _ in range(net.T):
        Y = fx.q(tf.matmul(A, X), "inv", 1)
        Z = fx.q(2 * eye - Y, "inv", 1)
        X = fx.q(tf.matmul(X, Z), "inv", 1)
    res = tf.sqrt(tf.reduce_sum(tf.abs(eye - tf.matmul(A, X)) ** 2, [1, 2]))
    C = fx.q(X * tf.cast(sp, C128)[:, None, :], "out", 1)
    W = fx.q(tf.matmul(hq, C, adjoint_a=True), "out", 1)
    pw = fx.q(tf.reduce_sum(tf.abs(W) ** 2, [1, 2]), "scal")
    F = fx.q(W * tf.cast(fx.rsqrt(pw, "scal"), C128)[:, None, None], "out", (1, 2))
    return tf.cast(F, tf.complex64), res


def eval_fx(net, h, snr, fx, bs=4096):
    rates, ress = [], []
    for i in range(0, h.shape[0], bs):
        hb = h[i:i + bs]
        F, res = structured_fx(net, hb, snr, fx)
        r = rate_per_sample(hb, F, 10.0 ** (-snr / 10.0))
        rates.append(r.numpy())
        ress.append(res.numpy())
    r, res = np.concatenate(rates), np.concatenate(ress)
    bad = ~np.isfinite(r)
    return {"debit": float(np.where(bad, 0.0, r).mean()), "non_fini": int(bad.sum()),
            "residu_ns_median": float(np.nanmedian(res)), "frac_residu_gt1": float(np.mean(~(res <= 1.0)))}


# ----- transformer en entier 8 bits (quantification post-entrainement, fausse quantification)


def int8_transformer(net, calib):
    """Poids int8 symetriques par canal de sortie ; entree de chaque Dense en int8 symetrique par tenseur,
    echelle statique = max |x| sur un lot de calibration (calib : liste (G complex64, snr)). LayerNorm,
    GELU, einsum triangulaire et softplus restent flottants (accumulateurs). Modifie net EN PLACE."""
    dens = [net.init, net.read] + [l for blk in net.blocks for l in (blk.a, blk.b, blk.f1, blk.f2)]
    amax = {id(l): 0.0 for l in dens}
    orig = {id(l): l.call for l in dens}

    def rec(l):
        def call(x, *a, **k):
            amax[id(l)] = max(amax[id(l)], float(tf.reduce_max(tf.abs(x))))
            return orig[id(l)](x, *a, **k)
        return call
    for l in dens:
        l.call = rec(l)
    for G, snr in calib:
        net.lambda_p(G, tf.fill([G.shape[0]], float(snr)))
    for l in dens:
        w = l.kernel.numpy()
        s = np.maximum(np.abs(w).max(0, keepdims=True), 1e-12) / 127.0
        l.kernel.assign(np.clip(np.round(w / s), -127, 127) * s)
        sa = max(amax[id(l)], 1e-12) / 127.0

        def call(x, *a, l=l, sa=sa, **k):
            return orig[id(l)](tf.clip_by_value(tf.round(x / sa), -127.0, 127.0) * sa, *a, **k)
        l.call = call
    return {l.name: amax[id(l)] for l in dens}


def part_fixed(int8):
    tf.config.threading.set_inter_op_parallelism_threads(8)
    raw = get_datasets()
    data = {s: subsample(raw[s], N_EVAL, seed=4242) for s in SITES_FX}             # memes echantillons que phase16_eval
    net = load(20, "")
    t0 = time.time()
    out = {"bits": BITS, "snrs_db": SNRS_FX, "n": N_EVAL, "table_bits": TBL, "doc": __doc__}
    configs = [("tout", STAGES), ("ns_seul", ("inv",)), ("ns+C+W", ("inv", "out", "scal")),
               ("in+gram", ("in", "gram"))]
    res = {}
    for site, h in data.items():
        S = res[site] = {}
        for snr in SNRS_FX:
            E = S[str(snr)] = {}
            E["flottant_modele"] = float(tf.reduce_mean(rate_per_sample(h, net(h, tf.fill([h.shape[0]], float(snr))),
                                                                        10.0 ** (-snr / 10.0))))
            E["flottant_emule"] = eval_fx(net, h, snr, FX(None))
            for name, st in configs:
                E[name] = {b: eval_fx(net, h, snr, FX(b, st)) for b in BITS}
                print(site, snr, name, {b: round(v["debit"], 3) for b, v in E[name].items()},
                      f"ref {E['flottant_emule']['debit']:.3f}  {time.time() - t0:.0f}s", flush=True)
                print("    residu>1 :", {b: v["frac_residu_gt1"] for b, v in E[name].items() if v["frac_residu_gt1"] > 0},
                      " non fini :", {b: v["non_fini"] for b, v in E[name].items() if v["non_fini"]}, flush=True)
            for name, _ in configs:
                fl = E["flottant_emule"]["debit"]
                good = [b for b in BITS if all(E[name][bb]["debit"] >= fl - 0.1 for bb in BITS if bb >= b)]
                E.setdefault("min_bits_0.1bps", {})[name] = min(good) if good else None
            print(site, snr, "bits min (perte <= 0,1 bps/Hz) :", E["min_bits_0.1bps"], flush=True)
    out["float_transformer"] = res
    merge_save("fixed", out)

    if int8:
        calib_h = subsample(raw["umi"], 2048, seed=999)                            # lot de calibration distinct
        Gc = tf.matmul(calib_h, calib_h, adjoint_b=True)
        amax = int8_transformer(net, [(Gc, s) for s in (0, 10, 20, 30, 40)])
        r8 = {"calib_amax": amax}
        for site, h in data.items():
            for snr in SNRS_FX:
                E = r8.setdefault(site, {}).setdefault(str(snr), {})
                E["flottant_ns"] = eval_fx(net, h, snr, FX(None))
                for b in BITS:
                    E[b] = eval_fx(net, h, snr, FX(b, STAGES))
                print("int8", site, snr, f"NS flottant {E['flottant_ns']['debit']:.3f}",
                      {b: round(E[b]["debit"], 3) for b in BITS}, flush=True)
        out["int8_transformer"] = r8
        merge_save("fixed", out)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["seed", "fixed"], required=True)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    if a.part == "seed":
        part_seed()
    else:
        part_fixed(a.int8)
