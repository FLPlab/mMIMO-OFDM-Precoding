"""Phase 17 -- les ETATS INTERNES du precodeur libre sont-ils lisses en frequence (comme (lambda, p)) ou
chahutes (comme W) ? EVALUATION SEULE, aucun entrainement.

Reseau : GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm"), poids
phase13 (experiments.phase14_finetune.load_base), applique independamment a chaque SC. Activations
intermediaires en appelant les blocs a la main (experiments.phase17_probe.embed / block_call, verifie
contre net(...)).

Jeux large bande : Sionna UMi / UMa, St-Catherine, Decarie, Ericsson (experiments.phase13_bench.channels,
memes generateurs que phase15_adaptive_step), ND premiers tirages, F = 612 SC, CSI parfait, M8K4,
SNR 20 / 30 dB. Ancres = 1 SC sur s (+ derniere SC), interpolation lineaire (convention phase15).

Deux jauges :
  raw     : canal tel quel. G_jk = h~_j h~_k^H tourne en frequence avec l'ecart de retard moyen des
            utilisateurs j, k (la normalisation d'entree ne retire que la phase COMMUNE a tous les
            utilisateurs, pas la phase par utilisateur).
  aligned : jauge fixee par utilisateur AVANT le reseau : h_k(f) <- h_k(f) e^{-i phi_k(f)}, phi_k suivi
            de proche en proche (phi_k(f) = phi_k(f-1) + arg <h_k(f), h_k(f-1)>), donc
            <h_k(f), h_k(f-1)> reel positif. Le debit est invariant a la phase par ligne de H : le W
            calcule sur le canal aligne est un precodeur valide pour le vrai canal (cout ~4KM MAC/SC).

1. Erreur relative d'interpolation (non-ancres) ||Q_interp - Q|| / ||Q|| (mediane sur tirages x SC), et
   version centree (denominateur ||Q - moyenne||), pour Q = G, etats apres couches 0,1,2,4,6,8, C, W,
   (lambda, p) ajustes au W du reseau (phase15_diag.fit_structure_batch ; p normalise a moyenne 1).
2. Test fonctionnel : etat apres la couche l interpole entre ancres, puis couches l+1..8 du reseau
   EXISTANT a chaque SC avec le G LOCAL, lecture C, W = h~^H C local ; l = 8 : C interpole, W local.
   Compare a (a) reseau a chaque SC, (b) W interpole, (c) (lambda, p) interpoles + forme exacte.
   MAC/SC (convention models.gram_precoder.macs_per_sc).

    python -u -m experiments.phase17_freq_smooth
"""
import json
import logging
import os
import time

import keras
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from experiments.phase13_bench import channels
from experiments.phase14_finetune import load_base
from experiments.phase15_diag import fit_structure_batch
from experiments.phase17_probe import block_call, embed, read_c
from precoders.classical import normalize_power
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase17")
OUT_PATH = os.path.join(OUT_DIR, "freq_smooth.json")
SETS = ["umi", "uma", "stecath", "decarie", "ericsson"]
SNRS = [20, 30]
STEPS = [3, 6, 12, 24]
ND, DB = 96, 8                       # tirages par jeu, tirages par lot
LAYERS_Q = [0, 1, 2, 4, 6, 8]        # etats mesures (1.)
LAYERS_F = [0, 1, 2, 4, 6, 8]        # couche d'interpolation (2.)
MODES = ["raw", "aligned"]


# ---------------------------------------------------------------------- MAC

def macs(K=4, M=8, d=16, hd=32, L=8):
    g = 4 * K * K * M
    init = K * K * 5 * 2 * d                               # convention de macs_per_sc (5 entrees)
    layer = K * K * ((4 * d * hd + hd * 2 * d) + 2 * (2 * d + 3) * (2 * d)) + 4 * K ** 3 * d
    read = K * K * 2 * d * 2
    w = 4 * M * K * K
    total = g + init + L * layer + read + w
    return {"G": g, "init": init, "layer": layer, "read": read, "W": w, "total": total}


def macs_split(l, s, mc, K=4, d=16, M=8, L=8):
    """Tronc (G, init, couches 1..l) aux ancres (/s) + interpolation de l'etat + couches l+1..L, lecture,
    W locaux (+ G local si l < L ; h~ local toujours)."""
    trunk = mc["G"] + mc["init"] + l * mc["layer"]
    interp = K * K * 2 * d                                 # a + t (b - a) : 1 MAC reelle par composante
    local = (L - l) * mc["layer"] + mc["read"] + mc["W"] + (mc["G"] if l < L else 0)
    return trunk / s + interp + local


# ---------------------------------------------------------------------- outils

def anchors_of(F, s):
    a = np.unique(np.append(np.arange(0, F, s), F - 1))
    j = np.clip(np.searchsorted(a, np.arange(F), side="right") - 1, 0, len(a) - 2)
    t = ((np.arange(F) - a[j]) / (a[j + 1] - a[j])).astype(np.float32)
    return a, j, t


def interp(x, a, j, t):
    """x [nb, F, ...] -> interpolation lineaire depuis les ancres a."""
    xa = x[:, a]
    tt = t.reshape((1, -1) + (1,) * (x.ndim - 2))
    return (1 - tt) * xa[:, j] + tt * xa[:, j + 1]


def gauge(h):
    """h [nb, F, K, M] -> h e^{-i phi_k(f)}, phi suivi de proche en proche."""
    inner = np.sum(h[:, 1:] * np.conj(h[:, :-1]), -1)                     # [nb, F-1, K]
    phi = np.concatenate([np.zeros_like(inner[:, :1].real), np.cumsum(np.angle(inner), 1)], 1)
    return (h * np.exp(-1j * phi)[..., None]).astype(np.complex64)


def forward(net, hflat, snr):
    hp, _, g, e = embed(net, hflat, snr)
    es = {0: e}
    for li, blk in enumerate(net.blocks):
        e = block_call(blk, e, g)
        es[li + 1] = e
    return hp, g, es


def tail(net, e, g, start):
    for li in range(start, len(net.blocks)):
        e = block_call(net.blocks[li], e, g)
    return read_c(net, e)


def w_of(hp, c):
    return normalize_power(tf.matmul(hp, c, adjoint_a=True))


def struct_w(hflat, lam, p, nv):
    h = tf.cast(hflat, tf.complex128)
    K = h.shape[1]
    a = tf.cast(lam, tf.complex128)[:, :, None] * tf.matmul(h, h, adjoint_b=True) + nv * tf.eye(K, dtype=tf.complex128)[None]
    c = tf.linalg.inv(a) * tf.cast(tf.sqrt(np.maximum(p, 0.0)), tf.complex128)[:, None, :]
    return tf.cast(normalize_power(tf.matmul(h, c, adjoint_a=True)), tf.complex64)


def relerr(qi, q, mu=None):
    ax = tuple(range(2, q.ndim))
    num = np.sqrt(np.sum(np.abs(qi - q) ** 2, ax))
    den = np.sqrt(np.sum(np.abs(q if mu is None else q - mu) ** 2, ax))
    return num / np.maximum(den, 1e-30)


def mean_token(e):
    """Moyenne de l'etat par type de jeton (diagonal / hors diagonal), e [nb, F, K, K, d]."""
    K = e.shape[2]
    dm = np.eye(K, dtype=bool)
    mu = np.zeros(e.shape[2:], e.dtype)
    mu[dm] = e[:, :, dm].mean((0, 1, 2))
    mu[~dm] = e[:, :, ~dm].mean((0, 1, 2))
    return mu


# ---------------------------------------------------------------------- un jeu, un SNR

def run(net, h_all, snr, name):
    nv = 10.0 ** (-snr / 10.0)
    _, F, K, M = h_all.shape
    L = len(net.blocks)
    anc = {s: anchors_of(F, s) for s in STEPS}
    nonanc = {s: np.setdiff1d(np.arange(F), anc[s][0]) for s in STEPS}
    err = {m: {} for m in MODES}                    # err[mode][Q][s] -> liste (plain, centre)
    rates = {m: {} for m in MODES}                  # rates[mode][key][s] -> liste [nb]
    verif = None
    for b0 in range(0, h_all.shape[0], DB):
        hb = h_all[b0:b0 + DB]
        nb = hb.shape[0]
        hflat_true = tf.constant(hb.reshape(-1, K, M))
        for mode in MODES:
            hin = gauge(hb) if mode == "aligned" else hb
            hflat = tf.constant(hin.reshape(-1, K, M))
            hp, g, es = forward(net, hflat, snr)
            c = read_c(net, es[L])
            w = w_of(hp, c)
            if verif is None:
                verif = float(np.max(np.abs(net(hflat[:512], tf.fill([512], float(snr))).numpy() - w[:512].numpy())))
            rr = rates[mode]
            rr.setdefault("s1", {}).setdefault(1, []).append(
                tf.reduce_mean(tf.reshape(rate_per_sample(hflat_true, w, nv), [nb, F]), 1).numpy())
            # ---- 1. erreurs d'interpolation
            Q = {"G": g.numpy().reshape(nb, F, K, K)}
            for l in LAYERS_Q:
                Q[f"E{l}"] = es[l].numpy().reshape(nb, F, K, K, -1)
            Q["C"] = c.numpy().reshape(nb, F, K, K)
            Q["W"] = w.numpy().reshape(nb, F, M, K)
            if mode == "raw":
                wst, lam, p, ferr = fit_structure_batch(hflat_true, w, nv)
                p = p / p.mean(-1, keepdims=True)
                lam, p = lam.reshape(nb, F, K), p.reshape(nb, F, K)
                Q["lambda"], Q["p"] = lam, p
                err[mode].setdefault("fit_err", {}).setdefault(1, []).append((ferr, ferr))
                rr.setdefault("struct_exact", {}).setdefault(1, []).append(
                    tf.reduce_mean(tf.reshape(rate_per_sample(hflat_true, wst, nv), [nb, F]), 1).numpy())
            for qn, q in Q.items():
                if qn.startswith("E"):
                    mu = mean_token(q)
                elif qn in ("lambda", "p"):
                    mu = q.mean((0, 1))
                else:
                    mu = q.mean((0, 1))
                for s in STEPS:
                    a, j, t = anc[s]
                    qi = interp(q, a, j, t)
                    na = nonanc[s]
                    err[mode].setdefault(qn, {}).setdefault(s, []).append(
                        (relerr(qi[:, na], q[:, na]).ravel(), relerr(qi[:, na], q[:, na], mu).ravel()))
            # ---- 2. test fonctionnel
            for s in STEPS:
                a, j, t = anc[s]
                for l in LAYERS_F:
                    ei = interp(Q[f"E{l}"], a, j, t).reshape(nb * F, K, K, -1)
                    ci = tail(net, tf.constant(ei), g, l)
                    r = rate_per_sample(hflat_true, w_of(hp, ci), nv)
                    rr.setdefault(f"l{l}", {}).setdefault(s, []).append(tf.reduce_mean(tf.reshape(r, [nb, F]), 1).numpy())
                wi = normalize_power(tf.constant(interp(Q["W"], a, j, t).reshape(nb * F, M, K)))
                r = rate_per_sample(hflat_true, wi, nv)
                rr.setdefault("W", {}).setdefault(s, []).append(tf.reduce_mean(tf.reshape(r, [nb, F]), 1).numpy())
                if mode == "raw":
                    li_ = interp(Q["lambda"], a, j, t).reshape(-1, K)
                    pi_ = interp(Q["p"], a, j, t).reshape(-1, K)
                    r = rate_per_sample(hflat_true, struct_w(hflat_true, li_, pi_, nv), nv)
                    rr.setdefault("lambda_p", {}).setdefault(s, []).append(tf.reduce_mean(tf.reshape(r, [nb, F]), 1).numpy())
        print(f"   {name} {snr} dB : tirages {b0 + nb}/{h_all.shape[0]}", flush=True)
    out = {"verif_max_abs_diff_W": verif}
    for mode in MODES:
        e_out = {}
        for qn, d in err[mode].items():
            e_out[qn] = {}
            for s, lst in d.items():
                pl = np.concatenate([x[0] for x in lst]); ce = np.concatenate([x[1] for x in lst])
                e_out[qn][str(s)] = {"med": float(np.median(pl)), "med_centre": float(np.median(ce)),
                                     "moy": float(np.mean(pl)), "p90": float(np.percentile(pl, 90))}
        r_out = {k: {str(s): float(np.mean(np.concatenate(v))) for s, v in d.items()} for k, d in rates[mode].items()}
        out[mode] = {"interp_err": e_out, "debit": r_out}
    return out


def main():
    keras.utils.set_random_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    net = load_base()
    mc = macs()
    assert mc["total"] == net.macs_per_sc(4, 8), (mc, net.macs_per_sc(4, 8))
    mac_tab = {f"l{l}": {str(s): macs_split(l, s, mc) for s in STEPS} for l in LAYERS_F}
    mac_tab["W"] = {str(s): mc["total"] / s + 2 * 8 * 4 for s in STEPS}
    mac_tab["s1"] = {"1": mc["total"]}
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {
        "meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "n_drops": ND,
                 "steps": STEPS, "snrs": SNRS, "layers_interp": LAYERS_F, "macs": mc}, "macs_per_sc": mac_tab, "res": {}}
    t0 = time.time()
    for name in SETS:
        h = None
        for snr in SNRS:
            key = f"{name}_{snr}"
            if key in out["res"]:
                continue
            if h is None:
                h = channels(name)[:ND].numpy().astype(np.complex64)
                print(f"== {name} : {h.shape}", flush=True)
            out["res"][key] = res = run(net, h, snr, name)
            json.dump(out, open(OUT_PATH, "w"), indent=1)
            for mode in MODES:
                e, r = res[mode]["interp_err"], res[mode]["debit"]
                print(f"  {key} {mode}: s=1 {r['s1']['1']:.2f}  verif {res['verif_max_abs_diff_W']:.1e}", flush=True)
                for s in STEPS:
                    ss = str(s)
                    print(f"    s={s:2d} err med " + " ".join(f"{q}:{e[q][ss]['med']:.3f}/{e[q][ss]['med_centre']:.3f}" for q in e if ss in e[q])
                          + " | debit " + " ".join(f"{k}:{r[k][ss]:.2f}" for k in r if ss in r[k]), flush=True)
            print(f"   ({time.time() - t0:.0f} s)", flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
