"""Phase 15 -- diagnostic d'EVALUATION SEULE (aucun entrainement) du precodeur par paires
GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm"), poids
phase13 (3GPP M8K4+M64K8), CSI parfait, M8K4 (demande F. Leduc-Primeau, 2026-11-26).

Contexte : sur les sites du labo (St-Catherine, Decarie, Ericsson) le reseau est a ~99% de WMMSE
300 it. jusqu'a 20 dB mais perd 0.2-0.6 bps/Hz a 30 dB, alors que sur UMi il depasse WMMSE a 30 dB.
Une it. WMMSE initialisee par le reseau depasse WMMSE 300 it. -> plancher d'interference residuelle
(inversion approchee) OU regime hors distribution (SINR jamais vus a l'entrainement).

Trois questions :
  floor     : fuite d'interference normalisee (dB) et debit, reseau / WMMSE 300 it. (init RZF) /
              reseau+1it WMMSE, SNR 10-40 dB, UMi (+ sous-ensemble quartile bas de cond(G)) et les
              3 sites (memes tirages test que phase14_finetune/phase14_ft_diag : seed=77, 64 tirages).
  structure : le reseau retrouve-t-il la structure de Bjornson 2014 (forme Gram) C=(Lambda G+sigma^2 I)^-1
              P^(1/2) ? Ajustement par gradient de (lambda,p) (2K scalaires>0, parametrage exp) pour
              minimiser l'ecart (domaine W, apres normalisation de puissance et alignement de phase
              par colonne -- le debit est invariant a cette phase). Inversion EXACTE.
  freedom   : (lambda,p) ajustes sur W de WMMSE 300 it. convergee, inversion par Newton-Schulz
              (X <- X(2I-AX), X0 = A^H/(||A||_1 ||A||_inf)), debit vs T = 2,4,6,8,10,15, et MAC/SC
              (G, NS, W) compares au reseau (macs_per_sc) et a RZF.

Reutilise (import seulement, SANS MODIFIER) : experiments.phase13_bench.channels (UMi), et
experiments.phase14_finetune.{load_base, site_drops, TEST_DROPS} + experiments.phase13_lab.read_cells
pour les sites (meme moitie "test", meme graine 77 que phase14_finetune/phase14_ft_diag).

    python -u -m experiments.phase15_diag --part floor|structure|freedom|macs|all
"""
import argparse
import json
import logging
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

tf.get_logger().setLevel(logging.ERROR)   # tf.cast complexe->reel au retour de gradient (Wirtinger), inoffensif

from config import SystemConfig
from experiments.phase13_bench import channels as bench_channels
from experiments.phase13_lab import read_cells
from experiments.phase14_finetune import TEST_DROPS, load_base, site_drops
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase15")
OUT_PATH = os.path.join(OUT_DIR, "diag.json")
SITES = ["stecath", "decarie", "ericsson"]
SNRS_FLOOR = [10, 20, 25, 30, 35, 40]
N_FLOOR = 16384
N_FIT = 256


def load_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        return json.load(open(OUT_PATH))
    return {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
            "floor": {}, "structure": {}, "freedom": {}, "macs": {}}


def save(out):
    json.dump(out, open(OUT_PATH, "w"), indent=1)


def get_datasets():
    """{"umi": [N,F,4,8], "stecath"/"decarie"/"ericsson": [64,F,4,8]} -- UMi = tirages phase13_bench
    (phase14_zeroshot) ; sites = moitie "test" de phase14_finetune.site_drops, seed=77, 64 tirages."""
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    data = {"umi": bench_channels("umi")}
    for site in SITES:
        cells = read_cells(site, "psi2az")
        test = cells[1::2]                                       # moitie test (phase14_finetune/ft_diag)
        data[site] = site_drops(test, TEST_DROPS, freqs, seed=77, per_sc=False)
    return data


def subsample(h, n, seed):
    hf = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])
    if hf.shape[0] <= n:
        return hf
    rng = np.random.default_rng(seed)
    idx = rng.choice(hf.shape[0], n, replace=False)
    return tf.gather(hf, idx)


def leak_db_sig_intf(h, w):
    """Fuite normalisee par utilisateur, en dB : sum_{j!=k}|h_k w_j|^2 / |h_k w_k|^2 (complex128)."""
    h128, w128 = tf.cast(h, tf.complex128), tf.cast(w, tf.complex128)
    p = tf.abs(tf.matmul(h128, w128)) ** 2                        # [N,K,K]
    sig = tf.linalg.diag_part(p)
    intf = tf.reduce_sum(p, axis=-1) - sig
    leak_db = 10.0 * tf.math.log(tf.maximum(intf, 1e-30) / tf.maximum(sig, 1e-30)) / np.log(10.0)
    return leak_db.numpy(), sig.numpy(), intf.numpy()


# ---------------------------------------------------------------------- 1. plancher


def run_floor(out):
    net = load_base()
    raw = get_datasets()
    sub = {name: subsample(h, N_FLOOR, seed=4242) for name, h in raw.items()}

    h_umi = sub["umi"]
    g = tf.matmul(tf.cast(h_umi, tf.complex128), tf.cast(h_umi, tf.complex128), adjoint_b=True)
    cond = np.linalg.cond(g.numpy())
    thresh = np.quantile(cond, 0.25)
    sub["umi_quartile_bas_condG"] = tf.gather(h_umi, np.where(cond <= thresh)[0])

    results = out.get("floor", {})
    for name, h in sub.items():
        res = results.get(name, {"n": int(h.shape[0])})
        for snr in SNRS_FLOOR:
            if str(snr) in res:
                continue
            nv = 10.0 ** (-snr / 10.0)
            w_net = net(h, tf.fill([h.shape[0]], float(snr)))
            w_wmmse = wmmse(h, nv, num_iter=300)
            w_net1it = wmmse(h, nv, num_iter=1, init=w_net)
            entry = {}
            for tag, w in (("reseau", w_net), ("wmmse300", w_wmmse), ("reseau+1it", w_net1it)):
                leak_db, sig, intf = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                sinr_mean = float(np.mean(sig / np.maximum(intf + nv, 1e-30)))
                entry[tag] = {"debit": rate, "sinr_moyen_lin": sinr_mean,
                              "fuite_db_moyenne": float(np.mean(leak_db)), "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"floor {name:24s} {snr:2d}dB : reseau {entry['reseau']['debit']:6.2f} "
                  f"(fuite med {entry['reseau']['fuite_db_mediane']:6.1f} dB)  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f} (fuite med {entry['wmmse300']['fuite_db_mediane']:6.1f} dB)  "
                  f"reseau+1it {entry['reseau+1it']['debit']:6.2f}", flush=True)
            results[name] = res
            out["floor"] = results
            save(out)
    return results


# ---------------------------------------------------------------------- 2/3. structure de Bjornson


def fit_structure_batch(h, w_target, nv, steps=400, lr=0.08):
    """Ajuste (lambda,p) > 0 (parametrage exp, un par echantillon x utilisateur) pour que
    C=(Lambda G + sigma^2 I)^-1 P^(1/2) (G=H H^H physique, inversion EXACTE) donne un W aussi proche
    que possible de w_target, apres normalisation de puissance et alignement de phase par colonne
    (le debit est invariant a cette phase : seule la mesure d'erreur de Frobenius en a besoin).
    Renvoie W_struct, lambda, p, erreur relative par echantillon."""
    N, K = h.shape[0], h.shape[1]
    h128 = tf.cast(h, tf.complex128)
    g = tf.matmul(h128, h128, adjoint_b=True)
    eye = tf.eye(K, dtype=tf.complex128)[None]
    nv128 = tf.cast(nv, tf.complex128)
    wt128 = tf.cast(w_target, tf.complex128)
    theta_l = tf.Variable(tf.zeros([N, K], tf.float64))
    theta_p = tf.Variable(tf.zeros([N, K], tf.float64))
    opt = keras.optimizers.Adam(lr)

    def forward():
        lam = tf.exp(theta_l)
        p = tf.exp(theta_p)
        a = tf.cast(lam, tf.complex128)[:, :, None] * g + nv128 * eye
        c = tf.linalg.inv(a) * tf.cast(tf.sqrt(p), tf.complex128)[:, None, :]
        w = tf.matmul(h128, c, adjoint_a=True)
        pw = tf.reduce_sum(tf.abs(w) ** 2, axis=[-2, -1], keepdims=True)
        return w / tf.cast(tf.sqrt(pw), w.dtype)

    def err2(w):
        inner = tf.reduce_sum(tf.math.conj(w) * wt128, axis=1)                     # [N,K]
        return tf.reduce_sum(tf.abs(w) ** 2, axis=1) + tf.reduce_sum(tf.abs(wt128) ** 2, axis=1) - 2 * tf.abs(inner)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum(tf.math.real(err2(forward())), axis=-1))
        grads = tape.gradient(loss, [theta_l, theta_p])
        opt.apply_gradients(zip(grads, [theta_l, theta_p]))

    w_final = forward()
    e2 = tf.reduce_sum(tf.math.real(err2(w_final)), axis=-1)
    err_rel = (tf.sqrt(e2) / tf.sqrt(tf.reduce_sum(tf.abs(wt128) ** 2, axis=[-2, -1]))).numpy()
    return tf.cast(w_final, tf.complex64), tf.exp(theta_l).numpy(), tf.exp(theta_p).numpy(), err_rel


def run_structure(out):
    net = load_base()
    data = get_datasets()
    results = out.get("structure", {})
    for name, h_full in data.items():
        hf = subsample(h_full, N_FIT, seed=777)
        res = results.get(name, {})
        for snr in (20, 30, 40):
            if str(snr) in res:
                continue
            nv = 10.0 ** (-snr / 10.0)
            w_net = net(hf, tf.fill([hf.shape[0]], float(snr)))
            w_struct, _, _, err = fit_structure_batch(hf, w_net, nv)
            rate_net = float(tf.reduce_mean(rate_per_sample(hf, w_net, nv)))
            rate_struct = float(tf.reduce_mean(rate_per_sample(hf, w_struct, nv)))
            res[str(snr)] = {"erreur_rel_moyenne": float(np.mean(err)), "erreur_rel_mediane": float(np.median(err)),
                              "debit_reseau": rate_net, "debit_struct": rate_struct}
            print(f"structure {name:10s} {snr}dB : erreur rel {np.mean(err):.3f} (med {np.median(err):.3f})  "
                  f"debit reseau {rate_net:.2f}  struct(exact) {rate_struct:.2f}", flush=True)
            results[name] = res
            out["structure"] = results
            save(out)
    return results


# ---------------------------------------------------------------------- 3. prix de la liberte (Newton-Schulz)


def newton_schulz(a, p_sqrt, t_steps):
    """X <- X(2I - A X), X0 = A^H / (||A||_1 ||A||_inf), sans autre division. C = X_T diag(p_sqrt)."""
    k = a.shape[-1]
    n1 = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=-2), axis=-1)       # ||A||_1 : max somme colonne
    ninf = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=-1), axis=-1)     # ||A||_inf : max somme ligne
    x = tf.linalg.adjoint(a) * tf.cast(1.0 / (n1 * ninf), a.dtype)[:, None, None]
    eye = tf.eye(k, dtype=a.dtype)[None]
    for _ in range(t_steps):
        x = tf.matmul(x, 2 * eye - tf.matmul(a, x))
    return x * p_sqrt[:, None, :]


def run_freedom(out):
    net = load_base()
    data = get_datasets()
    t_list = [2, 4, 6, 8, 10, 15]
    results = out.get("freedom", {})
    for name, h_full in data.items():
        hf = subsample(h_full, N_FIT, seed=888)
        h128 = tf.cast(hf, tf.complex128)
        g = tf.matmul(h128, h128, adjoint_b=True)
        eye = tf.eye(hf.shape[1], dtype=tf.complex128)[None]
        res = results.get(name, {})
        for snr in (20, 30):
            if str(snr) in res:
                continue
            nv = 10.0 ** (-snr / 10.0)
            w_net = net(hf, tf.fill([hf.shape[0]], float(snr)))
            w_wmmse = wmmse(hf, nv, num_iter=300)
            rate_net = float(tf.reduce_mean(rate_per_sample(hf, w_net, nv)))
            rate_wmmse = float(tf.reduce_mean(rate_per_sample(hf, w_wmmse, nv)))
            _, lam, p, err = fit_structure_batch(hf, w_wmmse, nv)
            a = tf.cast(lam, tf.complex128)[:, :, None] * g + tf.cast(nv, tf.complex128) * eye
            entry = {"debit_reseau": rate_net, "debit_wmmse300": rate_wmmse, "erreur_fit_structwmmse": float(np.mean(err)),
                      "T": {}}
            for t_steps in t_list:
                c_ns = newton_schulz(a, tf.cast(tf.sqrt(p), tf.complex128), t_steps)
                w_ns = tf.matmul(h128, c_ns, adjoint_a=True)
                pw = tf.reduce_sum(tf.abs(w_ns) ** 2, axis=[-2, -1], keepdims=True)
                w_ns = tf.cast(w_ns / tf.cast(tf.sqrt(pw), w_ns.dtype), tf.complex64)
                entry["T"][str(t_steps)] = float(tf.reduce_mean(rate_per_sample(hf, w_ns, nv)))
            res[str(snr)] = entry
            print(f"freedom {name:10s} {snr}dB : reseau {rate_net:.2f}  wmmse300 {rate_wmmse:.2f}  "
                  f"NS(T) " + " ".join(f"{t}:{v:.2f}" for t, v in entry["T"].items()), flush=True)
            results[name] = res
            out["freedom"] = results
            save(out)
    return results


# ---------------------------------------------------------------------- MAC/SC


def run_macs(out):
    net = load_base()
    k, m = 4, 8
    mac_net = int(net.macs_per_sc(k, m))
    mac_g = 4 * m * k * k                    # G = H H^H : m*k*k mult. complexes -> 4 MAC reelles chacune
    mac_w = 4 * m * k * k                    # W = H^H C : idem
    mac_rzf_inv = int(4 * (2 * k ** 3 / 3))  # inversion exacte (LU), ordre de grandeur
    mac_rzf = mac_g + mac_rzf_inv + mac_w
    entry = {"reseau_macs_par_sc": mac_net, "rzf_macs_par_sc_approx": mac_rzf, "G_macs": mac_g, "W_macs": mac_w}
    for t_steps in (2, 4, 6, 8, 10, 15):
        mac_ns_iter = 8 * k ** 3              # 2 matmuls KxK par pas (A X puis X.(2I-AX)), 4 MAC/mult. complexe
        entry[f"NS_T{t_steps}_macs_par_sc"] = mac_g + mac_w + t_steps * mac_ns_iter
    out["macs"] = entry
    save(out)
    print("macs/SC :", entry, flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["floor", "structure", "freedom", "macs", "all"])
    a = ap.parse_args()
    keras.utils.set_random_seed(0)
    out = load_out()
    t0 = time.time()
    if a.part in ("floor", "all"):
        run_floor(out)
    if a.part in ("structure", "all"):
        run_structure(out)
    if a.part in ("freedom", "all"):
        run_freedom(out)
    if a.part in ("macs", "all"):
        run_macs(out)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
