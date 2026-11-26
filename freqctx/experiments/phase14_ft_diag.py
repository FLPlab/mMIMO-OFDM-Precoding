"""Phase 14 -- diagnostic de l'ecart au fort SNR sur un site du labo (demande F. Leduc-Primeau, suite
du fine-tuning phase14_finetune.py : l'adaptation (complet/LoRA/valeurs singulieres, N=40/400/4000) ne
comble qu'un quart a un tiers de l'ecart a WMMSE convergé a 30 dB). Memes fonctions (load_base, site_drops,
read_cells), memes moities adaptation/test et meme tirage de test (seed=77, 64 tirages) que
phase14_finetune.py, pour que les chiffres soient comparables.

Experiences (une execution = un site, cles choisies avec --only) :
  oracle          : fine-tuning complet, TOUTES les cellules d'adaptation, echantillons frais a chaque pas
                    (pas de jeu fixe), ~20000 pas, SNR U(0,30), perte normalisee (= phase14_finetune), pas 1e-4 -> cosine.
                    Borne haute d'adaptation : si meme ca ne rejoint pas WMMSE, c'est l'architecture/l'optimisation.
  oracle_snrhigh  : identique, mais SNR U(20,30) a l'entrainement (le fort SNR pese plus dans chaque pas).
  scratch         : identique a oracle (meme budget), mais poids initialises au hasard (pas de depart 3GPP).
  ft_snrhigh_N{n} : reprise de phase14_finetune (methode "complet", memes N et pas que la-bas) avec SNR U(20,30).
  wmmse_refine    : k=1,2,3 iterations de WMMSE initialisees par la sortie du reseau (zero-shot) au lieu de RZF,
                    comparees a WMMSE k it. initialise par RZF (precoders/classical.wmmse, parametre init=).
  diag_cond       : (ericsson seulement, ou --force-diag) debit par tirage a 30 dB vs conditionnement de G=H H^H,
                    reseau (zero-shot) contre WMMSE 300 it. -- cause possible de l'ecart residuel.

    python -u -m experiments.phase14_ft_diag --site ericsson
    python -u -m experiments.phase14_ft_diag --site ericsson --only oracle,oracle_snrhigh
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_lab import read_cells
from experiments.phase14_finetune import NS, STEPS, TEST_DROPS, load_base, peft, site_drops
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

W14 = "/export/tmp/sala/freqctx/weights/phase14"
ORACLE_STEPS, ORACLE_BATCH, ORACLE_LR = 20000, 256, 1e-4
LOG_EVERY = 2000
OUT_PATH_TMPL = os.path.join(RESULTS, "phase14", "ft_diag_{site}.json")


def scratch_net():
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    return net


def train_step_fn(net, variables, opt, normalize):
    @tf.function
    def step(h, snr):
        nv = 10.0 ** (-snr / 10.0)
        with tf.GradientTape() as tape:
            out = net(h, snr)
            if normalize:
                ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
                loss = -tf.reduce_mean(rate_per_sample(h, out, nv) / tf.stop_gradient(ref))
            else:
                loss = -tf.reduce_mean(rate_per_sample(h, out, nv))
        g = tape.gradient(loss, variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, variables))
        return loss
    return step


def train_oracle(net, cells, freqs, steps, snr_lo, snr_hi, normalize, seed, tag):
    """Fine-tuning complet, echantillons frais a chaque pas (site_drops rappele a chaque pas, pas de jeu fixe)."""
    variables = net.trainable_variables
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(ORACLE_LR, steps))
    step = train_step_fn(net, variables, opt, normalize)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for s in range(1, steps + 1):
        h = site_drops(cells, ORACLE_BATCH, freqs, seed=int(rng.integers(1, 2**31 - 1)), per_sc=True)
        snr = tf.random.uniform([ORACLE_BATCH], snr_lo, snr_hi)
        loss = step(h, snr)
        if s % LOG_EVERY == 0:
            print(f"  {tag} pas {s}/{steps} objectif {-float(loss):.3f} ({time.time() - t0:.0f} s)", flush=True)
    return net


def finetune_fixed(net, data, steps, lr, snr_lo, snr_hi, normalize):
    """= phase14_finetune.finetune (methode "complet"), SNR et normalisation parametrables."""
    variables = peft(net, "complet")
    opt = keras.optimizers.Adam(lr)
    step = train_step_fn(net, variables, opt, normalize)
    n = data.shape[0]
    bs = min(n, 256)
    for _ in range(steps):
        idx = tf.random.uniform([bs], 0, n, dtype=tf.int32)
        h = tf.gather(data, idx)
        perm = tf.argsort(tf.random.uniform([bs, 4]), axis=-1)
        h = tf.gather(h, perm, batch_dims=1)
        snr = tf.random.uniform([bs], snr_lo, snr_hi)
        step(h, snr)


def eval_net(net, h_test):
    return rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h_test)


def wmmse_refine(h_test, net_zs):
    """k = 1, 2, 3 it. WMMSE, initialise par RZF (defaut) ou par la sortie du reseau zero-shot."""
    out = {}
    for k in (1, 2, 3):
        out[f"WMMSE {k} it. (init RZF)"] = rates(lambda x, nv, s, k=k: wmmse(x, nv, num_iter=k), h_test)
        out[f"WMMSE {k} it. (init reseau)"] = rates(
            lambda x, nv, s, k=k: wmmse(x, nv, num_iter=k, init=net_zs(x, tf.fill([x.shape[0]], float(s)))), h_test)
        print(f"  WMMSE {k} it. : init RZF 20/30dB {out[f'WMMSE {k} it. (init RZF)'][4]:.2f}/"
              f"{out[f'WMMSE {k} it. (init RZF)'][6]:.2f} ; init reseau "
              f"{out[f'WMMSE {k} it. (init reseau)'][4]:.2f}/{out[f'WMMSE {k} it. (init reseau)'][6]:.2f}", flush=True)
    return out


def diag_cond(cells_test, freqs, net_zs, n=256, snr_db=30.0):
    """Debit par tirage a 30 dB (reseau zero-shot vs WMMSE 300 it.) vs conditionnement de G = H H^H
    (moyenne sur les SC), et vs l'ecart de puissance entre utilisateurs (dB, max-min par tirage)."""
    h = site_drops(cells_test, n, freqs, seed=321, per_sc=False)            # [n, F, K, M]
    hf = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])
    nv = 10.0 ** (-snr_db / 10.0)
    r_net = rate_per_sample(hf, net_zs(hf, tf.fill([hf.shape[0]], snr_db)), nv).numpy()
    r_wmmse = rate_per_sample(hf, wmmse(hf, nv, num_iter=300), nv).numpy()
    gram = tf.matmul(hf, hf, adjoint_b=True).numpy()
    cond = np.linalg.cond(gram)
    pow_db = 10 * np.log10(np.maximum(np.sum(np.abs(hf.numpy()) ** 2, axis=-1), 1e-12))
    pow_db = pow_db.reshape(n, -1)
    spread = pow_db.max(1) - pow_db.min(1)
    gap = (r_wmmse - r_net).reshape(n, -1).mean(1)
    cond_mean = np.log10(cond).reshape(n, -1).mean(1)
    corr_cond = float(np.corrcoef(gap, cond_mean)[0, 1])
    corr_pow = float(np.corrcoef(gap, spread)[0, 1])
    print(f"  diag_cond : corr(ecart, log10 cond G) = {corr_cond:.2f} ; corr(ecart, ecart de puissance dB) = "
          f"{corr_pow:.2f} (n={n} tirages, SC moyennees)", flush=True)
    return {"n": n, "snr_db": snr_db, "corr_ecart_log10cond": corr_cond, "corr_ecart_spread_db": corr_pow,
            "gap_mean": float(gap.mean()), "log10cond_mean": float(cond_mean.mean()), "spread_db_mean": float(spread.mean())}


def save(out, site):
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(OUT_PATH_TMPL.format(site=site), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["stecath", "decarie", "ericsson"])
    ap.add_argument("--only", default="oracle,oracle_snrhigh,scratch,ft_snrhigh_N400,ft_snrhigh_N4000,"
                                       "wmmse_refine,diag_cond")
    ap.add_argument("--oracle_steps", type=int, default=ORACLE_STEPS)
    ap.add_argument("--force_diag", action="store_true")
    a = ap.parse_args()
    only = set(a.only.split(","))
    keras.utils.set_random_seed(0)
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    cells = read_cells(a.site, "psi2az")
    adapt, test = cells[0::2], cells[1::2]
    h_test = site_drops(test, TEST_DROPS, freqs, seed=77, per_sc=False)     # memes tirages que phase14_finetune

    path = OUT_PATH_TMPL.format(site=a.site)
    out = json.load(open(path)) if os.path.exists(path) else \
        {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "site": a.site,
                  "cells_adapt": len(adapt), "cells_test": len(test), "oracle_steps": a.oracle_steps},
         "snrs_db": SNRS, "rates": {}, "diag": {}}

    base_zs = load_base()
    out["rates"]["zero-shot"] = eval_net(base_zs, h_test)
    print(f"{a.site} zero-shot : 20dB {out['rates']['zero-shot'][4]:.2f} / 30dB {out['rates']['zero-shot'][6]:.2f}", flush=True)
    out["rates"]["WMMSE 300 it."] = rates(lambda x, nv, s: wmmse(x, nv, num_iter=300), h_test)
    save(out, a.site)

    if "oracle" in only:
        t0 = time.time()
        net = load_base()
        train_oracle(net, adapt, freqs, a.oracle_steps, 0.0, 30.0, True, seed=1001, tag="oracle")
        out["rates"]["oracle (complet, donnees fraiches)"] = eval_net(net, h_test)
        net.save_weights(os.path.join(W14, f"ft_diag_{a.site}_oracle.weights.h5"))
        print(f"{a.site} oracle : 20dB {out['rates']['oracle (complet, donnees fraiches)'][4]:.2f} / "
              f"30dB {out['rates']['oracle (complet, donnees fraiches)'][6]:.2f} ({time.time()-t0:.0f} s)", flush=True)
        save(out, a.site)

    if "oracle_snrhigh" in only:
        t0 = time.time()
        net = load_base()
        train_oracle(net, adapt, freqs, a.oracle_steps, 20.0, 30.0, True, seed=1002, tag="oracle_snrhigh")
        out["rates"]["oracle + SNR U(20,30)"] = eval_net(net, h_test)
        net.save_weights(os.path.join(W14, f"ft_diag_{a.site}_oracle_snrhigh.weights.h5"))
        print(f"{a.site} oracle_snrhigh : 20dB {out['rates']['oracle + SNR U(20,30)'][4]:.2f} / "
              f"30dB {out['rates']['oracle + SNR U(20,30)'][6]:.2f} ({time.time()-t0:.0f} s)", flush=True)
        save(out, a.site)

    if "scratch" in only:
        t0 = time.time()
        net = scratch_net()
        train_oracle(net, adapt, freqs, a.oracle_steps, 0.0, 30.0, True, seed=1003, tag="scratch")
        out["rates"]["de zero (meme budget)"] = eval_net(net, h_test)
        net.save_weights(os.path.join(W14, f"ft_diag_{a.site}_scratch.weights.h5"))
        print(f"{a.site} scratch : 20dB {out['rates']['de zero (meme budget)'][4]:.2f} / "
              f"30dB {out['rates']['de zero (meme budget)'][6]:.2f} ({time.time()-t0:.0f} s)", flush=True)
        save(out, a.site)

    for n in NS[1:]:                                                        # N = 400, 4000
        key = f"ft_snrhigh_N{n}"
        if key in only:
            t0 = time.time()
            data = site_drops(adapt, n, freqs, seed=100 + n, per_sc=True)   # meme graine que phase14_finetune
            net = load_base()
            finetune_fixed(net, data, STEPS[n], 1e-4, 20.0, 30.0, True)
            rkey = f"complet N={n}, SNR U(20,30)"
            out["rates"][rkey] = eval_net(net, h_test)
            print(f"{a.site} {rkey} : 20dB {out['rates'][rkey][4]:.2f} / 30dB {out['rates'][rkey][6]:.2f} "
                  f"({time.time()-t0:.0f} s)", flush=True)
            save(out, a.site)

    if "wmmse_refine" in only:
        out["rates"].update(wmmse_refine(h_test, base_zs))
        save(out, a.site)

    if "diag_cond" in only and (a.site == "ericsson" or a.force_diag):
        out["diag"]["cond_vs_gap"] = diag_cond(test, freqs, base_zs)
        save(out, a.site)

    print("->", path, flush=True)


if __name__ == "__main__":
    main()
