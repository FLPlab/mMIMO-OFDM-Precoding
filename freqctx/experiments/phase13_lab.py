"""Phase 13 -- sites du labo (MIMO GenEE, scenario "Raytracing with mobility" : St-Catherine, Decarie,
Ericsson) convertis en OFDM et passes dans NOTRE chaine Sionna (pilotes en peigne, controle de
puissance montant, estimation), SANS reentrainer. Exploration : une graine.

Lien labo -> Sionna : le fichier du labo donne, pour chaque cellule (i, j) d'une grille, ses trajets
(tau_s, gain complexe alpha, angles psi_1, psi_2, ux, uy). Retards et gains suffisent pour la frequence ;
la convention des angles n'est pas documentee, on teste donc plusieurs hypotheses (--conv) :
  psi2az   : azimut de depart = psi_2 (dans [-pi, pi]), zenith = |psi_1|
  psi1az   : azimut = psi_1, horizontal
  aleatoire: azimut tire au hasard par trajet (seule la structure en retard est gardee)
H(f) est construit pour le reseau BS de l'entrainement (4 colonnes, 2 polarisations) par
phase13_deepmimo.channel ; un tirage = 4 cellules au hasard.

    python -u -m experiments.phase13_lab --site stecath --conv psi2az
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase11_power import BATCH, SNRS, classical_rates, load, model_rates
from experiments.phase12_mix import load as load_mix
from experiments.phase12_robust import KAPPA
from experiments.phase12_wiener import load_wiener_net, wiener
from experiments.phase13_deepmimo import DROPS, drops
from precoders.classical import rzf, wmmse
from utils import RESULTS, git_commit

DATA = "/export/tmp/sala/freqctx/data/lab"
PILOTS = [0, 10]


def read_cells(site, conv, seed=0):
    """Liste (phi, theta, phase, toa, power) par cellule de la grille (meme format que DeepMIMO)."""
    df = pd.read_csv(os.path.join(DATA, f"{site}_ts.csv"), usecols=["i", "j", "tau_s", "alpha_re", "alpha_im",
                                                                   "psi_1", "psi_2"]).dropna()
    rng = np.random.default_rng(seed)
    users = []
    for _, c in df.groupby(["i", "j"], sort=False):
        a = c.alpha_re.values + 1j * c.alpha_im.values
        if conv == "psi2az":
            phi, theta = c.psi_2.values, np.abs(c.psi_1.values)
        elif conv == "psi1az":
            phi, theta = c.psi_1.values, np.full(len(c), np.pi / 2)
        else:
            phi, theta = rng.uniform(-np.pi, np.pi, len(c)), np.full(len(c), np.pi / 2)
        users.append((phi, theta, np.angle(a), c.tau_s.values, np.abs(a) ** 2))
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["stecath", "decarie", "ericsson"])
    ap.add_argument("--conv", default="psi2az", choices=["psi2az", "psi1az", "aleatoire"])
    ap.add_argument("--extra", action="store_true",
                    help="ajoute RZF, notre precodeur en CSI parfait et LMMSE avec la covariance du site")
    a = ap.parse_args()
    t0 = time.time()
    users = read_cells(a.site, a.conv)
    spread = np.median([np.sqrt(np.average((u[3] - np.average(u[3], weights=u[4])) ** 2, weights=u[4]))
                        for u in users if len(u[3]) > 1]) * 1e9
    print(f"{a.site} : {len(users)} cellules, {np.mean([len(u[3]) for u in users]):.1f} trajets en moyenne, "
          f"etalement des retards median {spread:.0f} ns ({time.time() - t0:.0f} s)", flush=True)
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    nets = {"base v2": load("pilot_rand_ph32_norm_relpow", "norm"),
            "encodeur convolution": load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv"),
            "Wiener formule fermee + notre precodeur": load_wiener_net(KAPPA)}
    cov = cov_mixture()
    lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
    if a.extra:                     # covariance frequentielle estimee sur le site (autres tirages)
        hc = drops(users, freqs, equal=True, seed=99).numpy()               # [N, F, K, M], puissance 1
        x = np.transpose(hc, (0, 2, 3, 1)).reshape(-1, cfg.num_sc)
        cov_site = tf.constant((x.T @ x.conj()) / x.shape[0], tf.complex64)
        lmm_site = UplinkSounder(cfg, "lmmse", cov_site)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "site": a.site,
                    "conv": a.conv, "cells": len(users), "delay_spread_ns": float(spread)},
           "snrs_db": SNRS, "rates": {}}
    h = drops(users, freqs, equal=False, seed=13)                     # puissances reelles
    for p in PILOTS:
        ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], float(p)) for j in range(0, DROPS, BATCH)], 0)
        pairs = [sound_power_controlled(lmm, h[j:j + BATCH], float(p), return_err_var=True)
                 for j in range(0, DROPS, BATCH)]
        lm, ev = tf.concat([x for x, _ in pairs], 0), tf.concat([y for _, y in pairs], 0)
        r = {k: model_rates(n, ls, h) for k, n in nets.items()}
        r["Wiener formule fermee + WMMSE 100 it."] = classical_rates(
            lambda x, nv, e: wmmse(x, nv, num_iter=100), wiener(ls, KAPPA), h)
        r["LMMSE + WMMSE robuste 100 it."] = classical_rates(
            lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev)
        r["WMMSE 100 it. (CSI parfait)"] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100), h, h)
        if a.extra:
            r["Wiener formule fermee + RZF"] = classical_rates(lambda x, nv, e: rzf(x, nv), wiener(ls, KAPPA), h)
            r["RZF (CSI parfait)"] = classical_rates(lambda x, nv, e: rzf(x, nv), h, h)
            prec = nets["encodeur convolution"].precoder                   # precodeur seul, sur le vrai canal
            r["notre precodeur (CSI parfait)"] = classical_rates(lambda x, nv, e: prec(x, tf.fill([x.shape[0]],
                                                                 float(-10 * np.log10(nv)))), h, h)
            pairs = [sound_power_controlled(lmm_site, h[j:j + BATCH], float(p), return_err_var=True)
                     for j in range(0, DROPS, BATCH)]
            lms, evs = tf.concat([x for x, _ in pairs], 0), tf.concat([y for _, y in pairs], 0)
            r["LMMSE (covariance du site) + WMMSE robuste 100 it."] = classical_rates(
                lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lms, h, evs)
        out["rates"][p] = r
        print(f"{a.site} {a.conv}, pilote {p} dB @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()),
              flush=True)
    os.makedirs(os.path.join(RESULTS, "phase13"), exist_ok=True)
    path = os.path.join(RESULTS, "phase13", f"lab_{a.site}_{a.conv}{'_extra' if a.extra else ''}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
