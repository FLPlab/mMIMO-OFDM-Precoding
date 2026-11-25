"""Phase 13 -- canal jamais vu : lancer de rayons DeepMIMO v2 (Boston5G, 28 GHz) du labo, converti en OFDM
et passe dans NOTRE chaine (pilotes en peigne Sionna, controle de puissance montant), SANS reentrainer.

Canal : pour chaque position, les trajets DeepMIMO (angle de depart, phase, retard, puissance ;
format de read_raytracing.m) donnent H(f) sur les 612 SC (frequences de Sionna) pour le meme reseau BS
que l'entrainement : 4 colonnes a lambda/2, 2 polarisations (antennes 0-3 puis 4-7) ; la 2e polarisation
recoit une phase aleatoire independante par trajet (le lancer de rayons ne donne pas la polarisation).
Retards relatifs au premier trajet (synchronisation). Un tirage = K=4 positions au hasard parmi celles
qui ont au moins un trajet ; normalisation par tirage (puissance moyenne 1), ecarts reels entre
utilisateurs, ou puissances egales.

    python -u -m experiments.phase13_deepmimo
"""
import json
import os
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase11_power import BATCH, SNRS, classical_rates, load, model_rates
from experiments.phase12_mix import load as load_mix
from experiments.phase12_robust import KAPPA
from experiments.phase12_wiener import load_wiener_net, wiener
from precoders.classical import wmmse
from utils import RESULTS, git_commit

SCEN = "/usr/datasets/deepMIMO/HH_channels/deepMIMO_v2/scenarios/Boston5G_28/Boston5G_28"
DROPS, PILOTS, MAX_PATHS = 192, [0, 10], 25


def read_paths():
    """Liste (phi, theta, phase, toa, power) par position ayant au moins un trajet."""
    dod = sio.loadmat(SCEN + ".1.DoD.mat")["DoD_array_full"].ravel()
    cir = sio.loadmat(SCEN + ".1.CIR.mat")["CIR_array_full"].ravel()
    n, ptr, users = int(dod[0]), 1, []
    for _ in range(n):
        npath = int(dod[ptr + 1])
        if npath > 0:
            d = dod[ptr + 2:ptr + 2 + 4 * npath].reshape(npath, 4)[:MAX_PATHS]
            c = cir[ptr + 2:ptr + 2 + 4 * npath].reshape(npath, 4)[:MAX_PATHS]
            users.append((np.deg2rad(d[:, 1]), np.deg2rad(d[:, 2]), np.deg2rad(c[:, 1]), c[:, 2],
                          10 ** (0.1 * c[:, 3])))
        ptr += 2 + 4 * npath
    return users


def channel(u, freqs, rng):
    """H(f) [F, M] d'une position : 4 colonnes a lambda/2 x 2 polarisations."""
    phi, theta, phase, toa, pw = u
    g = np.sqrt(pw) * np.exp(1j * phase)                                   # [L]
    a = np.exp(1j * np.pi * np.outer(np.sin(theta) * np.sin(phi), np.arange(4)))   # [L, 4]
    g2 = g * np.exp(1j * rng.uniform(0, 2 * np.pi, len(g)))                # 2e polarisation
    d = np.exp(-2j * np.pi * np.outer(freqs, toa - toa.min()))             # [F, L]
    return np.concatenate([d @ (g[:, None] * a), d @ (g2[:, None] * a)], axis=1) / np.sqrt(2)


def drops(users, freqs, equal, seed):
    rng = np.random.default_rng(seed)
    h = np.zeros((DROPS, len(freqs), 4, 8), np.complex64)
    for b in range(DROPS):
        for k, i in enumerate(rng.choice(len(users), 4, replace=False)):
            hk = channel(users[i], freqs, rng)
            h[b, :, k] = hk / np.sqrt(np.mean(np.abs(hk) ** 2)) if equal else hk
        h[b] /= np.sqrt(np.mean(np.abs(h[b]) ** 2))
    return tf.constant(h)


def main():
    t0 = time.time()
    users = read_paths()
    print(f"{len(users)} positions avec au moins un trajet ({time.time() - t0:.0f} s)", flush=True)
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    nets = {"base v2": load("pilot_rand_ph32_norm_relpow", "norm"),
            "encodeur convolution": load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv"),
            "Wiener formule fermee + notre precodeur": load_wiener_net(KAPPA)}
    cov = cov_mixture()
    lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "scenario": "DeepMIMO Boston5G_28", "positions": len(users)}, "snrs_db": SNRS, "rates": {}}
    for equal in (False, True):
        h = drops(users, freqs, equal, seed=13)
        tag = "puissances egales" if equal else "puissances reelles"
        out["rates"][tag] = {}
        for p in PILOTS:
            ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], float(p)) for j in range(0, DROPS, BATCH)], 0)
            pairs = [sound_power_controlled(lmm, h[j:j + BATCH], float(p), return_err_var=True)
                     for j in range(0, DROPS, BATCH)]
            lm, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
            r = {k: model_rates(n, ls, h) for k, n in nets.items()}
            r["Wiener formule fermee + WMMSE 100 it."] = classical_rates(
                lambda x, nv, e: wmmse(x, nv, num_iter=100), wiener(ls, KAPPA), h)
            r["LMMSE + WMMSE robuste 100 it."] = classical_rates(
                lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev)
            r["WMMSE 100 it. (CSI parfait)"] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100), h, h)
            out["rates"][tag][p] = r
            print(f"{tag}, pilote {p} dB @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase13"), exist_ok=True)
    path = os.path.join(RESULTS, "phase13", "deepmimo_boston.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
