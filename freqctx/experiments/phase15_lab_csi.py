"""Phase 15 -- generalisation a des sites jamais vus, en CSI estime (demande de F. Leduc-Primeau, 2026-11-26).

Deux questions :
(a) sur les sites du labo (St-Catherine, Decarie, Ericsson, jamais vus a l'entrainement), en CSI estime,
    les precodeurs entraines avec une erreur de CSI SYNTHETIQUE (phase14_csi.py, phase14_csi_errin.py,
    canaux 3GPP) egalent-ils ou depassent-ils WMMSE robuste ?
(b) l'ecart au fort SNR vu en CSI parfait (phase15_diag.py : plancher d'interference du reseau, ~-45 dB,
    Ericsson 30 dB reseau 35.26 contre WMMSE300 36.15) subsiste-t-il en CSI estime, ou l'erreur
    d'estimation l'efface-t-elle ?

Sites convertis en OFDM comme experiments/phase13_lab.py (convention psi2az), M8K4, puissances reelles,
192 tirages (phase13_deepmimo.DROPS), seed=13 -- memes tirages que phase13_lab.main() sans --extra.
Estimation comme experiments/phase14_csi_errin.py : pilotes Sionna, LS + interpolation LMMSE a covariance
du melange (experiments.phase7_context.cov_mixture()) ; variance d'erreur = celle renvoyee par
l'estimateur (csi.UplinkSounder, return_err_var=True), convertie en dB relatif par err_db_from (meme
fonction que phase14_csi_errin). Pilote 0 / 10 / 20 dB, SNR de donnees 10 / 20 / 25 / 30 dB, debit
calcule sur le VRAI canal.

Precodeurs (tous recoivent la meme estimation Ĥ, meme variance d'erreur) :
  reseau CSI parfait                              weights/phase13/precoder_relpow_mix.weights.h5
  reseau erreur synthetique (sans variance)        weights/phase14/precoder_synth_csi.weights.h5
  reseau erreur synthetique + variance en entree   weights/phase14/precoder_synth_csi_errin.weights.h5
                                                    SI PRESENT -- phase14_csi_errin.py ne les a jamais
                                                    sauvegardes (verifie le 2026-11-26) -> ligne omise,
                                                    AUCUN REENTRAINEMENT ne doit etre lance ici
  WMMSE robuste (err_var donnee) 100 et 300 it., RZF
Reference CSI PARFAIT (meme h, pour situer) : reseau CSI parfait, WMMSE 300 it.

EVALUATION SEULE : ce script n'importe que des poids deja entraines, aucun appel a train_synthetic()
ou train_errin().

    python -u -m experiments.phase15_lab_csi
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from experiments.phase14_csi import W14, load_perfect, metrics
from experiments.phase14_csi_errin import err_db_from, load_existing_no_err
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from utils import RESULTS, git_commit

SITES = ["stecath", "decarie", "ericsson"]
PILOTS = [0, 10, 20]
SNRS = [10, 20, 25, 30]
B = 16
OUT_PATH = os.path.join(RESULTS, "phase15", "lab_csi.json")


def load_errin_if_exists():
    """reseau erreur synthetique + variance en entree (GramPrecoder(err_in=True)) : None si les poids
    n'ont jamais ete sauvegardes (phase14_csi_errin.py a tourne mais ne les a pas ecrits sur disque)."""
    path = os.path.join(W14, "precoder_synth_csi_errin.weights.h5")
    if not os.path.exists(path):
        return None
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm", err_in=True)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]), tf.zeros([1, 4]))
    net.load_weights(path)
    return net


def evaluate(fns, h_est, h, err=None, chunk=8192):
    """fns : nom -> fn(h_est_chunk, nv, snr, err_chunk) -> W. Debit sur le VRAI canal h."""
    K, M = h.shape[-2], h.shape[-1]
    hf, he = tf.reshape(h, [-1, K, M]), tf.reshape(h_est, [-1, K, M])
    ef = None if err is None else tf.reshape(err, [-1, K, err.shape[-1]])
    out = {}
    for name, fn in fns.items():
        res = {"rate": [], "ber_qpsk": [], "ber_16qam": []}
        for snr in SNRS:
            nv = 10.0 ** (-snr / 10.0)
            m = [metrics(hf[i:i + chunk], fn(he[i:i + chunk], nv, snr, None if ef is None else ef[i:i + chunk]), nv)
                 for i in range(0, hf.shape[0], chunk)]
            for k, j in (("rate", 0), ("ber_qpsk", 1), ("ber_16qam", 2)):
                res[k].append(float(np.mean([x[j] for x in m])))
        out[name] = res
    return out


def main(errin_only=False):
    if errin_only:
        # Ajoute UNIQUEMENT la ligne manquante "reseau erreur synthetique + variance" aux sites deja
        # presents dans lab_csi.json (meme protocole : seed=13, memes pilotes/SNR/B), sans recalculer
        # ni ecraser les autres precodeurs/lignes.
        net_se = load_errin_if_exists()
        assert net_se is not None, "poids errin absents -- rien a ajouter"
        assert os.path.exists(OUT_PATH), f"{OUT_PATH} absent -- lance d'abord main() normal"
        out = json.load(open(OUT_PATH))
        cov = cov_mixture()
        cfg = SystemConfig()
        snd = UplinkSounder(cfg, "lmmse", cov)
        for site in SITES:
            if site not in out["sites"]:
                print(f"{site} absent de {OUT_PATH}, ignore.", flush=True)
                continue
            t0 = time.time()
            users = read_cells(site, "psi2az")
            freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
            h = drops(users, freqs, equal=False, seed=13)
            site_out = out["sites"][site]
            for p in PILOTS:
                pairs = [sound_power_controlled(snd, h[j:j + B], float(p), return_err_var=True)
                         for j in range(0, DROPS, B)]
                he, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
                fns = {"reseau erreur synthetique + variance":
                       lambda x, nv, s, e: net_se(x, tf.fill([x.shape[0]], float(s)), err_db_from(e, x))}
                r = evaluate(fns, he, h, ev)
                site_out["estime"][f"pilote{p}"].update(r)
                print(f"{site} pilote {p} dB (errin-only) : reseau erreur synthetique + variance "
                      f"{r['reseau erreur synthetique + variance']['rate'][SNRS.index(20)]:.2f}@20 "
                      f"{r['reseau erreur synthetique + variance']['rate'][SNRS.index(30)]:.2f}@30", flush=True)
            out["sites"][site] = site_out
            out["meta"]["errin_weights_found"] = True
            json.dump(out, open(OUT_PATH, "w"), indent=1)
            print(f"{site} (errin-only) termine ({time.time() - t0:.0f} s)", flush=True)
        print("->", OUT_PATH, flush=True)
        return

    net_p = load_perfect()
    net_s = load_existing_no_err()
    net_se = load_errin_if_exists()
    if net_se is None:
        print("ATTENTION : weights/phase14/precoder_synth_csi_errin.weights.h5 absent -- phase14_csi_errin.py "
              "n'a jamais sauvegarde ce reseau -- ligne 'erreur synthetique + variance' omise, "
              "AUCUN REENTRAINEMENT lance.", flush=True)

    cfg = SystemConfig()                                      # defaut : M8K4, comme phase13_lab
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    cov = cov_mixture()
    snd = UplinkSounder(cfg, "lmmse", cov)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "snrs_db": SNRS, "pilots_db": PILOTS, "drops": DROPS,
                    "errin_weights_found": net_se is not None}, "sites": {}}
    if os.path.exists(OUT_PATH):
        prev = json.load(open(OUT_PATH))
        out["sites"] = prev.get("sites", {})

    for site in SITES:
        if site in out["sites"]:
            print(f"{site} deja fait, passe.", flush=True)
            continue
        t0 = time.time()
        users = read_cells(site, "psi2az")
        h = drops(users, freqs, equal=False, seed=13)                 # [192, F, 4, 8], puissances reelles
        site_out = {"n_cells": len(users)}

        perfect_fns = {"reseau (CSI parfait)": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                       "WMMSE 300 it. (CSI parfait)": lambda x, nv, s, e: wmmse(x, nv, num_iter=300)}
        site_out["csi_parfait"] = evaluate(perfect_fns, h, h)
        print(f"{site} ({len(users)} cellules) CSI parfait : " + " | ".join(
            f"{k} {v['rate'][SNRS.index(20)]:.2f}@20 {v['rate'][SNRS.index(30)]:.2f}@30"
            for k, v in site_out["csi_parfait"].items()), flush=True)

        site_out["estime"] = {}
        for p in PILOTS:
            pairs = [sound_power_controlled(snd, h[j:j + B], float(p), return_err_var=True)
                     for j in range(0, DROPS, B)]
            he, ev = tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)
            fns = {
                "reseau CSI parfait": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                "reseau erreur synthetique": lambda x, nv, s, e: net_s(x, tf.fill([x.shape[0]], float(s))),
                "WMMSE robuste 100 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e),
                "WMMSE robuste 300 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=300, err_var=e),
                "RZF": lambda x, nv, s, e: rzf(x, nv)}
            if net_se is not None:
                fns["reseau erreur synthetique + variance"] = \
                    lambda x, nv, s, e: net_se(x, tf.fill([x.shape[0]], float(s)), err_db_from(e, x))
            r = evaluate(fns, he, h, ev)
            site_out["estime"][f"pilote{p}"] = r
            print(f"{site} pilote {p} dB : " + " | ".join(
                f"{k} {v['rate'][SNRS.index(20)]:.2f}@20 {v['rate'][SNRS.index(30)]:.2f}@30"
                for k, v in r.items()), flush=True)

        out["sites"][site] = site_out
        os.makedirs(os.path.join(RESULTS, "phase15"), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), indent=1)
        print(f"{site} termine ({time.time() - t0:.0f} s)", flush=True)

    print("->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main(errin_only="--errin-only" in sys.argv)
