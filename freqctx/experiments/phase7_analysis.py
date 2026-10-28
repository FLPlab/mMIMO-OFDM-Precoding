"""Phase 7 -- analyses sans entrainement du modele retenu (encodeur frequentiel
sur toute la bande + transformer par paires, phase 6, graine 0).

1. Que represente la sortie de l'encodeur ? Erreur quadratique normalisee
   (NMSE) par rapport au vrai canal : brute, apres un facteur complexe
   global par tirage, apres un facteur complexe par utilisateur et par
   tirage (les symetries auxquelles le debit est insensible). Comparee au LS
   et au LMMSE (melange). Debit de WMMSE sur la sortie realignee.
   + facteur complexe par utilisateur ET par SC (= erreur de DIRECTION du
   vecteur canal sur les M antennes, a laquelle le precodage est sensible).
2. Granularite : le precodeur calcule sur une SC par groupe de 1, 2 ou 4 RB
   (la SC du milieu) et reutilise sur tout le groupe (PRG en 5G) -- calcul
   du precodeur divise par 12, 24, 48. Meme chose pour WMMSE(LMMSE).
   Variantes qui gardent la representation par SC : seuls les coefficients C
   (la partie couteuse, W = h^H C) sont calcules par groupe, tenus constants
   ou interpoles lineairement entre les centres de groupe (idee de TA-RB).

    python -u -m experiments.phase7_analysis
"""
import json
import os
import time

import numpy as np
import tensorflow as tf

from config import CSIConfig, SystemConfig
from experiments.phase7_context import PILOT, SCENARIOS, cov_mixture, flat
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import normalize_power, wmmse
from train import csi_pool, rate_per_sample
from utils import RESULTS, git_commit

W4 = "/export/tmp/sala/freqctx/weights/phase4"
WEIGHTS = "delay_bande_os2_trigL8_untied_n1056_s80k_est32_D64_T4_linear_melange_p10.weights.h5"
SNRS = [0, 10, 20, 25]
GROUPS = {"SC": 1, "1 RB": 12, "2 RB": 24, "4 RB": 48}
EVAL_DROPS = 192


def nmse_db(e, h, axis=None):
    return float(10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(h) ** 2)))


def align(x, h, per_user):
    """Facteur complexe optimal a = <x, h> / <x, x>, par tirage (et par utilisateur)."""
    ax = (1, 3) if per_user else (1, 2, 3)                     # x, h [B, F, K, M]
    a = np.sum(np.conj(x) * h, axis=ax, keepdims=True) / np.sum(np.abs(x) ** 2, axis=ax, keepdims=True)
    return a * x


def align_sc(x, h):
    """Facteur complexe par (tirage, SC, utilisateur) : ne garde que la direction sur les M antennes."""
    a = np.sum(np.conj(x) * h, axis=-1, keepdims=True) / np.sum(np.abs(x) ** 2, axis=-1, keepdims=True)
    return a * x


def interp_centers(c, g):
    """c [B, F, ...] : valeurs aux centres des groupes de g SC, interpolees lineairement entre centres."""
    F = c.shape[1]
    centers = np.minimum(np.arange(0, F, g) + g // 2, F - 1)
    pos = np.clip(np.arange(F), centers[0], centers[-1])
    j = np.clip(np.searchsorted(centers, pos, side="right") - 1, 0, len(centers) - 2)
    t = ((pos - centers[j]) / np.maximum(centers[j + 1] - centers[j], 1)).astype(np.float32)
    shape = [1, F] + [1] * (len(c.shape) - 2)
    t = tf.cast(tf.reshape(t, shape), c.dtype)
    return (1 - t) * tf.gather(c, centers[j], axis=1) + t * tf.gather(c, centers[j + 1], axis=1)


def group_center(w, g):
    """w [B, F, ...] -> precodeur de la SC du milieu de chaque groupe de g SC, reutilise."""
    F = w.shape[1]
    idx = np.minimum((np.arange(F) // g) * g + g // 2, F - 1)
    return tf.gather(w, idx, axis=1)


def rate(h, w, snr):
    nv = 10.0 ** (-snr / 10.0)
    return float(tf.reduce_mean(rate_per_sample(flat(h), tf.reshape(w, [-1, w.shape[-2], w.shape[-1]]), nv)))


def main():
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", d=16, hidden=64, layers=8, mix="triangular_g", tied=False)
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "weights": WEIGHTS, "snrs_db": SNRS, "nmse_db": {}, "gauge_rates": {}, "granularity": {}}
    for i, scen in enumerate(SCENARIOS):
        c = SystemConfig(scenario=scen)
        ls, h = csi_pool(c, CSIConfig("ls_lin", PILOT, 0), EVAL_DROPS, 2026 + 10 * i, flat=False)
        lm, _ = csi_pool(c, CSIConfig("ls_lmmse", PILOT, 0), EVAL_DROPS, 2026 + 10 * i, cov_freq=cov, flat=False)
        if not net.built:
            net(ls[:1], tf.zeros([1]))
            net.load_weights(os.path.join(W4, WEIGHTS))
        enc = np.concatenate([net.encode(ls[j:j + 16]).numpy() for j in range(0, EVAL_DROPS, 16)])
        hn, lsn, lmn = h.numpy(), ls.numpy(), lm.numpy()
        g_al, u_al = align(enc, hn, False), align(enc, hn, True)
        out["nmse_db"][scen] = {"LS": nmse_db(lsn - hn, hn), "LMMSE (melange)": nmse_db(lmn - hn, hn),
                                "encodeur brut": nmse_db(enc - hn, hn),
                                "encodeur, facteur global": nmse_db(g_al - hn, hn),
                                "encodeur, facteur par utilisateur": nmse_db(u_al - hn, hn),
                                "LMMSE, facteur par utilisateur": nmse_db(align(lmn, hn, True) - hn, hn),
                                "encodeur, direction (facteur par utilisateur et SC)": nmse_db(align_sc(enc, hn) - hn, hn),
                                "LMMSE, direction": nmse_db(align_sc(lmn, hn) - hn, hn),
                                "LS, direction": nmse_db(align_sc(lsn, hn) - hn, hn),
                                "echelle moyenne |encodeur| / |h|": float(np.sqrt(np.mean(np.abs(enc) ** 2)
                                                                                   / np.mean(np.abs(hn) ** 2)))}
        print(scen, {k: round(v, 2) for k, v in out["nmse_db"][scen].items()}, flush=True)
        ua = tf.constant(u_al, tf.complex64)
        out["gauge_rates"][scen], out["granularity"][scen] = {}, {}
        for snr in SNRS:
            nv = 10.0 ** (-snr / 10.0)
            w_mod = tf.concat([net(ls[j:j + 16], tf.fill([16], float(snr))) for j in range(0, EVAL_DROPS, 16)], 0)
            w_al = tf.concat([tf.reshape(wmmse(flat(ua[j:j + 16]), nv), [-1, cfg.num_sc, 8, 4])
                              for j in range(0, EVAL_DROPS, 16)], 0)
            w_lm = tf.concat([tf.reshape(wmmse(flat(lm[j:j + 16]), nv), [-1, cfg.num_sc, 8, 4])
                              for j in range(0, EVAL_DROPS, 16)], 0)
            out["gauge_rates"][scen][snr] = {"modele": rate(h, w_mod, snr),
                                             "WMMSE sur encodeur realigne": rate(h, w_al, snr),
                                             "WMMSE (LMMSE melange)": rate(h, w_lm, snr)}
            hh = tf.constant(enc, tf.complex64)
            snr_v = tf.fill([cfg.num_sc], float(snr))
            c_all = tf.stack([net.precoder.coeffs(hh[b], snr_v) for b in range(EVAL_DROPS)])   # [B, F, K, K]

            def w_from(c):
                return normalize_power(tf.matmul(hh, c, adjoint_a=True))
            out["granularity"][scen][snr] = {
                g: {"modele": rate(h, group_center(w_mod, n), snr),
                    "modele, C tenu par groupe": rate(h, w_from(group_center(c_all, n)), snr),
                    "modele, C interpole": rate(h, w_from(interp_centers(c_all, n)), snr),
                    "WMMSE (LMMSE melange)": rate(h, group_center(w_lm, n), snr)} for g, n in GROUPS.items()}
            print(f"  {scen} {snr} dB", {k: round(v, 2) for k, v in out["gauge_rates"][scen][snr].items()},
                  {g: [round(x, 2) for x in v.values()] for g, v in out["granularity"][scen][snr].items()}, flush=True)
    os.makedirs(os.path.join(RESULTS, "phase7"), exist_ok=True)
    path = os.path.join(RESULTS, "phase7", "analysis.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
