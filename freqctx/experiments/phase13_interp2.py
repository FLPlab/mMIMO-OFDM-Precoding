"""Phase 13 -- (1) faut-il une meilleure interpolation entre ancres ? (2) les bords de la bande perdent-ils ?
Exploration, une graine, sans entrainement.

(1) CSI parfait, notre precodeur calcule sur 1 SC sur s, trois facons d'interpoler W :
    lineaire          : W(f) = (1 - t) W(a) + t W(a + s), puis renormalisation (methode actuelle) ;
    lineaire alignee  : avant d'interpoler, la phase de chaque colonne de W est alignee d'une ancre a la
                        suivante (le debit ne depend pas de la phase de chaque colonne : deux ancres peuvent
                        avoir des phases arbitraires qui s'annulent dans la moyenne) ;
    spline alignee    : spline cubique sur les ancres alignees.
    Sionna UMi (selectif) et St-Catherine en OFDM (presque plat), puissances reelles, 20 dB.
(2) Depuis les pilotes (pilote 10 dB, Sionna UMi) : debit par sous-porteuse, notre modele (encodeur
    convolutif) contre Wiener + WMMSE, pour voir si les bords de la bande perdent.

    python -u -m experiments.phase13_interp2
"""
import json
import os

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline
from sionna.phy.channel import subcarrier_frequencies
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase7_context import cov_mixture
from experiments.phase12_mix import load as load_mix
from experiments.phase12_robust import KAPPA, sound
from experiments.phase12_wiener import wiener
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS

STEPS, SNR, BATCH = [1, 3, 6, 12, 24, 48], 20.0, 16


def align(wa):
    """wa [B, A, M, K] : aligne la phase de chaque colonne sur l'ancre precedente (de proche en proche)."""
    wa = wa.copy()
    for j in range(1, wa.shape[1]):
        ip = np.sum(np.conj(wa[:, j - 1]) * wa[:, j], axis=1, keepdims=True)          # [B, 1, K]
        wa[:, j] *= np.exp(-1j * np.angle(ip))
    return wa


def interp(wa, anchors, F, how):
    f = np.arange(F)
    if how == "spline alignee":
        return CubicSpline(anchors, align(wa), axis=1)(f)
    if how == "lineaire alignee":
        wa = align(wa)
    j = np.clip(np.searchsorted(anchors, f, side="right") - 1, 0, len(anchors) - 2)
    t = ((f - anchors[j]) / (anchors[j + 1] - anchors[j]))[None, :, None, None]
    return (1 - t) * wa[:, j] + t * wa[:, j + 1]


def rate_interp(h, prec, s, how):
    B, F, K, M = h.shape
    anchors = np.unique(np.append(np.arange(0, F, s), F - 1))
    nv = 10 ** (-SNR / 10)
    out = []
    for b in range(0, B, BATCH):
        hb = h[b:b + BATCH]
        ha = tf.reshape(tf.gather(hb, anchors, axis=1), [-1, K, M])
        wa = prec(ha, tf.fill([ha.shape[0]], SNR)).numpy().reshape(hb.shape[0], len(anchors), M, K)
        w = interp(wa, anchors, F, how).reshape(-1, M, K)
        w = w / np.sqrt((np.abs(w) ** 2).sum((1, 2), keepdims=True))
        out.append(rate_per_sample(tf.reshape(hb, [-1, K, M]), tf.constant(w.astype(np.complex64)), nv))
    return float(tf.reduce_mean(tf.concat(out, 0)))


def main():
    cfg = SystemConfig()
    net = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv")
    sionna_config.seed = 4026
    c = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    gen = ChannelGenerator(c)
    h_umi = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    h_sc = drops(read_cells("stecath", "psi2az"), subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy(),
                 equal=False, seed=13)
    out = {"snr_db": SNR, "steps": STEPS, "interp": {}}
    for name, h in (("sionna_umi", h_umi), ("stecath", h_sc)):
        out["interp"][name] = {how: [rate_interp(h, net.precoder, s, how) for s in STEPS]
                               for how in ("lineaire", "lineaire alignee", "spline alignee")}
        print(name, {k: [round(v, 2) for v in x] for k, x in out["interp"][name].items()}, flush=True)
    # (2) debit par sous-porteuse depuis les pilotes
    ls, _, _ = sound(c, cov_mixture(), h_umi, 10.0)
    nv = 10 ** (-SNR / 10)
    w_m = tf.concat([net(ls[i:i + BATCH], tf.fill([BATCH], SNR)) for i in range(0, DROPS, BATCH)], 0)
    wn = wiener(ls, KAPPA)
    w_w = tf.reshape(wmmse(tf.reshape(wn, [-1, 4, 8]), nv, num_iter=100), w_m.shape)
    per_sc = {}
    for k, w in (("notre modele", w_m), ("Wiener + WMMSE", w_w)):
        r = rate_per_sample(tf.reshape(h_umi, [-1, 4, 8]), tf.reshape(w, [-1, 8, 4]), nv)
        per_sc[k] = tf.reduce_mean(tf.reshape(r, [DROPS, -1]), 0).numpy().tolist()
        p = np.array(per_sc[k])
        print(f"{k} : bords (12 premieres + 12 dernieres SC) {np.r_[p[:12], p[-12:]].mean():.2f} / milieu {p[12:-12].mean():.2f}",
              flush=True)
    out["per_sc_pilot10"] = per_sc
    json.dump(out, open(os.path.join(RESULTS, "phase13", "interp2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
