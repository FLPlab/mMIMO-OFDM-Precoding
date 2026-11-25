"""Phase 13 -- deux verifications rapides (exploration, une graine), puissances reelles :

1. WMMSE selon le nombre d'iterations (1 a 300) contre notre precodeur, en CSI PARFAIT : faut-il
   beaucoup d'iterations a WMMSE pour atteindre notre precodeur ? Peut-on le depasser ?
   Sur les sites du labo (St-Catherine, Decarie, Ericsson, convention psi2az) et sur Sionna 3GPP (UMi).
2. Sondage SANS Sionna : LS sur le peigne = vrai canal aux SC pilotes + bruit gaussien, fait a la main
   (l'encodeur n'utilise que les SC pilotes). Si les debits sont les memes qu'avec le sondeur Sionna,
   Sionna n'apporte rien de cache a la chaine (seulement la grille et les pilotes standard).

    python -u -m experiments.phase13_checks
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import sound_power_controlled
from experiments.phase11_power import BATCH, SNRS, classical_rates, model_rates
from experiments.phase12_mix import load as load_mix
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from precoders.classical import wmmse
from utils import RESULTS, git_commit

ITERS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300]


def sound_by_hand(h, pilot_db, rng):
    """LS sur le peigne de Sionna (utilisateur k : SC k, k+K, ...) fait a la main, controle de puissance
    montant : bruit relatif a la puissance de chaque utilisateur. Les autres SC sont mises a zero
    (l'encodeur ne les lit pas)."""
    h = h.numpy()
    B, F, K, M = h.shape
    p = np.mean(np.abs(h) ** 2, axis=(1, 3), keepdims=True)
    n = (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) * np.sqrt(p * 10 ** (-pilot_db / 10) / 2)
    ls = np.zeros_like(h)
    for k in range(K):
        ls[:, k::K, k] = h[:, k::K, k] + n[:, k::K, k]
    return tf.constant(ls.astype(np.complex64))


def main():
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    net = load_mix("pilot_rand_ph32_emconv_norm_relpow", "conv")
    prec = net.precoder
    lin = UplinkSounder(cfg, "lin")
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS,
           "iters": ITERS, "wmmse_iters": {}, "sounder": {}}
    sets = {}
    for s in ("stecath", "decarie", "ericsson"):
        sets[s] = drops(read_cells(s, "psi2az"), freqs, equal=False, seed=13)
    c = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen = ChannelGenerator(c)
    sets["sionna_umi"] = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    for name, h in sets.items():
        r = {"notre precodeur": classical_rates(
            lambda x, nv, e: prec(x, tf.fill([x.shape[0]], float(-10 * np.log10(nv)))), h, h)}
        for it in ITERS:
            r[f"WMMSE {it} it."] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=it), h, h)
        out["wmmse_iters"][name] = r
        print(f"{name} CSI parfait @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
        rng = np.random.default_rng(7)
        out["sounder"][name] = {}
        for p in (0, 10):
            ls_s = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], float(p)) for j in range(0, DROPS, BATCH)], 0)
            ls_h = sound_by_hand(h, p, rng)
            idx = np.arange(cfg.num_sc) % 4                                  # SC pilotes de chaque utilisateur
            e_s = np.mean([np.mean(np.abs(ls_s.numpy()[:, idx == k, k] - h.numpy()[:, idx == k, k]) ** 2) for k in range(4)])
            e_h = np.mean([np.mean(np.abs(ls_h.numpy()[:, idx == k, k] - h.numpy()[:, idx == k, k]) ** 2) for k in range(4)])
            out["sounder"][name][p] = q = {"Sionna": model_rates(net, ls_s, h), "a la main": model_rates(net, ls_h, h),
                                           "bruit pilote Sionna": float(e_s), "bruit pilote a la main": float(e_h)}
            print(f"{name} pilote {p} dB @20dB : Sionna {q['Sionna'][2]:.2f} | a la main {q['a la main'][2]:.2f} "
                  f"(bruit {e_s:.2e} / {e_h:.2e})", flush=True)
    os.makedirs(os.path.join(RESULTS, "phase13"), exist_ok=True)
    path = os.path.join(RESULTS, "phase13", "checks.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
