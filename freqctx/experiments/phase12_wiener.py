"""Phase 12 -- Wiener dans le domaine des retards en FORMULE FERMEE (models/delay_domain.WienerDelayEstimator).
Meme grille 2x que la base v2, masque max(P - kappa s2, 0)/P avec profil et bruit estimes sur le tirage,
aucun parametre appris. Cadre de la phase 11 (puissances reelles, controle de puissance montant, pilote
10 dB, 192 tirages / scenario, memes tirages).

1. Choix de kappa sur des tirages d'ENTRAINEMENT (graines 1+i, 32 tirages / scenario) : WMMSE 30 it.
2. Evaluation : Wiener + RZF, Wiener + WMMSE 100 it. (chaine classique sans apprentissage) ;
   si les poids existent, Wiener + notre precodeur (--variant wiener de phase7_context) et la base v2
   (controle : doit redonner la phase 11).

    python -u -m experiments.phase12_wiener [--kappa K]   # sans --kappa : choix sur l'entrainement
"""
import argparse
import json
import os
import time

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase7_context import PRECODER, SCENARIOS, W7, sound_power_controlled
from experiments.phase11_power import BATCH, DROPS, PILOT, SNRS, classical_rates, load, model_rates
from models.delay_domain import DelayDomainPrecoder, WienerDelayEstimator
from precoders.classical import rzf, wmmse
from utils import RESULTS, git_commit

KAPPAS = [0.5, 1.0, 2.0, 4.0]


def draws(scen, i, seed, n):
    cfg = SystemConfig(scenario=scen, num_slots=1, relative_power=True)
    sionna_config.seed = seed
    gen, lin = ChannelGenerator(cfg), UplinkSounder(cfg, "lin")
    h = tf.concat([gen(BATCH)[:, 0] for _ in range(n // BATCH)], 0)
    ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], PILOT) for j in range(0, n, BATCH)], 0)
    return ls, h


def wiener(ls, kappa):
    cfg = SystemConfig()
    est = WienerDelayEstimator(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, kappa=kappa)
    return tf.concat([est(ls[j:j + BATCH]) for j in range(0, ls.shape[0], BATCH)], 0)


def load_wiener_net(kappa):
    """Wiener en formule fermee + notre precodeur (--variant wiener de phase7_context), ou None."""
    path = os.path.join(W7, f"wiener_k{kappa:g}_ph32_norm_relpow.weights.h5")
    if not os.path.exists(path):
        return None
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", wiener_kappa=kappa, **dict(PRECODER, hidden=32, inp="norm"))
    net(tf.zeros([1, cfg.num_sc, cfg.num_users, cfg.num_bs_ant], tf.complex64), tf.zeros([1]))
    net.load_weights(path)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=None)
    a = ap.parse_args()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "snrs_db": SNRS, "pilot_snr_db": PILOT}
    kappa = a.kappa
    if kappa is None:                                      # choix sur des tirages d'entrainement
        tr = [draws(s, i, 1 + i, 32) for i, s in enumerate(SCENARIOS)]
        sel = {}
        for k in KAPPAS:
            r = [classical_rates(lambda x, nv, e: wmmse(x, nv), wiener(ls, k), h) for ls, h in tr]
            sel[k] = sum(x[2] for x in r) / len(r)
            print(f"kappa {k} : WMMSE 30 it. a 20 dB, moyenne des scenarios {sel[k]:.2f}", flush=True)
        kappa = max(sel, key=sel.get)
        out["kappa_selection_wmmse30_20db"] = sel
    out["kappa"] = kappa
    nets = {"base v2 (encodeur appris)": load("pilot_rand_ph32_norm_relpow", "norm")}
    net = load_wiener_net(kappa)
    if net is not None:
        nets["Wiener formule fermee + notre precodeur"] = net
        out["macs_per_sc"] = {k: int(n.macs_per_sc(4)) for k, n in nets.items()}
        out["params"] = {k: int(n.count_params()) for k, n in nets.items()}
    out["rates"] = {}
    for i, scen in enumerate(SCENARIOS):
        ls, h = draws(scen, i, 4026 + 10 * i, DROPS)        # memes tirages que la phase 11
        wn = wiener(ls, kappa)
        r = {k: model_rates(n, ls, h) for k, n in nets.items()}
        r["Wiener formule fermee + RZF"] = classical_rates(lambda x, nv, e: rzf(x, nv), wn, h)
        r["Wiener formule fermee + WMMSE 100 it."] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100), wn, h)
        out["rates"][scen] = r
        print(f"{scen} @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)
    os.makedirs(os.path.join(RESULTS, "phase12"), exist_ok=True)
    path = os.path.join(RESULTS, "phase12", "wiener.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
