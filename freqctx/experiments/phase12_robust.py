"""Phase 12 -- robustesse de la reference Wiener (formule fermee) face a la base v2, SANS reentrainer.
Cadre de la phase 11 (puissances reelles, controle de puissance montant), 192 tirages / scenario :
1. SNR pilote 0, 5, 10, 15, 20 dB (donnees au slot du sondage) ;
2. vieillissement : donnees 0 / 2,5 / 5 ms apres le sondage (pilote 10 dB) ;
3. puissances egales (test standard des phases 3-8), pilote 10 dB.
Methodes : base v2, Wiener + notre precodeur, Wiener + WMMSE 100 it., LMMSE (melange) + WMMSE robuste
100 it. (variance d'erreur annoncee par Sionna, qui ignore le vieillissement). Kappa = 4 (phase 12).

    python -u -m experiments.phase12_robust --scen umi|uma|rma      # un scenario par GPU
    python -u -m experiments.phase12_robust --merge
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
from experiments.phase7_context import SCENARIOS, cov_mixture, sound_power_controlled
from experiments.phase11_power import BATCH, DROPS, SNRS, classical_rates, load, model_rates
from experiments.phase12_wiener import load_wiener_net, wiener
from precoders.classical import wmmse
from utils import RESULTS, git_commit

KAPPA, PILOTS, DELAYS_MS = 4.0, [0, 5, 10, 15, 20], [0.0, 2.5, 5.0]
OUT = os.path.join(RESULTS, "phase12")


def sound(cfg, cov, h, pilot):
    lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
    ls = tf.concat([sound_power_controlled(lin, h[j:j + BATCH], pilot) for j in range(0, DROPS, BATCH)], 0)
    pairs = [sound_power_controlled(lmm, h[j:j + BATCH], pilot, return_err_var=True) for j in range(0, DROPS, BATCH)]
    return ls, tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)


def rates(nets, ls, lm, ev, h):
    r = {k: model_rates(n, ls, h) for k, n in nets.items()}
    r["Wiener formule fermee + WMMSE 100 it."] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100),
                                                                 wiener(ls, KAPPA), h)
    r["LMMSE + WMMSE robuste 100 it."] = classical_rates(lambda x, nv, e: wmmse(x, nv, num_iter=100, err_var=e),
                                                         lm, h, ev)
    return r


def show(tag, r):
    print(f"{tag} @20dB : " + " | ".join(f"{k} {v[2]:.2f}" for k, v in r.items()), flush=True)


def run(scen):
    i = SCENARIOS.index(scen)
    nets = {"base v2": load("pilot_rand_ph32_norm_relpow", "norm"),
            "Wiener formule fermee + notre precodeur": load_wiener_net(KAPPA)}
    cov = cov_mixture()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "kappa": KAPPA},
           "snrs_db": SNRS, "pilot_sweep": {}, "aging": {}}
    cfg = SystemConfig(scenario=scen, num_slots=1, relative_power=True)
    sionna_config.seed = 4026 + 10 * i                     # tirages de la phase 11
    gen = ChannelGenerator(cfg)
    h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    for p in PILOTS:
        out["pilot_sweep"][p] = r = rates(nets, *sound(cfg, cov, h, float(p)), h)
        show(f"{scen} pilote {p} dB", r)
    cfg = SystemConfig(scenario=scen, num_slots=3, slot_step=5, relative_power=True)
    sionna_config.seed = 5026 + 10 * i
    hs = tf.concat([ChannelGenerator(cfg)(BATCH) for _ in range(DROPS // BATCH)], 0)   # [N, 3, F, K, M]
    ls, lm, ev = sound(cfg, cov, hs[:, 0], 10.0)
    for s, dms in enumerate(DELAYS_MS):
        out["aging"][dms] = r = rates(nets, ls, lm, ev, hs[:, s])
        show(f"{scen} retard {dms} ms", r)
    cfg = SystemConfig(scenario=scen, num_slots=1)
    sionna_config.seed = 2026 + 10 * i                     # tirages standard des phases 3-8
    gen = ChannelGenerator(cfg)
    h = tf.concat([gen(BATCH)[:, 0] for _ in range(DROPS // BATCH)], 0)
    out["equal_power"] = r = rates(nets, *sound(cfg, cov, h, 10.0), h)
    show(f"{scen} puissances egales", r)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"robust_{scen}.json"), "w"), indent=1)


def merge():
    parts = {s: json.load(open(os.path.join(OUT, f"robust_{s}.json"))) for s in SCENARIOS}
    out = {"meta": parts[SCENARIOS[0]]["meta"], "snrs_db": SNRS, "scenarios": parts}
    json.dump(out, open(os.path.join(OUT, "robust.json"), "w"), indent=1)
    for s in SCENARIOS:
        os.remove(os.path.join(OUT, f"robust_{s}.json"))
    print("->", os.path.join(OUT, "robust.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", choices=SCENARIOS)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a.scen)
