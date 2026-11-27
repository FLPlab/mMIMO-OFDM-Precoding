"""Phase 15 -- evaluation SEULE (aucun entrainement) de la passe 2 (RefineNet, models/refine.py,
poids weights/phase15/refine_L4.weights.h5) ajoutee a la base figee (GramPrecoder phase13).
Reutilise (import seulement, SANS MODIFIER) :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf} et
    experiments.phase14_finetune.load_base                         (partie 1, CSI parfait)
  - experiments.phase15_lab_csi.{SITES, PILOTS, SNRS, B} et les memes briques que ce script
    (config.SystemConfig, csi.UplinkSounder, experiments.phase7_context.{cov_mixture,
    sound_power_controlled}, experiments.phase13_deepmimo.{DROPS, drops},
    experiments.phase13_lab.read_cells)                            (partie 2, CSI estime)
  - results/phase15/lab_csi.json pour WMMSE robuste 300 it. (PAS recalcule)             (partie 2)
  - experiments.phase15_macs (convention MAC) + RefineNet.macs_per_sc                   (partie 3)

Criere fixe a l'avance (voir prompt 26 nov. 2026) : RETENU si >= +0.2 bps/Hz a 30 dB en CSI
parfait (moyenne des 3 sites) ET pas plus de -0.05 entre 0 et 20 dB (UMi et sites) ET perte
reduite en CSI estime a pilote 20 dB. Sinon ECARTE.

    python -u -m experiments.phase15_refine_eval
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from csi import UplinkSounder
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from experiments.phase14_finetune import load_base
from experiments.phase15_diag import get_datasets, leak_db_sig_intf, subsample
from experiments.phase15_lab_csi import B as LAB_B
from experiments.phase15_lab_csi import PILOTS as LAB_PILOTS
from experiments.phase15_lab_csi import SITES as LAB_SITES
from experiments.phase15_lab_csi import SNRS as LAB_SNRS
from experiments.phase7_context import cov_mixture, sound_power_controlled
from models.refine import RefineNet
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase15")
OUT_PATH = os.path.join(OUT_DIR, "refine.json")
LAB_CSI_PATH = os.path.join(OUT_DIR, "lab_csi.json")
REFINE_WEIGHTS = "/export/tmp/sala/freqctx/weights/phase15/refine_L4.weights.h5"
SNRS_PARFAIT = [0, 10, 20, 25, 30, 35, 40]
N_EVAL = 16384
PILOTE_ESTIME = 20                                  # seul pilote demande pour la partie 2
SNRS_ESTIME = [20, 30]


def load_refine():
    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(REFINE_WEIGHTS)
    return net


# ---------------------------------------------------------------------- 1. CSI parfait


def run_parfait(net_base, net_refine):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_diag/snr40_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            w_base = net_base(h, tf.fill([h.shape[0]], float(snr)))
            w_refine = net_refine(h, tf.fill([h.shape[0]], float(snr)))
            w_wmmse = wmmse(h, nv, num_iter=300)
            entry = {}
            for tag, w in (("base", w_base), ("base+passe2", w_refine), ("wmmse300", w_wmmse)):
                leak_db, _, _ = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                entry[tag] = {"debit": rate, "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"parfait {name:10s} {snr:2d}dB : base {entry['base']['debit']:6.2f} (fuite {entry['base']['fuite_db_mediane']:6.1f})  "
                  f"base+passe2 {entry['base+passe2']['debit']:6.2f} (fuite {entry['base+passe2']['fuite_db_mediane']:6.1f})  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f} (fuite {entry['wmmse300']['fuite_db_mediane']:6.1f})", flush=True)
    return out


# ---------------------------------------------------------------------- 2. CSI estime (pilote 20 dB)


def run_estime(net_base, net_refine):
    assert os.path.exists(LAB_CSI_PATH), f"{LAB_CSI_PATH} absent -- lance d'abord experiments.phase15_lab_csi"
    lab_csi = json.load(open(LAB_CSI_PATH))
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    cov = cov_mixture()
    snd = UplinkSounder(cfg, "lmmse", cov)

    out = {}
    for site in LAB_SITES:
        users = read_cells(site, "psi2az")
        h = drops(users, freqs, equal=False, seed=13)                              # meme graine que phase15_lab_csi
        pairs = [sound_power_controlled(snd, h[j:j + LAB_B], float(PILOTE_ESTIME), return_err_var=True)
                 for j in range(0, DROPS, LAB_B)]
        he = tf.concat([a for a, _ in pairs], 0)
        hf, hef = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]]), tf.reshape(he, [-1, h.shape[-2], h.shape[-1]])

        entry = {}
        for snr in SNRS_ESTIME:
            nv = 10.0 ** (-snr / 10.0)
            w_base = net_base(hef, tf.fill([hef.shape[0]], float(snr)))
            w_refine = net_refine(hef, tf.fill([hef.shape[0]], float(snr)))
            rate_base = float(tf.reduce_mean(rate_per_sample(hf, w_base, nv)))
            rate_refine = float(tf.reduce_mean(rate_per_sample(hf, w_refine, nv)))
            idx = lab_csi["meta"]["snrs_db"].index(snr)
            rate_wmmse300 = lab_csi["sites"][site]["estime"][f"pilote{PILOTE_ESTIME}"]["WMMSE robuste 300 it."]["rate"][idx]
            entry[str(snr)] = {"reseau_csi_parfait": rate_base, "reseau_csi_parfait+passe2": rate_refine,
                                "wmmse_robuste_300it_depuis_lab_csi_json": rate_wmmse300}
            print(f"estime  {site:10s} pilote{PILOTE_ESTIME} {snr:2d}dB : reseau {rate_base:6.2f}  "
                  f"reseau+passe2 {rate_refine:6.2f}  wmmse300(lab_csi.json) {rate_wmmse300:6.2f}", flush=True)
        out[site] = entry
    return out


# ---------------------------------------------------------------------- 3. MAC passe 2


def run_macs(net_refine):
    base = net_refine.base
    K, M = 4, 8                                           # M8K4, meme convention que phase15_macs.py
    mac_base = int(base.macs_per_sc(K, M))
    mac_passe2 = int(net_refine.macs_per_sc(K, M))
    entry = {"K": K, "M": M, "base_macs_par_sc": mac_base, "passe2_macs_par_sc": mac_passe2,
              "total_macs_par_sc": mac_base + mac_passe2}
    print(f"MAC/SC (M{M}K{K}) : base {mac_base}  passe2 {mac_passe2}  total {entry['total_macs_par_sc']}", flush=True)
    return entry


# ---------------------------------------------------------------------- verdict


def verdict(parfait, estime):
    sites3 = ["stecath", "decarie", "ericsson"]
    gain_30 = [parfait[s]["30"]["base+passe2"]["debit"] - parfait[s]["30"]["base"]["debit"] for s in sites3]
    gain_30_mean = float(np.mean(gain_30))
    low_snr_sets = ["umi"] + sites3
    low_losses = []
    for s in low_snr_sets:
        for snr in (0, 10, 20):
            if str(snr) not in parfait.get(s, {}):
                continue
            low_losses.append(parfait[s][str(snr)]["base+passe2"]["debit"] - parfait[s][str(snr)]["base"]["debit"])
    worst_low = float(np.min(low_losses)) if low_losses else None
    estime_gains = []
    for s in sites3:
        for snr in map(str, SNRS_ESTIME):
            e = estime[s][snr]
            estime_gains.append(e["reseau_csi_parfait+passe2"] - e["reseau_csi_parfait"])
    estime_gain_min = float(np.min(estime_gains)) if estime_gains else None
    ok_30 = gain_30_mean >= 0.2
    ok_low = (worst_low is not None) and (worst_low >= -0.05)
    ok_estime = (estime_gain_min is not None) and (estime_gain_min > 0)
    retenu = bool(ok_30 and ok_low and ok_estime)
    return {"gain_30db_moyen_3sites": gain_30_mean, "pire_perte_0_20db": worst_low,
            "pire_variation_estime_pilote20": estime_gain_min,
            "criteres": {"gain_30db>=0.2": ok_30, "perte_0_20db>=-0.05": ok_low, "perte_estime_reduite": ok_estime},
            "retenu": retenu}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_refine = load_refine()

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait(net_base, net_refine)
    out["csi_estime_pilote20"] = run_estime(net_base, net_refine)
    out["macs"] = run_macs(net_refine)
    out["verdict"] = verdict(out["csi_parfait"], out["csi_estime_pilote20"])
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
