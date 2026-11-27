"""Phase 15 -- evaluation SEULE (aucun entrainement) de la variante passe 2 entrainee jusqu'a 40 dB
(poids weights/phase15/refine_L4_snr0-40.weights.h5, experiments/phase15_refine40.py) comparee a la
passe 2 U(0, 30) retenue (poids weights/phase15/refine_L4.weights.h5, experiments/phase15_refine.py,
results/phase15/refine.json) et a la base figee seule (GramPrecoder phase13). Reutilise (import
seulement, SANS MODIFIER) les memes tirages et fonctions que experiments/phase15_refine_eval.py :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf} et
    experiments.phase14_finetune.load_base                         (partie 1, CSI parfait)
  - experiments.phase15_lab_csi.{SITES, PILOTS, SNRS, B} et les memes briques que phase15_refine_eval
    (config.SystemConfig, csi.UplinkSounder, experiments.phase7_context.{cov_mixture,
    sound_power_controlled}, experiments.phase13_deepmimo.{DROPS, drops},
    experiments.phase13_lab.read_cells)                            (partie 2, CSI estime)
  - results/phase15/lab_csi.json pour WMMSE robuste 300 it. (PAS recalcule)             (partie 2)

Critere fixe a l'avance (prompt 26 nov. 2026), par rapport a la passe 2 U(0, 30) : RETENU si
>= +0.2 bps/Hz a 35-40 dB en moyenne sur les 3 sites ET pas plus de -0.05 entre 0 et 30 dB
(UMi et sites). Sinon ECARTE.

    python -u -m experiments.phase15_refine40_eval
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
OUT_PATH = os.path.join(OUT_DIR, "refine40.json")
LAB_CSI_PATH = os.path.join(OUT_DIR, "lab_csi.json")
REFINE30_WEIGHTS = "/export/tmp/sala/freqctx/weights/phase15/refine_L4.weights.h5"
REFINE40_WEIGHTS = "/export/tmp/sala/freqctx/weights/phase15/refine_L4_snr0-40.weights.h5"
SNRS_PARFAIT = [0, 10, 20, 25, 30, 35, 40]
N_EVAL = 16384
PILOTE_ESTIME = 20
SNRS_ESTIME = [20, 30]
SITES3 = ["stecath", "decarie", "ericsson"]


def load_refine(weights_path):
    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(weights_path)
    return net


# ---------------------------------------------------------------------- 1. CSI parfait


def run_parfait(net_base, net_r30, net_r40):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_refine_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            w_base = net_base(h, tf.fill([h.shape[0]], float(snr)))
            w_r30 = net_r30(h, tf.fill([h.shape[0]], float(snr)))
            w_r40 = net_r40(h, tf.fill([h.shape[0]], float(snr)))
            w_wmmse = wmmse(h, nv, num_iter=300)
            entry = {}
            for tag, w in (("base", w_base), ("base+passe2_u0-30", w_r30),
                           ("base+passe2_u0-40", w_r40), ("wmmse300", w_wmmse)):
                leak_db, _, _ = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                entry[tag] = {"debit": rate, "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"parfait {name:10s} {snr:2d}dB : base {entry['base']['debit']:6.2f}  "
                  f"passe2(u30) {entry['base+passe2_u0-30']['debit']:6.2f}  "
                  f"passe2(u40) {entry['base+passe2_u0-40']['debit']:6.2f} (fuite {entry['base+passe2_u0-40']['fuite_db_mediane']:6.1f})  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f}", flush=True)
    return out


# ---------------------------------------------------------------------- 2. CSI estime (pilote 20 dB)


def run_estime(net_base, net_r30, net_r40):
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
            w_r30 = net_r30(hef, tf.fill([hef.shape[0]], float(snr)))
            w_r40 = net_r40(hef, tf.fill([hef.shape[0]], float(snr)))
            rate_base = float(tf.reduce_mean(rate_per_sample(hf, w_base, nv)))
            rate_r30 = float(tf.reduce_mean(rate_per_sample(hf, w_r30, nv)))
            rate_r40 = float(tf.reduce_mean(rate_per_sample(hf, w_r40, nv)))
            idx = lab_csi["meta"]["snrs_db"].index(snr)
            rate_wmmse300 = lab_csi["sites"][site]["estime"][f"pilote{PILOTE_ESTIME}"]["WMMSE robuste 300 it."]["rate"][idx]
            entry[str(snr)] = {"reseau_csi_parfait": rate_base, "reseau_csi_parfait+passe2_u0-30": rate_r30,
                                "reseau_csi_parfait+passe2_u0-40": rate_r40,
                                "wmmse_robuste_300it_depuis_lab_csi_json": rate_wmmse300}
            print(f"estime  {site:10s} pilote{PILOTE_ESTIME} {snr:2d}dB : reseau {rate_base:6.2f}  "
                  f"+passe2(u30) {rate_r30:6.2f}  +passe2(u40) {rate_r40:6.2f}  "
                  f"wmmse300(lab_csi.json) {rate_wmmse300:6.2f}", flush=True)
        out[site] = entry
    return out


# ---------------------------------------------------------------------- verdict


def verdict(parfait):
    low_snr_sets = ["umi"] + SITES3
    gains_high = []
    for s in SITES3:
        for snr in (35, 40):
            if str(snr) not in parfait.get(s, {}):
                continue
            gains_high.append(parfait[s][str(snr)]["base+passe2_u0-40"]["debit"]
                               - parfait[s][str(snr)]["base+passe2_u0-30"]["debit"])
    gain_high_mean = float(np.mean(gains_high)) if gains_high else None
    low_losses = []
    for s in low_snr_sets:
        for snr in (0, 10, 20, 25, 30):
            if str(snr) not in parfait.get(s, {}):
                continue
            low_losses.append(parfait[s][str(snr)]["base+passe2_u0-40"]["debit"]
                               - parfait[s][str(snr)]["base+passe2_u0-30"]["debit"])
    worst_low = float(np.min(low_losses)) if low_losses else None
    ok_high = (gain_high_mean is not None) and (gain_high_mean >= 0.2)
    ok_low = (worst_low is not None) and (worst_low >= -0.05)
    retenu = bool(ok_high and ok_low)
    return {"gain_35_40db_moyen_3sites_vs_passe2_u030": gain_high_mean,
            "pire_variation_0_30db_vs_passe2_u030": worst_low,
            "criteres": {"gain_35_40db>=0.2": ok_high, "perte_0_30db>=-0.05": ok_low},
            "retenu": retenu}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_r30 = load_refine(REFINE30_WEIGHTS)
    net_r40 = load_refine(REFINE40_WEIGHTS)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait(net_base, net_r30, net_r40)
    out["csi_estime_pilote20"] = run_estime(net_base, net_r30, net_r40)
    out["verdict"] = verdict(out["csi_parfait"])
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
