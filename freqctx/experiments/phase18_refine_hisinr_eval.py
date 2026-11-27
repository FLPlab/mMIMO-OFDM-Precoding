"""Phase 18 -- EVALUATION SEULE (aucun entrainement) de la passe 2 reentrainee sur le melange
enrichi en SINR eleve (experiments/phase18_refine_hisinr.py, poids
weights/phase18/refine_L4_hisinr.weights.h5), comparee a la passe 2 ACTUELLE (weights/phase15/
refine_L4.weights.h5) et a WMMSE 300 it., appliquees 1 et 2 fois (meme generalisation n passes que
experiments/phase18_refine_npass.py, demande F. Leduc-Primeau 2026-11-27).

Critere fixe a l'avance (prompt 26 nov. 2026) : RETENU si Ericsson 30 dB >= 36.14 (WMMSE - 0.05)
en zero-shot (nouvelle passe, 1x OU 2x) ET aucune perte > 0.05 face a la passe 2 actuelle ailleurs
(UMi, St-Catherine, Decarie, 10-40 dB, comparaison a meme nombre de passes : 1x contre 1x, 2x
contre 2x). Indique l'ecart restant a 40 dB (meilleure config nouvelle passe contre WMMSE 300).

Reutilise (import seulement, SANS MODIFIER) :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf},
    experiments.phase14_finetune.load_base                                    (CSI parfait)
  - experiments.phase15_refine_eval.{SNRS_PARFAIT, N_EVAL, load_refine, PILOTE_ESTIME,
    SNRS_ESTIME, LAB_CSI_PATH}                                                (CSI estime, pilote 20dB)
  - experiments.phase15_lab_csi.{SITES, B} + experiments.phase7_context.{cov_mixture,
    sound_power_controlled} + experiments.phase13_deepmimo.{DROPS, drops} +
    experiments.phase13_lab.read_cells                                       (CSI estime)
  - experiments.phase18_refine_npass.npass_coeffs                             (n passes, import SANS MODIFIER)
  - models.refine.RefineNet, precoders.classical.wmmse, train.rate_per_sample

    python -u -m experiments.phase18_refine_hisinr_eval
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
from experiments.phase15_lab_csi import SITES as LAB_SITES
from experiments.phase15_refine_eval import LAB_CSI_PATH, N_EVAL, PILOTE_ESTIME, SNRS_ESTIME, SNRS_PARFAIT, load_refine
from experiments.phase18_refine_npass import npass_coeffs
from experiments.phase7_context import cov_mixture, sound_power_controlled
from models.refine import RefineNet
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase18")
OUT_PATH = os.path.join(OUT_DIR, "refine_hisinr_eval.json")
HISINR_WEIGHTS = "/export/tmp/sala/freqctx/weights/phase18/refine_L4_hisinr.weights.h5"
N_PASSES = [1, 2]
SITES3 = ["stecath", "decarie", "ericsson"]
ERICSSON_30DB_SEUIL = 36.14               # WMMSE - 0.05
LOSS_BUDGET = 0.05


def load_hisinr():
    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(HISINR_WEIGHTS)
    return net


# ---------------------------------------------------------------------- 1. CSI parfait


def run_parfait(net_base, net_cur, net_new):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_refine_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            snr_t = tf.fill([h.shape[0]], float(snr))
            w_base = net_base(h, snr_t)
            w_wmmse = wmmse(h, nv, num_iter=300)
            entry = {}
            for tag, w in (("base", w_base), ("wmmse300", w_wmmse)):
                leak_db, _, _ = leak_db_sig_intf(h, w)
                entry[tag] = {"debit": float(tf.reduce_mean(rate_per_sample(h, w, nv))),
                              "fuite_db_mediane": float(np.median(leak_db))}
            for n in N_PASSES:
                w_cur = npass_coeffs(net_cur, h, snr_t, n)
                w_new = npass_coeffs(net_new, h, snr_t, n)
                for tag, w in ((f"actuelle_{n}x", w_cur), (f"hisinr_{n}x", w_new)):
                    leak_db, _, _ = leak_db_sig_intf(h, w)
                    entry[tag] = {"debit": float(tf.reduce_mean(rate_per_sample(h, w, nv))),
                                  "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"parfait {name:10s} {snr:2d}dB : base {entry['base']['debit']:6.2f}  "
                  f"actuelle 1x/2x {entry['actuelle_1x']['debit']:6.2f}/{entry['actuelle_2x']['debit']:6.2f}  "
                  f"hisinr 1x/2x {entry['hisinr_1x']['debit']:6.2f}/{entry['hisinr_2x']['debit']:6.2f}  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f}", flush=True)
    return out


# ---------------------------------------------------------------------- 2. CSI estime (pilote 20 dB)


def run_estime(net_cur, net_new):
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
            snr_t = tf.fill([hef.shape[0]], float(snr))
            idx = lab_csi["meta"]["snrs_db"].index(snr)
            rate_wmmse300 = lab_csi["sites"][site]["estime"][f"pilote{PILOTE_ESTIME}"]["WMMSE robuste 300 it."]["rate"][idx]
            row = {"wmmse_robuste_300it": rate_wmmse300}
            for n in N_PASSES:
                w_cur = npass_coeffs(net_cur, hef, snr_t, n)
                w_new = npass_coeffs(net_new, hef, snr_t, n)
                row[f"actuelle_{n}x"] = float(tf.reduce_mean(rate_per_sample(hf, w_cur, nv)))
                row[f"hisinr_{n}x"] = float(tf.reduce_mean(rate_per_sample(hf, w_new, nv)))
            entry[str(snr)] = row
            print(f"estime  {site:10s} pilote{PILOTE_ESTIME} {snr:2d}dB : "
                  f"actuelle 1x/2x {row['actuelle_1x']:6.2f}/{row['actuelle_2x']:6.2f}  "
                  f"hisinr 1x/2x {row['hisinr_1x']:6.2f}/{row['hisinr_2x']:6.2f}  "
                  f"wmmse300 {rate_wmmse300:6.2f}", flush=True)
        out[site] = entry
    return out


# ---------------------------------------------------------------------- verdict


def verdict(parfait):
    ericsson_30 = {n: parfait["ericsson"]["30"][f"hisinr_{n}x"]["debit"] for n in N_PASSES}
    ok_ericsson = any(v >= ERICSSON_30DB_SEUIL for v in ericsson_30.values())

    pertes = []                                   # (new_nx - cur_nx) a meme n, hors ericsson, 10-40dB
    for name in ["umi", "stecath", "decarie"]:
        if name not in parfait:
            continue
        for snr_s, entry in parfait[name].items():
            if snr_s == "n" or int(snr_s) < 10 or int(snr_s) > 40:
                continue
            for n in N_PASSES:
                diff = entry[f"hisinr_{n}x"]["debit"] - entry[f"actuelle_{n}x"]["debit"]
                pertes.append({"site": name, "snr": int(snr_s), "n": n, "diff": diff})
    worst = min(pertes, key=lambda r: r["diff"]) if pertes else None
    ok_no_loss = (worst is None) or (worst["diff"] >= -LOSS_BUDGET)

    ecart_40db = {n: parfait["ericsson"]["40"]["wmmse300"]["debit"] - parfait["ericsson"]["40"][f"hisinr_{n}x"]["debit"]
                  for n in N_PASSES}
    retenu = bool(ok_ericsson and ok_no_loss)
    return {"ericsson_30db_hisinr_1x_2x": ericsson_30, "seuil_ericsson_30db": ERICSSON_30DB_SEUIL,
            "ok_ericsson_30db": ok_ericsson, "pire_perte_vs_actuelle": worst, "ok_pas_de_perte_0_05": ok_no_loss,
            "ecart_wmmse300_moins_hisinr_40db_ericsson": ecart_40db, "retenu": retenu}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_cur = load_refine()
    net_new = load_hisinr()

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait(net_base, net_cur, net_new)
    out["csi_estime_pilote20"] = run_estime(net_cur, net_new)
    out["verdict"] = verdict(out["csi_parfait"])
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
