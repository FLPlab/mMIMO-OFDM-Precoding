"""Phase 18 -- EVALUATION SEULE (aucun entrainement) : la passe 2 ACTUELLE (RefineNet,
models/refine.py, poids weights/phase15/refine_L4.weights.h5, base figee phase13) appliquee 3 et
4 fois (en plus de 1 et 2, pour comparaison), memes poids, sans reentrainer. Generalise
experiments.phase15_refine_more.two_pass_coeffs (import SANS MODIFIER ce fichier) a n passes :
C_{i+1} = C_i + DELTA_C(G, E_i), E_i = canal effectif de C_i (RefineNet._delta_from reconstruit E
depuis le c1 fourni -- meme normalisation a chaque rappel).

Question (demande F. Leduc-Primeau, 2026-11-27) : au-dela de 2 passes, le gain continue-t-il ou
sature/oscille-t-il ? Memes jeux et tirages que phase15_refine_eval (UMi + 3 sites du labo,
N_EVAL=16384, seed=4242) et phase15_refine_more (SNRS_PARFAIT = [0,10,20,25,30,35,40]).

Reutilise (import seulement, SANS MODIFIER) : experiments.phase15_diag.{get_datasets, subsample,
leak_db_sig_intf}, experiments.phase14_finetune.load_base, experiments.phase15_refine_eval.{
SNRS_PARFAIT, N_EVAL, load_refine}, precoders.classical.{wmmse, normalize_power}, train.rate_per_sample.

    python -u -m experiments.phase18_refine_npass
"""
import json
import os
import time

import numpy as np
import tensorflow as tf

from experiments.phase14_finetune import load_base
from experiments.phase15_diag import get_datasets, leak_db_sig_intf, subsample
from experiments.phase15_refine_eval import N_EVAL, SNRS_PARFAIT, load_refine
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase18")
OUT_PATH = os.path.join(OUT_DIR, "refine_npass.json")
N_PASSES = [1, 2, 3, 4]


def npass_coeffs(net_refine, h, snr_db, n):
    """C_n obtenu en rappelant n fois RefineNet._delta_from (non modifie) : C_0 = C1 (base),
    C_{i+1} = C_i + DELTA_C(G, E_i) ou E_i est reconstruit depuis C_i a chaque rappel (meme
    normalisation que pour obtenir C1->C2 la premiere fois). n=1 == net_refine(h, snr_db)
    exactement ; n=2 == experiments.phase15_refine_more.two_pass_coeffs."""
    base = net_refine.base
    hp, pdb = base._norm(h)
    c = base.coeffs(h, snr_db)
    for _ in range(n):
        c = c + net_refine._delta_from(hp, pdb, c, snr_db)
    return normalize_power(tf.matmul(hp, c, adjoint_a=True))


def run(net_base, net_refine):
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
            leak_db, _, _ = leak_db_sig_intf(h, w_base)
            entry["base"] = {"debit": float(tf.reduce_mean(rate_per_sample(h, w_base, nv))),
                              "fuite_db_mediane": float(np.median(leak_db))}
            for n in N_PASSES:
                w_n = npass_coeffs(net_refine, h, snr_t, n)
                leak_db, _, _ = leak_db_sig_intf(h, w_n)
                entry[f"{n}x_passe2"] = {"debit": float(tf.reduce_mean(rate_per_sample(h, w_n, nv))),
                                          "fuite_db_mediane": float(np.median(leak_db))}
            leak_db, _, _ = leak_db_sig_intf(h, w_wmmse)
            entry["wmmse300"] = {"debit": float(tf.reduce_mean(rate_per_sample(h, w_wmmse, nv))),
                                  "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            msg = " ".join(f"{n}x {entry[f'{n}x_passe2']['debit']:6.2f}(fuite{entry[f'{n}x_passe2']['fuite_db_mediane']:5.1f})"
                            for n in N_PASSES)
            print(f"{name:10s} {snr:2d}dB : base {entry['base']['debit']:6.2f}  {msg}  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f}", flush=True)
    return out


def analyse(out):
    """Gain au-dela de 2 (debit(3)-debit(2), debit(4)-debit(3)) et stabilite (n+1 pire que n,
    tolerance 1e-3) a 30/35/40 dB, tous jeux."""
    rows = []
    for name, by_snr in out.items():
        for snr_s, entry in by_snr.items():
            if snr_s == "n":
                continue
            snr = int(snr_s)
            d = {n: entry[f"{n}x_passe2"]["debit"] for n in N_PASSES}
            rows.append({"site": name, "snr": snr, "d1": d[1], "d2": d[2], "d3": d[3], "d4": d[4],
                         "gain_3_vs_2": d[3] - d[2], "gain_4_vs_3": d[4] - d[3],
                         "instable_3": d[3] < d[2] - 1e-3, "instable_4": d[4] < d[3] - 1e-3})
    high = [r for r in rows if r["snr"] in (30, 35, 40)]
    gains_3 = [r["gain_3_vs_2"] for r in high]
    gains_4 = [r["gain_4_vs_3"] for r in high]
    instables = [r for r in rows if r["instable_3"] or r["instable_4"]]
    return {"detail": rows,
            "gain_3_vs_2_moyen_30_35_40": float(np.mean(gains_3)), "gain_3_vs_2_max": float(np.max(gains_3)),
            "gain_4_vs_3_moyen_30_35_40": float(np.mean(gains_4)), "gain_4_vs_3_max": float(np.max(gains_4)),
            "cas_instables": instables}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_refine = load_refine()

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run(net_base, net_refine)
    out["analyse"] = analyse(out["csi_parfait"])
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("gain 3x vs 2x (moyen/max, 30-40dB) :", out["analyse"]["gain_3_vs_2_moyen_30_35_40"],
          out["analyse"]["gain_3_vs_2_max"], flush=True)
    print("gain 4x vs 3x (moyen/max, 30-40dB) :", out["analyse"]["gain_4_vs_3_moyen_30_35_40"],
          out["analyse"]["gain_4_vs_3_max"], flush=True)
    print("cas instables :", out["analyse"]["cas_instables"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
