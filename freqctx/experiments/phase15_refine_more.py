"""Phase 15 -- deux questions complementaires sur la passe 2 (RefineNet, models/refine.py,
poids weights/phase15/refine_L4.weights.h5), EVALUATION SEULE (aucun entrainement), SANS
MODIFIER experiments/phase15_refine_eval.py NI models/refine.py. Reutilise les memes briques
(import seulement) :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf}
  - experiments.phase14_finetune.load_base
  - experiments.phase15_refine_eval.load_refine (meme poids, meme construction)
  - precoders.classical.{wmmse, normalize_power}
  - train.rate_per_sample

Question 1 -- passe 2 appliquee deux fois (memes poids, sans reentrainer) :
  C3 = C2 + DELTA_C(G, E2), E2 = canal effectif de C2 (meme normalisation que pour E).
  On obtient C3 en rappelant RefineNet._delta_from (deja ecrit dans refine.py, meme fonction
  que pour le calcul de C2 a partir de C1) une seconde fois en lui passant C2 a la place de C1 --
  _delta_from recalcule alors E a partir du c1 fourni, donc E2 est bien le canal effectif de C2
  avec exactement la meme normalisation (W1n) que pour le calcul de C2. G ne change pas (il
  depend de hp = canal normalise par utilisateur, pas de c1). C'est une fonction du present
  script (two_pass_coeffs), refine.py n'est pas touche.

Question 2 -- passe 2 activee seulement au-dessus d'un seuil de SNR (le SNR est connu) :
  calcule a partir des debits "base" / "1x_passe2" deja obtenus a la question 1 (pas de nouvel
  appel reseau) : pour un seuil s, debit(snr) = base si snr<s sinon 1x_passe2. Perte max par
  rapport a "toujours activee" (1x_passe2) sur 0-40dB = max(0, 1x_passe2 - base) aux SNR < s
  (UMi + 3 sites). Cout moyen en MAC/SC si SNR ~ Uniforme(0,30) = base_mac + passe2_mac *
  P(SNR>=s) = base_mac + passe2_mac * (30-s)/30 (717312 / 363264 MAC/SC en M8K4, deja verifies
  dans results/phase15/refine.json "macs").

    python -u -m experiments.phase15_refine_more
"""
import json
import os
import time

import numpy as np
import tensorflow as tf

from experiments.phase14_finetune import load_base
from experiments.phase15_diag import get_datasets, leak_db_sig_intf, subsample
from experiments.phase15_refine_eval import SNRS_PARFAIT, N_EVAL, load_refine
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase15")
OUT_PATH = os.path.join(OUT_DIR, "refine_more.json")
THRESHOLDS = [10, 15, 20, 25]
MAC_BASE = 717312          # M8K4, verifie dans refine.json "macs" (base_macs_par_sc)
MAC_PASSE2 = 363264        # M8K4, verifie dans refine.json "macs" (passe2_macs_par_sc)
UNIFORM_LO, UNIFORM_HI = 0.0, 30.0
LOSS_BUDGET = 0.02         # bps/Hz, critere "ne perd pas plus de" de la question 2


# ------------------------------------------------------------- Q1 : passe 2 appliquee 2 fois


def two_pass_coeffs(net_refine, h, snr_db):
    """C3 = C2 + DELTA_C(G, E2), E2 = canal effectif de C2 (meme normalisation que pour E).
    Rappelle RefineNet._delta_from (non modifie) avec c1=C2 : E est alors reconstruit depuis
    C2 (W1 = h~^H C2, renormalise comme la sortie base), donc E2 est exactement le canal
    effectif de C2 avec la meme normalisation que celle utilisee pour obtenir C2 depuis C1."""
    base = net_refine.base
    hp, pdb = base._norm(h)
    c1 = base.coeffs(h, snr_db)
    d1 = net_refine._delta_from(hp, pdb, c1, snr_db)
    c2 = c1 + d1
    d2 = net_refine._delta_from(hp, pdb, c2, snr_db)                 # E2 recalcule depuis c2
    c3 = c2 + d2
    return normalize_power(tf.matmul(hp, c3, adjoint_a=True))


def run_parfait_more(net_base, net_refine):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_refine_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            snr_t = tf.fill([h.shape[0]], float(snr))
            w_base = net_base(h, snr_t)
            w_1x = net_refine(h, snr_t)
            w_2x = two_pass_coeffs(net_refine, h, snr_t)
            w_wmmse = wmmse(h, nv, num_iter=300)
            entry = {}
            for tag, w in (("base", w_base), ("1x_passe2", w_1x), ("2x_passe2", w_2x), ("wmmse300", w_wmmse)):
                leak_db, _, _ = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                entry[tag] = {"debit": rate, "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"parfait {name:10s} {snr:2d}dB : base {entry['base']['debit']:6.2f} (fuite {entry['base']['fuite_db_mediane']:6.1f})  "
                  f"1x {entry['1x_passe2']['debit']:6.2f} (fuite {entry['1x_passe2']['fuite_db_mediane']:6.1f})  "
                  f"2x {entry['2x_passe2']['debit']:6.2f} (fuite {entry['2x_passe2']['fuite_db_mediane']:6.1f})  "
                  f"wmmse300 {entry['wmmse300']['debit']:6.2f} (fuite {entry['wmmse300']['fuite_db_mediane']:6.1f})", flush=True)
    return out


def analyse_q1(parfait):
    """Le plancher descend-il encore (2x vs 1x a haut SNR) ? Y a-t-il des pertes (2x < base, ou
    instabilite : 2x pire que 1x) a un SNR donne ?"""
    rows = []
    for name, by_snr in parfait.items():
        for snr_s, entry in by_snr.items():
            if snr_s == "n":
                continue
            d_base, d_1x, d_2x = entry["base"]["debit"], entry["1x_passe2"]["debit"], entry["2x_passe2"]["debit"]
            f_1x, f_2x = entry["1x_passe2"]["fuite_db_mediane"], entry["2x_passe2"]["fuite_db_mediane"]
            rows.append({"site": name, "snr": int(snr_s), "gain_2x_vs_1x_debit": d_2x - d_1x,
                         "gain_2x_vs_base_debit": d_2x - d_base, "fuite_2x_vs_1x_db": f_2x - f_1x,
                         "perte_2x_vs_base": d_2x < d_base, "instabilite_2x_vs_1x": d_2x < d_1x - 1e-3})
    floor_gain_30_40 = [r["fuite_2x_vs_1x_db"] for r in rows if r["snr"] in (30, 35, 40)]
    pertes = [r for r in rows if r["perte_2x_vs_base"] or r["instabilite_2x_vs_1x"]]
    return {"detail": rows, "plancher_descend_encore_db_moyen_30_40": float(np.mean(floor_gain_30_40)),
            "cas_instables_ou_perte": pertes}


# ------------------------------------------------------------- Q2 : seuil de SNR


def analyse_q2(parfait):
    """debit(snr) = base si snr<s sinon 1x_passe2 ; perte max vs 1x_passe2 toujours actif sur
    0-40dB ; cout MAC moyen si SNR ~ Uniforme(0,30)."""
    out = {"budget_bps_hz": LOSS_BUDGET, "seuils": {}}
    for s in THRESHOLDS:
        worst, worst_loc = 0.0, None
        for name, by_snr in parfait.items():
            for snr_s, entry in by_snr.items():
                if snr_s == "n":
                    continue
                snr = int(snr_s)
                d_1x = entry["1x_passe2"]["debit"]
                d_thresh = entry["base"]["debit"] if snr < s else d_1x
                loss = d_1x - d_thresh                                # >=0, nul si snr>=s
                if loss > worst:
                    worst, worst_loc = loss, (name, snr)
        p_on = max(0.0, min(1.0, (UNIFORM_HI - s) / (UNIFORM_HI - UNIFORM_LO)))
        mac_moyen = MAC_BASE + MAC_PASSE2 * p_on
        out["seuils"][str(s)] = {"perte_max_bps_hz": worst, "pire_cas": worst_loc,
                                  "proba_passe2_active_uniforme_0_30": p_on, "mac_moyen_par_sc": mac_moyen,
                                  "respecte_budget": worst <= LOSS_BUDGET}
    seuils_ok = [int(s) for s, v in out["seuils"].items() if v["respecte_budget"]]
    out["seuil_max_sous_budget"] = max(seuils_ok) if seuils_ok else None
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_refine = load_refine()

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait_more(net_base, net_refine)
    out["q1_deux_passes"] = analyse_q1(out["csi_parfait"])
    out["q2_seuil_snr"] = analyse_q2(out["csi_parfait"])
    out["macs"] = {"base_macs_par_sc": MAC_BASE, "passe2_macs_par_sc": MAC_PASSE2}
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("Q1 plancher (fuite 2x-1x dB moyen 30/35/40) :", out["q1_deux_passes"]["plancher_descend_encore_db_moyen_30_40"], flush=True)
    print("Q1 cas instables/perte :", out["q1_deux_passes"]["cas_instables_ou_perte"], flush=True)
    print("Q2 seuils :", out["q2_seuil_snr"]["seuils"], flush=True)
    print("Q2 seuil max sous budget :", out["q2_seuil_snr"]["seuil_max_sous_budget"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
