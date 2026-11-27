"""Phase 18 -- pas frequentiel adaptatif, regle a DEUX parametres (etalement + SNR). Suite de
experiments/phase15_adaptive_step.py (FINDINGS.md "Pas frequentiel adaptatif par tirage") : la regle a un
seul alpha (s*Delta_f <= 1/(alpha*tau_rms), alpha fixe) perd 1 a 5% a 30 dB car le bon pas diminue aussi
quand le SNR monte. Ici alpha depend du SNR : alpha(SNR) = alpha0 * 10**(gamma*SNR_dB/10), donc
s*Delta_f <= 1/(alpha0 * 10**(gamma*SNR_dB/10) * tau_rms).

EVALUATION SEULE, aucun entrainement. Reutilise par import (sans dupliquer) les fonctions de
experiments/phase15_adaptive_step.py : rates_per_drop (precodeur sur les ancres 1 SC sur s, W interpole
lineairement, cf. phase13_interp.interp_rate), tau_rms_idft (ajustement de l'autocorrelation en frequence,
petits decalages -- l'IDFT pleine bande surestime tau_rms x10-50 par fuite spectrale), rule_step (s = plus
grand pas candidat tel que s*Delta_f <= 1/(alpha*tau_rms)), oracle_step, gather_by_step, s_stats. Precodeur
3GPP GramPrecoder M8K4 (CSI parfait), poids weights/phase13/precoder_relpow_mix.weights.h5, memes jeux et
memes tirages que phase15_adaptive_step (phase13_bench.channels, seed 4026 pour les 3GPP, seed=13 pour les
sites). SNR de donnees 10/20/30 dB (phase15_adaptive_step n'avait que 20/30).

(1) Calibration de (alpha0, gamma) UNE SEULE FOIS sur Sionna UMi (le plus selectif), a 10/20/30 dB, cible
    perte MOYENNE (sur les 3 SNR) <= 0.5% : pour chaque gamma sur une grille, le plus petit alpha0 (donc le
    plus grand pas, le moins de MAC) qui respecte la cible ; on garde le (alpha0, gamma) qui maximise le s
    moyen harmonique moyenne sur les 3 SNR. Puis appliquee TELLE QUELLE (zero-shot, aucun reglage par jeu)
    a UMi, UMa, RMa, St-Catherine, Decarie, Ericsson.
(2) Debit, s moyen (harmonique) et MAC/SC moyens (base 717 k MAC/SC du GramPrecoder M8K4,
    results/phase15/macs.json) a 10/20/30 dB : regle vs pas fixe 3 (239 k MAC/SC) vs oracle par tirage
    (perte <= 1%, cf. phase15_adaptive_step.oracle_step).
(3) Si simple : meme regle (memes alpha0, gamma, PAS reajustes), mais tau_rms estime depuis les pilotes LS
    en peigne + interpolation LMMSE (covariance du melange, comme l'encodeur -- csi.UplinkSounder,
    experiments/phase15_lab_csi.py), pilote 20 dB, plutot que depuis le vrai canal. Mesure seulement la
    robustesse de la regle a cette estimation (meme debit/s/MAC, tau_rms vrai remplace par tau_rms estime).

    python -u -m experiments.phase18_step_rule
"""
import json
import os
import time

import numpy as np
import tensorflow as tf

from config import SystemConfig
from csi import UplinkSounder
from experiments.phase13_bench import SETS, channels
from experiments.phase14_finetune import load_base
from experiments.phase15_adaptive_step import (STEPS, gather_by_step, oracle_step, rates_per_drop,
                                                rule_step, s_stats, tau_rms_idft)
from experiments.phase7_context import cov_mixture, sound_power_controlled
from utils import RESULTS, git_commit

SNRS = [10.0, 20.0, 30.0]
CALIB_SET = "umi"
CALIB_TARGET = 0.005          # perte moyenne <= 0.5%
FIXED_STEP = 3
PILOT_SNR_DB = 20.0
OUT_PATH = os.path.join(RESULTS, "phase18", "step_rule.json")


def alpha_of_snr(alpha0, gamma, snr_db):
    return alpha0 * 10.0 ** (gamma * snr_db / 10.0)


def rule_step_2p(tau_rms, snr_db, alpha0, gamma, delta_f, steps):
    return rule_step(tau_rms, alpha_of_snr(alpha0, gamma, snr_db), delta_f, steps)


def calibrate(tau_umi, rates_umi, delta_f, steps, snrs, target=CALIB_TARGET):
    """Grille (gamma, alpha0) sur UMi. Pour chaque gamma : le plus petit alpha0 (grille croissante, donc le
    plus grand pas / le moins de MAC) dont la perte MOYENNE sur `snrs` (vs s=1) est <= target. Parmi les
    gamma qui en trouvent un, garde celui qui maximise le s moyen harmonique moyenne sur les 3 SNR (regle la
    moins chere qui respecte la cible)."""
    gammas = np.linspace(0.0, 2.5, 51)
    alpha0_grid = np.geomspace(0.05, 300.0, 400)
    rate1_mean = {snr: float(np.mean(rates_umi[snr][1])) for snr in snrs}
    best = None
    for gamma in gammas:
        for a0 in alpha0_grid:
            losses, smeans = [], []
            for snr in snrs:
                s_r = rule_step_2p(tau_umi, snr, a0, gamma, delta_f, steps)
                r = gather_by_step(rates_umi[snr], s_r)
                losses.append((rate1_mean[snr] - float(np.mean(r))) / rate1_mean[snr])
                smeans.append(1.0 / np.mean(1.0 / s_r))
            if np.mean(losses) <= target:
                smean = float(np.mean(smeans))
                if best is None or smean > best[0]:
                    best = (smean, a0, gamma, float(max(losses)), float(np.mean(losses)))
                break   # alpha0 croissant : premier qui respecte la cible = le moins cher pour ce gamma
    assert best is not None, "aucun (alpha0, gamma) ne respecte la cible meme a alpha0 max de la grille"
    smean, alpha0, gamma, worst_loss, mean_loss = best
    return alpha0, gamma, smean, worst_loss, mean_loss


def tau_from_pilots(h, snd, delta_f, pilot_snr_db=PILOT_SNR_DB):
    """tau_rms [B] estime depuis l'estimation LS + LMMSE (covariance du melange) a pilote `pilot_snr_db`,
    au lieu du vrai canal -- meme fonction d'ajustement (tau_rms_idft) que pour le vrai canal."""
    h_hat = sound_power_controlled(snd, h, pilot_snr_db)
    return tau_rms_idft(h_hat, delta_f)


def add_rule_entry(entry, key, tau, rates, alpha0, gamma, delta_f, mac_base):
    entry[key] = {}
    for snr in SNRS:
        s_r = rule_step_2p(tau, snr, alpha0, gamma, delta_f, STEPS)
        stats = s_stats(s_r)
        r = float(np.mean(gather_by_step(rates[snr], s_r)))
        rate1_mean = float(np.mean(rates[snr][1]))
        entry[key][str(snr)] = {**stats, "rate": r, "mac_per_sc": mac_base / stats["s_mean_harmonic"],
                                 "pct_loss": 100 * (rate1_mean - r) / rate1_mean}


def main():
    t0 = time.time()
    cfg = SystemConfig()
    delta_f = cfg.subcarrier_spacing
    net = load_base()
    mac_base = int(net.macs_per_sc(4, 8))
    mac_fixed3 = mac_base / FIXED_STEP
    print(f"MAC/SC base (M8K4) = {mac_base}, pas fixe {FIXED_STEP} -> {mac_fixed3:.0f}", flush=True)

    data = {}
    for name in SETS:
        t1 = time.time()
        h = channels(name)
        tau = tau_rms_idft(h, delta_f)
        rates = {snr: {s: rates_per_drop(h, net, s, snr) for s in STEPS} for snr in SNRS}
        data[name] = {"h": h, "tau_rms_ns": tau, "rates": rates}
        print(f"{name} : tau_rms median {np.median(tau) * 1e9:.1f} ns "
              f"(p10 {np.percentile(tau, 10) * 1e9:.1f} / p90 {np.percentile(tau, 90) * 1e9:.1f}), "
              f"rate(s=1) 10/20/30 dB = {np.mean(rates[10.0][1]):.2f} / {np.mean(rates[20.0][1]):.2f} / "
              f"{np.mean(rates[30.0][1]):.2f} ({time.time() - t1:.0f} s)", flush=True)

    alpha0, gamma, s_cal, worst_cal, mean_cal = calibrate(
        data[CALIB_SET]["tau_rms_ns"], data[CALIB_SET]["rates"], delta_f, STEPS, SNRS)
    print(f"(alpha0, gamma) calibres sur {CALIB_SET} (10/20/30 dB, perte moyenne <= 0.5%) : "
          f"alpha0={alpha0:.4f} gamma={gamma:.3f} (s_moy cal={s_cal:.2f}, perte moy={100*mean_cal:.2f}%, "
          f"pire SNR={100*worst_cal:.2f}%)", flush=True)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "delta_f_hz": delta_f, "mac_per_sc_base": mac_base, "fixed_step": FIXED_STEP,
                    "pilot_snr_db": PILOT_SNR_DB},
           "steps": STEPS, "snrs_db": SNRS, "calib_set": CALIB_SET, "calib_target_pct": 100 * CALIB_TARGET,
           "alpha0": alpha0, "gamma": gamma, "calib_s_mean": s_cal, "calib_worst_loss_pct": 100 * worst_cal,
           "calib_mean_loss_pct": 100 * mean_cal, "sets": {}}

    cov = cov_mixture()
    snd = UplinkSounder(cfg, "lmmse", cov)

    for name in SETS:
        t1 = time.time()
        tau = data[name]["tau_rms_ns"]
        rates = data[name]["rates"]
        entry = {"tau_rms_ns": {"median": float(np.median(tau) * 1e9),
                                 "p10": float(np.percentile(tau, 10) * 1e9),
                                 "p90": float(np.percentile(tau, 90) * 1e9)},
                 "fixed3": {}, "oracle_1pct": {}}

        add_rule_entry(entry, "rule", tau, rates, alpha0, gamma, delta_f, mac_base)

        for snr in SNRS:
            r1 = rates[snr][1]
            rate1_mean = float(np.mean(r1))
            entry["fixed3"][str(snr)] = {
                "rate": float(np.mean(rates[snr][FIXED_STEP])),
                "pct_loss": 100 * (rate1_mean - float(np.mean(rates[snr][FIXED_STEP]))) / rate1_mean}
            s_o = oracle_step(rates[snr], STEPS, 0.01)
            ostats = s_stats(s_o)
            r_o = float(np.mean(gather_by_step(rates[snr], s_o)))
            entry["oracle_1pct"][str(snr)] = {**ostats, "rate": r_o,
                                               "mac_per_sc": mac_base / ostats["s_mean_harmonic"],
                                               "pct_loss": 100 * (rate1_mean - r_o) / rate1_mean}

        # (3) meme regle, tau_rms estime depuis les pilotes LS en peigne (pilote 20 dB) plutot que le vrai
        # canal -- memes (alpha0, gamma), aucun reglage supplementaire.
        tau_p = tau_from_pilots(data[name]["h"], snd, delta_f)
        entry["pilot_tau_rms_ns"] = {"median": float(np.median(tau_p) * 1e9),
                                     "p10": float(np.percentile(tau_p, 10) * 1e9),
                                     "p90": float(np.percentile(tau_p, 90) * 1e9)}
        add_rule_entry(entry, "rule_pilot_tau", tau_p, rates, alpha0, gamma, delta_f, mac_base)

        out["sets"][name] = entry
        print(f"{name} : fixe3 {entry['fixed3']['20.0']['rate']:.2f}@20 "
              f"| regle s_moy {entry['rule']['20.0']['s_mean_harmonic']:.2f} "
              f"rate {entry['rule']['20.0']['rate']:.2f}@20 (perte {entry['rule']['20.0']['pct_loss']:.2f}%) "
              f"mac {entry['rule']['20.0']['mac_per_sc']:.0f} "
              f"| oracle1% s_moy {entry['oracle_1pct']['20.0']['s_mean_harmonic']:.2f} "
              f"rate {entry['oracle_1pct']['20.0']['rate']:.2f}@20 "
              f"| regle(pilote) s_moy {entry['rule_pilot_tau']['20.0']['s_mean_harmonic']:.2f} "
              f"rate {entry['rule_pilot_tau']['20.0']['rate']:.2f}@20 "
              f"({time.time() - t1:.0f} s)", flush=True)

        os.makedirs(os.path.join(RESULTS, "phase18"), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), indent=1)

    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
