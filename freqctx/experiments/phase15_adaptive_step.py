"""Phase 15 -- le pas d'interpolation s (precodeur calcule sur 1 SC sur s, W interpole lineairement,
cf. FINDINGS.md "CSI parfait : precodeur calcule sur 1 SC sur s" et "Interpolation et bords de la bande",
experiments/phase13_interp.py, phase13_interp2.py, results/phase13/interp.json) depend de l'etalement des
retards (UMi ~1/3, sites du labo 1/12 a 1/24). Un pas ADAPTATIF par tirage vaut-il le coup ?

EVALUATION SEULE, aucun entrainement. Precodeur 3GPP GramPrecoder M8K4 (CSI parfait), poids
weights/phase13/precoder_relpow_mix.weights.h5 charges comme experiments.phase14_finetune.load_base()
(meme reseau que phase13_interp utiliserait pour M8K4 ; interpolation lineaire identique a
phase13_interp.interp_rate). CSI parfait, puissances reelles, SNR 20 et 30 dB. Jeux : Sionna UMi/UMa/RMa
et St-Catherine/Decarie/Ericsson -- memes generateurs que experiments.phase13_bench.channels (= memes
tirages que phase13_interp pour "sionna_umi" : seed 4026 ; etendu a UMa/RMa/sites du labo). Pas candidats
s = 1,2,3,4,6,8,12,16,24.

Trois volets :
(1) Oracle par tirage : plus grand s dont la perte relative de debit (vs s=1) <= 1% (et 0.5%). Distribution
    de s, s moyen effectif (harmonique, car le cout est ~1/s) et debit, par jeu ; compare au pas fixe 3 et
    au plus grand pas fixe dont la perte MOYENNE du jeu reste <= 1%.
(2) Regle simple sans apprentissage : tau_rms par tirage depuis le VRAI canal (IDFT sur la bande), bande
    de coherence B_c ~= 1/(alpha*tau_rms), s = plus grand pas candidat tel que s*Delta_f <= B_c. alpha
    regle UNE SEULE FOIS sur Sionna UMi (le plus selectif, SNR 20 dB, cible perte moyenne <= 1%), puis
    reutilise sur tous les jeux.
(3) Synthese : MAC effectives par SC (base 717 k MAC/SC du GramPrecoder M8K4, results/phase15/macs.json)
    et perte de debit, regle et oracle contre le pas fixe 3.

    python -u -m experiments.phase15_adaptive_step
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from experiments.phase13_bench import SETS, channels
from experiments.phase14_finetune import load_base
from precoders.classical import normalize_power
from train import rate_per_sample
from utils import RESULTS, git_commit

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
SNRS = [20.0, 30.0]
CALIB_SNR = 20.0
BATCH = 16
OUT_PATH = os.path.join(RESULTS, "phase15", "adaptive_step.json")


def rates_per_drop(h, net, s, snr, batch=BATCH):
    """h [B, F, K, M] : precodeur calcule sur les ancres (1 SC sur s), W interpole lineairement sur toute
    la bande (comme experiments.phase13_interp.interp_rate), puis renormalise. Renvoie le debit moyenne
    sur F (mais pas sur B) : [B]."""
    B, F, K, M = h.shape
    anchors = np.unique(np.append(np.arange(0, F, s), F - 1))
    A = len(anchors)
    nv = 10.0 ** (-snr / 10.0)
    j = np.clip(np.searchsorted(anchors, np.arange(F), side="right") - 1, 0, A - 2)
    t_np = ((np.arange(F) - anchors[j]) / (anchors[j + 1] - anchors[j])).astype(np.float32)[None, :, None, None]
    out = []
    for b in range(0, B, batch):
        hb = h[b:b + batch]
        nb = hb.shape[0]
        ha = tf.gather(hb, anchors, axis=1)
        wa = net(tf.reshape(ha, [-1, K, M]), tf.fill([nb * A], float(snr)))
        wa = tf.reshape(wa, [nb, A, M, K])
        t = tf.cast(t_np, wa.dtype)
        w = (1 - t) * tf.gather(wa, j, axis=1) + t * tf.gather(wa, j + 1, axis=1)
        w = normalize_power(tf.reshape(w, [-1, M, K]))
        r = rate_per_sample(tf.reshape(hb, [-1, K, M]), w, nv)
        out.append(tf.reduce_mean(tf.reshape(r, [nb, F]), axis=1))
    return tf.concat(out, 0).numpy()


def tau_rms_idft(h, delta_f, dmax=8):
    """h [B, F, K, M] complexe (freq. uniformement espacees de delta_f). tau_rms [B] (s) depuis le VRAI
    canal, via la fonction d'autocorrelation en frequence (equivalente a l'IDFT du profil puissance-retard,
    Wiener-Khinchin) : rho(d) = E_f,k,m[H(f) conj(H(f+d))] / E[|H|^2], ajustement moindres carres de
    1 - |rho(d)|^2 ~= (2*pi*d*delta_f)^2 * tau_rms^2 pour d=1..dmax (petits decalages seulement).
    L'IDFT PLEINE BANDE (moment directement sur tout le profil retard circulaire) a ete essayee en premier
    mais la fuite spectrale (sidelobes sinc, dus aux retards non multiples du pas d'echantillonnage
    1/(F*delta_f)) gonfle tau_rms d'un facteur ~10-50x (verifie contre les retards reels des sites du labo :
    IDFT pleine bande donne ~1-4 us, alors que le vrai etalement est de quelques dizaines de ns) ; cette
    version petits decalages, valide contre les memes retards reels (erreur < 30%), l'evite."""
    B, F, K, M = h.shape
    hf = tf.reshape(h, [B, F, K * M]).numpy()
    r0 = np.mean(np.abs(hf) ** 2, axis=(1, 2))
    num, den = 0.0, 0.0
    for d in range(1, dmax + 1):
        rd = np.mean(hf[:, :F - d] * np.conj(hf[:, d:]), axis=(1, 2))
        y = 1.0 - (np.abs(rd) / r0) ** 2
        x = (2 * np.pi * d * delta_f) ** 2
        num, den = num + x * y, den + x * x
    return np.sqrt(np.maximum(num / den, 0.0))


def rule_step(tau_rms, alpha, delta_f, steps):
    """s = plus grand pas candidat tel que s*delta_f <= 1/(alpha*tau_rms) ; s=1 si aucun ne convient."""
    bc = 1.0 / (alpha * np.maximum(tau_rms, 1e-15))
    s_max = np.floor(bc / delta_f)
    steps_arr = np.array(sorted(steps), dtype=np.float64)
    idx = np.clip(np.searchsorted(steps_arr, s_max, side="right") - 1, 0, len(steps_arr) - 1)
    return steps_arr[idx].astype(int)


def oracle_step(rates_by_s, steps, thresh):
    """rates_by_s : {s: [B] debit}. Pour chaque tirage, le plus grand s (parmi `steps` tries) dont la
    perte relative vs s=1 est <= thresh."""
    steps_sorted = sorted(steps)
    rate1 = rates_by_s[1]
    B = len(rate1)
    s_chosen = np.ones(B, dtype=int)
    for s in steps_sorted:
        if s == 1:
            continue
        loss = (rate1 - rates_by_s[s]) / rate1
        s_chosen = np.where(loss <= thresh, s, s_chosen)
    return s_chosen


def gather_by_step(rates_by_s, s_chosen):
    return np.array([rates_by_s[int(s)][b] for b, s in enumerate(s_chosen)])


def best_fixed_step(rates_by_s, steps, thresh):
    """Plus grand pas fixe dont la perte MOYENNE du jeu (vs s=1) reste <= thresh."""
    rate1_mean = float(np.mean(rates_by_s[1]))
    best = 1
    for s in sorted(steps):
        if s == 1:
            continue
        loss = (rate1_mean - float(np.mean(rates_by_s[s]))) / rate1_mean
        if loss <= thresh:
            best = s
    return best


def s_stats(s_chosen):
    vals, counts = np.unique(s_chosen, return_counts=True)
    dist = {str(int(v)): int(c) for v, c in zip(vals, counts)}
    return {"dist": dist, "s_mean_arith": float(np.mean(s_chosen)),
            "s_mean_harmonic": float(1.0 / np.mean(1.0 / s_chosen))}


def main():
    t0 = time.time()
    cfg = SystemConfig()
    delta_f = cfg.subcarrier_spacing
    net = load_base()
    mac_base = int(net.macs_per_sc(4, 8))
    print(f"MAC/SC base (M8K4) = {mac_base}", flush=True)

    data = {}
    for name in SETS:
        t1 = time.time()
        h = channels(name)
        tau = tau_rms_idft(h, delta_f)
        rates = {snr: {s: rates_per_drop(h, net, s, snr) for s in STEPS} for snr in SNRS}
        data[name] = {"tau_rms_ns": tau, "rates": rates}
        print(f"{name} : tau_rms median {np.median(tau) * 1e9:.1f} ns "
              f"(p10 {np.percentile(tau, 10) * 1e9:.1f} / p90 {np.percentile(tau, 90) * 1e9:.1f}), "
              f"rate(s=1)@20dB {np.mean(rates[20.0][1]):.2f}, rate(s=1)@30dB {np.mean(rates[30.0][1]):.2f} "
              f"({time.time() - t1:.0f} s)", flush=True)

    # ---- calibration de alpha sur Sionna UMi, SNR 20 dB, cible perte moyenne <= 1% ----
    tau_umi = data["umi"]["tau_rms_ns"]
    rates_umi20 = data["umi"]["rates"][CALIB_SNR]
    rate1_umi = float(np.mean(rates_umi20[1]))
    grid = np.geomspace(0.2, 80.0, 200)
    alpha = float(grid[-1])
    for a in grid:
        s_r = rule_step(tau_umi, a, delta_f, STEPS)
        r = gather_by_step(rates_umi20, s_r)
        loss = (rate1_umi - float(np.mean(r))) / rate1_umi
        if loss <= 0.01:
            alpha = float(a)
            break
    print(f"alpha calibre sur UMi (20 dB, perte moyenne <= 1%) : alpha={alpha:.3f}", flush=True)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "delta_f_hz": delta_f, "mac_per_sc_base": mac_base},
           "steps": STEPS, "snrs_db": SNRS, "calib_snr_db": CALIB_SNR, "alpha": alpha, "sets": {}}

    for name in SETS:
        tau = data[name]["tau_rms_ns"]
        rates = data[name]["rates"]
        s_rule = rule_step(tau, alpha, delta_f, STEPS)
        entry = {"tau_rms_ns": {"median": float(np.median(tau) * 1e9), "p10": float(np.percentile(tau, 10) * 1e9),
                                "p90": float(np.percentile(tau, 90) * 1e9)},
                 "fixed3": {}, "fixed_best_1pct": {}, "oracle_1pct": {}, "oracle_0.5pct": {}, "rule": {}}

        s_best1 = best_fixed_step(rates[CALIB_SNR], STEPS, 0.01)
        entry["fixed_best_1pct"]["s"] = s_best1

        rule_stats = s_stats(s_rule)
        entry["rule"].update(rule_stats)
        entry["rule"]["mac_per_sc"] = mac_base / rule_stats["s_mean_harmonic"]

        for snr in SNRS:
            r1 = rates[snr][1]
            rate1_mean = float(np.mean(r1))
            entry["fixed3"][str(snr)] = {"rate": float(np.mean(rates[snr][3])),
                                          "pct_loss": 100 * (rate1_mean - float(np.mean(rates[snr][3]))) / rate1_mean}
            entry["fixed_best_1pct"][str(snr)] = {"rate": float(np.mean(rates[snr][s_best1])),
                                                   "pct_loss": 100 * (rate1_mean - float(np.mean(rates[snr][s_best1]))) / rate1_mean}
            entry["rule"][str(snr)] = {"rate": float(np.mean(gather_by_step(rates[snr], s_rule))),
                                        "pct_loss": 100 * (rate1_mean - float(np.mean(gather_by_step(rates[snr], s_rule)))) / rate1_mean}
            for tag, thresh in (("oracle_1pct", 0.01), ("oracle_0.5pct", 0.005)):
                s_o = oracle_step(rates[snr], STEPS, thresh)
                stats = s_stats(s_o)
                r_o = float(np.mean(gather_by_step(rates[snr], s_o)))
                entry[tag][str(snr)] = {**stats, "rate": r_o,
                                        "pct_loss": 100 * (rate1_mean - r_o) / rate1_mean,
                                        "mac_per_sc": mac_base / stats["s_mean_harmonic"]}

        out["sets"][name] = entry
        print(f"{name} : fixe3 {entry['fixed3']['20.0']['rate']:.2f}@20 "
              f"| oracle1% s_moy {entry['oracle_1pct']['20.0']['s_mean_harmonic']:.2f} "
              f"rate {entry['oracle_1pct']['20.0']['rate']:.2f}@20 "
              f"| regle s_moy {rule_stats['s_mean_harmonic']:.2f} rate {entry['rule']['20.0']['rate']:.2f}@20 "
              f"(perte {entry['rule']['20.0']['pct_loss']:.2f}%)", flush=True)

        os.makedirs(os.path.join(RESULTS, "phase15"), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), indent=1)

    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
