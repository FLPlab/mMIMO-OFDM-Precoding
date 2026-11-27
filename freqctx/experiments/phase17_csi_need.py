"""Phase 17 -- en CSI ESTIME, quelle precision du precodeur faut-il reellement ? (question de
l'utilisateur, 2026-11-27). EVALUATION SEULE (aucun entrainement -- seules des sondes lineaires
ajustees en forme fermee par moindres carres, comme experiments/phase17_probe.py).

Constat de depart : en CSI PARFAIT, le precodeur libre a un plancher de precision relative sur W
d'environ 0,5 % (fuite d'interference ~-45 dB ; FINDINGS phases 15 et 17). En CSI ESTIME, l'erreur
d'estimation impose deja un plancher sur le VRAI canal. Si ce plancher est large (-20/-30 dB, soit
3 a 10 % d'erreur relative), un precodeur beaucoup moins precis (2-5 %) -- ou un tronc de reseau
beaucoup moins profond -- pourrait suffire sans perte mesurable.

Protocole et tirages identiques a experiments/phase15_lab_csi.py (sites St-Catherine / Decarie /
Ericsson, convention psi2az, seed=13, 192 tirages) et a l'equivalent UMi (experiments/phase14_csi.py,
ChannelGenerator UMi, seed Sionna 4026, 192 tirages) ; pilotes Sionna, LS + interpolation LMMSE
(covariance du melange, experiments.phase7_context.cov_mixture), sondage avec controle de puissance
montant (sound_power_controlled). Pilote 0 / 10 / 20 dB, SNR de donnees 10 / 20 / 30 dB, M8K4.
Reseau : poids phase13 (experiments.phase14_csi.load_perfect), le meme que dans phase15_lab_csi.

1. Fuite mediane et debit sur le VRAI canal pour : reseau CSI parfait applique a l'estimation,
   WMMSE robuste 300 it., et reference CSI parfait (meme reseau applique directement au vrai canal,
   pour isoler l'effet de l'estimation seule).
2. Courbe "precision utile" : C = net.coeffs(h_in, snr) (domaine normalise du modele), degrade par un
   bruit relatif CONTROLE de niveau eps sur C (par echantillon, ||delta||_F / ||C||_F = eps exactement),
   W = normalize_power(hp_in^H (C + delta)), debit mesure sur le VRAI canal. h_in = l'ESTIMATION (courbe
   "estime", par pilote x SNR) ou le VRAI canal (courbe "parfait", par SNR seulement, pour comparaison).
   eps in {0, 0.5, 1, 2, 5, 10, 20} %. La precision requise = le plus grand eps teste dont la perte de
   debit (vs eps=0) reste <= 0,05 bps/Hz ; le premier eps qui la depasse est aussi rapporte.
3. Meme question, version "profondeur" : C lu a la couche l (sonde lineaire moindres carres, ajustee sur
   une moitie des echantillons, testee sur l'autre, vers le C du reseau complet -- memes fonctions que
   experiments/phase17_probe.py), l in {4, 6, 7} contre le reseau complet (8), en CSI estime, SNR 20 dB,
   par pilote. Un reseau tronque a 4-6 couches suffit-il en CSI estime ?

    python -u -m experiments.phase17_csi_need
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
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase13_deepmimo import DROPS, drops
from experiments.phase13_lab import read_cells
from experiments.phase14_csi import load_perfect
from experiments.phase15_diag import leak_db_sig_intf
from experiments.phase17_probe import c_of_states, feats, probe_matrix, states, w_from_c
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

SITES = ["stecath", "decarie", "ericsson"]
DATASETS = SITES + ["umi"]
PILOTS = [0, 10, 20]
SNRS = [10, 20, 30]
EPS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
PROBE_LAYERS = [4, 6, 7]
B = 16
CHUNK = 8192
N_TR, N_TE = 3072, 2048          # pour l'etape 3 (sondes), comme phase17_probe
DROP_TOL = 0.05                  # bps/Hz : seuil de perte acceptable
OUT_DIR = os.path.join(RESULTS, "phase17")
OUT_PATH = os.path.join(OUT_DIR, "csi_need.json")

tf.random.set_seed(42)


# ---------------------------------------------------------------------- donnees


def site_h(site):
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    users = read_cells(site, "psi2az")
    return drops(users, freqs, equal=False, seed=13)                  # [192, F, 4, 8]


def umi_h():
    cfg = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen = ChannelGenerator(cfg)
    return tf.concat([gen(B)[:, 0] for _ in range(DROPS // B)], 0)    # [192, F, 4, 8]


def get_h(name):
    return site_h(name) if name in SITES else umi_h()


def sound(snd, h, pilot_db):
    pairs = [sound_power_controlled(snd, h[j:j + B], float(pilot_db), return_err_var=True)
             for j in range(0, h.shape[0], B)]
    return tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)


# ---------------------------------------------------------------------- mesures sur le vrai canal


def eval_on_true(h_true, h_in, w_fn, nv, err=None, chunk=CHUNK):
    """w_fn(h_in_chunk, err_chunk) -> W. Debit moyen + fuite mediane sur le VRAI canal."""
    rs, lks = [], []
    for i in range(0, h_true.shape[0], chunk):
        ht, hi = h_true[i:i + chunk], h_in[i:i + chunk]
        e = None if err is None else err[i:i + chunk]
        w = w_fn(hi, e)
        rs.append(rate_per_sample(ht, w, nv).numpy())
        lk, _, _ = leak_db_sig_intf(ht, w)
        lks.append(lk)
    return float(np.concatenate(rs).mean()), float(np.median(np.concatenate(lks)))


def eps_curve(net, h_true, h_in, snr, nv, eps_list=EPS, chunk=CHUNK):
    """Degrade C = net.coeffs(h_in, snr) par un bruit relatif controle eps, W = normalize_power
    (hp_in^H (C + delta)), debit mesure sur h_true. Renvoie {eps: debit moyen}."""
    acc = {e: [] for e in eps_list}
    for i in range(0, h_true.shape[0], chunk):
        ht, hi = h_true[i:i + chunk], h_in[i:i + chunk]
        n = int(hi.shape[0])
        hp = net._norm(hi)[0]
        c = net.coeffs(hi, tf.fill([n], float(snr)))                  # [n, K, K] complex64
        cnorm = tf.norm(tf.reshape(c, [n, -1]), axis=-1)
        for e in eps_list:
            if e == 0.0:
                cn = c
            else:
                z = tf.complex(tf.random.normal(c.shape), tf.random.normal(c.shape))
                znorm = tf.norm(tf.reshape(z, [n, -1]), axis=-1)
                delta = z * tf.cast((e * cnorm / znorm)[:, None, None], z.dtype)
                cn = c + delta
            w = normalize_power(tf.matmul(hp, cn, adjoint_a=True))
            acc[e].append(rate_per_sample(ht, w, nv).numpy())
    return {e: float(np.concatenate(v).mean()) for e, v in acc.items()}


def required_eps(curve, eps_list=EPS, tol=DROP_TOL):
    base = curve[eps_list[0]]
    ok, fail = eps_list[0], None
    for e in eps_list[1:]:
        if base - curve[e] > tol:
            fail = e
            break
        ok = e
    return {"base_debit": base, "eps_max_ok": ok, "eps_premier_echec": fail,
            "perte_a_eps_max_ok": float(base - curve[ok])}


# ---------------------------------------------------------------------- etape 3 : sondes par couche


def pick_subset(h_true, h_in, n_tr, n_te, seed):
    N = int(h_true.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, n_tr + n_te, replace=False)
    g = lambda t, ix: tf.gather(t, ix)
    return (g(h_true, idx[:n_tr]), g(h_in, idx[:n_tr]), g(h_true, idx[n_tr:]), g(h_in, idx[n_tr:]))


def layer_probe(net, h_tr_true, h_tr_in, h_te_true, h_te_in, snr, layers=PROBE_LAYERS):
    nv = 10.0 ** (-snr / 10.0)
    hp_tr, g_tr, es_tr = states(net, h_tr_in, snr)
    hp_te, g_te, es_te = states(net, h_te_in, snr)
    L = es_tr.shape[1] - 1
    c_net_tr, c_net_te = c_of_states(net, es_tr[:, L]), c_of_states(net, es_te[:, L])
    out = {}
    for l in list(layers) + [L]:
        if l == L:
            c_pred = c_net_te
        else:
            x_tr, x_te = feats(es_tr[:, l]), feats(es_te[:, l])
            c_pred = probe_matrix(x_tr, c_net_tr, x_te)
        w = w_from_c(hp_te, c_pred)
        out[f"couche{l}"] = float(tf.reduce_mean(rate_per_sample(h_te_true, w, nv)))
    return out


# ---------------------------------------------------------------------- main


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {
        "meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                 "pilots_db": PILOTS, "snrs_db": SNRS, "eps": EPS, "drop_tol_bps_hz": DROP_TOL,
                 "drops": DROPS, "n_tr": N_TR, "n_te": N_TE},
        "step1": {}, "step2_parfait": {}, "step2_estime": {}, "step3": {}}

    net = load_perfect()
    cfg = SystemConfig()
    cov = cov_mixture()
    snd = UplinkSounder(cfg, "lmmse", cov)
    t0 = time.time()

    for name in DATASETS:
        if name in out["step1"] and name in out["step2_estime"] and name in out["step3"]:
            print(f"{name} deja fait, passe.", flush=True)
            continue
        h = get_h(name)
        K, M = h.shape[-2], h.shape[-1]
        hf = tf.reshape(h, [-1, K, M])
        print(f"== {name} ({tuple(h.shape)} -> {hf.shape[0]} echantillons)", flush=True)

        # ---- etape 2, courbe "parfait" (independante du pilote) ----
        if name not in out["step2_parfait"]:
            out["step2_parfait"][name] = {}
            for snr in SNRS:
                nv = 10.0 ** (-snr / 10.0)
                curve = eps_curve(net, hf, hf, snr, nv)
                out["step2_parfait"][name][f"snr{snr}"] = {"courbe": curve, **required_eps(curve)}
                print(f"  parfait snr{snr}: {curve} -> {out['step2_parfait'][name][f'snr{snr}']['eps_max_ok']}",
                      flush=True)
            json.dump(out, open(OUT_PATH, "w"), indent=1)

        out["step1"].setdefault(name, {})
        out["step2_estime"].setdefault(name, {})
        out["step3"].setdefault(name, {})
        for p in PILOTS:
            key = f"pilote{p}"
            if key in out["step1"][name] and key in out["step2_estime"][name] and key in out["step3"][name]:
                continue
            he, ev = sound(snd, h, p)
            hef = tf.reshape(he, [-1, K, M])
            evf = tf.reshape(ev, [-1, K, ev.shape[-1]])

            # ---- etape 1 : fuite + debit sur le vrai canal ----
            s1 = {}
            for snr in SNRS:
                nv = 10.0 ** (-snr / 10.0)
                r_net, lk_net = eval_on_true(hf, hef, lambda x, e: net(x, tf.fill([x.shape[0]], float(snr))), nv)
                r_wm, lk_wm = eval_on_true(hf, hef, lambda x, e: wmmse(x, nv, num_iter=300, err_var=e), nv, err=evf)
                r_ref, lk_ref = eval_on_true(hf, hf, lambda x, e: net(x, tf.fill([x.shape[0]], float(snr))), nv)
                s1[f"snr{snr}"] = {"reseau_sur_estime": {"debit": r_net, "fuite_med_db": lk_net},
                                    "wmmse300_robuste": {"debit": r_wm, "fuite_med_db": lk_wm},
                                    "reference_csi_parfait": {"debit": r_ref, "fuite_med_db": lk_ref}}
            out["step1"][name][key] = s1
            print(f"  {name} {key} etape1 (30dB): reseau {s1['snr30']['reseau_sur_estime']['debit']:.2f} "
                  f"({s1['snr30']['reseau_sur_estime']['fuite_med_db']:.1f}dB) wmmse300 "
                  f"{s1['snr30']['wmmse300_robuste']['debit']:.2f} ref_parfait "
                  f"{s1['snr30']['reference_csi_parfait']['debit']:.2f}", flush=True)

            # ---- etape 2 : courbe "estime" ----
            s2 = {}
            for snr in SNRS:
                nv = 10.0 ** (-snr / 10.0)
                curve = eps_curve(net, hf, hef, snr, nv)
                s2[f"snr{snr}"] = {"courbe": curve, **required_eps(curve)}
            out["step2_estime"][name][key] = s2
            print(f"  {name} {key} etape2 eps_max_ok (snr10/20/30): "
                  f"{s2['snr10']['eps_max_ok']}/{s2['snr20']['eps_max_ok']}/{s2['snr30']['eps_max_ok']}", flush=True)

            # ---- etape 3 : sondes par couche, snr 20 dB ----
            h_tr_t, h_tr_i, h_te_t, h_te_i = pick_subset(hf, hef, N_TR, N_TE, seed=17)
            out["step3"][name][key] = layer_probe(net, h_tr_t, h_tr_i, h_te_t, h_te_i, 20)
            print(f"  {name} {key} etape3 (20dB): {out['step3'][name][key]}", flush=True)

            json.dump(out, open(OUT_PATH, "w"), indent=1)
        print(f"{name} termine ({time.time() - t0:.0f} s)", flush=True)

    print("->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
