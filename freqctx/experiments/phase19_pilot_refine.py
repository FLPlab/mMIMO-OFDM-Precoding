"""Phase 19 -- la 2e passe (models/refine.py) dans la chaine de l'article : pilotes -> encodeur
frequentiel -> precodeur (base v2, poids phase7/pilot_rand_ph32_norm_relpow), EVALUATION SEULE.

Question : faut-il garder la 2e passe dans le modele v0 ? Critere fixe a l'avance : GARDEE si elle apporte
>= +0,2 bps/Hz quelque part (pilote x SNR x jeu) sans perte > 0,05 ailleurs.

Lignes (meme estimation pour toutes, debit sur le VRAI canal, puissances reelles) :
  v2                         : encodeur + son precodeur (= DelayDomainPrecoder, sans pas frequentiel)
  v2 + passe (transfert)     : RefineNet(base = precodeur v2) avec les poids PROPRES de la passe phase 15
                               (weights/phase15/refine_L4, entrainee sur une AUTRE base, phase 13, CSI
                               parfait) ; G et E calcules sur la sortie de l'encodeur h^ ; x2 = meme passe
                               appliquee deux fois (C3 = C2 + DELTA_C(G, E2))
  v2 + passe adaptee         : (option --adapted) poids weights/phase19/refine_on_v2 (phase19_pilot_refine_train)
  LMMSE + WMMSE robuste 100 it., Wiener formule fermee + WMMSE 100 it. (references de phase13_lab)
  CSI parfait : WMMSE 300 it., precodeur v2 seul, precodeur v2 + passe (transfert / adaptee),
                precodeur phase 13 + passe phase 15 (controle : doit redonner les chiffres de phase 15)

Jeux : UMi / UMa / RMa (tirages d'evaluation de phase11_power : puissances reelles, graine 4026 + 10 i,
192 tirages) et les 3 sites du labo (chaine de phase13_lab, convention psi2az, 192 tirages, graine 13).
Pilote 0 / 10 / 20 dB (sondage avec controle de puissance montant), SNR de donnees 10 / 20 / 25 / 30 dB.
Le bruit de sondage est tire avec une graine fixe par (jeu, pilote) : --adapted ajoute les lignes de la
passe adaptee sur les memes estimations (la ligne v2 est recalculee et comparee).

    python -u -m experiments.phase19_pilot_refine [--quick] [--adapted] [--sets umi,ericsson]
"""
import argparse
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
from experiments.phase7_context import SCENARIOS, cov_mixture, sound_power_controlled
from experiments.phase11_power import load
from experiments.phase12_robust import KAPPA
from experiments.phase12_wiener import wiener
from experiments.phase13_deepmimo import drops
from experiments.phase13_lab import read_cells
from experiments.phase14_finetune import load_base
from models.refine import RefineNet
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

SITES = ["stecath", "decarie", "ericsson"]
SETS = SCENARIOS + SITES
PILOTS, SNRS = [0, 10, 20], [10, 20, 25, 30]
DROPS, B, CHUNK = 192, 16, 8192
REFINE15 = "/export/tmp/sala/freqctx/weights/phase15/refine_L4.weights.h5"
REFINE19 = "/export/tmp/sala/freqctx/weights/phase19/refine_on_v2.weights.h5"
OUT = os.path.join(RESULTS, "phase19", "pilot_refine.json")


def own_layers(r):
    return [r.init, *r.blocks, r.read]


def load_v2():
    return load("pilot_rand_ph32_norm_relpow", "norm")


def refine_on(base, weights_src=None):
    """RefineNet sur `base` (deja chargee, figee). weights_src : RefineNet dont on copie les couches
    propres (init, blocs, lecture) -- la base n'est JAMAIS rechargee."""
    r = RefineNet(base, d=16, hidden=32, layers=4)
    r(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    if weights_src is not None:
        for a, b in zip(own_layers(weights_src), own_layers(r)):
            b.set_weights(a.get_weights())
    return r


def load_refine15():
    """Passe phase 15 avec SA base (phase 13) : controle, et source des poids pour le transfert."""
    base13 = load_base()
    r = refine_on(base13)
    r.load_weights(REFINE15)
    return r


def load_refine19(base_v2):
    """Passe adaptee : les poids sauves contiennent aussi la base v2 (identique, figee) ; on charge dans un
    RefineNet sur une COPIE de la base v2 puis on copie les couches propres sur la vraie base."""
    tmp = refine_on(load_v2().precoder)
    tmp.load_weights(REFINE19)
    return refine_on(base_v2, tmp)


def npass(r, x, snr, n):
    """W apres n applications des memes poids : C_{i+1} = C_i + DELTA_C(G, E_i) (n=1 == r(x, snr))."""
    hp, pdb = r.base._norm(x)
    c = r.base.coeffs(x, snr)
    for _ in range(n):
        c = c + r._delta_from(hp, pdb, c, snr)
    return normalize_power(tf.matmul(hp, c, adjoint_a=True))


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def rates(fn, x, h, err=None):
    """fn(x_chunk [n,K,M], nv, snr, err_chunk) -> W [n,M,K] ; debit moyen sur le vrai h, par SNR."""
    x, h = flat(x), flat(h)
    err = None if err is None else flat(err)
    out = []
    for snr in SNRS:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h[i:i + CHUNK], fn(x[i:i + CHUNK], nv, snr,
                                                None if err is None else err[i:i + CHUNK]), nv)
             for i in range(0, x.shape[0], CHUNK)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def channels(name, n_drops):
    """(h [n, F, K, M] puissances reelles, cfg du sondeur)."""
    if name in SCENARIOS:
        i = SCENARIOS.index(name)
        cfg = SystemConfig(scenario=name, num_slots=1, relative_power=True)
        sionna_config.seed = 4026 + 10 * i                                  # tirages de phase11_power
        gen = ChannelGenerator(cfg)
        return tf.concat([gen(B)[:, 0] for _ in range(n_drops // B)], 0), cfg
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    h = drops(read_cells(name, "psi2az"), freqs, equal=False, seed=13)     # tirages de phase13_lab
    return h[:n_drops], cfg


def sound(name, p, h, cfg, cov):
    """(ls, lmmse, err_var) avec une graine de bruit fixe par (jeu, pilote)."""
    lin, lmm = UplinkSounder(cfg, "lin"), UplinkSounder(cfg, "lmmse", cov)
    sionna_config.seed = 7000 + 100 * SETS.index(name) + p
    ls = tf.concat([sound_power_controlled(lin, h[j:j + B], float(p)) for j in range(0, h.shape[0], B)], 0)
    sionna_config.seed = 8000 + 100 * SETS.index(name) + p
    pairs = [sound_power_controlled(lmm, h[j:j + B], float(p), return_err_var=True)
             for j in range(0, h.shape[0], B)]
    return ls, tf.concat([a for a, _ in pairs], 0), tf.concat([b for _, b in pairs], 0)


def encode(net, ls):
    return tf.concat([net.encode(ls[j:j + B]) for j in range(0, ls.shape[0], B)], 0)


def fill(x, s):
    return tf.fill([x.shape[0]], float(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="32 tirages, UMi et Ericsson, fichier _quick")
    ap.add_argument("--adapted", action="store_true", help="ajoute les lignes de la passe adaptee (phase 19)")
    ap.add_argument("--sets", default=None)
    a = ap.parse_args()
    n_drops = 32 if a.quick else DROPS
    sets = a.sets.split(",") if a.sets else (["umi", "ericsson"] if a.quick else SETS)
    path = OUT.replace(".json", "_quick.json") if a.quick else OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)

    net = load_v2()
    prec = net.precoder
    w_before = [w.numpy().copy() for w in prec.weights]
    r15 = load_refine15()
    rt = refine_on(prec, r15)                                               # transfert
    assert all(np.array_equal(u, v.numpy()) for u, v in zip(w_before, prec.weights)), "base v2 modifiee"
    ra = load_refine19(prec) if a.adapted else None
    cov = cov_mixture()

    out = json.load(open(path)) if os.path.exists(path) else {"sets": {}}
    out["meta"] = {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "snrs_db": SNRS,
                   "pilots_db": PILOTS, "drops": n_drops, "base_v2": "phase7/pilot_rand_ph32_norm_relpow",
                   "refine_transfer": REFINE15, "refine_adapted": REFINE19 if a.adapted else None,
                   "conv_sites": "psi2az"}
    for name in sets:
        t0 = time.time()
        h, cfg = channels(name, n_drops)
        so = out["sets"].get(name, {}) if a.adapted else {}
        if not a.adapted:
            P = lambda x, nv, s, e: prec(x, fill(x, s))
            so["csi_parfait"] = {
                "WMMSE 300 it.": rates(lambda x, nv, s, e: wmmse(x, nv, num_iter=300), h, h),
                "precodeur v2": rates(P, h, h),
                "precodeur v2 + passe (transfert)": rates(lambda x, nv, s, e: rt(x, fill(x, s)), h, h),
                "precodeur v2 + passe (transfert) x2": rates(lambda x, nv, s, e: npass(rt, x, fill(x, s), 2), h, h),
                "precodeur phase 13 + passe phase 15 (controle)": rates(lambda x, nv, s, e: r15(x, fill(x, s)), h, h),
                "precodeur phase 13 seul (controle)": rates(lambda x, nv, s, e: r15.base(x, fill(x, s)), h, h)}
        else:
            so["csi_parfait"]["precodeur v2 + passe adaptee"] = rates(lambda x, nv, s, e: ra(x, fill(x, s)), h, h)
            so["csi_parfait"]["precodeur v2 + passe adaptee x2"] = rates(
                lambda x, nv, s, e: npass(ra, x, fill(x, s), 2), h, h)
        print(f"{name} CSI parfait : " + " | ".join(f"{k} {v[1]:.2f}@20 {v[3]:.2f}@30"
                                                   for k, v in so["csi_parfait"].items()), flush=True)
        for p in PILOTS:
            ls, lm, ev = sound(name, p, h, cfg, cov)
            hh = encode(net, ls)
            v2 = rates(lambda x, nv, s, e: prec(x, fill(x, s)), hh, h)
            if not a.adapted:
                r = {"v2": v2,
                     "v2 + passe (transfert)": rates(lambda x, nv, s, e: rt(x, fill(x, s)), hh, h),
                     "v2 + passe (transfert) x2": rates(lambda x, nv, s, e: npass(rt, x, fill(x, s), 2), hh, h),
                     "LMMSE + WMMSE robuste 100 it.": rates(
                         lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e), lm, h, ev),
                     "Wiener formule fermee + WMMSE 100 it.": rates(
                         lambda x, nv, s, e: wmmse(x, nv, num_iter=100), wiener(ls, KAPPA), h)}
            else:
                r = so[f"pilote{p}"]
                print(f"   controle v2 recalcule {['%.3f' % x for x in v2]} vs stocke "
                      f"{['%.3f' % x for x in r['v2']]}", flush=True)
                r["v2 + passe adaptee"] = rates(lambda x, nv, s, e: ra(x, fill(x, s)), hh, h)
                r["v2 + passe adaptee x2"] = rates(lambda x, nv, s, e: npass(ra, x, fill(x, s), 2), hh, h)
            so[f"pilote{p}"] = r
            print(f"{name} pilote {p} dB : " + " | ".join(f"{k} {v[1]:.2f}@20 {v[3]:.2f}@30" for k, v in r.items()),
                  flush=True)
        out["sets"][name] = so
        json.dump(out, open(path, "w"), indent=1)
        print(f"{name} termine ({time.time() - t0:.0f} s)", flush=True)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
