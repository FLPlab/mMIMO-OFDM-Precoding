"""Phase 18 -- evaluation SEULE (aucun entrainement) de la passe 2 ETROITE (RefineNet d=8/h=16/L=4,
~96 k MAC) ajoutee aux troncs RETRECIS E1 (d8/h16/L8, ~190 k MAC) et E2 (d8/h16/L12, ~290 k MAC),
poids experiments/phase18_refine_narrow.py (weights/phase18/refine_narrow_on_L{8,12}.weights.h5),
demande F. Leduc-Primeau 26-27 nov. 2026.

Compare : base (phase13, 717 k), base+passe2 large (phase15, 1.08 M), E1, E1+passe etroite
(~286 k), E2, E2+passe etroite (~386 k), WMMSE 300 it.

Reutilise (import seulement, SANS MODIFIER), memes briques que experiments/phase15_refine_eval.py
et experiments/phase18_eval.py :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf} +
    experiments.phase14_finetune.load_base                         (partie 1, CSI parfait)
  - experiments.phase15_lab_csi.{SITES, PILOTS, SNRS, B} + experiments.phase7_context.{cov_mixture,
    sound_power_controlled} + experiments.phase13_deepmimo.{DROPS, drops} +
    experiments.phase13_lab.read_cells                              (partie 2, CSI estime)
  - results/phase15/lab_csi.json pour WMMSE robuste 300 it. (PAS recalcule)             (partie 2)
  - experiments.phase15_refine_eval.load_refine (base+passe2 large, reference)
  - models.gram_precoder.GramPrecoder (E1/E2) + models.refine.RefineNet (passe etroite) + macs_per_sc

Critere fixe a l'avance (prompt 27 nov. 2026), par candidat (E1+passe etroite, E2+passe etroite) :
RETENU si >= base - 0.05 bps/Hz de 0 a 20 dB (UMi ET les 3 sites) ET >= base+passe2 large - 0.05
bps/Hz a 30 dB (moyenne des 3 sites) ET MAC total <= 360 k (moitie de la base).

    python -u -m experiments.phase18_refine_narrow_eval
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
from experiments.phase15_refine_eval import load_refine
from experiments.phase7_context import cov_mixture, sound_power_controlled
from models.gram_precoder import GramPrecoder
from models.refine import RefineNet
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase18")
OUT_PATH = os.path.join(OUT_DIR, "refine_narrow_E1.json")
LAB_CSI_PATH = os.path.join(RESULTS, "phase15", "lab_csi.json")
W18 = "/export/tmp/sala/freqctx/weights/phase18"
TRUNKS = {"E1": dict(d=8, hidden=16, layers=8, tag="narrow_d8_h16_L8", trunk_key="L8"),
}
SNRS_PARFAIT = [0, 10, 20, 25, 30, 35, 40]
N_EVAL = 16384
PILOTE_ESTIME = 20
SNRS_ESTIME = [20, 30]
SITES3 = ["stecath", "decarie", "ericsson"]
MAC_BUDGET = 360000
ABOVE_SNR = 20.0                        # variante : passe etroite seulement au-dessus de ce SNR


def load_trunk(kw):
    net = GramPrecoder(64, d=kw["d"], hidden=kw["hidden"], layers=kw["layers"],
                        mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W18, f"{kw['tag']}.weights.h5"))
    return net


def load_narrow_refine(trunk_net, trunk_key):
    trunk_net.trainable = False
    net = RefineNet(trunk_net, d=8, hidden=16, layers=4)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W18, f"refine_narrow_on_{trunk_key}.weights.h5"))
    return net


def build_nets():
    base = load_base()
    refine_base = load_refine()
    nets = {"base": base, "base+passe2": refine_base}
    macs = {}
    for name, kw in TRUNKS.items():
        trunk = load_trunk(kw)
        narrow = load_narrow_refine(trunk, kw["trunk_key"])
        nets[name] = trunk
        nets[f"{name}+passe_etroite"] = narrow
        macs[name] = {"trunk_macs": int(trunk.macs_per_sc(4, 8)),
                      "passe_etroite_macs": int(narrow.macs_per_sc(4, 8)),
                      "total_macs": int(trunk.macs_per_sc(4, 8) + narrow.macs_per_sc(4, 8))}
    macs["base"] = {"total_macs": int(base.macs_per_sc(4, 8))}
    macs["base+passe2"] = {"total_macs": int(refine_base.macs_per_sc(4, 8) + refine_base.base.macs_per_sc(4, 8))}
    return nets, macs


# ---------------------------------------------------------------------- 1. CSI parfait


def run_parfait(nets):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_refine_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            ws = {tag: net(h, tf.fill([h.shape[0]], float(snr))) for tag, net in nets.items()}
            ws["wmmse300"] = wmmse(h, nv, num_iter=300)
            entry = {}
            for tag, w in ws.items():
                leak_db, _, _ = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                entry[tag] = {"debit": rate, "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            print(f"parfait {name:10s} {snr:2d}dB : " +
                  "  ".join(f"{t} {entry[t]['debit']:6.2f}" for t in entry), flush=True)
    return out


# ---------------------------------------------------------------------- 2. CSI estime (pilote 20 dB)


def run_estime(nets):
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
            rates = {}
            for tag, net in nets.items():
                w = net(hef, tf.fill([hef.shape[0]], float(snr)))
                rates[tag] = float(tf.reduce_mean(rate_per_sample(hf, w, nv)))
            idx = lab_csi["meta"]["snrs_db"].index(snr)
            rates["wmmse_robuste_300it_depuis_lab_csi_json"] = \
                lab_csi["sites"][site]["estime"][f"pilote{PILOTE_ESTIME}"]["WMMSE robuste 300 it."]["rate"][idx]
            entry[str(snr)] = rates
            print(f"estime  {site:10s} pilote{PILOTE_ESTIME} {snr:2d}dB : " +
                  "  ".join(f"{t} {v:6.2f}" for t, v in rates.items()), flush=True)
        out[site] = entry
    return out


# ---------------------------------------------------------------------- 3. variante "passe etroite > 20 dB seulement"


def run_variant_above20(parfait, macs):
    """Cout moyen (SNR uniforme sur [0, 30] dB) si la passe etroite n'est appliquee qu'au-dessus de
    ABOVE_SNR dB (le debit aux points evalues est deja celui du tronc seul ou tronc+passe, inchange ;
    seul le cout moyen differe)."""
    frac_above = max(0.0, 30.0 - ABOVE_SNR) / 30.0
    out = {}
    for name in TRUNKS:
        trunk_mac = macs[name]["trunk_macs"]
        passe_mac = macs[name]["passe_etroite_macs"]
        avg_mac = trunk_mac + frac_above * passe_mac
        out[name] = {"above_snr_db": ABOVE_SNR, "frac_temps_passe_active": frac_above,
                     "mac_moyen_snr_uniforme_0_30db": avg_mac,
                     "debit_variante": {snr: (parfait["umi"][str(snr)][f"{name}+passe_etroite"]["debit"]
                                               if snr > ABOVE_SNR else parfait["umi"][str(snr)][name]["debit"])
                                        for snr in SNRS_PARFAIT}}
    return out


# ---------------------------------------------------------------------- verdict


def verdict(parfait, macs):
    sites_low = ["umi"] + SITES3
    out = {}
    for name in TRUNKS:
        tag = f"{name}+passe_etroite"
        low_losses = []
        for s in sites_low:
            for snr in (0, 10, 20):
                if str(snr) not in parfait.get(s, {}):
                    continue
                low_losses.append(parfait[s][str(snr)][tag]["debit"] - parfait[s][str(snr)]["base"]["debit"])
        worst_low = float(np.min(low_losses)) if low_losses else None
        gain_30 = [parfait[s]["30"][tag]["debit"] - parfait[s]["30"]["base+passe2"]["debit"] for s in SITES3]
        gain_30_mean = float(np.mean(gain_30))
        mac_total = macs[name]["total_macs"]
        ok_low = (worst_low is not None) and (worst_low >= -0.05)
        ok_30 = gain_30_mean >= -0.05
        ok_mac = mac_total <= MAC_BUDGET
        retenu = bool(ok_low and ok_30 and ok_mac)
        out[tag] = {"pire_perte_0_20db_vs_base": worst_low, "ecart_30db_moyen_3sites_vs_base+passe2": gain_30_mean,
                    "mac_total": mac_total, "mac_budget": MAC_BUDGET,
                    "criteres": {"perte_0_20db>=-0.05_vs_base": ok_low,
                                 "30db>=base+passe2-0.05_moy3sites": ok_30,
                                 "mac<=360k": ok_mac},
                    "retenu": retenu}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    nets, macs = build_nets()

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["macs"] = macs
    out["csi_parfait"] = run_parfait(nets)
    out["csi_estime_pilote20"] = run_estime(nets)
    out["variante_passe_etroite_above20db"] = run_variant_above20(out["csi_parfait"], macs)
    out["verdict"] = verdict(out["csi_parfait"], macs)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
