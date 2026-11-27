"""Phase 18 -- evaluation SEULE (aucun entrainement) des precodeurs RETRECIS entraines DE ZERO
par experiments/phase18_narrow.py :
  E1 : GramPrecoder(d=8, hidden=16, layers=8)  -- poids weights/phase18/narrow_d8_h16_L8.weights.h5
  E2 : GramPrecoder(d=8, hidden=16, layers=12) -- poids weights/phase18/narrow_d8_h16_L12.weights.h5
Compares a la base figee (phase13, d=16/hidden=32/layers=8), a base+passe2 (RefineNet, phase15,
weights/phase15/refine_L4.weights.h5) et a WMMSE 300 it.

Reutilise (import seulement, SANS MODIFIER), memes briques que experiments/phase15_refine_eval.py :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf} et
    experiments.phase14_finetune.load_base                         (partie 1, CSI parfait)
  - experiments.phase15_lab_csi.{SITES, PILOTS, SNRS, B} et les memes briques (config.SystemConfig,
    csi.UplinkSounder, experiments.phase7_context.{cov_mixture, sound_power_controlled},
    experiments.phase13_deepmimo.{DROPS, drops}, experiments.phase13_lab.read_cells)  (partie 2)
  - results/phase15/lab_csi.json pour WMMSE robuste 300 it. (PAS recalcule)             (partie 2)
  - models.refine.RefineNet + weights/phase15/refine_L4.weights.h5 (base+passe2, reference)
  - models.gram_precoder.GramPrecoder (chargement E1/E2) + macs_per_sc

Critere fixe a l'avance (prompt 27 nov. 2026), par variante (E1, E2) : ACCEPTABLE comme nouveau
tronc si >= base - 0,05 bps/Hz de 0 a 20 dB (UMi ET les 3 sites) ; indique en plus la perte a
30-40 dB (qu'une 2e passe devra combler ensuite).

    python -u -m experiments.phase18_eval
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
from models.gram_precoder import GramPrecoder
from models.refine import RefineNet
from precoders.classical import wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase18")
OUT_PATH = os.path.join(OUT_DIR, "narrow_d12.json")
LAB_CSI_PATH = os.path.join(RESULTS, "phase15", "lab_csi.json")
REFINE_WEIGHTS = "/export/tmp/sala/freqctx/weights/phase15/refine_L4.weights.h5"
W18 = "/export/tmp/sala/freqctx/weights/phase18"
NARROW = {"E5_d12_h16_L8": dict(d=12, hidden=16, layers=8)}
SNRS_PARFAIT = [0, 10, 20, 25, 30, 35, 40]
N_EVAL = 16384
PILOTE_ESTIME = 20
SNRS_ESTIME = [20, 30]
SITES3 = ["stecath", "decarie", "ericsson"]


def load_narrow(tag, kw):
    net = GramPrecoder(64, mix="triangular_g", tied=False, inp="norm", **kw)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W18, f"narrow_{tag.split('_', 1)[1]}.weights.h5"))
    return net


def load_refine():
    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(REFINE_WEIGHTS)
    return net


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


# ---------------------------------------------------------------------- 3. MAC


def run_macs(nets):
    K, M = 4, 8
    out = {}
    for tag, net in nets.items():
        out[tag] = {"macs_par_sc": int(net.macs_per_sc(K, M)), "params": int(net.count_params())}
        print(f"MAC/SC (M{M}K{K}) {tag}: {out[tag]['macs_par_sc']}  params {out[tag]['params']}", flush=True)
    return out


# ---------------------------------------------------------------------- verdict


def verdict(parfait, variant_tag):
    low_snr_sets = ["umi"] + SITES3
    low_losses = []
    for s in low_snr_sets:
        for snr in (0, 10, 20):
            if str(snr) not in parfait.get(s, {}):
                continue
            low_losses.append(parfait[s][str(snr)][variant_tag]["debit"] - parfait[s][str(snr)]["base"]["debit"])
    worst_low = float(np.min(low_losses)) if low_losses else None
    high_losses = {}
    for s in low_snr_sets:
        for snr in (30, 35, 40):
            if str(snr) not in parfait.get(s, {}):
                continue
            high_losses[f"{s}_{snr}"] = parfait[s][str(snr)][variant_tag]["debit"] - parfait[s][str(snr)]["base"]["debit"]
    ok_low = (worst_low is not None) and (worst_low >= -0.05)
    return {"pire_perte_0_20db": worst_low, "pertes_30_40db": high_losses,
            "critere": "perte_0_20db>=-0.05", "acceptable_comme_nouveau_tronc": bool(ok_low)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    base = load_base()
    refine = load_refine()
    narrow = {tag: load_narrow(tag, kw) for tag, kw in NARROW.items()}
    nets = {"base": base, **narrow, "base+passe2": refine}

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait(nets)
    out["csi_estime_pilote20"] = run_estime(nets)
    out["macs"] = run_macs(nets)
    out["verdict"] = {tag: verdict(out["csi_parfait"], tag) for tag in NARROW}
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
