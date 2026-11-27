"""Phase 16 -- evaluation SEULE (aucun entrainement) de la variante STRUCTUREE
(models/structured_precoder.py, poids weights/phase16/structured_T{10,20}.weights.h5), comparee au
reseau LIBRE (base), au reseau libre + 2e passe (RefineNet, phase15), a WMMSE 300 it. et a RZF.
Question : quel est le prix de la liberte, quand la forme imposee (Bjornson 2014, Newton-Schulz
deroule a T pas) est elle-meme ENTRAINEE de bout en bout (pas un fit post-hoc comme phase15_diag) ?

Reutilise (import seulement, SANS MODIFIER) :
  - experiments.phase15_diag.{get_datasets, subsample, leak_db_sig_intf} et
    experiments.phase14_finetune.load_base                              (CSI parfait)
  - experiments.phase15_refine_eval.load_refine                          (base + 2e passe)
  - experiments.phase15_lab_csi.{SITES, PILOTS, B, SNRS} + results/phase15/lab_csi.json pour
    WMMSE robuste 300 it. (PAS recalcule), memes briques que phase15_refine_eval (CSI estime)
  - models.structured_precoder.StructuredPrecoder

    python -u -m experiments.phase16_eval
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
from experiments.phase15_refine_eval import load_refine
from experiments.phase7_context import cov_mixture, sound_power_controlled
from models.structured_precoder import StructuredPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase16")
OUT_PATH = os.path.join(OUT_DIR, "structured.json")
LAB_CSI_PATH = os.path.join(RESULTS, "phase15", "lab_csi.json")
W16 = "/export/tmp/sala/freqctx/weights/phase16"
SNRS_PARFAIT = [0, 10, 20, 25, 30, 35, 40]
N_EVAL = 16384
PILOTE_ESTIME = 20
SNRS_ESTIME = [20, 30]
T_LIST = [10, 20]


def load_structured(t_steps):
    net = StructuredPrecoder(8, ns_steps=t_steps, d=16, hidden=32, layers=2)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W16, f"structured_T{t_steps}.weights.h5"))
    return net


# ---------------------------------------------------------------------- 1. CSI parfait


def run_parfait(nets):
    raw = get_datasets()
    sub = {name: subsample(h, N_EVAL, seed=4242) for name, h in raw.items()}   # meme graine que phase15_diag/refine_eval
    out = {}
    for name, h in sub.items():
        res = out.setdefault(name, {"n": int(h.shape[0])})
        for snr in SNRS_PARFAIT:
            nv = 10.0 ** (-snr / 10.0)
            ws = {tag: fn(h, tf.fill([h.shape[0]], float(snr))) for tag, fn in nets.items()}
            ws["wmmse300"] = wmmse(h, nv, num_iter=300)
            ws["rzf"] = rzf(h, nv)
            entry = {}
            for tag, w in ws.items():
                leak_db, _, _ = leak_db_sig_intf(h, w)
                rate = float(tf.reduce_mean(rate_per_sample(h, w, nv)))
                entry[tag] = {"debit": rate, "fuite_db_mediane": float(np.median(leak_db))}
            res[str(snr)] = entry
            line = "  ".join(f"{t} {entry[t]['debit']:6.2f}" for t in ws)
            print(f"parfait {name:24s} {snr:2d}dB : {line}", flush=True)
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
            idx = lab_csi["meta"]["snrs_db"].index(snr)
            rate_wmmse300 = lab_csi["sites"][site]["estime"][f"pilote{PILOTE_ESTIME}"]["WMMSE robuste 300 it."]["rate"][idx]
            e = {"wmmse_robuste_300it_depuis_lab_csi_json": rate_wmmse300}
            for tag, fn in nets.items():
                w = fn(hef, tf.fill([hef.shape[0]], float(snr)))
                e[tag] = float(tf.reduce_mean(rate_per_sample(hf, w, nv)))
            entry[str(snr)] = e
            line = "  ".join(f"{t} {e[t]:6.2f}" for t in e)
            print(f"estime  {site:10s} pilote{PILOTE_ESTIME} {snr:2d}dB : {line}", flush=True)
        out[site] = entry
    return out


# ---------------------------------------------------------------------- 3. MAC/SC


def run_macs(net_refine, nets_struct):
    K, M = 4, 8
    base = net_refine.base
    entry = {"K": K, "M": M, "base_macs_par_sc": int(base.macs_per_sc(K, M)),
              "base+passe2_macs_par_sc": int(base.macs_per_sc(K, M) + net_refine.macs_per_sc(K, M))}
    for t, net in nets_struct.items():
        entry[f"structure_{t}_macs_par_sc"] = int(net.macs_per_sc(K, M))
    mac_g, mac_w = 4 * M * K * K, 4 * M * K * K
    entry["rzf_macs_par_sc_approx"] = int(mac_g + 4 * (2 * K ** 3 / 3) + mac_w)
    entry["wmmse300_note"] = "reference hors ligne (300 it. x solve KxK en double precision) -- pas quantifie en MAC comparable"
    print("MAC/SC (M8K4) :", entry, flush=True)
    return entry


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    net_base = load_base()
    net_refine = load_refine()
    nets_struct = {f"structure_T{t}": load_structured(t) for t in T_LIST}

    nets = {"reseau_libre": lambda h, snr: net_base(h, snr),
            "reseau_libre+passe2": lambda h, snr: net_refine(h, snr)}
    nets.update({f"structure_T{t}": (lambda h, snr, n=n: n(h, snr)) for t, n in nets_struct.items()})

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}}
    out["csi_parfait"] = run_parfait(nets)
    out["csi_estime_pilote20"] = run_estime(nets)
    out["macs"] = run_macs(net_refine, nets_struct)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
