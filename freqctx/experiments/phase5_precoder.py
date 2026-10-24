"""Phase 5b -- diagnostic du precodeur seul en massif, CSI PARFAIT (M=64,
K=8, 600 SC, melange UMi + UMa + RMa). Le modele transpose tel quel plafonne
vers 35 bps/Hz : est-ce le precodeur ? Variantes (un seul changement) :
    base  : precodeur actuel (D64, 1 bloc lineaire x4, sortie directe)
    d128  : precodeur plus large
    span  : sortie W = h^H C (le modele ne produit que les K x K coefficients)
    d256  : precodeur beaucoup plus large (force brute)
    gram  : precodeur a attention de Gram (etat par paire d'utilisateurs,
            bloc repete 4 fois, sortie W = h^H C ; models/gram_precoder.py)
Evaluation sur le vrai canal des donnees d'evaluation du massif (memes tirages
que les references, results/phase5/massive_refs.json).

    python -m experiments.phase5_precoder --variant base|d128|span
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from config import CSIConfig
from experiments.phase5_massive import DATA, PILOT, SCENARIOS, SNRS_DB, cfg_of
from models.gram_precoder import GramPrecoder
from models.per_sc import PerSCPrecoder
from train import csi_pool, evaluate, run_variant
from utils import RESULTS, save

VARIANTS = {"base": dict(d=64), "d128": dict(d=128), "span": dict(d=64, out_mode="span"),
            "d256": dict(d=256), "gram": dict(d=16, hidden=64)}
W5 = "/export/tmp/sala/freqctx/weights/phase5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--train_drops", type=int, default=352)
    ap.add_argument("--steps", type=int, default=40000)
    args = ap.parse_args()
    refs = json.load(open(os.path.join(RESULTS, "phase5", "massive_refs.json")))["references"]
    pool = tf.concat([csi_pool(cfg_of(s), CSIConfig("perfect", PILOT, 0), args.train_drops, seed=1 + i,
                               batch=16)[1] for i, s in enumerate(SCENARIOS)], 0)    # vrai canal, aplati
    ev = {s: tf.reshape(tf.constant(np.load(os.path.join(DATA, f"massive_eval_{s}.npz"))["true"]),
                        [-1, 8, 64]) for s in SCENARIOS}
    if args.variant == "gram":
        net = GramPrecoder(64, layers=4, **VARIANTS["gram"])
    else:
        net = PerSCPrecoder(64, layers=4, tied=True, attn="linear", **VARIANTS[args.variant])
    run = run_variant(f"massive_precodeur_parfait_{args.variant}", net, pool, ev["umi"][::10],
                      SNRS_DB, {"RZF": refs["umi"]["RZF (CSI parfait)"]}, args.steps, seed=0,
                      loss="rel_rzf", weights_dir=W5)
    rates = {s: evaluate(net, ev[s], SNRS_DB) for s in SCENARIOS}
    for s in SCENARIOS:
        print(s, "@20dB", round(rates[s][4], 2), "| RZF parfait", round(refs[s]["RZF (CSI parfait)"][4], 2),
              "| WMMSE parfait", round(refs[s]["WMMSE (CSI parfait)"][4], 2), flush=True)
    save("phase5", f"precoder_parfait_{args.variant}",
         {"snrs_db": SNRS_DB, "variant": args.variant, "model": VARIANTS[args.variant],
          "params": run["params"], "macs_per_sc": run["macs_per_sc"], "rates": rates,
          "references": {s: {k: refs[s][k] for k in ("RZF (CSI parfait)", "WMMSE (CSI parfait)")}
                         for s in SCENARIOS}, "training_history": run["history"]})


if __name__ == "__main__":
    main()
