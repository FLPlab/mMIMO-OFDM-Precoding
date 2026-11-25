"""Recalcule l'energie des 48 modeles BitAdapt v5 de St-Catherine avec Pipeline/Energy.py v5, a partir des
bits par couche de l'export (bitwidths_all_v5.csv), et compare a l'energie publiee (pareto_data_v5.csv).
Ordre des bits = ordre de build_Q_lists (main_v3.py) : Emb, puis par couche Att (in_proj et out_proj, meme
bit), FF0, FF1, puis Out.

    python -m torch_port.check_v5_energy
"""
import csv
import json
import os
import sys
from collections import defaultdict

import torch

V5 = "/export/tmp/sala/home_archive/ELE6310/ELE6310E"
sys.path.insert(0, V5)
from Pipeline.Energy import analyze_and_compute_energy  # noqa: E402
from Pipeline.Transformer import StackedTransformer  # noqa: E402

D = f"{V5}/export_francois_v5_20261024/donnees"
CFG = {"tiny": (4, 64, 4, 512), "small": (4, 128, 4, 1024), "medium": (4, 128, 4, 2048),
       "base": (6, 128, 4, 2048), "large": (6, 256, 8, 2048), "xlarge": (8, 256, 8, 4096)}


def main():
    rows = defaultdict(list)
    for r in csv.DictReader(open(f"{D}/bitwidths_all_v5.csv")):
        if r["scenario"] == "stecath":
            rows[(r["config"], r["lambda"])].append(r)
    out, worst = [], 0.0
    for (cfg, lam), rs in sorted(rows.items()):
        L, d, h, f = CFG[cfg]
        m = StackedTransformer(num_layers=L, embedding_dim=d, num_heads=h, hidden_dim=f, dropout=0.0)
        qw, qa = [], []
        for r in rs:
            n = 2 if r["layer"].endswith("Att") else 1
            qw += [float(r["bit_w"])] * n
            qa += [float(r["bit_a"])] * n
        e = float(analyze_and_compute_energy(m, Q=qw, Q_activations=qa, input_size=(1, 4, 128))) / 1e6
        pub = float(rs[0]["energy"])
        worst = max(worst, abs(e / pub - 1))
        out.append({"config": cfg, "lambda": lam, "sr": float(rs[0]["sr"]), "energy_publiee": pub, "energy_recalculee": e})
    print(f"{len(out)} modeles, ecart relatif maximal recalcule / publie : {100 * worst:.2f} %")
    for o in out[:6]:
        print(o)
    json.dump(out, open("/users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx/results/stecath/check_v5_energy.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
