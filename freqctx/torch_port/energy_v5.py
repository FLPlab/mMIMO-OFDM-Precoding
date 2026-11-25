"""Energie de notre precodeur par paires avec le MEME code que BitAdapt v5 (Pipeline/Energy.py du depot
BitAdaptHLS, branche demo2d-fpga), pour comparer au front de Pareto v5 envoye a F. Leduc-Primeau le
2026-10-24 (export_francois_v5_20261024/donnees/pareto_data_v5.csv).

1. Controle : Energy.analyze_and_compute_energy sur le modele v5 'tiny' (Pipeline/Transformer.py,
   configuration de Pipeline/main_v3.py) a 8 bits doit redonner la colonne E8 de RESULTATS_v5.txt (1,049 uJ).
2. Notre precodeur (St-Catherine, entree normalisee, ART lambda=5, produits quantifies) : chaque couche
   dense compte comme dans analyze_and_compute_energy (MAC = in x out x tokens, activations = tokens x out,
   avec K^2 = 16 tokens de paires), et les produits sans poids (G = h h^H, produit triangulaire, W = h^H C)
   comme des couches sans poids ; tout passe par Energy.compute_energy (v5), avec les bits appris.

    python -m torch_port.energy_v5
"""
import json
import os
import sys

import torch

from torch_port.energy import pair_precoder_layers, pair_precoder_products
from torch_port.models import PairPrecoder
from torch_port.stecath import RESULTS

V5 = "/export/tmp/sala/home_archive/ELE6310/ELE6310E"
sys.path.insert(0, V5)
from Pipeline.Energy import analyze_and_compute_energy, compute_energy  # noqa: E402
from Pipeline.Transformer import StackedTransformer  # noqa: E402

TINY = dict(num_layers=4, embedding_dim=64, num_heads=4, hidden_dim=512, dropout=0.05)
K, M, D, L = 4, 64, 16, 8


def n_q(model):
    """Nombre d'indices de bits consommes par analyze_and_compute_energy (2 par MHA)."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            n += 2
        elif isinstance(m, torch.nn.Linear) and not isinstance(m, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            n += 1
    return n


def ours(bits=None, q=None, hidden=32, layers=L):
    """Energie (pJ) de notre precodeur : bits = dict appris, ou q = precision uniforme."""
    net = PairPrecoder(d=D, hidden=hidden, layers=layers, inp="norm")
    layers_ = pair_precoder_layers(net, K)
    prods = pair_precoder_products(K, M, D, layers)
    layers = layers_
    e = 0.0
    for n, (mac, w, a) in {**layers, **prods}.items():
        qw, qa = (q, q) if q is not None else bits[n]
        e += float(compute_energy(mac, w, a, float(qw), float(qa)))
    return e, sum(m for m, _, _ in layers.values()) + sum(m for m, _, _ in prods.values())


def main():
    tiny = StackedTransformer(**TINY)
    n = n_q(tiny)
    e8 = float(analyze_and_compute_energy(tiny, Q=[8] * n, Q_activations=[8] * n, input_size=(1, K, 2 * M)))
    print(f"controle : v5 tiny a 8 bits = {e8 / 1e6:.3f} uJ (attendu 1,049)", flush=True)
    q = json.load(open(os.path.join(RESULTS, "art_pair_h32_lam5_qprod_norm.json")))
    eq, mac = ours(bits=q["bits"])
    e32, _ = ours(q=32)
    e8o, _ = ours(q=8)
    out = {"controle_v5_tiny_8b_uJ": e8 / 1e6, "notre_precodeur": {
        "mac": mac, "sr_fp32": q["sr_fp32"], "sr_quant": q["sr_quant"],
        "energy_uJ_32b": e32 / 1e6, "energy_uJ_8b": e8o / 1e6, "energy_uJ_quant": eq / 1e6}}
    out["tailles"] = {}
    for tag, hid, nl in (("art_pair_h32_lam5_qprod_norm", 32, 8), ("art_pair_h32_L4_lam5_qprod_norm", 32, 4),
                         ("art_pair_h16_L4_lam5_qprod_norm", 16, 4), ("art_pair_h16_L2_lam5_qprod_norm", 16, 2)):
        qq = json.load(open(os.path.join(RESULTS, f"{tag}.json")))
        e, mac = ours(bits=qq["bits"], hidden=hid, layers=nl)
        out["tailles"][tag] = {"layers": nl, "hidden": hid, "mac": mac, "energy_uJ_quant": e / 1e6,
                               "sr_quant_20k": qq["sr_quant"]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RESULTS, "energy_v5_ours.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
