"""Energie corrigee (torch_port/energy.py) des modeles BitAdapt quantifies du labo (St-Catherine),
a partir des bits appris enregistres (phase2_quant/lambda_*/results_L_*.json, derniere epoque),
comparee a notre precodeur par paires quantifie (results/stecath/art_pair_h32_lam*.json).
Debits du labo : ceux qu'ils rapportent (leur jeu de test, meme distribution que le notre).

    python -m torch_port.lab_energy_compare
"""
import glob
import json
import math
import os
import sys

from torch_port.energy import lab_transformer_layers, total_energy
from torch_port.stecath import LAB, RESULTS

sys.path.insert(0, LAB)
from Pipeline.Transformer_FPGA import StackedTransformer   # noqa: E402

DIMS = {"tiny": (64, 4, 512), "small": (128, 4, 512), "medium": (192, 6, 768)}


def main():
    out = {"labo": {}, "nous": {}}
    for cfg, (d, h, f) in DIMS.items():
        m = StackedTransformer(embedding_dim=d, num_heads=h, hidden_dim=f)
        layers = lab_transformer_layers(m, 4)
        e32 = float(total_energy(layers, {n: (32.0, 32.0) for n in layers}, True))
        base = f"{LAB}/checkpoints/FPGA_{cfg}_fpga_stecath"
        fp32 = json.load(open(f"{base}/summary.json"))["fp32_stats"]["sr"]
        out["labo"][cfg] = {"params": sum(p.numel() for p in m.parameters()), "sr_fp32": fp32,
                            "energy_fp32_uJ": e32 / 1e6, "quant": {}}
        for path in sorted(glob.glob(f"{base}/phase2_quant/lambda_*/results_L_*.json")):
            lam = path.split("results_L_")[1][:-5]
            r = json.load(open(path))
            last = r[max(r, key=int)]
            bits = {n: (math.ceil(last["bit"][n]), math.ceil(last["bit_act"][n])) for n in layers}
            e = float(total_energy(layers, bits, True))
            out["labo"][cfg]["quant"][lam] = {"sr": last["sum_rate"], "energy_uJ": e / 1e6,
                                              "energie_labo_publiee_uJ": last["energy"]}
    for path in sorted(glob.glob(f"{RESULTS}/art_pair_h32_lam*.json")):
        r = json.load(open(path))
        key = os.path.basename(path)[:-5]
        out["nous"][key] = {"sr": r["sr_quant"], "sr_fp32": r["sr_fp32"],
                            "energy_uJ": r["energy_pJ"]["quant_corrige"] / 1e6,
                            "energy_fp32_uJ": r["energy_pJ"]["fp32_corrige"] / 1e6,
                            "bits_w": r["mean_bits_w"], "bits_a": r["mean_bits_a"]}
    for cfg, v in out["labo"].items():
        print(f"BitAdapt {cfg:6s} ({v['params'] // 1000} k) FP32 : SR {v['sr_fp32']:.2f}, E {v['energy_fp32_uJ']:.3f} uJ")
        for lam, q in v["quant"].items():
            print(f"    lambda {lam:5s} : SR {q['sr']:.2f} | E corrigee {q['energy_uJ']:.4f} uJ "
                  f"(publiee {q['energie_labo_publiee_uJ']:.4f})")
    for k, q in out["nous"].items():
        print(f"nous {k:28s} : SR {q['sr']:.2f} (FP32 {q['sr_fp32']:.2f}) | E corrigee {q['energy_uJ']:.4f} uJ "
              f"(FP32 {q['energy_fp32_uJ']:.3f}) | bits {q['bits_w']:.1f}/{q['bits_a']:.1f}")
    json.dump(out, open(f"{RESULTS}/energy_comparison.json", "w"), indent=1)


if __name__ == "__main__":
    main()
