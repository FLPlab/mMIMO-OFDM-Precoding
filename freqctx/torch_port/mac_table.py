"""MAC reelles et parametres par echantillon (St-Catherine, M=64, K=4), comptes comme Pipeline/Energy.py v5
(couches denses : in x out x tokens ; attention : 4 T d^2 + 2 T^2 d) pour les Transformers du labo v5, notre
precodeur (torch_port.energy_v5) et RZF / WMMSE (forme efficace en dimension K, et convention de l'article
SiPS : eq. 12, N_c par iteration).

    python -m torch_port.mac_table
"""
import json
import sys

import torch

from torch_port.check_v5_energy import CFG, V5
from torch_port.energy_classical import counts_rzf, counts_wmmse

sys.path.insert(0, V5)
from Pipeline.Transformer import StackedTransformer  # noqa: E402
from eval_baselines import _wmmse_nc_per_iter  # noqa: E402

T = 4


def transformer_macs(m):
    mac = 0
    for mod in m.modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            d = mod.embed_dim
            mac += 4 * T * d * d + 2 * T * T * d
        elif isinstance(mod, torch.nn.Linear) and not isinstance(mod, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            mac += mod.in_features * mod.out_features * T
    return mac


def main():
    out = {}
    for cfg, (L, d, h, f) in CFG.items():
        m = StackedTransformer(num_layers=L, embedding_dim=d, num_heads=h, hidden_dim=f, dropout=0.0)
        out[f"Transformer labo {cfg}"] = {"mac": transformer_macs(m), "params": sum(p.numel() for p in m.parameters())}
    ev = json.load(open("results/stecath/energy_v5_ours.json"))["notre_precodeur"]
    out["notre precodeur"] = {"mac": ev["mac"], "params": 44000}
    out["RZF"] = {"mac": counts_rzf()[0]}
    for i in (1, 10, 30, 100):
        out[f"WMMSE {i} it. (forme efficace)"] = {"mac": counts_wmmse(i)[0]}
        out[f"WMMSE {i} it. (article, eq. 12)"] = {"mac": int(i * _wmmse_nc_per_iter())}
    for k, v in out.items():
        print(f"{k:40s} {v['mac'] / 1e3:10.1f} k MAC" + (f"   {v['params'] / 1e3:8.0f} k param." if "params" in v else ""))
    json.dump(out, open("results/stecath/mac_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
