"""Phase 15 -- repartition des MAC du precodeur par bloc, sans reentrainement.
Configs de base et leviers (profondeur, largeur, interpolation, Pareto).

    python -u -m experiments.phase15_macs
"""
import json
import os

import tensorflow as tf

from models.gram_precoder import GramPrecoder
from utils import RESULTS

os.makedirs(os.path.join(RESULTS, "phase15"), exist_ok=True)


def macs_breakdown(K, M, d, hidden, layers, mix="triangular_g", inp="norm"):
    """Calcule les MAC reelles du precodeur par paires.

    Formule de GramPrecoder.macs_per_sc:
      return 4*K*K*M + K*K*per_edge + L*4*K^3*d + 4*M*K*K

    Composantes principales :
    - G: Gram matrix h*h^H
    - Edges: init Dense, FFN par couche, projections triangulaires, lecture
    - Triangular: produit triangulaire par couche
    - W: sortie h^H C
    """
    net = GramPrecoder(M, d=d, hidden=hidden, layers=layers, mix=mix, tied=False, inp=inp)
    total_macs = net.macs_per_sc(K, M)

    # Decomposition analytique simple
    hd = hidden
    L = layers

    # Composantes principales du total
    macs_g = 4 * K * K * M  # Gram matrix h*h^H
    macs_w = 4 * M * K * K   # W = h^H C
    macs_tri = L * 4 * K ** 3 * d  # produit triangulaire

    # Reste : par paire
    macs_edges = total_macs - macs_g - macs_tri - macs_w
    macs_per_pairbatch = macs_edges / (K * K) if K * K > 0 else 0

    return {
        "K": K, "M": M, "d": d, "hidden": hidden, "layers": layers,
        "config": f"M{M}K{K}",
        "total_macs": total_macs,
        "breakdown": {
            "G_matrix": int(macs_g),
            "edges_per_pair": int(macs_per_pairbatch),
            "triangular_products": int(macs_tri),
            "W_matmul": int(macs_w),
        },
        "percentages": {
            "G_matrix": 100.0 * macs_g / total_macs,
            "edges": 100.0 * macs_edges / total_macs,
            "triangular_products": 100.0 * macs_tri / total_macs,
            "W_matmul": 100.0 * macs_w / total_macs,
        }
    }


def main():
    # Configuration de base : d=16, hidden=32, layers=8, mix="triangular_g", inp="norm"
    # Verification : M8K4, M64K4, M64K8

    configs = [
        # Base : hidden=32
        {"K": 4, "M": 8, "d": 16, "hidden": 32, "layers": 8, "tag": "base_M8K4"},
        {"K": 4, "M": 64, "d": 16, "hidden": 32, "layers": 8, "tag": "base_M64K4"},
        {"K": 8, "M": 64, "d": 16, "hidden": 32, "layers": 8, "tag": "base_M64K8"},

        # Profondeur : hidden=32
        {"K": 4, "M": 8, "d": 16, "hidden": 32, "layers": 4, "tag": "L4_M8K4"},
        {"K": 4, "M": 64, "d": 16, "hidden": 32, "layers": 4, "tag": "L4_M64K4"},
        {"K": 4, "M": 8, "d": 16, "hidden": 32, "layers": 2, "tag": "L2_M8K4"},
        {"K": 4, "M": 64, "d": 16, "hidden": 32, "layers": 2, "tag": "L2_M64K4"},

        # Largeur : hidden=16
        {"K": 4, "M": 8, "d": 16, "hidden": 16, "layers": 4, "tag": "h16_L4_M8K4"},
        {"K": 4, "M": 64, "d": 16, "hidden": 16, "layers": 4, "tag": "h16_L4_M64K4"},
        {"K": 4, "M": 8, "d": 16, "hidden": 16, "layers": 2, "tag": "h16_L2_M8K4"},
        {"K": 4, "M": 64, "d": 16, "hidden": 16, "layers": 2, "tag": "h16_L2_M64K4"},
    ]

    out = {"meta": {"created": "2026-11-26", "config_base": "d=16, hidden=32, layers=8, mix=triangular_g, inp=norm"},
           "breakdown_configs": {}}

    for cfg in configs:
        result = macs_breakdown(cfg["K"], cfg["M"], cfg["d"], cfg["hidden"], cfg["layers"])
        out["breakdown_configs"][cfg["tag"]] = result
        g_pct = result['percentages']['G_matrix']
        tri_pct = result['percentages']['triangular_products']
        edge_pct = result['percentages']['edges']
        print(f"{cfg['tag']:20s}: {result['total_macs']:10d} MAC "
              f"(G:{g_pct:5.1f}%, edge:{edge_pct:5.1f}%, tri:{tri_pct:5.1f}%)", flush=True)

    # Verifications avec les chiffres connus
    print("\nVerification des totaux :")
    print(f"M8K4 base (caché en phase11): ~700k attendu, obtenu {out['breakdown_configs']['base_M8K4']['total_macs']:d}")
    print(f"M64K4 St-Catherine (738k attendu): obtenu {out['breakdown_configs']['base_M64K4']['total_macs']:d}")
    print(f"M64K8 (1418k attendu en phase13): obtenu {out['breakdown_configs']['base_M64K8']['total_macs']:d}")

    # Leviers sans forme imposee : lire les fichiers existants phase13
    leviers = {}

    # Levier 1 : interpolation 1 SC sur s (phase13/interp.json)
    # A 1 SC sur 12, perte ~-1.5% en M8K4, stable en St-Catherine (-0.4%)
    # MAC divises par le pas
    leviers["interp_1sc_per_12"] = {
        "description": "Pas frequentiel 1 SC sur 12 (au lieu de 1 sur 1)",
        "rate_loss_bps_hz": {
            "sionna_umi": -1.46,  # 17.65 - 16.87 = 0.78 mais c'est -4.4%, a 20dB -1.5 bps/Hz interpolé
            "stecath": -0.46,     # 22.09 - 21.66 = 0.43
        },
        "mac_reduction_factor": 1.0 / 12,  # 1 SC sur 12 coûte 1/12 du calcul
        "mac_per_sc_effective": int(724480 / 12),  # ~60k MAC
        "source": "phase13/interp.json (rate interpolee), 1 SC sur 12 divise par 12"
    }

    # Levier 2 : reduction de profondeur (4 couches au lieu de 8)
    # Perte ~0.4-0.7 bps/Hz en St-Catherine (phase13 bench ou FINDINGS)
    # MAC reduits de moitié
    leviers["reduce_L_8to4"] = {
        "description": "4 couches au lieu de 8 (hidden=32)",
        "mac_per_sc": 375296,  # from energy_v5_ours.json
        "rate_loss_bps_hz": 0.7,  # ~26.65 -> 25.94
        "mac_reduction_percent": 50,  # L divise par 2
        "source": "energy_v5_ours.json + FINDINGS pareto"
    }

    # Levier 3 : reduction de largeur (hidden=16 au lieu de 32)
    # Perte additionnelle ~0.3-0.4 bps/Hz
    leviers["reduce_h_32to16"] = {
        "description": "FFN largeur 16 au lieu de 32 (4 couches)",
        "mac_per_sc": 273920,  # from energy_v5_ours.json
        "rate_loss_bps_hz": 0.36,  # 25.94 -> 25.58
        "mac_reduction_percent": 27,  # 274k / 375k
        "source": "energy_v5_ours.json art_pair_h16_L4_lam5_qprod_norm"
    }

    # Levier 4 : reduction L=2, h=16
    leviers["reduce_L2_h16"] = {
        "description": "2 couches, FFN largeur 16",
        "mac_per_sc": 143360,  # from energy_v5_ours.json
        "rate_loss_bps_hz": 2.52,  # 26.65 - 24.13
        "mac_reduction_percent": 81,  # 143k / 738k
        "source": "energy_v5_ours.json art_pair_h16_L2_lam5_qprod_norm"
    }

    out["leviers_no_form"] = leviers

    path = os.path.join(RESULTS, "phase15", "macs.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n-> {path}")
    print("\nResume des leviers (MAC effectives et perte debit) :")
    for name, lever in leviers.items():
        if "mac_per_sc" in lever:
            print(f"  {name}: {lever['mac_per_sc']:7d} MAC, {lever['rate_loss_bps_hz']:+.2f} bps/Hz")
        elif "mac_per_sc_effective" in lever:
            print(f"  {name}: {lever['mac_per_sc_effective']:7d} MAC, {lever['rate_loss_bps_hz']['stecath']:+.2f} bps/Hz (St-Cath)")


if __name__ == "__main__":
    main()
