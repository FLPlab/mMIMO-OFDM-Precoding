#!/usr/bin/env python3
"""
Ajuste un modele de cout simple a partir de cost_table.csv et ecrit les
constantes retenues dans fit_coeffs.json (le modele utilisable est ecrit a
la main dans fpga_cost.py ; ce script ne fait que l'ajustement + verif).

Constat cle (grille complete, 68 points) : le cas A (poids constants,
implementation par defaut de Vitis HLS) n'est "sans DSP" que pour b_w <= 4.
Pour b_w in {6, 8}, l'outil commence a inferer des DSP48 pour une FRACTION
des multiplicateurs constants (ex. bw=8 -> ~100-110 DSP sur 256 MACs, pas un
ratio simple comme 0.5). Ce changement de regime n'est pas lineaire et nous
ne le modelisons pas par une formule : on fitte le modele LUT/MAC uniquement
sur le regime propre (bw in {1,2,3,4}, DSP=0 observe sur tous ces points) et
on documente les points bw in {6,8} comme table brute (cf. RAPPORT.md).
Ce choix est raisonnable car ART vise la quantification agressive (bw<=4
est le regime d'interet pratique) ; au-dela, le modele doit etre complete
par resynthese si besoin.

Second constat, plus fin : a l'interieur du regime propre, bw=1 (poids
ternaire {-1,0}) est un cas DEGENERE -- il n'y a pas de vrai multiplieur
constant, juste un mux/negation, donc sa pente en b_a n'a pas de raison de
suivre la meme loi que bw in {2,3,4} (ou un reseau shift-add existe
reellement). Un fit global trilineaire sur bw in {1..4} donne un residu de
signe incorrect sur bw=1 (le cout mesure CROIT avec b_a, le fit global le
fait decroitre). On separe donc le cas A en deux morceaux :
  - bw=1   : regression affine en b_a seule (5 points)
  - bw in {2,3,4} : regression trilineaire (bw*ba, ba, 1) (15 points)
Les deux morceaux sont raccordes (pas de discontinuite imposee, juste deux
formules valides chacune sur son domaine).

Modeles ajustes (regression lineaire moindres carres, numpy) :

Cas A, regime propre (poids constants, bw<=4, DSP=0 observe) :
    bw=1       : LUT_A(1, ba)  ~= a_A1 * ba + b_A1                (par MAC)
    bw in{2,3,4}: LUT_A(bw, ba) ~= a_A * bw * ba + b_A * ba + c_A  (par MAC)
    DSP_A(bw, ba)  = 0                                        (observe, bw<=4)

Cas B (poids variables, implementation par defaut, DSP48 avec "SIMD packing"
pour les petites largeurs) :
    DSP_B(bw, ba)  = 0 si bw==1 (pas de multiplieur, poids ternaire/binaire),
                     sinon 0.5 DSP/MAC (observe EXACTEMENT sur tous les bw in
                     {2,3,4,6,8} et tous les ba -- Vitis empaquette 2 petites
                     multiplications par DSP48E2 via SIMD, independamment de
                     ba). C'est un palier, pas une regression.
    LUT_B(bw, ba)  ~= a_B * bw * ba + b_B * (bw + ba) + c_B   (par MAC, logique
                     de routage/accumulation autour des DSP, hors multiplieurs)

On ajuste aussi, a partir du cas B force en LUT (BIND_OP impl=fabric) :
    LUT_mult(bw, ba) ~= a_L * bw * ba + b_L * (bw + ba) + c_L  (cout d'un
                     multiplieur pur en LUT, sans DSP)

Et on verifie le cas B force DSP (BIND_OP impl=dsp) : DSP/MAC == 1.0 exact
(un DSP par MAC, pas de packing) sur les 4 points testes.
"""
import csv
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_rows():
    with open(os.path.join(ROOT, "cost_table.csv")) as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fit_linear(rows, ycol, feature_fn, label):
    X, y = [], []
    for r in rows:
        yv = to_float(r.get(ycol))
        if yv is None:
            continue
        bw, ba = float(r["bw"]), float(r["ba"])
        X.append(feature_fn(bw, ba))
        y.append(yv)
    X = np.array(X)
    y = np.array(y)
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    print(f"[{label}] n={len(y)}  R2={r2:.4f}  RMSE={rmse:.3f}  coef={coef}")
    return coef, r2, rmse, len(y)


def feat_bwba_ba_1(bw, ba):
    return [bw * ba, ba, 1.0]


def feat_bwba_sum_1(bw, ba):
    return [bw * ba, bw + ba, 1.0]


def feat_ba_1(bw, ba):
    return [ba, 1.0]


def main():
    rows = load_rows()

    rows_A = [r for r in rows if r["case"] == "A" and r["bindop"] == "const"]
    rows_A_clean = [r for r in rows_A if int(r["bw"]) <= 4]
    rows_A_bw1 = [r for r in rows_A if int(r["bw"]) == 1]
    rows_A_bw234 = [r for r in rows_A if 2 <= int(r["bw"]) <= 4]
    rows_A_dsp_regime = [r for r in rows_A if int(r["bw"]) >= 6]
    rows_B_default = [r for r in rows if r["case"] == "B" and r["bindop"] == "default"]
    rows_B_lut = [r for r in rows if r["case"] == "B" and r["bindop"] == "lut"]
    rows_B_dsp = [r for r in rows if r["case"] == "B" and r["bindop"] == "dsp"]

    print(f"Total points charges : {len(rows)} (attendu 68)")

    print("\n=== Cas A, regime propre (bw<=4, DSP=0) : LUT/MAC, modele en 2 morceaux ===")
    dsp_A_clean = set(to_float(r["DSP"]) for r in rows_A_clean)
    print(f"  DSP observes sur ce regime = {dsp_A_clean} (doit etre {{0.0}})")
    print("  -- morceau bw=1 (degenere, ternaire, pas de vrai multiplieur) --")
    coef_A_bw1, r2_A_bw1, rmse_A_bw1, n_A_bw1 = fit_linear(rows_A_bw1, "lut_per_mac", feat_ba_1, "A LUT/MAC (bw=1)")
    print("  -- morceau bw in {2,3,4} (vrai reseau shift-add) --")
    coef_A_lut, r2_A, rmse_A, n_A = fit_linear(rows_A_bw234, "lut_per_mac", feat_bwba_ba_1, "A LUT/MAC (bw in 2..4)")

    print("\n=== Cas A, regime DSP (bw in {6,8}) : NON modelise, table brute ===")
    for r in sorted(rows_A_dsp_regime, key=lambda r: (int(r["bw"]), int(r["ba"]))):
        print(f"    bw={r['bw']:>2} ba={r['ba']:>2} -> LUT={r['LUT']:>5} DSP={r['DSP']:>4} DSP/MAC={float(r['dsp_per_mac']):.4f}")

    print("\n=== Cas B (defaut) : LUT/MAC et DSP/MAC (poids variables) ===")
    dsp_B_by_bw = {}
    for r in rows_B_default:
        dsp_B_by_bw.setdefault(int(r["bw"]), set()).add(to_float(r["dsp_per_mac"]))
    print(f"  DSP/MAC par bw (doit etre {{0.0}} pour bw=1, {{0.5}} sinon) : {dsp_B_by_bw}")
    coef_B_lut, r2_B, rmse_B, n_B = fit_linear(rows_B_default, "lut_per_mac", feat_bwba_sum_1, "B LUT/MAC")

    print("\n=== Cas B force LUT (BIND_OP impl=fabric) : cout pur multiplieur en LUT ===")
    coef_L, r2_L, rmse_L, n_L = fit_linear(rows_B_lut, "lut_per_mac", feat_bwba_sum_1, "B-LUT mult/MAC")

    print("\n=== Cas B force DSP (BIND_OP impl=dsp, attendu 1 DSP/MAC exact) ===")
    dsp_forced_vals = set()
    for r in sorted(rows_B_dsp, key=lambda r: (int(r["bw"]), int(r["ba"]))):
        dsp_forced_vals.add(to_float(r["dsp_per_mac"]))
        print(f"    bw={r['bw']:>2} ba={r['ba']:>2} -> DSP={r['DSP']:>4} DSP/MAC={float(r['dsp_per_mac']):.3f}  LUT={r['LUT']:>5}")
    print(f"  DSP/MAC observes = {dsp_forced_vals} (attendu {{1.0}})")

    fit = {
        "n_points_total": len(rows),
        "caseA_lut_per_mac_bw1_degenerate": {
            "coef": list(coef_A_bw1), "feature": "[ba, 1]",
            "r2": r2_A_bw1, "rmse": rmse_A_bw1, "n": n_A_bw1,
            "valid_range": "bw == 1 (poids ternaire, pas de vrai multiplieur)",
        },
        "caseA_lut_per_mac_bw_2_3_4": {
            "coef": list(coef_A_lut), "feature": "[bw*ba, ba, 1]",
            "r2": r2_A, "rmse": rmse_A, "n": n_A,
            "valid_range": "bw in {2,3,4}, DSP=0",
        },
        "caseA_dsp_regime_bw_ge_6_raw": {
            "note": "pas de formule -- cf. cost_table.csv / RAPPORT.md pour les valeurs brutes",
            "bw_values": [6, 8],
        },
        "caseB_default_lut_per_mac": {
            "coef": list(coef_B_lut), "feature": "[bw*ba, bw+ba, 1]",
            "r2": r2_B, "rmse": rmse_B, "n": n_B,
        },
        "caseB_default_dsp_per_mac_threshold": {
            "rule": "0.0 if bw==1 else 0.5",
            "observed": {str(k): sorted(v) for k, v in dsp_B_by_bw.items()},
        },
        "caseB_lut_forced_mult_per_mac": {
            "coef": list(coef_L), "feature": "[bw*ba, bw+ba, 1]",
            "r2": r2_L, "rmse": rmse_L, "n": n_L,
        },
        "caseB_dsp_forced_per_mac": {
            "value": 1.0, "observed": sorted(dsp_forced_vals),
        },
    }
    with open(os.path.join(ROOT, "fit_coeffs.json"), "w") as f:
        json.dump(fit, f, indent=2)
    print("\nCoefficients sauvegardes dans fit_coeffs.json")


if __name__ == "__main__":
    main()
