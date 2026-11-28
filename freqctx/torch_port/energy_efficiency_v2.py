"""Tableau d'efficacite energetique, v2 : ajoute la convention "WMMSE litterature standard" a cote des
trois conventions deja calculees (SiPS/BitAdapt v5, forme efficace en dimension K, forme Gram).
N'ecrit PAS energy_efficiency.json (le module v1 reste tel quel) ; sortie :
results/stecath/energy_efficiency_v2.json. St-Catherine et Ericsson, M = 64, K = 4, I = 5, 30, 100, 300,
modele d'energie Moons et al. 2017 v5 (torch_port.energy, corrected=True).

Les quatre conventions de comptage de WMMSE (meme algorithme, memes debits) :
  1. "SiPS / BitAdapt v5" : E = K_pJ . I . N_c, N_c = 2 880 309 MAC/iteration (terme dominant
     (8/3) M^3 N_U : une inversion M x M PAR UTILISATEUR et par iteration ; c'est aussi le terme
     dominant (14/3) N_U N_T^3 FLOP du tableau II d'Emery et al., arXiv:2507.18587) ; constante
     calibree 296 uJ a I = 92.8, independante des bits (repris de energy_efficiency.py).
  2. "litterature standard" (NOUVEAU) : l'algorithme de Shi et al. 2011 tel qu'il est analyse dans
     Pellaco et al. 2020 (O(L M^3)), Zhao et al. 2023 (R-WMMSE, "WMMSE ... O(M^3)"), Schynol &
     Pesavento 2023 (O(... + K_BS M^3), une inversion M x M par station) et Zhu-Chang-Shen 2026
     (O(N_t^3 + K N_t^2 d) par pas, nombre de pas de bissection traite comme constant) : UNE
     inversion M x M par iteration, partagee par tous les utilisateurs (A = H^H diag(a) H + mu I est
     commune), en dimension M, exactement notre wmmse() de torch_port/stecath.py :
        H W                    4 K^2 M
        recepteurs / poids     2 K^2 + 6 K + 3 K div        (comme energy_classical.counts_wmmse)
        A = H^H diag(a) H + mu I   2 K M (mise a l'echelle reelle) + 4 K M^2 + M
        B = H^H diag(omega g*)     4 M K
        inverse M x M          4 M^3 + M div                (meme convention que l'inverse K x K des
                                                             autres formes : 4 n^3 + n divisions)
        W = A^-1 B             4 M^2 K
        normalisation          2 M K + 1 racine
     initialisation par RZF (meme que les autres formes, energy_classical.counts_rzf).
     Division / racine = 10 MAC (DIV, comme energy_classical.py). Activations : H W, A, A^-1, B, W.
     BISSECTION SUR mu : NON comptee dans la ligne principale, pour deux raisons : (i) la litterature
     de reference l'ignore ou la traite comme un facteur constant (Pellaco 2020, Schynol 2023,
     Zhu 2026 ; Shi 2011 l'ignorerait aussi selon une citation secondaire, texte original non
     verifie) ; (ii) notre WMMSE (celui dont on mesure le debit) calcule mu en forme fermee
     (mu = sigma^2 sum_k a_k, reformulation sans contrainte de Hu et al. 2021 / R-WMMSE) : il n'y a
     pas de bissection dans l'algorithme mesure. Deux variantes AVEC bissection sont donnees en plus
     (sensibilite, debits non remesures -- la v5 a bissection donne des debits a +-0.3 bps/Hz des
     notres, FINDINGS 2026-11-25) :
       - "evd" (forme originale de Shi 2011 / Pellaco 2020, eq. (8)) : une decomposition propre de
         l'M x M hermitienne (cout SUPPOSE 18 M^3 MAC reelles = 9 n^3 flops de Golub & Van Loan pour
         le QR symetrique avec vecteurs, x4 pour le complexe, /2 flops -> MAC), D^H B 4 M^2 K,
         phi 2 M K, N_B pas scalaires de M (2 + div) chacun, W = D (Lambda + mu I)^-1 D^H B :
         M div + 2 M K + 4 M^2 K ; N_B = 20 ;
       - "inversion repetee" (lecture de Zhang et al. 2023 / Zhu 2026, "implementation directe") :
         N_B = 20 inversions M x M + produits par pas -- borne haute.
     Inversion par Cholesky + substitutions (au lieu d'un inverse explicite) : variante "cholesky"
     (L L^H : M^3/3 MAC complexes = 4/3 M^3 reelles + M div ; deux substitutions a K seconds membres :
     4 M^2 K) donnee aussi, borne basse de la forme standard.
     BITS : 16 (ligne principale). Les 13 / 15 bits mesures (fixedpoint_classical.json) le sont pour
     l'inversion K x K de la forme Gram ; la matrice M x M a les memes K valeurs propres non nulles
     (+ mu) plus M - K valeurs propres egales a mu, donc un conditionnement >= celui de la K x K :
     transposer 13 / 15 bits n'est pas justifiable -> 16 bits ; la valeur a 13 / 15 bits est donnee
     en plus (optimiste).
  3. "forme efficace" (dimension K, identite push-through, energy_classical.counts_wmmse) : aux bits
     mesures de la forme Gram (13 / 15 : meme matrice K x K inversee, mu I + D H H^H), et a 16 bits.
  4. "forme Gram" (= R-WMMSE de Zhao et al. 2023 pour K utilisateurs mono-antenne) : repris de
     energy_efficiency.py (13 / 15 bits mesures).

    CUDA_VISIBLE_DEVICES="" python -u -m torch_port.energy_efficiency_v2
"""
import json
import os

from torch_port import energy_efficiency as ee
from torch_port.energy_classical import counts_rzf, counts_wmmse

OUT = os.path.join(ee.RESULTS, "energy_efficiency_v2.json")
M, K, DIV = 64, 4, 10
N_B = 20                       # pas de bissection (variantes avec bissection seulement)
EVD_MAC_PER_M3 = 18            # hypothese : EVD hermitienne complexe avec vecteurs ~ 18 M^3 MAC reelles
STD_BITS_MAIN = 16


def _iter_common():
    """Partie de l'iteration standard commune a toutes les variantes (hors resolution du systeme)."""
    mac = 4 * K * K * M                                   # H W
    mac += 2 * K * K + 6 * K + 3 * DIV * K                # recepteurs / poids
    mac += 2 * K * M + 4 * K * M * M + M                  # A = H^H diag(a) H + mu I
    mac += 4 * M * K                                      # B = H^H diag(omega g*)
    act = 2 * K * K + 6 * K + 2 * M * M + 2 * M * K       # H W, scalaires, A, B
    return mac, act


def counts_wmmse_standard(n, variant="inverse"):
    """WMMSE en dimension M (notre wmmse()), UNE inversion M x M par iteration.
    variant : 'inverse' (principal), 'cholesky', 'bisect_evd', 'bisect_inv'."""
    mac0, act0 = counts_rzf()                             # initialisation (identique aux autres formes)
    c_mac, c_act = _iter_common()
    norm_mac, norm_act = 2 * M * K + DIV, 2 * M * K       # normalisation, W
    if variant == "inverse":
        s_mac = 4 * M ** 3 + M * DIV + 4 * M * M * K      # inverse explicite + W = A^-1 B
        s_act = 2 * M * M
    elif variant == "cholesky":
        s_mac = (4 * M ** 3) // 3 + M * DIV + 2 * 2 * M * M * K   # L L^H (M^3/3 complexes) + 2 substitutions
        s_act = 2 * M * M // 2 + 2 * M * K
    elif variant == "bisect_evd":
        s_mac = EVD_MAC_PER_M3 * M ** 3                   # A = D Lambda D^H
        s_mac += 4 * M * M * K + 2 * M * K                # D^H B, phi_m
        s_mac += N_B * M * (2 + DIV)                      # bissection : sum_m phi_m / (lambda_m + mu)^2
        s_mac += M * DIV + 2 * M * K + 4 * M * M * K      # W = D (Lambda + mu I)^-1 D^H B
        s_act = 2 * M * M + 2 * M * K + M + N_B
    elif variant == "bisect_inv":
        s_mac = N_B * (4 * M ** 3 + M * DIV + 4 * M * M * K + 2 * M * K)  # inversion + W + puissance par pas
        s_act = N_B * (2 * M * M + 2 * M * K)
    else:
        raise ValueError(variant)
    return mac0 + n * (c_mac + s_mac + norm_mac), act0 + n * (c_act + s_act + norm_act)


def main():
    v1 = json.load(open(ee.OUT)) if os.path.exists(ee.OUT) else None
    out = {"meta": {
        "source": "torch_port/energy_efficiency_v2.py (importe energy_efficiency.py, non modifie)",
        "modele": "Moons et al. 2017 v5 (torch_port.energy, corrected=True), M=64, K=4, CSI parfait",
        "div_cost_mac": DIV,
        "conventions": {
            "WMMSE_SiPS_eq12": "E = K_pJ I N_c, N_c = 2 880 309 MAC/it (inversion M x M par utilisateur), "
                               "calibre 296 uJ a I = 92.8, sans bits",
            "WMMSE_litterature_standard": "UNE inversion M x M par iteration (dimension M, = notre wmmse()), "
                                           "4 M^3 + M div pour l'inverse, bissection non comptee "
                                           "(mu en forme fermee dans l'algorithme mesure), 16 bits",
            "WMMSE_efficace_dimK": "push-through, inversion K x K, produits en M a chaque iteration "
                                   "(energy_classical.counts_wmmse), bits mesures de la forme Gram",
            "WMMSE_gram": "iteration sur C (K x K), G une fois, W = H^H C a la fin (= R-WMMSE, Zhao et al. "
                          "2023, pour des utilisateurs mono-antenne), bits mesures",
        },
        "bisection": {"N_B": N_B, "evd_mac_par_M3": EVD_MAC_PER_M3,
                      "note": "variantes de sensibilite seulement ; debits = ceux de notre WMMSE a mu en forme "
                              "fermee (non remesures avec bissection)"},
        "bits_standard": {"principal": STD_BITS_MAIN,
                          "justification": "conditionnement de la M x M >= celui de la K x K (memes K valeurs "
                                           "propres non nulles + mu, M-K valeurs egales a mu) ; 13/15 bits "
                                           "mesures pour la K x K seulement -> 16 bits ; 13/15 donnes en "
                                           "variante optimiste"},
    }, "sites": {}}

    for site in ("stecath", "ericsson"):
        wm = ee.SR[site]["WMMSE"]
        b_meas = ee.WMMSE_GRAM_BITS[site]
        s = {"RZF": {"debit": ee.SR[site]["RZF"], "energy_uJ_bits_mesures": ee.RZF_ENERGY_FIXED[site]["energy_uJ"],
                     "bits": ee.RZF_ENERGY_FIXED[site]["bits"], "energy_uJ_16b": ee.RZF_ENERGY_16B},
             "notre_precodeur": ee.OURS[site],
             "forme_structuree_T20_persc": {"debit": ee.STRUCT_SR[site][20], "energy_uJ": ee.struct_energy(20)},
             "WMMSE": {}}
        for i in ee.WMMSE_IS:
            row = {"debit": wm[i]}
            row["SiPS_eq12_uJ"] = ee.E_PER_ITER_STANDARD_uJ * i
            mac, act = counts_wmmse_standard(i, "inverse")
            row["standard_mac"] = mac
            row["standard_16b_uJ"] = ee.e_moons(mac, 0, act, STD_BITS_MAIN, STD_BITS_MAIN)
            row[f"standard_{b_meas}b_uJ_optimiste"] = ee.e_moons(mac, 0, act, b_meas, b_meas)
            for var in ("cholesky", "bisect_evd", "bisect_inv"):
                m2, a2 = counts_wmmse_standard(i, var)
                row[f"standard_{var}_16b_uJ"] = ee.e_moons(m2, 0, a2, STD_BITS_MAIN, STD_BITS_MAIN)
                row[f"standard_{var}_mac"] = m2
            mac_e, act_e = counts_wmmse(i)
            row["efficace_mac"] = mac_e
            row[f"efficace_{b_meas}b_uJ"] = ee.e_moons(mac_e, 0, act_e, b_meas, b_meas)
            row["efficace_16b_uJ"] = ee.e_moons(mac_e, 0, act_e, 16, 16)
            mac_g, act_g = ee.counts_wmmse_gram(i)
            row["gram_mac"] = mac_g
            row[f"gram_{b_meas}b_uJ"] = ee.e_moons(mac_g, 0, act_g, b_meas, b_meas)
            row["gram_16b_uJ"] = ee.e_moons(mac_g, 0, act_g, 16, 16)
            if v1 is not None:                                # controle : Gram identique a la v1
                row["controle_gram_v1_uJ"] = v1["sites"][site]["WMMSE_gram_bits"][str(i)]["energy_uJ"]
            for k in [k for k in row if k.endswith("_uJ") or k.endswith("_uJ_optimiste")]:
                row["eff_" + k] = round(wm[i] / row[k], 3)
            s["WMMSE"][str(i)] = row
        out["sites"][site] = s

    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)
    for site in ("stecath", "ericsson"):
        print(f"\n=== {site} (bits mesures {ee.WMMSE_GRAM_BITS[site]}) ===")
        b = ee.WMMSE_GRAM_BITS[site]
        for i, r in out["sites"][site]["WMMSE"].items():
            print(f"I={i:>3} debit {r['debit']:.2f} | SiPS {r['SiPS_eq12_uJ']:.3g} | std16 {r['standard_16b_uJ']:.3g} "
                  f"(chol {r['standard_cholesky_16b_uJ']:.3g}, evd {r['standard_bisect_evd_16b_uJ']:.3g}, "
                  f"inv-rep {r['standard_bisect_inv_16b_uJ']:.3g}) | eff{b} {r[f'efficace_{b}b_uJ']:.3g} "
                  f"| gram{b} {r[f'gram_{b}b_uJ']:.3g} | MAC std {r['standard_mac']}")


if __name__ == "__main__":
    main()
