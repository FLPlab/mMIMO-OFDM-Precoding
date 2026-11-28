"""Tableau d'efficacite energetique (demande F. Leduc-Primeau, 2026-11-27) : RZF, WMMSE (DEUX
conventions d'energie), notre precodeur quantifie, la forme STRUCTUREE de phase 16, et (pour
St-Catherine) le front BitAdapt v5 -- St-Catherine et Ericsson, M=64, K=4, modele d'energie Moons
et al. 2017 v5 (torch_port.energy, corrected=True).

Deux efficacites, par methode :
  (a) efficacite de calcul = debit / energie de calcul (bps/Hz par uJ) ;
  (b) efficacite energetique SYSTEME = bits transmis / (energie de calcul + energie rayonnee), pour
      un symbole OFDM, a une puissance d'emission TOTALE P_tx donnee (hypothese explicite, parametrable
      -- voir META ci-dessous) : avec une bande B, un espacement de sous-porteuses SCS, N_sc = B/SCS,
      P_tx_sc = P_tx/N_sc, duree utile du symbole T_u = 1/SCS (donc SCS*T_u = 1, bits/symbole/SC =
      debit [bps/Hz] EXACTEMENT), duree totale avec PCP T_sym = T_u (1+cp_overhead) (pour l'energie
      rayonnee, qui est bien depensee pendant tout le symbole, CP compris) :
        E_rad = P_tx_sc * T_sym ;  E_total = E_calcul + E_rad ;  (b) = debit / E_total [bits/uJ].

Sources (Journal des resultats, phases 13/14/16) :
  - RZF/WMMSE forme efficace (dimension K) et forme Gram : torch_port/energy_classical.py,
    results/stecath/classical_energy*.json ;
  - RZF/WMMSE en virgule fixe (bits minimaux) : torch_port/fixedpoint_classical.py,
    results/stecath/fixedpoint_classical.json ;
  - WMMSE convention "article" (SiPS eq. 12, E = K I N_c, inversion M x M par utilisateur et par
    iteration) : constante calibree sur 296 uJ a I=92.8 dans
    /export/tmp/sala/home_archive/.../ELE6310E/eval_baselines.py (_wmmse_nc_per_iter, _K_pJ) ;
    N_c (MAC) = 2 880 309 pour M=64, K=4 (controle : torch_port/mac_table.py) -> E(I) = (296/92.8) I uJ,
    independant de la precision (convention de l'article, pas de variante bits) ;
  - notre precodeur quantifie (8 et 2 couches ART) : results/stecath/energy_v5_ours.json,
    results/stecath/v5_testset_ours.json, results/stecath/art_pair_h16_L2_lam5_qprod_norm*.json ;
  - forme structuree (phase 16, Bjornson 2014 + Newton-Schulz T pas APPRIS de bout en bout) :
    models/structured_precoder.py, poids weights/phase16/structured_T{10,20}.weights.h5 ; bits
    Newton-Schulz/G/W mesures en results/phase16/structured_check.json (13-17 bits) ; MAC/activations
    du petit transformer (2 couches, d=16, FFN 32) reutilises de torch_port.energy (architecture
    IDENTIQUE a nos PairPrecoder d=16, L=2) -- transformer NON quantifie par ART : hypothese PLATE 8
    bits (a documenter comme telle, pas une quantification verifiee) ;
  - front BitAdapt v5 (St-Catherine seulement, pas de point Ericsson disponible) :
    results/stecath/check_v5_energy.json (48 modeles, energie recalculee).

Debit de la forme structuree -- EVALUATION ZERO-SHOT reelle (pas une simple transposition) : le
modele (poids Sionna M8K4+M64K8, puissances reelles) est APPELE tel quel sur les canaux bruts
St-Catherine/Ericsson du protocole torch_port (M=64, K=4, mono-porteuse), avec h physique NORMALISE
a energie moyenne 1 et nv_eff = NOISE/e, snr_db = 10 log10(e/NOISE) PAR ECHANTILLON (meme convention
que torch_port.stecath.ours_input ; verifie par un controle RZF sur le meme chemin TF, qui retombe a
moins de 0.1 bps/Hz des references torch 17.03 / 33.44 St-Catherine/Ericsson -- la mise a l'echelle
est donc validee). La variante INTERPOLEE (lambda, p calcules 1 SC sur s, G/NS/W exacts par SC) ne
peut PAS etre mesuree sur ce protocole mono-porteuse (aucun axe frequentiel) : seule l'ENERGIE est
calculee ici, le debit est une estimation reportee de M8K4 Sionna (FINDINGS phase 16 : perte <= 0.1
bps/Hz a s=24 par rapport a s=1), signalee comme telle.

    CUDA_VISIBLE_DEVICES="" python -u -m torch_port.energy_efficiency
"""
import json
import os

from torch_port.energy import layer_energy

RESULTS = "/users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx/results/stecath"
OUT = os.path.join(RESULTS, "energy_efficiency.json")

# ---------------------------------------------------------------- META / hypotheses systeme
SCS_HZ = 30e3                 # 5G NR numerologie 1 (30 kHz)
BW_HZ = 100e6                 # bande macro 100 MHz
N_SC = round(BW_HZ / SCS_HZ)  # ~3276 sous-porteuses utiles
CP_OVERHEAD = 0.07            # ~7 % (CP normal, 14 symboles / slot de 0.5 ms)
T_U = 1.0 / SCS_HZ            # duree utile (s) ; SCS * T_U = 1 => bits/symbole/SC = debit [bps/Hz]
T_SYM = T_U * (1 + CP_OVERHEAD)
PTX_SCENARIOS_W = {"macro_20W": 20.0, "macro_30W": 30.0, "macro_40W": 40.0, "petite_cellule_1W": 1.0}


def e_rad_per_symbol_uJ(ptx_w):
    """Energie rayonnee par SYMBOLE OFDM et par SOUS-PORTEUSE (uJ), pour une puissance totale ptx_w
    repartie uniformement sur N_SC sous-porteuses."""
    ptx_sc = ptx_w / N_SC
    return ptx_sc * T_SYM * 1e6


def eff_calcul(debit, e_uJ):
    return debit / e_uJ if e_uJ else None


def eff_systeme(debit, e_calcul_uJ, ptx_w):
    return debit / (e_calcul_uJ + e_rad_per_symbol_uJ(ptx_w))


def crossover_ptx_w(debit_a, e_a, debit_b, e_b):
    """Puissance totale P_tx (W) a partir de laquelle la methode a (plus de debit, plus d'energie de
    calcul) devient plus efficace EN SYSTEME que b (moins de debit, moins d'energie) : resout
    debit_a/(e_a+E_rad) > debit_b/(e_b+E_rad) pour E_rad, puis E_rad -> P_tx (uJ -> W)."""
    if debit_a <= debit_b:
        return None
    rhs_uJ = (debit_b * e_a - debit_a * e_b) / (debit_a - debit_b)
    if rhs_uJ <= 0:
        return 0.0
    return rhs_uJ * N_SC / (T_SYM * 1e6)


def e_moons(mac, w, a, Qw, Qa):
    return float(layer_energy(mac, w, a, Qw, Qa, corrected=True)) / 1e6


# ---------------------------------------------------------------- 1. RZF / WMMSE (deux conventions)
# Debits : identiques quelle que soit la convention d'energie (meme algorithme) -- classical_energy*.json
SR = {
    "stecath": {"RZF": 17.03152, "WMMSE": {1: 19.0107, 5: 21.6172, 10: 22.7651, 30: 24.4141,
                                             100: 25.9153, 300: 26.8810}},
    "ericsson": {"RZF": 33.43820, "WMMSE": {1: 34.19085, 5: 35.44809, 10: 36.13078, 30: 37.31246,
                                              100: 38.66543, 300: 39.84234}},
}
WMMSE_IS = (5, 30, 100, 300)

# RZF forme Gram (dim K), bits minimaux mesures (fixedpoint_classical.json, critere <=0.1 bps/Hz) :
RZF_BITS = {"stecath": 13, "ericsson": 14}            # "a la precision qui tient vraiment"
RZF_ENERGY_FIXED = {                                   # energie_uJ_1pct / 0.1bps de fixedpoint_classical.json
    "stecath": {"bits": 13, "energy_uJ": 0.0115461},
    "ericsson": {"bits": 14, "energy_uJ": 0.0130189},
}
RZF_ENERGY_16B = 0.0161959                             # classical_energy*.json (independant du site, M,K fixes)
RZF_MAC = 9010

# WMMSE forme Gram, bits mesures (fixedpoint_classical.json, meme critere) : St-Catherine 12-13,
# Ericsson 13-15 selon I -> on retient la valeur HAUTE (conservatrice, uniforme sur I=5..300) par site.
WMMSE_GRAM_BITS = {"stecath": 13, "ericsson": 15}


def counts_wmmse_gram(n, K=4, M=64, DIV=10):
    mac = 4 * K * K * M + 4 * K ** 3
    mac += n * (12 * K ** 3 + 10 * K * K + 4 * DIV * K)
    mac += 4 * M * K * K + 2 * M * K + DIV
    act = 2 * K * M + 2 * K * K + 2 * K * K
    act += n * (10 * K * K + 6 * K)
    act += 2 * M * K
    return mac, act


# WMMSE convention "article" (SiPS eq. 12, E = K I N_c) : lineaire en I, independante du site/bits
# (constante calibree une fois sur 296 uJ a I=92.8 -- cf. eval_baselines.py du depot BitAdaptHLS).
NC_PER_ITER = 2880309                                  # controle : torch_port/mac_table.py ("article, eq. 12")
E_PER_ITER_STANDARD_uJ = 296.0 / 92.8                  # ~3.1897 uJ/iteration, independant du site (memes M,K)

# ---------------------------------------------------------------- 2. Notre precodeur quantifie (ART)
OURS = {
    "stecath": {
        "8couches_quant": {"debit": 26.650097, "energy_uJ": 0.113343},   # v5_testset_ours.json / energy_v5_ours.json
        "2couches_ART": {"debit": 24.070309, "energy_uJ": 0.024520},      # art_pair_h16_L2_lam5_qprod_norm.json
    },
    "ericsson": {
        "2couches_ART": {"debit": 34.354603, "energy_uJ": 0.030468},      # art_pair_h16_L2_lam5_qprod_norm_ericsson.json
    },
}

# ---------------------------------------------------------------- 3. Forme structuree (phase 16)
# MAC/activations du petit transformer (2 couches, d=16, FFN 32) = architecture IDENTIQUE a nos
# PairPrecoder(d=16, hidden=32, layers=2, inp="norm") ; valeurs issues de torch_port.energy
# (pair_precoder_layers/products, K=4, M=64) -- calcul fait une fois, reporte ici (pas de dependance
# a torch pour ce module, qui doit pouvoir tourner avec seulement TF installe).
STRUCT_TRANSFORMER = {"mac": 177664 + 8192, "w": 11266, "a": 4640 + 1024}   # dense + produit triangulaire (edge_g)
STRUCT_K, STRUCT_M = 4, 64
STRUCT_GW_MAC = 4 * STRUCT_K * STRUCT_K * STRUCT_M * 2                     # G = H H^H, W = H^H C
STRUCT_GW_A = 32 + 512
STRUCT_BITS_TRANSFORMER = 8           # NON quantifie par ART -- hypothese plate, a documenter comme telle
STRUCT_BITS_PHYS = 16                 # G / Newton-Schulz / W, dans la fourchette mesuree 13-17 bits (phase16_check)

# Debits zero-shot mesures (ce script, controle RZF <0.1 bps/Hz des references torch) :
STRUCT_SR = {
    "stecath": {10: 26.283031, 20: 27.280668},
    "ericsson": {10: 38.802711, 20: 40.903217},
}


def struct_energy(t_steps, s=1):
    mac_ns = t_steps * 8 * STRUCT_K ** 3
    a_ns = t_steps * 2 * STRUCT_K * STRUCT_K
    e_tr = e_moons(STRUCT_TRANSFORMER["mac"] / s, STRUCT_TRANSFORMER["w"] / s, STRUCT_TRANSFORMER["a"] / s,
                   STRUCT_BITS_TRANSFORMER, STRUCT_BITS_TRANSFORMER)
    e_phys = e_moons(STRUCT_GW_MAC + mac_ns, 0, STRUCT_GW_A + a_ns, STRUCT_BITS_PHYS, STRUCT_BITS_PHYS)
    return e_tr + e_phys


def main():
    out = {"meta": {
        "sites": "St-Catherine et Ericsson, M=64, K=4, CSI parfait, modele d'energie Moons et al. 2017 "
                 "(torch_port.energy, corrected=True)",
        "systeme": {"scs_hz": SCS_HZ, "bande_hz": BW_HZ, "n_sc": N_SC, "cp_overhead": CP_OVERHEAD,
                    "T_symbole_utile_s": T_U, "T_symbole_total_s": T_SYM,
                    "ptx_scenarios_W": PTX_SCENARIOS_W,
                    "note": "bits/symbole/SC = debit [bps/Hz] exactement (SCS * T_u = 1) ; "
                            "E_rad = (P_tx/N_sc) * T_sym ; (b) = debit / (E_calcul + E_rad)"},
        "wmmse_standard_eq12": {"N_c_MAC": NC_PER_ITER, "E_par_iteration_uJ": E_PER_ITER_STANDARD_uJ,
                                 "note": "E = K.I.N_c, calibre sur 296 uJ a I=92.8 (BitAdaptHLS/eval_baselines.py) ; "
                                         "lineaire en I, memes M=64,K=4 -> MEME valeur aux deux sites"},
        "structure_phase16": {"bits_transformer": STRUCT_BITS_TRANSFORMER, "bits_phys_G_NS_W": STRUCT_BITS_PHYS,
                               "note": "transformer NON quantifie par ART (hypothese plate 8 bits, pas verifiee) ; "
                                       "G/NS/W aux bits mesures en phase16_check.json (13-17 bits, on retient 16)"},
    }, "sites": {}}

    for site in ("stecath", "ericsson"):
        s = {}
        sr, wm = SR[site]["RZF"], SR[site]["WMMSE"]

        # RZF
        s["RZF_12-14bit"] = {"debit": sr, "energy_uJ": RZF_ENERGY_FIXED[site]["energy_uJ"],
                              "bits": RZF_ENERGY_FIXED[site]["bits"]}
        s["RZF_16bit"] = {"debit": sr, "energy_uJ": RZF_ENERGY_16B}

        # WMMSE forme standard (article, E = K.I.N_c)
        s["WMMSE_standard_eq12"] = {str(i): {"debit": wm[i], "energy_uJ": E_PER_ITER_STANDARD_uJ * i}
                                     for i in WMMSE_IS}

        # WMMSE forme Gram, bits mesures
        bits_g = WMMSE_GRAM_BITS[site]
        s["WMMSE_gram_bits"] = {}
        for i in WMMSE_IS:
            mac_g, act_g = counts_wmmse_gram(i)
            s["WMMSE_gram_bits"][str(i)] = {"debit": wm[i], "energy_uJ": e_moons(mac_g, 0, act_g, bits_g, bits_g),
                                             "bits": bits_g, "mac": mac_g}

        # notre precodeur quantifie
        s["notre_precodeur"] = OURS[site]

        # forme structuree
        s["forme_structuree_T20_persc"] = {"debit": STRUCT_SR[site][20], "energy_uJ": struct_energy(20),
                                            "note": "zero-shot, poids Sionna M8K4+M64K8 non reentraines"}
        s["forme_structuree_T10_persc"] = {"debit": STRUCT_SR[site][10], "energy_uJ": struct_energy(10)}
        s["forme_structuree_T20_interp_s24"] = {
            "debit_estime_transpose_M8K4": round(STRUCT_SR[site][20] - 0.1, 2),
            "energy_uJ": struct_energy(20, s=24),
            "note": "debit NON mesurable sur ce protocole mono-porteuse (pas d'axe frequentiel) ; "
                    "estime par transposition de la perte mesuree en M8K4 Sionna (<=0.1 bps/Hz a s=24, phase16)"}
        s["forme_structuree_T20_wmmse_anchor"] = {"note": "non calcule (fit post-hoc, hors budget 'si simple')"}

        # (a) efficacite de calcul, pour toutes les entrees qui ont (debit, energy_uJ)
        for k, v in list(s.items()):
            if isinstance(v, dict) and "debit" in v and "energy_uJ" in v:
                v["eff_calcul_bps_hz_par_uJ"] = round(eff_calcul(v["debit"], v["energy_uJ"]), 3)
            elif isinstance(v, dict) and all(isinstance(vv, dict) and "debit" in vv for vv in v.values()):
                for vv in v.values():
                    vv["eff_calcul_bps_hz_par_uJ"] = round(eff_calcul(vv["debit"], vv["energy_uJ"]), 3)

        # (b) efficacite systeme, pour quelques scenarios de Ptx -- une selection de methodes repere
        reper = {
            "RZF_16bit": s["RZF_16bit"],
            "notre_precodeur_8couches" if site == "stecath" else "notre_precodeur_2couches": (
                s["notre_precodeur"]["8couches_quant"] if site == "stecath" else s["notre_precodeur"]["2couches_ART"]),
            "forme_structuree_T20": s["forme_structuree_T20_persc"],
            "WMMSE_gram_I300": s["WMMSE_gram_bits"]["300"],
        }
        s["eff_systeme_par_Ptx"] = {}
        for name, ptx in PTX_SCENARIOS_W.items():
            s["eff_systeme_par_Ptx"][name] = {
                m: round(eff_systeme(v["debit"], v["energy_uJ"], ptx), 4) for m, v in reper.items()}

        # point de croisement : notre meilleur precodeur contre RZF 16 bits
        best = s["notre_precodeur"]["8couches_quant"] if site == "stecath" else s["notre_precodeur"]["2couches_ART"]
        s["crossover_ours_vs_RZF16b_W"] = crossover_ptx_w(best["debit"], best["energy_uJ"], sr, RZF_ENERGY_16B)
        s["crossover_structure_T20_vs_RZF16b_W"] = crossover_ptx_w(
            s["forme_structuree_T20_persc"]["debit"], s["forme_structuree_T20_persc"]["energy_uJ"], sr, RZF_ENERGY_16B)

        out["sites"][site] = s

    # front BitAdapt v5 (St-Catherine seulement)
    pts = json.load(open(os.path.join(RESULTS, "check_v5_energy.json")))
    pts = sorted(pts, key=lambda p: p["energy_recalculee"])
    front, best_sr = [], -1
    for p in pts:
        if p["sr"] > best_sr:
            front.append({"config": p["config"], "lambda": p["lambda"], "debit": p["sr"],
                          "energy_uJ": p["energy_recalculee"],
                          "eff_calcul_bps_hz_par_uJ": round(p["sr"] / p["energy_recalculee"], 3)})
            best_sr = p["sr"]
    out["sites"]["stecath"]["bitadapt_v5_front_non_domine"] = front
    out["sites"]["ericsson"]["bitadapt_v5_front_non_domine"] = "non disponible (pas de point BitAdapt v5 a Ericsson)"

    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)
    for site in ("stecath", "ericsson"):
        print(f"\n=== {site} ===")
        for name in ("RZF_12-14bit", "RZF_16bit", "notre_precodeur", "forme_structuree_T20_persc",
                     "forme_structuree_T20_interp_s24"):
            print(name, out["sites"][site][name])
        print("crossover (notre precodeur vs RZF 16 bits) :", out["sites"][site]["crossover_ours_vs_RZF16b_W"], "W")
        print("crossover (structuree T20 vs RZF 16 bits) :", out["sites"][site]["crossover_structure_T20_vs_RZF16b_W"], "W")


if __name__ == "__main__":
    main()
