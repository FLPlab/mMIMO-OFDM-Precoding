"""AUDIT independant des calculs d'energie (2026-11-28, CPU, lecture seule du code existant).

Recompte a la main (formules ecrites ici, sans reutiliser counts_* de energy_classical) les MAC de RZF,
R-WMMSE (forme Gram), WMMSE standard O(M^3), SiPS, et de nos precodeurs ; recalcule l'energie avec
Pipeline/Energy.compute_energy (v5) ; teste les sensibilites :
  (a) convention LSQ du labo : clamp [-(2^b-1), 2^b-1] = 2^(b+1)-1 niveaux -> ap_int<b+1> (b+1 bits signes),
      alors que la virgule fixe des classiques compte b bits SIGNE COMPRIS ;
  (b) operations non comptees : chez nous LayerNorm, GELU, residus, normalisation par utilisateur (rsqrt,
      log), |G|, requantification LSQ (un produit par sortie), normalisation de W ; chez les classiques
      mise a l'echelle finale de W, normalisation de C a chaque iteration, surcout exposant de bloc
      (detection de zeros de tete + decalage par mot ecrit), et economie possible (G C reutilise) ;
  (c) efficacite systeme : seuils de P_tx recalcules, avec reutilisation du precodeur sur un creneau ;
  (d) OFDM : cout par SC avec un pas frequentiel s (meme interpolation pour tous).

    CUDA_VISIBLE_DEVICES="" python -u -m torch_port.energy_audit
"""
import json
import math
import os
import sys

V5 = "/export/tmp/sala/home_archive/ELE6310/ELE6310E"
sys.path.insert(0, V5)
from Pipeline.Energy import compute_energy  # noqa: E402

RES = "/users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx/results/stecath"
M, K, DIV = 64, 4, 10
uJ = lambda pj: float(pj) / 1e6


def E(mac, act, q, w=0):
    return uJ(compute_energy(mac, w, act, float(q), float(q)))


# ------------------------------------------------------------------ 1. MAC classiques (recompte)
def rzf_counts(extra=False):
    mac = 4 * K * K * M            # G = H H^H (pleine, comme chez nous)
    mac += 4 * K ** 3 + K * DIV    # inverse K x K (Gauss-Jordan complexe ~ K^3 MAC complexes)
    mac += 4 * M * K * K           # W = H^H X
    mac += 2 * M * K + DIV         # ||W||^2 + rsqrt
    if extra:
        mac += 2 * M * K           # W * rsqrt (mise a l'echelle reelle) : NON compte dans le code
    act = 2 * K * M + 2 * 2 * K * K + 2 * M * K
    return mac, act


def gram_counts(n, extra=False, reuse=False):
    it = 4 * K ** 3                # T = G C
    it += 2 * K * K                # |T|^2
    it += 6 * K + 3 * K * DIV      # g, omega, a (3 reciproques)
    it += 2 * K * K                # A = diag(a) G + mu I
    it += 4 * K ** 3 + K * DIV     # inverse
    it += 4 * K * K                # C = A^-1 diag(.)
    it += 4 * K ** 3 + 2 * K * K   # tr(C^H G C)
    if extra:
        it += DIV + 2 * K * K      # rsqrt + mise a l'echelle de C : non compte
    if reuse:
        it -= 4 * K ** 3           # T(t+1) = (G C_t) / sqrt(p) deja calcule pour la puissance
    mac = 4 * K * K * M + 4 * K ** 3 + n * it + 4 * M * K * K + 2 * M * K + DIV
    if extra:
        mac += 2 * M * K
    act = 2 * K * M + 4 * K * K + n * (10 * K * K + 6 * K) + 2 * M * K
    return mac, act


def standard_counts(n):
    r_mac, r_act = rzf_counts()
    it = 4 * K * K * M + (2 * K * K + 6 * K + 3 * K * DIV) + (2 * K * M + 4 * K * M * M + M) + 4 * M * K
    it += 4 * M ** 3 + M * DIV + 4 * M * M * K + 2 * M * K + DIV
    act = 2 * K * K + 6 * K + 2 * M * M + 2 * M * K + 2 * M * M + 2 * M * K
    return r_mac + n * it, r_act + n * act


def sips_nc():
    return (8 / 3) * M ** 3 * K + 4 * M ** 2 * K + 4 * M * (4 * K ** 2 + 2 * K) + 4 * K ** 2 + (56 / 3) * K


# ------------------------------------------------------------------ 2. nos precodeurs (recompte)
def pair_layers(hidden, L, d=16, inp=7):
    T = K * K
    lay = {"init": (T * inp * 2 * d, inp * 2 * d + 2 * d, T * 2 * d)}
    for i in range(L):
        x = 2 * d + 3
        lay[f"blocks.{i}.a"] = (T * x * 2 * d, x * 2 * d, T * 2 * d)
        lay[f"blocks.{i}.b"] = (T * x * 2 * d, x * 2 * d, T * 2 * d)
        lay[f"blocks.{i}.f1"] = (T * (x + 2 * d) * hidden, (x + 2 * d) * hidden + hidden, T * hidden)
        lay[f"blocks.{i}.f2"] = (T * hidden * 2 * d, hidden * 2 * d + 2 * d, T * 2 * d)
    lay["read"] = (T * 2 * d * 2, 2 * d * 2 + 2, T * 2)
    lay["G"] = (4 * K * K * M, 0, 2 * K * K)
    lay["W"] = (4 * M * K * K, 0, 2 * M * K)
    for i in range(L):
        lay[f"tri{i}"] = (4 * K ** 3 * d, 0, 2 * K * K * d)
    return lay


def ours_energy(bits, hidden, L, plus=0, act_store=None):
    """act_store : largeur des activations ECRITES (act_t = 19 bits dans notre HLS, hls/README.md) ;
    None = convention v5 (activations a Q_A)."""
    lay = pair_layers(hidden, L)
    e = 0.0
    for n, (mac, w, a) in lay.items():
        qw, qa = bits[n]
        if act_store is None:
            e += uJ(compute_energy(mac, w, a, float(qw + plus), float(qa + plus)))
        else:
            e += uJ(compute_energy(mac, w, 0, float(qw + plus), float(qa + plus)))
            e += uJ(compute_energy(0, 0, a, float(act_store), float(act_store)))
    return e, sum(v[0] for v in lay.values()), sum(v[1] for v in lay.values())


def ours_uncounted(hidden, L, d=16, gelu_cost=5, q_ln=16, q_misc=8):
    """Operations non comptees chez nous (equivalent MAC), (ops a q_ln bits, ops a q_misc bits, sorties)."""
    T = K * K
    ln = L * T * (6 * 2 * d + DIV)                 # LN : moyenne, centrage, variance, rsqrt, *rstd, affine
    ln += K * (2 * M + DIV) + K * 2 * M + K * DIV  # normalisation par utilisateur (||h||^2, rsqrt, h*r, log)
    ln += T * (2 + DIV)                            # |G| = sqrt(re^2 + im^2)
    ln += 2 * M * K + DIV + 2 * M * K              # normalize_power(W)
    misc = L * T * hidden * gelu_cost              # GELU
    misc += L * T * 2 * d                          # residus
    lay = pair_layers(hidden, L)
    requant = sum(a for n, (m, w, a) in lay.items())   # un produit (echelle LSQ) par sortie
    return ln + requant, misc, requant


# ------------------------------------------------------------------ main
def main():
    out = {"meta": {"note": __doc__.split("\n")[0], "div_mac": DIV, "M": M, "K": K}}
    # --- MAC
    mac = {"RZF": rzf_counts()[0], "RZF_avec_mise_a_echelle": rzf_counts(True)[0]}
    for n in (1, 5, 30, 100, 300):
        mac[f"Gram_{n}"] = gram_counts(n)[0]
        mac[f"Gram_{n}_extra+reuse"] = gram_counts(n, True, True)[0]
        mac[f"standard_{n}"] = standard_counts(n)[0]
        mac[f"SiPS_{n}"] = n * sips_nc()
    mac["8c"] = sum(v[0] for v in pair_layers(32, 8).values())
    mac["2c_h16"] = sum(v[0] for v in pair_layers(16, 2).values())
    mac["4c_h32"] = sum(v[0] for v in pair_layers(32, 4).values())
    mac["4c_h16"] = sum(v[0] for v in pair_layers(16, 4).values())
    out["mac"] = mac
    # controle contre les JSON
    ce = json.load(open(os.path.join(RES, "classical_energy.json")))["methods"]
    ev2 = json.load(open(os.path.join(RES, "energy_efficiency_v2.json")))["sites"]["stecath"]["WMMSE"]
    out["controle_json"] = {
        "RZF_mac_json": ce["RZF"]["mac"], "Gram30_mac_json": ce["WMMSE 30 it. (forme Gram)"]["mac"],
        "std30_mac_json": ev2["30"]["standard_mac"], "sips_nc": sips_nc()}

    # --- nos points, convention LSQ
    pts = {"stecath_8c": ("art_pair_h32_lam5_qprod_norm", 32, 8), "stecath_2c": ("art_pair_h16_L2_lam5_qprod_norm", 16, 2),
           "stecath_4c_h32": ("art_pair_h32_L4_lam5_qprod_norm", 32, 4),
           "ericsson_2c": ("art_pair_h16_L2_lam5_qprod_norm_ericsson", 16, 2),
           "ericsson_2c_lam1": ("art_pair_h16_L2_lam1_qprod_norm_ericsson", 16, 2),
           "ericsson_4c": ("art_pair_h16_L4_lam5_qprod_norm_ericsson", 16, 4)}
    ours = {}
    for k, (f, hid, L) in pts.items():
        q = json.load(open(os.path.join(RES, f + ".json")))
        e_b, m_, w_ = ours_energy(q["bits"], hid, L)
        e_b1, _, _ = ours_energy(q["bits"], hid, L, plus=1)
        e_b1_a19, _, _ = ours_energy(q["bits"], hid, L, plus=1, act_store=19)
        ln_ops, misc_ops, req = ours_uncounted(hid, L)
        e_unc_hi = E(ln_ops, 0, 16) + E(misc_ops, misc_ops, 8)        # LN/rsqrt/requant a 16 b, GELU/residus a 8 b
        e_unc_lo = E(ln_ops, 0, 12) + E(misc_ops // 5, 0, 8)          # optimiste : 12 b, GELU par table
        ours[k] = {"sr_quant_20k": q["sr_quant"], "mac": m_, "weights": w_,
                   "E_json_torch_energy_uJ": q["energy_pJ"]["quant_corrige"] / 1e6,
                   "E_v5_b_uJ": e_b, "E_v5_b+1_uJ": e_b1, "E_v5_b+1_act19_uJ": e_b1_a19,
                   "uncounted_ops": {"ln_rsqrt_requant_norm": ln_ops, "gelu_residus": misc_ops, "requant": req},
                   "E_uncounted_uJ_[opt,cons]": [e_unc_lo, e_unc_hi],
                   "E_corrige_uJ_[b+1, b+1+noncomptes]": [e_b1, e_b1 + e_unc_hi]}
    out["ours"] = ours

    # --- classiques : bits mesures, + operations non comptees
    fx = json.load(open(os.path.join(RES, "fixedpoint_classical.json")))["sites"]
    SR = {"stecath": {"RZF": 17.0315, 1: 19.0107, 5: 21.6172, 30: 24.4141, 100: 25.9153, 300: 26.8810},
          "ericsson": {"RZF": 33.4382, 1: 34.1909, 5: 35.4481, 30: 37.3125, 100: 38.6654, 300: 39.8423}}
    BITS = {"stecath": {"RZF": 13, "W": 13}, "ericsson": {"RZF": 14, "W": 15}}
    cl = {}
    for site in ("stecath", "ericsson"):
        b = BITS[site]
        r = {}
        m_, a_ = rzf_counts()
        m2, a2 = rzf_counts(True)
        r["RZF"] = {"sr": SR[site]["RZF"], "bits": b["RZF"], "E_uJ": E(m_, a_, b["RZF"]),
                    "E_extra_bfp_uJ": E(m2 + a2, a2, b["RZF"] + 1), "E_16b": E(m_, a_, 16), "E_32b": E(m_, a_, 32)}
        for n in (5, 30, 100, 300):
            m_, a_ = gram_counts(n)
            m2, a2 = gram_counts(n, True, True)
            sm, sa = standard_counts(n)
            r[f"Gram_{n}"] = {"sr": SR[site][n], "bits": b["W"], "mac": m_, "E_uJ": E(m_, a_, b["W"]),
                              "E_extra_reuse_bfp_uJ": E(m2 + a2, a2, b["W"] + 1),
                              "bits_mesures": n <= 30}
            r[f"standard_{n}"] = {"sr": SR[site][n], "mac": sm, "E16_uJ": E(sm, sa, 16),
                                  f"E{b['W']}_uJ": E(sm, sa, b["W"])}
            r[f"SiPS_{n}"] = {"sr": SR[site][n], "E_uJ": 296.0 / 92.8 * n,
                              "pJ_par_MAC_implicite": 296e6 / (92.8 * sips_nc()),
                              "E_moons16_uJ": E(n * sips_nc(), 0, 16)}
        r["fixedpoint_min_bits_json"] = fx[site]["min_bits"]
        cl[site] = r
    out["classiques"] = cl

    # --- efficacite systeme
    def cross(sa, ea, sb, eb, nsc, tsym, reuse=1, pa_eff=1.0):
        """P_tx (W rayonnes) au-dela duquel a (plus de debit, plus de calcul) bat b ; calcul amorti sur
        `reuse` symboles ; energie rayonnee divisee par le rendement d'ampli pa_eff (consommee)."""
        rhs = (sb * ea - sa * eb) / reuse / (sa - sb)       # uJ par symbole et par SC
        return max(rhs, 0) * nsc * pa_eff / (tsym * 1e6)
    T_SYM = (1 / 30e3) * 1.07
    sysr = {}
    for nsc in (3333, 3276):
        for lab, (sa, ea, sb, eb) in {
                "stecath_8c_vs_RZF16": (26.650, 0.11334, 17.0315, 0.0161959),
                "stecath_8c_vs_RZF13": (26.650, 0.11334, 17.0315, 0.0115461),
                "ericsson_2c_vs_RZF16": (34.3546, 0.030468, 33.4382, 0.0161959),
                "ericsson_struct_vs_RZF16": (40.903, 0.1066, 33.4382, 0.0161959),
                "stecath_8c_b+1_vs_RZF13": (26.650, ours["stecath_8c"]["E_v5_b+1_uJ"], 17.0315, 0.0115461),
                "ericsson_2c_b+1_vs_RZF14": (34.3546, ours["ericsson_2c"]["E_v5_b+1_uJ"], 33.4382, 0.0130189)}.items():
            sysr[f"{lab}_nsc{nsc}"] = {"pas_de_reutilisation": cross(sa, ea, sb, eb, nsc, T_SYM),
                                       "reutilise_14_symboles": cross(sa, ea, sb, eb, nsc, T_SYM, 14),
                                       "14_symboles_PA_35pct": cross(sa, ea, sb, eb, nsc, T_SYM, 14, 1 / 0.35)}
    out["systeme_seuils_W"] = sysr
    out["E_rad_par_SC_symbole_uJ_20W"] = 20 / 3333 * T_SYM * 1e6

    # --- OFDM : cout par SC avec un pas s (interpolation lineaire de W : 2 x 2MK MAC reelles par SC)
    interp = (4 * M * K, 2 * M * K)
    apply_prec = (4 * M * K, 2 * M)                           # x = W s par SC et par SYMBOLE (commun a tous)
    ofdm = {"E_interp_par_SC_uJ_12b": E(*interp, 12), "E_application_W_par_SC_symbole_uJ_12b": E(*apply_prec, 12)}
    for s in (1, 4, 9, 20):
        row = {}
        for lab, e in (("RZF_13b", cl["stecath"]["RZF"]["E_uJ"]), ("Gram5", cl["stecath"]["Gram_5"]["E_uJ"]),
                       ("Gram30", cl["stecath"]["Gram_30"]["E_uJ"]), ("nous_2c_b+1", ours["stecath_2c"]["E_v5_b+1_uJ"]),
                       ("nous_8c_b+1", ours["stecath_8c"]["E_v5_b+1_uJ"])):
            row[lab] = e / s + (ofdm["E_interp_par_SC_uJ_12b"] if s > 1 else 0)
        ofdm[f"s={s}"] = row
    out["ofdm_par_SC_uJ"] = ofdm

    path = os.path.join(RES, "energy_audit.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
