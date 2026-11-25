#!/usr/bin/env python3
"""
Modele de cout FPGA pour ART (remplace le modele d'energie de Moons et al.
2017 par un modele calibre sur synthese HLS reelle, Vitis HLS 2022.2,
xcu280-fsvh2892-2L-e, clock cible 10 ns). Source des donnees : cost_table.csv
/ cost_table.json (68 points synthetises, grille b_w in {1,2,3,4,6,8}, b_a in
{2,3,4,6,8}, cas A poids constants et cas B poids variables, plus points
BIND_OP force LUT/DSP). Constantes ajustees par fit_model.py -> fit_coeffs.json.

Contexte / hypothese de deploiement vise par ART :
  Les poids sont fixes (constants) une fois quantifies, et le design FPGA
  est "deroule" (weight-stationary, un multiplieur par poids). C'est le
  CAS A du banc de synthese (hls_work/cost_model/caseA_*).

RESULTAT IMPORTANT (a lire avant d'utiliser ce module) :
  Le cas A n'est "sans DSP" (tout en additionneurs/decaleurs) QUE pour
  b_w <= 4 (observe sur les 20 points bw in {1,2,3,4} : DSP=0 partout).
  Pour b_w in {6,8}, Vitis HLS commence a inferer des DSP48 pour une
  fraction des multiplicateurs constants (ex. bw=8 -> 100-110 DSP sur 256
  MAC, soit ~0.39-0.43 DSP/MAC, un ratio qui varie avec b_a et n'est PAS une
  fonction lineaire simple). Ce changement de regime n'est pas modelise par
  une formule continue : le modele LUT ci-dessous n'est valide et ajuste QUE
  sur bw in {1,2,3,4}. Au-dela, utiliser la table brute (cost_table.csv) ou
  resynthetiser. Comme ART cible la quantification agressive (1-4 bits est
  le regime d'interet), cette restriction est jugee acceptable (cf.
  RAPPORT.md, section limites).

Convention de largeur : toutes les largeurs sont SIGNEES (ap_int<b>), y
compris b_w=1 (poids in {-1,0} -- cas degenere documente dans RAPPORT.md).
"""
import json
import math
import os
import warnings

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

ROOT = os.path.dirname(os.path.abspath(__file__))

# Domaine de validite du modele cas A (poids constants, regime sans DSP).
CASEA_BW_MAX_VALID = 4


# ---------------------------------------------------------------------------
# Chargement des constantes ajustees (fit_coeffs.json, produit par
# fit_model.py). Fallback si le fichier est absent.
# ---------------------------------------------------------------------------
_FALLBACK_CASEA = (0.8151, -1.4757, 9.6765)          # cas A, bw<=4
_FALLBACK_CASEB_LUT = (-0.3313, 3.3251, -2.2110)     # cas B, LUT/MAC (defaut)
_FALLBACK_CASEB_LUT_FORCED = (1.1161, -0.4523, 21.9377)  # multiplieur pur LUT


def _load_fit_coeffs():
    path = os.path.join(ROOT, "fit_coeffs.json")
    if not os.path.exists(path):
        return {
            "caseA_lut_per_mac_clean_bw_le_4": {"coef": list(_FALLBACK_CASEA)},
            "caseB_default_lut_per_mac": {"coef": list(_FALLBACK_CASEB_LUT)},
            "caseB_lut_forced_mult_per_mac": {"coef": list(_FALLBACK_CASEB_LUT_FORCED)},
        }
    with open(path) as f:
        return json.load(f)


_FIT = _load_fit_coeffs()
# Cas A, morceau bw=1 (degenere : poids ternaire {-1,0}, pas de vrai
# multiplieur -- juste un mux/negation. Sa pente en b_a differe du regime
# bw>=2, d'ou un fit separe).
_A1_CASEA, _B1_CASEA = _FIT.get("caseA_lut_per_mac_bw1_degenerate", {"coef": [0.3586, 3.8620]})["coef"]
# Cas A, morceau bw in {2,3,4} (vrai reseau shift-add constant).
_A_CASEA, _B_CASEA, _C_CASEA = _FIT.get(
    "caseA_lut_per_mac_bw_2_3_4", _FIT.get("caseA_lut_per_mac_clean_bw_le_4", {"coef": _FALLBACK_CASEA})
)["coef"]
_A_CASEB, _B_CASEB, _C_CASEB = _FIT["caseB_default_lut_per_mac"]["coef"]
_A_CASEB_LUTF, _B_CASEB_LUTF, _C_CASEB_LUTF = _FIT["caseB_lut_forced_mult_per_mac"]["coef"]

_warned_bw_range = False


def _check_bw_range(bw):
    """Avertit une seule fois si bw sort du domaine valide (extrapolation)."""
    global _warned_bw_range
    if _warned_bw_range:
        return
    try:
        bw_val = float(bw)
    except (TypeError, ValueError):
        return
    if bw_val > CASEA_BW_MAX_VALID:
        warnings.warn(
            f"fpga_cost: b_w={bw_val} > {CASEA_BW_MAX_VALID} -- hors du regime "
            "'sans DSP' valide/ajuste (cas A). Le modele extrapole une formule "
            "LUT pure alors que la synthese reelle alloue des DSP a partir de "
            "bw=6 (cf. RAPPORT.md). Resultat a ne pas utiliser pour un choix "
            "final sans resynthese.",
            stacklevel=2,
        )
        _warned_bw_range = True


# ---------------------------------------------------------------------------
# Cas A (poids constants, bw<=4) : modele en DEUX morceaux, raccordes a
# bw=1.5 (entre les deux choix entiers bw=1 et bw=2) :
#   bw == 1      (degenere, mux/negation, pas de vrai multiplieur) :
#       LUT_par_MAC(1, ba) = A1*ba + B1
#   bw in {2,3,4} (vrai reseau shift-add constant) :
#       LUT_par_MAC(bw, ba) = A*bw*ba + B*ba + C
# Raison : un fit trilineaire unique sur bw in {1..4} donne une pente de
# signe incorrect sur bw=1 (le cout mesure CROIT avec b_a ; cf. RAPPORT.md).
# ---------------------------------------------------------------------------

def lut_per_mac_caseA(bw, ba):
    """Cout LUT par MAC (poids constant, regime sans DSP, bw<=4) -- modele ajuste
    en 2 morceaux (bw=1 degenere vs bw in {2,3,4})."""
    _check_bw_range(bw)
    if _HAS_TORCH and (torch.is_tensor(bw) or torch.is_tensor(ba)):
        bw_t = torch.as_tensor(bw, dtype=torch.float32)
        ba_t = torch.as_tensor(ba, dtype=torch.float32)
        cost_bw1 = _A1_CASEA * ba_t + _B1_CASEA
        cost_bw234 = _A_CASEA * bw_t * ba_t + _B_CASEA * ba_t + _C_CASEA
        return torch.where(bw_t <= 1.5, cost_bw1, cost_bw234)
    if bw <= 1.5:
        return _A1_CASEA * ba + _B1_CASEA
    return _A_CASEA * bw * ba + _B_CASEA * ba + _C_CASEA


# ---------------------------------------------------------------------------
# Cas B (poids variables) : pour reference / comparaison uniquement. Pas le
# scenario de deploiement vise par ART (qui fige les poids), mais utile pour
# juger le surcout d'un design a poids reconfigurables.
# ---------------------------------------------------------------------------

def lut_per_mac_caseB(bw, ba):
    return _A_CASEB * bw * ba + _B_CASEB * (bw + ba) + _C_CASEB


def dsp_per_mac_caseB(bw):
    """Palier observe : 0 DSP pour bw=1 (poids ternaire, pas de multiplieur
    reel), 0.5 DSP/MAC sinon (Vitis empaquette 2 petites multiplications par
    DSP48E2 via SIMD, independamment de ba, pour bw in {2,3,4,6,8})."""
    bw_int = int(round(float(bw))) if not (_HAS_TORCH and torch.is_tensor(bw)) else None
    if bw_int is not None:
        return 0.0 if bw_int == 1 else 0.5
    return torch.where(bw <= 1.5, torch.zeros_like(bw), torch.full_like(bw, 0.5))


def lut_per_mac_mult_pure_lut(bw, ba):
    """Cout d'un multiplieur variable x variable pur LUT (BIND_OP impl=fabric),
    sans DSP du tout -- borne superieure utile pour juger le gain du packing DSP."""
    return _A_CASEB_LUTF * bw * ba + _B_CASEB_LUTF * (bw + ba) + _C_CASEB_LUTF


# ---------------------------------------------------------------------------
# Decomposition "cout par poids + cout par activation" pour la perte ART.
#
# Choix de modelisation (documente, cf. RAPPORT.md section limites) :
# en topologie deroulee a poids constants (cas A, bw<=4), chaque poids
# possede EXACTEMENT un multiplieur constant (1 poids = 1 MAC). Le cout
# materiel de ce multiplieur depend a la fois de b_w (nombre de termes
# shift-add) et de b_a (largeur de chaque terme) -- d'ou le terme croise
# A*bw*ba. Ce terme croise ne peut pas etre separe proprement en "une part
# qui vient du poids" et "une part qui vient de l'activation" sans
# resynthetiser avec IN/OUT variables (hors budget de ce travail). On adopte
# donc la convention suivante, simple et explicite :
#
#   cout_poids(b_w, b_a)      = A*bw*ba + C      (porte le terme croise +
#                                                  l'overhead fixe du
#                                                  multiplieur/accumulateur)
#   cout_activation(b_a)      = B*ba             (porte le terme purement
#                                                  lineaire en b_a, qui ne
#                                                  depend pas de b_w)
#
# Cette decomposition redonne EXACTEMENT le modele cas-A par MAC quand on
# somme cout_poids + cout_activation pour un seul poids associe a une seule
# activation. Pour une couche a N_w poids et N_a activations (N_w = N_a *
# fan-out en dense deroule), le cout de couche s'ecrit :
#
#   cout_layer = N_w * cout_poids(bw, ba) + N_a * cout_activation(ba)
#
# NB: si une meme activation est reutilisee par plusieurs poids (fan-out),
# son cout_activation n'est compte qu'une fois (cout de routage/registre),
# alors que cout_poids est compte une fois par poids (un multiplieur par
# poids). C'est la raison d'etre de la decomposition poids/activation.
#
# Valide pour bw<=4 (cf. avertissement _check_bw_range). Pour bw>=6, utiliser
# la table brute (DSP non nul, cf. cost_table.csv) -- ART devrait en pratique
# rarement explorer ces largeurs si l'objectif est la quantification agressive.
#
# Raccord avec le modele en 2 morceaux de lut_per_mac_caseA (bw=1 degenere
# vs bw in {2,3,4}) : pour bw=1 il n'y a pas de terme croise bw*ba (bw est
# fixe), donc la decomposition devient cout_poids=B1 (constante, intercept)
# et cout_activation=A1*ba (toute la pente mesuree vient alors de b_a). Pour
# bw>=2 on garde la decomposition cout_poids=A*bw*ba+C, cout_activation=B*ba.
# Dans les deux cas, cout_poids+cout_activation == lut_per_mac_caseA(bw, ba).
# ---------------------------------------------------------------------------

def cost_per_weight(bw, ba):
    """Cout (unite LUT-equivalent) attribue a un poids, fonction de (b_w, b_a)."""
    _check_bw_range(bw)
    if _HAS_TORCH and (torch.is_tensor(bw) or torch.is_tensor(ba)):
        bw_t = torch.as_tensor(bw, dtype=torch.float32)
        ba_t = torch.as_tensor(ba, dtype=torch.float32)
        cost_bw1 = torch.full_like(ba_t, _B1_CASEA)
        cost_bw234 = _A_CASEA * bw_t * ba_t + _C_CASEA
        return torch.where(bw_t <= 1.5, cost_bw1, cost_bw234)
    if bw <= 1.5:
        return _B1_CASEA
    return _A_CASEA * bw * ba + _C_CASEA


def cost_per_activation(ba, bw=None):
    """Cout (unite LUT-equivalent) attribue a une activation, fonction de b_a
    (et de b_w, car la pente en b_a differe selon le morceau bw=1 ou bw>=2).
    Si bw est omis, on suppose le regime bw>=2 (le plus frequent)."""
    if bw is None:
        return _B_CASEA * ba
    if _HAS_TORCH and (torch.is_tensor(bw) or torch.is_tensor(ba)):
        bw_t = torch.as_tensor(bw, dtype=torch.float32)
        ba_t = torch.as_tensor(ba, dtype=torch.float32)
        cost_bw1 = _A1_CASEA * ba_t
        cost_bw234 = _B_CASEA * ba_t
        return torch.where(bw_t <= 1.5, cost_bw1, cost_bw234)
    if bw <= 1.5:
        return _A1_CASEA * ba
    return _B_CASEA * ba


def layer_cost(n_weights, n_activations, bw, ba):
    """Cout total (LUT-equivalent) d'une couche dense deroulee a poids constants."""
    return n_weights * cost_per_weight(bw, ba) + n_activations * cost_per_activation(ba, bw=bw)


# ---------------------------------------------------------------------------
# Version torch, differentiable en bw, ba (pour relaxation continue dans ART
# : bw, ba sont des tenseurs a virgule flottante pendant l'apprentissage,
# arrondis seulement a l'inference / export).
# ---------------------------------------------------------------------------

def layer_cost_torch(n_weights, n_activations, bw, ba):
    """
    bw, ba : torch.Tensor (peuvent etre fractionnaires pendant l'apprentissage,
    ART les relaxe en continu puis arrondit). Retourne un cout scalaire ou par
    couche (meme shape que bw/ba), differentiable.

    ATTENTION : valide pour bw <= 4 (regime sans DSP observe). Si l'ART
    explore bw > 4 pendant la recherche, soit clamper bw avant l'appel, soit
    ajouter une penalite / un terme DSP a part (non fourni ici, cf. RAPPORT.md).
    """
    if not _HAS_TORCH:
        raise ImportError("torch n'est pas installe dans cet environnement")
    cost_w = cost_per_weight(bw, ba)
    cost_a = cost_per_activation(ba, bw=bw)
    return n_weights * cost_w + n_activations * cost_a


# ---------------------------------------------------------------------------
# Modele de Moons et al. 2017 (pour comparaison -- modele d'energie de
# circuit dedie generique, PAS calibre sur FPGA). E(b) ~= (b/16)**1.9 avec
# b = max(b_w, b_a) (convention usuelle dans la litterature de quantification).
# ---------------------------------------------------------------------------

def moons_cost_per_mac(bw, ba):
    if _HAS_TORCH and (torch.is_tensor(bw) or torch.is_tensor(ba)):
        b = torch.maximum(torch.as_tensor(bw, dtype=torch.float32),
                           torch.as_tensor(ba, dtype=torch.float32))
    else:
        b = max(bw, ba)
    return (b / 16.0) ** 1.9


def moons_layer_cost(n_mac, bw, ba):
    return n_mac * moons_cost_per_mac(bw, ba)


# ---------------------------------------------------------------------------
# Comparaison de classement FPGA (cas A, cette etude) vs Moons : pour une
# grille de choix (bw, ba) valides (bw<=4, regime calibre), est-ce que les
# deux modeles ordonnent les configurations de la meme facon (du moins cher
# au plus cher) ? Utilise une correlation de rang de Spearman (implementee
# a la main, sans scipy).
# ---------------------------------------------------------------------------

def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


def compare_rankings(bw_values=(1, 2, 3, 4), ba_values=(2, 3, 4, 6, 8), verbose=True):
    """Compare le classement (du moins cher au plus cher) des configurations
    (bw, ba) selon le modele FPGA cas A ajuste ici vs le modele de Moons.
    Restreint a bw<=4 (seul domaine calibre proprement sans DSP)."""
    grid = [(bw, ba) for bw in bw_values for ba in ba_values]
    fpga_costs = [lut_per_mac_caseA(bw, ba) for bw, ba in grid]
    moons_costs = [moons_cost_per_mac(bw, ba) for bw, ba in grid]

    rho = _spearman(fpga_costs, moons_costs)

    order_fpga = sorted(range(len(grid)), key=lambda i: fpga_costs[i])
    order_moons = sorted(range(len(grid)), key=lambda i: moons_costs[i])

    if verbose:
        print(f"Correlation de rang de Spearman (FPGA cas A vs Moons) : rho={rho:.4f}")
        print(f"{'bw':>3}{'ba':>3}{'FPGA_LUT/MAC':>14}{'Moons/MAC':>12}")
        for bw, ba in grid:
            print(f"{bw:>3}{ba:>3}{lut_per_mac_caseA(bw, ba):>14.3f}{moons_cost_per_mac(bw, ba):>12.4f}")
        print("\nTop-5 configs les moins cheres selon FPGA (cas A) :")
        for i in order_fpga[:5]:
            print(f"  bw={grid[i][0]} ba={grid[i][1]}  FPGA={fpga_costs[i]:.2f}  Moons={moons_costs[i]:.4f}")
        print("Top-5 configs les moins cheres selon Moons :")
        for i in order_moons[:5]:
            print(f"  bw={grid[i][0]} ba={grid[i][1]}  FPGA={fpga_costs[i]:.2f}  Moons={moons_costs[i]:.4f}")
        same_top1 = order_fpga[0] == order_moons[0]
        print(f"\nMeme configuration optimale (top-1) sous les deux modeles ? {same_top1}")

    return {
        "spearman_rho": rho,
        "grid": grid,
        "fpga_costs": fpga_costs,
        "moons_costs": moons_costs,
        "order_fpga": order_fpga,
        "order_moons": order_moons,
    }


if __name__ == "__main__":
    fb1 = _FIT.get("caseA_lut_per_mac_bw1_degenerate")
    fb234 = _FIT.get("caseA_lut_per_mac_bw_2_3_4")
    print("Constantes cas A, regime propre bw<=4, modele en 2 morceaux :")
    print(f"  bw=1        : LUT/MAC = {_A1_CASEA:.4f}*ba + {_B1_CASEA:.4f}"
          + (f"   (R2={fb1['r2']:.4f} RMSE={fb1['rmse']:.3f} n={fb1['n']})" if fb1 else ""))
    print(f"  bw in{{2,3,4}}: LUT/MAC = {_A_CASEA:.4f}*bw*ba + {_B_CASEA:.4f}*ba + {_C_CASEA:.4f}"
          + (f"   (R2={fb234['r2']:.4f} RMSE={fb234['rmse']:.3f} n={fb234['n']})" if fb234 else ""))
    print()
    print("Exemples (LUT-equivalent par MAC, cas A, bw<=4) :")
    for bw in [1, 2, 3, 4]:
        for ba in [2, 4, 6, 8]:
            print(f"  bw={bw} ba={ba} -> cost/MAC(FPGA cas A)={lut_per_mac_caseA(bw, ba):7.2f}"
                  f"   Moons/MAC={moons_cost_per_mac(bw, ba):.4f}")
    print()
    print("=== Comparaison de classement FPGA (cas A) vs Moons ===")
    compare_rankings()
