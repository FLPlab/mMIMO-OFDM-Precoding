#!/usr/bin/env python3
"""
Genere les noyaux HLS pour le modele de cout FPGA (ART-FPGA).

Noyau : couche dense IN=16 -> OUT=16, entierement deroulee et pipelinee (II=1),
y[o] = sum_i w[o][i] * x[i].

Convention : toutes les largeurs (poids b_w, activations b_a) sont SIGNEES
(ap_int<b>), y compris b_w=1 (donc poids in {-1,0} seulement pour b_w=1 -- un
choix degenerate mais garde pour la coherence de la grille).

Deux cas :
  A : poids CONSTANTS (tableau static const, generes avec une graine fixe par
      point pour la reproductibilite) -- design "deroule" typique d'un
      accelerateur a poids figes.
  B : poids VARIABLES, passes en argument du noyau (meme convention de
      largeur) -- pour comparer au cas A.

Pour quelques points du cas B, on genere aussi des variantes ou la
multiplication est forcee en LUT (BIND_OP impl=fabric) ou en DSP
(BIND_OP impl=dsp), pour mesurer l'effet du choix d'implementation.

Chaque point est un dossier contenant :
  - dense_kernel.cpp
  - run_synth.tcl   (csynth seulement, part xcu280-fsvh2892-2L-e, clk 10ns)
"""
import os
import random
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
IN_N = 16
OUT_N = 16
PART = "xcu280-fsvh2892-2L-e"
CLK_NS = 10

BW_GRID = [1, 2, 3, 4, 6, 8]
BA_GRID = [2, 3, 4, 6, 8]  # grille complete (chaque csynth ~40s, cf. test prealable)

# points pour la comparaison BIND_OP LUT vs DSP (cas B uniquement)
BINDOP_POINTS = [(1, 2), (4, 4), (6, 6), (8, 8)]


def acc_width(bw, ba):
    # bw + ba (produit signe) + log2(IN_N) pour la somme + 1 garde
    import math
    return bw + ba + int(math.ceil(math.log2(IN_N))) + 1


def gen_weights(bw, seed):
    rnd = random.Random(seed)
    lo, hi = -(2 ** (bw - 1)), (2 ** (bw - 1)) - 1
    return [[rnd.randint(lo, hi) for _ in range(IN_N)] for _ in range(OUT_N)]


def cpp_header(bw, ba, acc_w):
    return f"""// Genere automatiquement par gen_kernels.py -- ne pas editer a la main.
#include <ap_int.h>

#define IN_N {IN_N}
#define OUT_N {OUT_N}
#define BW {bw}
#define BA {ba}
#define ACC_W {acc_w}

typedef ap_int<BA> data_t;
typedef ap_int<BW> weight_t;
typedef ap_int<ACC_W> acc_t;
"""


def weights_array_literal(weights):
    rows = []
    for row in weights:
        rows.append("{" + ", ".join(str(v) for v in row) + "}")
    return "{\n  " + ",\n  ".join(rows) + "\n}"


def gen_case_a(bw, ba, seed):
    acc_w = acc_width(bw, ba)
    weights = gen_weights(bw, seed)
    body = cpp_header(bw, ba, acc_w)
    body += f"""
// Cas A : poids CONSTANTS (design deroule), graine fixe = {seed}
static const weight_t W[OUT_N][IN_N] = {weights_array_literal(weights)};

void dense_kernel(data_t x[IN_N], acc_t y[OUT_N]) {{
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
    for (int o = 0; o < OUT_N; o++) {{
#pragma HLS UNROLL
        acc_t sum = 0;
        for (int i = 0; i < IN_N; i++) {{
#pragma HLS UNROLL
            weight_t wv = W[o][i];
            data_t xv = x[i];
            acc_t prod = (acc_t)wv * (acc_t)xv;
            sum += prod;
        }}
        y[o] = sum;
    }}
}}
"""
    return body, weights


def gen_case_b(bw, ba, bindop=None):
    acc_w = acc_width(bw, ba)
    body = cpp_header(bw, ba, acc_w)
    bindop_pragma = ""
    if bindop == "lut":
        bindop_pragma = "#pragma HLS BIND_OP variable=prod op=mul impl=fabric\n            "
    elif bindop == "dsp":
        bindop_pragma = "#pragma HLS BIND_OP variable=prod op=mul impl=dsp\n            "
    body += f"""
// Cas B : poids VARIABLES (argument du noyau)
void dense_kernel(data_t x[IN_N], weight_t w[OUT_N][IN_N], acc_t y[OUT_N]) {{
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=w complete dim=0
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
    for (int o = 0; o < OUT_N; o++) {{
#pragma HLS UNROLL
        acc_t sum = 0;
        for (int i = 0; i < IN_N; i++) {{
#pragma HLS UNROLL
            weight_t wv = w[o][i];
            data_t xv = x[i];
            acc_t prod = (acc_t)wv * (acc_t)xv;
            {bindop_pragma}sum += prod;
        }}
        y[o] = sum;
    }}
}}
"""
    return body


TCL_TEMPLATE = """open_project -reset proj
set_top dense_kernel
add_files dense_kernel.cpp -cflags "-std=c++14"
open_solution -reset "solution1" -flow_target vivado
set_part {{{part}}}
create_clock -period {clk} -name default
csynth_design
exit
""".format


def write_point(name, cpp_body):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "dense_kernel.cpp"), "w") as f:
        f.write(cpp_body)
    with open(os.path.join(d, "run_synth.tcl"), "w") as f:
        f.write(TCL_TEMPLATE(part=PART, clk=CLK_NS))
    return d


def main():
    manifest = []
    seed_base = 20260923  # graine fixe, documentee

    for bw in BW_GRID:
        for ba in BA_GRID:
            seed = seed_base + bw * 100 + ba
            # Cas A
            name_a = f"caseA_bw{bw}_ba{ba}"
            body_a, weights = gen_case_a(bw, ba, seed)
            write_point(name_a, body_a)
            manifest.append({"point": name_a, "case": "A", "bw": bw, "ba": ba,
                              "acc_w": acc_width(bw, ba), "seed": seed, "bindop": None})

            # Cas B (defaut, pas de BIND_OP force)
            name_b = f"caseB_bw{bw}_ba{ba}"
            body_b = gen_case_b(bw, ba, bindop=None)
            write_point(name_b, body_b)
            manifest.append({"point": name_b, "case": "B", "bw": bw, "ba": ba,
                              "acc_w": acc_width(bw, ba), "seed": None, "bindop": "default"})

    # Variantes BIND_OP (cas B seulement) sur un sous-ensemble de points
    for bw, ba in BINDOP_POINTS:
        for mode in ("lut", "dsp"):
            name = f"caseB_bw{bw}_ba{ba}_{mode}"
            body = gen_case_b(bw, ba, bindop=mode)
            write_point(name, body)
            manifest.append({"point": name, "case": "B", "bw": bw, "ba": ba,
                              "acc_w": acc_width(bw, ba), "seed": None, "bindop": mode})

    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(manifest)} points generes dans {ROOT}")
    for m in manifest:
        print(" ", m["point"])


if __name__ == "__main__":
    main()
