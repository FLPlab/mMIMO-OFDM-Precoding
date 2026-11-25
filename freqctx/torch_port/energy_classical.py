"""Energie de RZF et WMMSE dans le MEME modele d'energie que les reseaux (Moons et al. 2017, constantes
Horowitz, version corrigee = torch_port.energy corrected=True = Pipeline/Energy.py v5 du depot
BitAdaptHLS), sur St-Catherine (M=64, K=4, CSI parfait, protocole du labo, 20 000 echantillons fixes).

Comptage (MAC reelles par echantillon ; un produit complexe = 4 MAC reelles), forme EFFICACE en
dimension K (identite push-through : (mu I + H^H D H)^-1 H^H = H^H (mu I + D H H^H)^-1), qui donne
exactement le meme precodeur que notre implementation en dimension M :
  G = H H^H (une fois)            4 K^2 M
  RZF : inverse K x K             4 K^3 + K divisions ;   W = H^H X : 4 M K^2 ;  normalisation 2 M K + 1 racine
  WMMSE, par iteration : H W 4 K^2 M ; recepteurs / poids 2 K^2 + 6 K + 3 K divisions ;
                         matrice K x K 4 K^2 ; inverse 4 K^3 + K divisions ; W = H^H X 4 M K^2 ;
                         normalisation 2 M K + 1 racine ; initialisation par RZF.
Une division ou une racine est comptee comme 10 MAC (hypothese). Activations (valeurs ecrites puis
relues) : G, X, inverse, W, H W ; lecture de H (2 K M) une fois. Aucun poids appris.
Precision : 32 bits (virgule flottante, comptee comme Q = 32) et 16 bits (virgule fixe optimiste : en
pratique l'inversion a fort ecart de puissances demande plus).

    python -m torch_port.energy_classical
"""
import json
import os

import torch

from torch_port.energy import layer_energy
from torch_port.stecath import DEV, K, RESULTS, batched, draw, lab_rate, pool, rzf, wmmse

M, DIV = 64, 10
ITERS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300]


def counts_rzf():
    mac = 4 * K * K * M + 4 * K ** 3 + DIV * K + 4 * M * K * K + 2 * M * K + DIV
    act = 2 * K * M + 2 * K * K * 2 + 2 * M * K
    return mac, act


def counts_wmmse(n):
    mac, act = counts_rzf()
    it_mac = 4 * K * K * M + 2 * K * K + 6 * K + 3 * DIV * K + 4 * K * K + 4 * K ** 3 + DIV * K + 4 * M * K * K + 2 * M * K + DIV
    it_act = 2 * K * K + 6 * K + 2 * K * K * 2 + 2 * M * K
    return mac + n * it_mac, act + n * it_act


def energy_uJ(mac, act, Q):
    return float(layer_energy(mac, 0, act, Q, Q, corrected=True)) / 1e6


def main():
    P = pool()
    Hte = draw(P, 20000, seed=2026)
    out = {"meta": {"note": __doc__.split("\n")[0], "div_cost_mac": DIV}, "methods": {}}
    with torch.no_grad():
        r = float(batched(lambda H: lab_rate(rzf(H), H), Hte).mean())
        mac, act = counts_rzf()
        out["methods"]["RZF"] = {"sr": r, "mac": mac, "energy_uJ_32b": energy_uJ(mac, act, 32),
                                 "energy_uJ_16b": energy_uJ(mac, act, 16)}
        print("RZF", out["methods"]["RZF"], flush=True)
        for n in ITERS:
            r = float(batched(lambda H: lab_rate(wmmse(H, n), H), Hte, bs=2048).mean())
            mac, act = counts_wmmse(n)
            out["methods"][f"WMMSE {n} it."] = {"sr": r, "iters": n, "mac": mac,
                                                "energy_uJ_32b": energy_uJ(mac, act, 32),
                                                "energy_uJ_16b": energy_uJ(mac, act, 16)}
            print(f"WMMSE {n}", out["methods"][f"WMMSE {n} it."], flush=True)
    path = os.path.join(RESULTS, "classical_energy.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
