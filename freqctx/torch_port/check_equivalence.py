"""Verifie que le modele PyTorch (torch_port/models.py) charge avec les poids Keras exportes
redonne les memes sorties que Keras. A lancer dans l'environnement PyTorch (bitadapt) :

    python -m torch_port.check_equivalence [--weights pilot_rand_ph32]
"""
import argparse
import os

import numpy as np
import torch

from torch_port.models import FreqctxModel, load_keras_npz

OUT = "/export/tmp/sala/freqctx/torch_port"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="pilot_rand_ph32")
    ap.add_argument("--hidden", type=int, default=32)
    a = ap.parse_args()
    model = load_keras_npz(FreqctxModel(hidden=a.hidden), os.path.join(OUT, f"{a.weights}.npz")).eval()
    ref = np.load(os.path.join(OUT, f"{a.weights}_ref.npz"))
    with torch.no_grad():
        h = torch.tensor(ref["h"])
        h_hat = model.estimator(h).numpy()
        w = model(h, torch.tensor(ref["snr"])).numpy()
    rel = lambda x, y: float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
    e_enc, e_w = rel(h_hat, ref["h_hat"]), rel(w, ref["w"])
    n = sum(p.numel() for p in model.parameters())
    print(f"parametres {n} | ecart relatif max : encodeur {e_enc:.1e}, precodeur W {e_w:.1e}")
    assert e_enc < 1e-4 and e_w < 1e-4, "les deux versions different"
    print("EQUIVALENCE KERAS / PYTORCH : OK")


if __name__ == "__main__":
    main()
