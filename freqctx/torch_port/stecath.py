"""St-Catherine (lancer de rayons, mono-porteuse, M=64, K=4) : notre precodeur par paires
contre les modeles BitAdapt du labo (mMIMO-ART-FPGA, un token par utilisateur), RZF et WMMSE,
avec le protocole du labo :
  - pool : 9 032 positions (LOS + NLOS) ; un echantillon = 4 positions tirees au hasard ;
  - debit : Learner.rate_calculator_3d, W_eff = conj(H) F, F de norme de Frobenius 1,
    bruit 1e-13 sur le canal brut (affaiblissement de parcours reel : SNR par utilisateur
    de -16 a +35 dB) ;
  - entree de leurs modeles : conj(H) normalise (Frobenius), [Im, Re] par antenne.
Notre precodeur voit h = conj(H) ramene a une energie moyenne 1 par element et le SNR
moyen de l'echantillon (meme convention que freqctx) ; CSI parfait, precodeur seul.
Les echantillons de test sont tires avec une graine fixe, identiques pour toutes les methodes.

    python -m torch_port.stecath --train          # entraine notre precodeur
    python -m torch_port.stecath                  # evalue tout (leurs checkpoints FP32 compris)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat

from torch_port.models import PairPrecoder, normalize_power

DATA = "/usr/datasets/deepMIMO/HH_channels/Dataset_matlab/"
LAB = "/users/sala/Documents/test_projet/external/mMIMO-ART-FPGA"
OUT_W = "/export/tmp/sala/freqctx/weights/stecath"
RESULTS = "/users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx/results/stecath"
NOISE, K = 1e-13, 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pool():
    H = [loadmat(f"{DATA}dataset_stecath_{l}_raytracing.mat")[f"CSIs_{l}"] for l in ("LOS", "NLOS")]
    return np.concatenate(H, 0).astype(np.complex64)                      # [9032, 64]


def draw(P, n, seed):
    idx = np.random.default_rng(seed).integers(0, P.shape[0], size=(n, K))
    return torch.tensor(P[idx])                                           # H brut [n, K, M]


def lab_rate(F, H):
    """Learner.rate_calculator_3d du labo, F [n, M, K] normalise, H brut [n, K, M]."""
    W = torch.einsum("nij,njk->nik", torch.conj(H), F)
    d = torch.diagonal(W.abs() ** 2, dim1=1, dim2=2)
    return torch.log2(1 + d / ((W.abs() ** 2).sum(2) - d + NOISE)).sum(1)


def ours_input(H):
    h = torch.conj(H)                                                      # notre h : y = h W s
    e = (h.abs() ** 2).mean(dim=(1, 2))
    return h / torch.sqrt(e)[:, None, None].to(h.dtype), 10 * torch.log10(e / NOISE)


def rzf(H):
    h = torch.conj(H).to(torch.complex128)
    G = h @ h.conj().transpose(-1, -2)
    I = torch.eye(K, dtype=h.dtype, device=h.device)
    return normalize_power(h.conj().transpose(-1, -2) @ torch.linalg.inv(G + K * NOISE * I)).to(torch.complex64)


def wmmse(H, iters=30):
    """Notre WMMSE (precoders/classical.py), en complex128, initialise par RZF."""
    h = torch.conj(H).to(torch.complex128)
    M = h.shape[-1]
    w = rzf(H).to(torch.complex128)
    eye = torch.eye(M, dtype=h.dtype, device=h.device)
    for _ in range(iters):
        hw = h @ w
        sig = torch.diagonal(hw, dim1=-2, dim2=-1)
        tot = (hw.abs() ** 2).sum(-1) + NOISE
        g = sig.conj() / tot
        omega = tot / (tot - sig.abs() ** 2)
        a = omega * g.abs() ** 2
        A = h.conj().transpose(-1, -2) @ (h * a[..., None]) + (NOISE * a.sum(-1))[:, None, None] * eye
        Bm = h.conj().transpose(-1, -2) * (omega * g.conj())[:, None, :]
        w = normalize_power(torch.linalg.solve(A, Bm))
    return w.to(torch.complex64)


def lab_model(cfg):
    sys.path.insert(0, LAB)
    from Pipeline.Transformer_FPGA import StackedTransformer
    dims = {"tiny": (64, 4, 512), "small": (128, 4, 512), "medium": (192, 6, 768)}[cfg]
    m = StackedTransformer(seq_length=K, token_dim=128, embedding_dim=dims[0], num_heads=dims[1],
                           hidden_dim=dims[2], num_layers=4)
    sd = torch.load(f"{LAB}/checkpoints/FPGA_{cfg}_fpga_stecath/phase1_pretrain/fp32_best.pth",
                    map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd.state_dict()
    m.load_state_dict(sd)
    return m.to(DEV).eval()


def lab_input(H):
    r = torch.conj(H)
    r = r / torch.linalg.norm(r, dim=(1, 2), keepdim=True)
    return torch.stack([r.imag, r.real], 1)                                # [n, 2, K, N]


def batched(fn, H, bs=4096):
    return torch.cat([fn(H[i:i + bs].to(DEV)).cpu() for i in range(0, H.shape[0], bs)])


def train(a, P):
    torch.manual_seed(0)
    net = PairPrecoder(d=16, hidden=a.hidden, layers=8, inp=a.inp).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    Htr = draw(P, a.train_samples, seed=1)
    t0, hist = time.time(), []
    for s in range(1, a.steps + 1):
        H = Htr[torch.randint(0, Htr.shape[0], (a.batch,))].to(DEV)
        h, snr = ours_input(H)
        loss = -lab_rate(net(h, snr), H).mean()           # perte du labo : -debit somme (RZF est une
                                                          # mauvaise reference ici : 17 contre 24 pour WMMSE)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 2000 == 0:
            hist.append({"step": s, "obj": -float(loss), "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    os.makedirs(OUT_W, exist_ok=True)
    wname = f"pair_h{a.hidden}" + ("" if a.inp == "raw" else f"_{a.inp}")
    torch.save(net.state_dict(), f"{OUT_W}/{wname}.pt")
    json.dump(hist, open(f"{OUT_W}/{wname}_history.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--train_samples", type=int, default=800000)
    ap.add_argument("--test_samples", type=int, default=20000)
    ap.add_argument("--inp", default="raw", choices=["raw", "log", "norm"])
    a = ap.parse_args()
    P = pool()
    if a.train:
        train(a, P)
        return
    Hte = draw(P, a.test_samples, seed=2026)
    res = {}
    with torch.no_grad():
        res["RZF"] = batched(lambda H: lab_rate(rzf(H), H), Hte)
        res["WMMSE (30 it.)"] = batched(lambda H: lab_rate(wmmse(H), H), Hte, bs=2048)
        # avec un affaiblissement de parcours tres inegal, WMMSE converge lentement : 30 it. ne suffisent pas
        res["WMMSE (300 it.)"] = batched(lambda H: lab_rate(wmmse(H, 300), H), Hte, bs=2048)
        for cfg in ("tiny", "small", "medium"):
            m = lab_model(cfg)
            n = sum(p.numel() for p in m.parameters())

            def f(H, m=m):
                Fp = m(lab_input(H))
                return lab_rate(Fp / torch.linalg.norm(Fp, dim=(1, 2), keepdim=True), H)
            res[f"BitAdapt {cfg} FP32 ({n // 1000} k params)"] = batched(f, Hte)
        for inp in ("raw", "log", "norm"):
            wpath = f"{OUT_W}/pair_h{a.hidden}" + ("" if inp == "raw" else f"_{inp}") + ".pt"
            if not os.path.exists(wpath):
                continue
            net = PairPrecoder(d=16, hidden=a.hidden, layers=8, inp=inp).to(DEV)
            net.load_state_dict(torch.load(wpath, map_location=DEV, weights_only=True))
            net.eval()
            n = sum(p.numel() for p in net.parameters())

            def g(H):
                h, snr = ours_input(H)
                return lab_rate(net(h, snr), H)
            res[f"notre precodeur par paires FP32, entree {inp} ({n // 1000} k params)"] = batched(g, Hte)
    out = {k: {"sum_rate_mean": float(v.mean()), "p5": float(np.percentile(v.numpy(), 5))} for k, v in res.items()}
    for k, v in out.items():
        print(f"{k:50s} debit moyen {v['sum_rate_mean']:6.2f} | 5e centile {v['p5']:6.2f}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump({"test_samples": a.test_samples, "seed": 2026, "noise_pwr": NOISE, "results": out},
              open(f"{RESULTS}/fp32_comparison.json", "w"), indent=1)


if __name__ == "__main__":
    main()
