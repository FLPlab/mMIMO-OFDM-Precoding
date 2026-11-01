"""Quantification ART (BitAdapt, mMIMO-ART-FPGA) de notre precodeur par paires, sur St-Catherine.

Reprend les quantificateurs du labo tels quels (quan/quantizer/lsq.py : LSQ, nombre de bits
appris par couche, poids 1-8 bits, activations 2-8 bits ; quan/func.py : QuanLinear) et leur
perte : -debit / debit_ref + lambda * energie / energie_ref, bits arrondis vers le haut avec un
gradient direct (ceil_pass). Energie : modele du labo CORRIGE (torch_port/energy.py), bon nombre
de tokens (K^2 paires). Puis, comme leur phase 3 : bits arrondis et figes, reglage fin sur le
seul debit.

    python -m torch_port.art_stecath --lam 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from torch_port.energy import pair_precoder_fixed, pair_precoder_layers, total_energy
from torch_port.models import PairPrecoder
from torch_port.stecath import DEV, LAB, OUT_W, RESULTS, K, batched, draw, lab_rate, ours_input, pool

sys.path.insert(0, LAB)
from quan.func import QuanLinear                               # noqa: E402  (code du labo)
from quan.quantizer.lsq import LsqQuan                         # noqa: E402

M, D, LAYERS = 64, 16, 8


def ceil_pass(n, gamma=0.01):                                  # Learner.ceil_pass du labo
    n_ceil = n.ceil()
    n_grad = n + gamma * n
    return (n_ceil - n_grad).detach() + n_grad


def quantize(net):
    """Remplace chaque nn.Linear par la QuanLinear du labo ; renvoie {nom: (q_w, q_a)}."""
    qs = {}
    for name, mod in list(net.named_modules()):
        for cname, child in list(mod.named_children()):
            if type(child) is torch.nn.Linear:
                full = f"{name}.{cname}" if name else cname
                qw = LsqQuan(bit=8, min_bit=1, max_bit=8)
                qa = LsqQuan(bit=8, min_bit=2, max_bit=8, all_positive=False)
                setattr(mod, cname, QuanLinear(child, quan_w_fn=qw, quan_a_fn=qa))
                qs[full] = (qw, qa)
    return qs


def calibrate(net, qs, H):
    """Echelle initiale des activations : moyenne |x| * 2 / sqrt(2^b - 1), comme le labo."""
    stats, hooks = {}, []
    for name, mod in net.named_modules():
        if name in qs:
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, n=name: stats.setdefault(n, []).append(i[0].detach().abs().mean().item())))
    with torch.no_grad():
        h, snr = ours_input(H.to(DEV))
        net(h, snr)
    for hk in hooks:
        hk.remove()
    for n, (qw, qa) in qs.items():
        qa.s = torch.nn.Parameter(torch.tensor(np.mean(stats[n]) * 2 / (255 ** 0.5), device=DEV))


def bits(qs, hard=False):
    f = (lambda b: b.detach().ceil()) if hard else ceil_pass
    return {n: (f(qw.bit), f(qa.bit)) for n, (qw, qa) in qs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ft_steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(0)
    P = pool()
    Htr, Hte = draw(P, 800000, seed=1), draw(P, 20000, seed=2026)
    net = PairPrecoder(d=D, hidden=a.hidden, layers=LAYERS).to(DEV)
    net.load_state_dict(torch.load(f"{OUT_W}/pair_h{a.hidden}.pt", map_location=DEV, weights_only=True))
    layers = pair_precoder_layers(net, K)                      # comptes avant le remplacement
    fixed = pair_precoder_fixed(K, M, D, LAYERS)

    def test_rate(model):
        model.eval()
        with torch.no_grad():
            r = batched(lambda H: lab_rate(model(*ours_input(H)), H), Hte)
        model.train()
        return float(r.mean())

    sr_fp32 = test_rate(net)
    e32 = {c: float(total_energy(layers, {n: (32.0, 32.0) for n in layers}, c, fixed)) for c in (True, False)}
    qs = quantize(net)
    net.to(DEV)
    calibrate(net, qs, Htr[:4096])
    bit_params = [p for n, p in net.named_parameters() if n.endswith(".bit")]
    other = [p for n, p in net.named_parameters() if not n.endswith(".bit")]
    opt = torch.optim.Adam([{"params": other, "lr": 1e-4}, {"params": bit_params, "lr": 1e-3}])
    t0, hist = time.time(), []
    for s in range(1, a.steps + a.ft_steps + 1):
        if s == a.steps + 1:                                   # phase 3 : bits arrondis et figes
            for qw, qa in qs.values():
                for q in (qw, qa):
                    q.bit.data = q.bit.data.ceil().clamp(q.min_bit, q.max_bit)
                    q.bit.requires_grad_(False)
            opt = torch.optim.Adam(other, lr=5e-5)
        H = Htr[torch.randint(0, Htr.shape[0], (a.batch,))].to(DEV)
        sr = lab_rate(net(*ours_input(H)), H).mean()
        loss = -sr / sr_fp32
        if s <= a.steps:
            loss = loss + a.lam * total_energy(layers, bits(qs), True, fixed) / e32[True]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 2500 == 0:
            e = float(total_energy(layers, bits(qs, True), True, fixed))
            hist.append({"step": s, "sr_train": float(sr), "energy_corr_pJ": e, "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    b = {n: (float(qw.bit.ceil()), float(qa.bit.ceil())) for n, (qw, qa) in qs.items()}
    out = {"lambda": a.lam, "hidden": a.hidden, "sr_fp32": sr_fp32, "sr_quant": test_rate(net),
           "energy_pJ": {"fp32_corrige": e32[True], "fp32_labo": e32[False],
                         "quant_corrige": float(total_energy(layers, b, True, fixed)),
                         "quant_labo": float(total_energy(layers, b, False, fixed))},
           "bits": b, "mean_bits_w": float(np.mean([v[0] for v in b.values()])),
           "mean_bits_a": float(np.mean([v[1] for v in b.values()])), "history": hist}
    print(json.dumps({k: v for k, v in out.items() if k not in ("bits", "history")}, indent=1), flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(f"{RESULTS}/art_pair_h{a.hidden}_lam{a.lam:g}.json", "w"), indent=1)
    torch.save(net.state_dict(), f"{OUT_W}/art_pair_h{a.hidden}_lam{a.lam:g}.pt")


if __name__ == "__main__":
    main()
