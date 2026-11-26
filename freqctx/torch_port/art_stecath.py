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

import types

import torch.nn.functional as Fn

from hls.cost_model.fpga_cost import CASEA_BW_MAX_VALID, layer_cost_torch
from torch_port.energy import pair_precoder_fixed, pair_precoder_layers, pair_precoder_products, total_energy
from torch_port.models import PairPrecoder, _cplx, _real, normalize_power
from torch_port.stecath import DEV, LAB, OUT_W, RESULTS, SITES, K, batched, draw, lab_rate, ours_input, pool, site_tag

sys.path.insert(0, LAB)
from quan.func import QuanLinear                               # noqa: E402  (code du labo)
from quan.quantizer.lsq import LsqQuan                         # noqa: E402

M, D, LAYERS = 64, 16, 8


def ceil_pass(n, gamma=0.01):                                  # Learner.ceil_pass du labo
    n_ceil = n.ceil()
    n_grad = n + gamma * n
    return (n_ceil - n_grad).detach() + n_grad


def quantize(net, wmax=8):
    """Remplace chaque nn.Linear par la QuanLinear du labo ; renvoie {nom: (q_w, q_a)}.
    wmax : bits de poids maximaux (4 avec --cost fpga : regime sans DSP ou le modele de cout est valide)."""
    qs = {}
    for name, mod in list(net.named_modules()):
        for cname, child in list(mod.named_children()):
            if type(child) is torch.nn.Linear:
                full = f"{name}.{cname}" if name else cname
                qw = LsqQuan(bit=wmax, min_bit=1, max_bit=wmax)
                qa = LsqQuan(bit=8, min_bit=2, max_bit=8, all_positive=False)
                setattr(mod, cname, QuanLinear(child, quan_w_fn=qw, quan_a_fn=qa))
                qs[full] = (qw, qa)
    return qs


def _qz(q, z):
    """Quantifie un tenseur complexe (parties reelle et imaginaire, meme echelle)."""
    return torch.complex(q(z.real), q(z.imag))


def quantize_products(net):
    """--qprod : quantificateurs LSQ (bits appris, 2-8) sur les operandes des produits sans poids :
    h (pour G = h h^H et W = h^H C), C, et a(x), b(x) de chaque couche triangulaire.
    Renvoie {nom: (q_operande_1, q_operande_2)} pour l'energie."""
    mk = lambda: LsqQuan(bit=8, min_bit=2, max_bit=8, all_positive=False).to(DEV)
    net.q_h, net.q_c = mk(), mk()
    prods = {"G": (net.q_h, net.q_h), "W": (net.q_c, net.q_h)}
    for i, blk in enumerate(net.blocks):
        blk.q_a, blk.q_b = mk(), mk()
        prods[f"tri{i}"] = (blk.q_a, blk.q_b)

        def fwd(self, e, ef):
            x = torch.cat([self.norm(_real(e)), ef], -1)
            a = _qz(self.q_a, _cplx(self.a(x)))
            b = _qz(self.q_b, _cplx(self.b(x)))
            m = torch.einsum("njlc,nlkc->njkc", a, b) / e.shape[1]
            return e + _cplx(self.f2(Fn.gelu(self.f1(torch.cat([x, _real(m)], -1)))))
        blk.forward = types.MethodType(fwd, blk)

    def coeffs(self, h, snr_db):
        hp, extra = self.prepare(h)
        hq = _qz(self.q_h, hp)
        return self.coeffs_from(self.gram(hq), snr_db, extra)

    def forward(self, h, snr_db):
        hp, extra = self.prepare(h)
        hq = _qz(self.q_h, hp)
        c = _qz(self.q_c, self.coeffs_from(self.gram(hq), snr_db, extra))
        return normalize_power(hq.conj().transpose(-1, -2) @ c)
    net.coeffs = types.MethodType(coeffs, net)
    net.forward = types.MethodType(forward, net)
    return prods


def calibrate_products(net, prods, H):
    """Echelle initiale des quantificateurs de produits : h et a(x), b(x) d'abord, puis C mesure
    avec ces echelles (moyenne |x| * 2 / sqrt(255), comme le labo)."""
    sc = lambda v: torch.nn.Parameter(torch.tensor(v * 2 / 255 ** 0.5, device=DEV))
    with torch.no_grad():
        h, snr = ours_input(H.to(DEV))
        net.q_h.s = sc(net.prepare(h)[0].abs().mean().item())
        for blk in net.blocks:
            blk.q_a.s, blk.q_b.s = sc(0.5), sc(0.5)
        net.q_c.s = sc(1.0)
        net.q_c.s = sc(net.coeffs(h, snr).abs().mean().item())


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


def _clamp_bw(bw, bw_cap=CASEA_BW_MAX_VALID):
    """Borne b_w au domaine valide du modele FPGA cas A (sans DSP), sans toucher au modele
    lui-meme ni aux bits reellement appris/rapportes (qui restent 1-8)."""
    return bw.clamp(max=bw_cap) if torch.is_tensor(bw) else min(bw, bw_cap)


def fpga_cost(dense_layers, bits_dict, K):
    """Cout FPGA (LUT-equivalent, cas A : poids constants, design deroule) des couches denses,
    pour UN token (dense_layers vient de pair_precoder_layers, MAC/activations comptes sur les
    T=K*K tokens -> on divise par T pour retrouver le nombre de multiplieurs/registres materiels
    d'une seule passe). bits_dict : nom -> (b_w, b_a), meme convention que total_energy."""
    T = K * K
    total = 0.0
    for n, (mac, _w, act) in dense_layers.items():
        bw, ba = bits_dict[n]
        total = total + layer_cost_torch(mac / T, act / T, _clamp_bw(bw), ba)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=LAYERS, help="nombre de couches du precodeur (front de Pareto)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ft_steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--qprod", action="store_true", help="quantifie aussi G, le produit triangulaire et W = h^H C")
    ap.add_argument("--inp", default="raw", choices=["raw", "log", "norm"], help="entree du precodeur (phase 10)")
    ap.add_argument("--cost", default="energy", choices=["energy", "fpga"],
                     help="objectif de cout (phase 2) : energie Moons corrigee (defaut, inchange) ou "
                          "cout FPGA LUT-equivalent (cas A, poids constants, calibre HLS)")
    ap.add_argument("--site", default="stecath", choices=SITES,
                     help="site du labo (comportement et noms de fichiers inchanges pour stecath, le defaut)")
    a = ap.parse_args()
    torch.manual_seed(0)
    P = pool(a.site)
    Htr, Hte = draw(P, 800000, seed=1), draw(P, 20000, seed=2026)
    net = PairPrecoder(d=D, hidden=a.hidden, layers=a.layers, inp=a.inp).to(DEV)
    lt = "" if a.layers == LAYERS else f"_L{a.layers}"
    wname = f"pair_h{a.hidden}" + lt + ("" if a.inp == "raw" else f"_{a.inp}") + site_tag(a.site)
    net.load_state_dict(torch.load(f"{OUT_W}/{wname}.pt", map_location=DEV, weights_only=True))
    layers = pair_precoder_layers(net, K)                      # comptes avant le remplacement
    fixed = pair_precoder_fixed(K, M, D, a.layers)
    dense_layers = dict(layers)                                # couches denses seules (hors qprod) ; seules
                                                                # visees par le cout FPGA (poids constants)
    fpga_ref = float(fpga_cost(dense_layers, {n: (float(CASEA_BW_MAX_VALID), 8.0) for n in dense_layers}, K))

    def test_rate(model):
        model.eval()
        with torch.no_grad():
            r = batched(lambda H: lab_rate(model(*ours_input(H)), H), Hte)
        model.train()
        return float(r.mean())

    sr_fp32 = test_rate(net)
    e32 = {c: float(total_energy(layers, {n: (32.0, 32.0) for n in layers}, c, fixed)) for c in (True, False)}
    qs = quantize(net, wmax=CASEA_BW_MAX_VALID if a.cost == "fpga" else 8)
    net.to(DEV)
    calibrate(net, qs, Htr[:4096])
    prods = {}
    if a.qprod:                                                # produits sans poids, bits appris eux aussi
        products = pair_precoder_products(K, M, D, a.layers)
        prods = quantize_products(net)
        calibrate_products(net, prods, Htr[:4096])
        layers = {**layers, **products}
        fixed = None
        e32 = {c: float(total_energy(layers, {n: (32.0, 32.0) for n in layers}, c, None)) for c in (True, False)}

    def all_bits(hard=False):
        f = (lambda b: b.detach().ceil()) if hard else ceil_pass
        out = {n: (f(qw.bit), f(qa.bit)) for n, (qw, qa) in qs.items()}
        out.update({n: (f(q1.bit), f(q2.bit)) for n, (q1, q2) in prods.items()})
        return out
    bit_params = [p for n, p in net.named_parameters() if n.endswith(".bit")]
    other = [p for n, p in net.named_parameters() if not n.endswith(".bit")]
    opt = torch.optim.Adam([{"params": other, "lr": 1e-4}, {"params": bit_params, "lr": 1e-3}])
    t0, hist = time.time(), []
    for s in range(1, a.steps + a.ft_steps + 1):
        if s == a.steps + 1:                                   # phase 3 : bits arrondis et figes
            for qw, qa in list(qs.values()) + list(prods.values()):
                for q in (qw, qa):
                    q.bit.data = q.bit.data.ceil().clamp(q.min_bit, q.max_bit)
                    q.bit.requires_grad_(False)
            opt = torch.optim.Adam(other, lr=5e-5)
        H = Htr[torch.randint(0, Htr.shape[0], (a.batch,))].to(DEV)
        sr = lab_rate(net(*ours_input(H)), H).mean()
        loss = -sr / sr_fp32
        if s <= a.steps:
            ab = all_bits()
            if a.cost == "fpga":
                cost_term = fpga_cost(dense_layers, {n: ab[n] for n in dense_layers}, K) / fpga_ref
            else:
                cost_term = total_energy(layers, ab, True, fixed) / e32[True]
            loss = loss + a.lam * cost_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 2500 == 0:
            hb = all_bits(True)
            e = float(total_energy(layers, hb, True, fixed))
            fc = float(fpga_cost(dense_layers, {n: hb[n] for n in dense_layers}, K))
            hist.append({"step": s, "sr_train": float(sr), "energy_corr_pJ": e, "fpga_cost_lut": fc,
                         "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)
    b = {n: (float(v[0]), float(v[1])) for n, v in all_bits(True).items()}
    fpga_final = float(fpga_cost(dense_layers, {n: b[n] for n in dense_layers}, K))
    out = {"lambda": a.lam, "hidden": a.hidden, "qprod": a.qprod, "inp": a.inp, "cost": a.cost, "site": a.site,
           "sr_fp32": sr_fp32, "sr_quant": test_rate(net),
           "energy_pJ": {"fp32_corrige": e32[True], "fp32_labo": e32[False],
                         "quant_corrige": float(total_energy(layers, b, True, fixed)),
                         "quant_labo": float(total_energy(layers, b, False, fixed))},
           "fpga_cost_lut": {"ref_bw4_ba8": fpga_ref, "quant": fpga_final},
           "bits": b, "mean_bits_w": float(np.mean([v[0] for v in b.values()])),
           "mean_bits_a": float(np.mean([v[1] for v in b.values()])), "history": hist}
    print(json.dumps({k: v for k, v in out.items() if k not in ("bits", "history")}, indent=1), flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    cost_tag = "" if a.cost == "energy" else f"_cost{a.cost}"
    tag = (f"art_pair_h{a.hidden}{lt}_lam{a.lam:g}{cost_tag}" + ("_qprod" if a.qprod else "")
           + ("" if a.inp == "raw" else f"_{a.inp}") + site_tag(a.site))
    json.dump(out, open(f"{RESULTS}/{tag}.json", "w"), indent=1)
    torch.save(net.state_dict(), f"{OUT_W}/{tag}.pt")


if __name__ == "__main__":
    main()
