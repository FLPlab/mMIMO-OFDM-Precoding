"""RZF et WMMSE (forme Gram, dimension K) en VIRGULE FIXE emulee : combien de bits demandent-ils vraiment ?

Meme protocole que energy_classical.py (pool du site, 20 000 echantillons, graine 2026, M=64, K=4, CSI
parfait, debit lab_rate sur le canal brut). Chaque operation est suivie d'un arrondi au plus proche +
saturation sur b bits (signe compris, parties reelle et imaginaire separees). Choix d'implementation
(ceux d'un concepteur FPGA soigneux, donc plutot GENEREUX pour RZF/WMMSE) :

  * virgule fixe a exposant de bloc (block floating point) : mantisse b bits, exposant partage choisi
    apres calcul (detection de zeros de tete sur le bloc) ; accumulateurs larges exacts (arrondi a
    l'ecriture seulement) ;
  * variante 'peruser' (retenue) : normalisation par utilisateur de H (exposant puissance de 2 par
    ligne, h'_k = h_k / d_k), tout le calcul en domaine normalise (le bruit devient nu_k = N0 / d_k^2) ;
    exposants par COLONNE pour les matrices K x K et W, par element pour les scalaires par utilisateur
    (tot, g, omega, a, mu, beta : K valeurs, petit flottant a mantisse b bits) ;
    variante 'naive' : un exposant par echantillon pour H et un par matrice ;
  * inversion : equilibrage de Jacobi (B'' = U^H B U, diagonale unite, rsqrt par table 256 entrees +
    Newton), LDL^H sans racine avec reciproque par table 256 entrees + Newton (1 tour si b <= 16,
    2 sinon), inverse triangulaire unitaire par substitution, X = L^-H D^-1 L^-1 ; aucune division ;
  * WMMSE reecrit pour eviter les divisions par de petits nombres :
      C' = (G' + diag(mu/a'))^-1 diag(1/g') = diag(u) X diag(rho),
      beta_k = |g'_k|^2 G'_kk + mu / omega_k, rho = rsqrt(beta), u = conj(g') rho,
      X = (U^H (G' + diag(mu/a')) U)^-1 ; interference + bruit calculee directement (pas tot - |sig|^2) ;
  * sortie F = W rsqrt(||W||^2) sur b bits a exposant commun (|F| <= 1, format fixe du CNA).
Le flottant (b = None) redonne stecath.wmmse_gram / rzf a 1e-6 pres (controle imprime).

Diagnostic 'ou ca casse' : quantification d'un seul groupe d'etapes (in, gram, inv, scal, out).

    CUDA_VISIBLE_DEVICES="" python -u -m torch_port.fixedpoint_classical
"""
import argparse
import json
import os
import time

import torch

from torch_port.energy_classical import counts_rzf, counts_wmmse_gram, energy_uJ
from torch_port.stecath import K, NOISE, RESULTS, draw, lab_rate, pool, rzf, wmmse_gram

C128, F64 = torch.complex128, torch.float64
STAGES = ("in", "gram", "inv", "scal", "out")
BITS = list(range(8, 25))
ITERS = [1, 5, 30]
TBL = 8                                                   # bits d'adresse des tables (256 entrees)


def _blockq(x, b, dims):
    """Arrondi + saturation sur b bits, exposant partage sur les dimensions dims (() = par element)."""
    a = torch.maximum(x.real.abs(), x.imag.abs()) if x.is_complex() else x.abs()
    if dims:
        a = a.amax(dim=dims, keepdim=True)
    s = torch.exp2(torch.ceil(torch.log2(a.clamp_min(1e-300))) - (b - 1))
    lo, hi = -2 ** (b - 1), 2 ** (b - 1) - 1

    def f(v):
        return torch.round(v / s).clamp(lo, hi) * s
    return torch.complex(f(x.real), f(x.imag)) if x.is_complex() else f(x)


def _qf(x, b, frac):
    """Format fixe statique Q(b-frac).frac (interne aux Newton)."""
    s = 2.0 ** frac
    return torch.round(x * s).clamp(-2 ** (b - 1), 2 ** (b - 1) - 1) / s


class FX:
    def __init__(self, b=None, stages=STAGES, peruser=True):
        self.b, self.stages, self.peruser = b, set(stages), peruser
        self.md = (1,) if peruser else (1, 2)              # matrices K x K : par colonne / par matrice
        self.newton = 1 if (b or 0) <= 16 else 2

    def on(self, st):
        return self.b is not None and st in self.stages

    def q(self, x, st, dims=()):
        return _blockq(x, self.b, dims) if self.on(st) else x

    def recip(self, x, st):
        if not self.on(st):
            return 1.0 / x
        b = self.b
        x = x.clamp_min(1e-300)
        e = torch.floor(torch.log2(x))
        m = x / torch.exp2(e)                                          # [1, 2)
        idx = torch.floor((m - 1) * 2 ** TBL).clamp(0, 2 ** TBL - 1)
        y = _qf(1.0 / (1 + (idx + 0.5) / 2 ** TBL), b, b - 2)
        for _ in range(self.newton):
            y = _qf(y * _qf(2 - _qf(m * y, b, b - 2), b, b - 2), b, b - 2)
        return y / torch.exp2(e)

    def rsqrt(self, x, st):
        if not self.on(st):
            return x.rsqrt()
        b = self.b
        x = x.clamp_min(1e-300)
        e = torch.floor(torch.log2(x) / 2)
        m = x / torch.exp2(2 * e)                                      # [1, 4)
        idx = torch.floor((m - 1) / 3 * 2 ** TBL).clamp(0, 2 ** TBL - 1)
        y = _qf((1 + 3 * (idx + 0.5) / 2 ** TBL) ** -0.5, b, b - 3)
        for _ in range(self.newton):
            y = _qf(y * _qf(1.5 - _qf(0.5 * m * _qf(y * y, b, b - 3), b, b - 3), b, b - 3), b, b - 3)
        return y / torch.exp2(e)


def ldl_inv(B, fx):
    """Inverse d'une matrice hermitienne definie positive (diagonale ~1) par LDL^H en virgule fixe."""
    n, Kd, _ = B.shape
    L = torch.zeros_like(B)
    d = torch.zeros(n, Kd, dtype=F64)
    invd = torch.zeros(n, Kd, dtype=F64)
    floor = 2.0 ** -(fx.b or 60)
    for j in range(Kd):
        dj = fx.q(B[:, j, j].real - (L[:, j, :j].abs() ** 2 * d[:, :j]).sum(-1), "inv").clamp_min(floor)
        d[:, j], invd[:, j] = dj, fx.recip(dj, "inv")
        if j < Kd - 1:
            num = B[:, j + 1:, j] - (L[:, j + 1:, :j] * (L[:, j, :j].conj() * d[:, :j])[:, None, :]).sum(-1)
            num = fx.q(num, "inv", (1,))
            L[:, j + 1:, j] = fx.q(num * invd[:, j, None], "inv", (1,))          # colonne j de L, un bloc
    Li = torch.eye(Kd, dtype=C128).repeat(n, 1, 1)
    for i in range(1, Kd):
        row = Li[:, i, :] - (L[:, i, :i, None] * Li[:, :i, :]).sum(1)
        Li[:, i, :] = fx.q(row, "inv", (1,))                                     # ligne i de L^-1, un bloc
    X = Li.conj().transpose(-1, -2) @ (invd[:, :, None] * Li)
    return fx.q(X, "inv", fx.md)


def prep(H, fx):
    h = torch.conj(H).to(C128)
    a = torch.maximum(h.real.abs(), h.imag.abs()).amax(dim=2 if fx.peruser else (1, 2), keepdim=True)
    d = torch.exp2(torch.ceil(torch.log2(a)))                                    # [n, K|1, 1], puissance de 2
    hp = fx.q(h / d, "in", (2,) if fx.peruser else (1, 2))
    nu = fx.q((NOISE / d[..., 0] ** 2).expand(-1, h.shape[1]).contiguous(), "scal")
    G = fx.q(hp @ hp.conj().transpose(-1, -2), "gram", fx.md)
    return hp, d[..., 0], nu, G


def diag(x):
    return torch.diagonal(x, dim1=-2, dim2=-1)


def rzf_C(hp, d, nu, G, fx):
    r = fx.q(K * nu, "scal")
    rho = fx.q(fx.rsqrt(fx.q(diag(G).real + r, "scal"), "scal"), "scal")
    B = G + torch.diag_embed(r).to(C128)
    Bpp = fx.q(rho[:, :, None] * B * rho[:, None, :], "inv", fx.md)
    X = ldl_inv(Bpp, fx)
    return fx.q(rho[:, :, None] * X * (rho / d)[:, None, :], "out", fx.md)      # C' = D C


def normalize_C(C, G, fx, C_prev=None):
    """tr(C^H G C) en forme Gram. Garde (comparateur + multiplexeur) : si la puissance calculee est <= 0
    (G quantifiee non definie positive, inversion ratee), on garde l'iteree precedente (|p| a l'init)."""
    GC = fx.q(G @ C, "gram", fx.md)
    p = fx.q((C.conj() * GC).real.sum((1, 2)), "scal")
    bad = ~(p > 0)
    Cn = fx.q(C * fx.rsqrt(p.abs().nan_to_num(1.0), "scal")[:, None, None], "out", fx.md)
    if C_prev is not None:
        Cn = torch.where(bad[:, None, None], C_prev, Cn)
    return Cn


def wmmse_iter(C, G, nu, fx):
    T = fx.q(G @ C, "gram", fx.md)                                              # T' = H' W
    Tkk = diag(T)
    P2 = T.abs() ** 2
    s2 = fx.q(diag(P2), "scal")
    ipn = fx.q(P2.sum(-1) - diag(P2) + nu, "scal")                              # somme hors diagonale, exacte
    tot = fx.q(ipn + s2, "scal")
    rt = fx.q(fx.recip(tot, "scal"), "scal")
    g = fx.q(Tkk.conj() * rt, "scal")
    winv = fx.q(ipn * rt, "scal")                                               # 1 / omega
    omega = fx.q(tot * fx.q(fx.recip(ipn, "scal"), "scal"), "scal")
    g2 = fx.q(g.abs() ** 2, "scal")
    a = fx.q(omega * g2, "scal")
    mu = fx.q((a * nu).sum(-1, keepdim=True), "scal")
    beta = fx.q(g2 * diag(G).real + fx.q(mu * winv, "scal"), "scal")
    rho = fx.q(fx.rsqrt(beta, "scal"), "scal")
    u = fx.q(g.conj() * rho, "scal")
    Bpp = u.conj()[:, :, None] * G * u[:, None, :]
    Bpp = Bpp - torch.diag_embed(diag(Bpp)) + torch.diag_embed(rho ** 2 * beta).to(C128)
    X = ldl_inv(fx.q(Bpp, "inv", fx.md), fx)
    return normalize_C(fx.q(u[:, :, None] * X * rho[:, None, :], "out", fx.md), G, fx, C_prev=C)


def output(hp, C, fx):
    W = fx.q(hp.conj().transpose(-1, -2) @ C, "out", (1,) if fx.peruser else (1, 2))
    p = fx.q((W.abs() ** 2).sum((1, 2)), "scal")
    return fx.q(W * fx.rsqrt(p, "scal")[:, None, None], "out", (1, 2))         # exposant commun (CNA)


def run(H, fx, iters_list):
    """Debits moyens {RZF, WMMSE n it.} pour un lot H brut [n, K, M]."""
    hp, d, nu, G = prep(H, fx)
    Hr = H.to(C128)
    C = rzf_C(hp, d, nu, G, fx)
    out = {"RZF": lab_rate(output(hp, C, fx), Hr)}
    C = normalize_C(C, G, fx)
    for it in range(1, max(iters_list) + 1):
        C = wmmse_iter(C, G, nu, fx)
        if it in iters_list:
            out[f"WMMSE {it} it."] = lab_rate(output(hp, C, fx), Hr)
    return out


def evaluate(Hte, fx, iters_list=ITERS, bs=5000):
    """Debit moyen ; un echantillon dont le precodeur n'est pas fini compte pour un debit nul (nombre dans 'fail')."""
    acc = {}
    for i in range(0, Hte.shape[0], bs):
        for k, v in run(Hte[i:i + bs], fx, iters_list).items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        v = torch.cat(v)
        bad = ~torch.isfinite(v)
        out[k] = float(torch.where(bad, 0.0, v).mean())
        if bad.any():
            out.setdefault("fail", {})[k] = int(bad.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sites", nargs="+", default=["stecath", "ericsson"])
    a = ap.parse_args()
    torch.set_num_threads(8)
    path = os.path.join(RESULTS, "fixedpoint_classical.json")
    out = {"meta": {"note": __doc__.split("\n")[0], "n": a.n, "seed": 2026, "bits": BITS, "table_bits": TBL,
                    "newton": "1 tour si b <= 16, 2 sinon", "doc": __doc__}, "sites": {}}
    for site in a.sites:
        t0 = time.time()
        Hte = draw(pool(site), a.n, seed=2026)
        S = out["sites"][site] = {}
        with torch.no_grad():
            ref = {"RZF": float(lab_rate(rzf(Hte).to(C128), Hte.to(C128)).mean())}
            for n in ITERS:
                ref[f"WMMSE {n} it."] = float(lab_rate(wmmse_gram(Hte, n).to(C128), Hte.to(C128)).mean())
            S["float_stecath_py"] = ref
            S["float"] = evaluate(Hte, FX(None))
            print(site, "controle flottant (stecath.py / ce script) :", ref, S["float"], flush=True)
            for var in ("peruser", "naive"):
                S[var] = {}
                for b in BITS:
                    S[var][b] = evaluate(Hte, FX(b, peruser=(var == "peruser")))
                    print(site, var, b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S[var][b].items()},
                          f"{time.time() - t0:.0f}s", flush=True)
            S["stages"] = {}
            for b in (8, 10, 12, 14, 16):
                S["stages"][b] = {}
                for st in STAGES:
                    S["stages"][b][st] = evaluate(Hte, FX(b, stages=(st,)))
                    print(site, "etape seule", b, st, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S["stages"][b][st].items()},
                          flush=True)
        # bits minimaux (variante retenue : peruser ; on verifie aussi que toutes les largeurs au-dessus tiennent)
        S["min_bits"], S["energy"] = {}, {}
        for meth, fl in S["float"].items():
            res = {}
            for var in ("peruser", "naive"):
                for crit, ok in (("1pct", lambda r: r >= 0.99 * fl), ("0.1bps", lambda r: r >= fl - 0.1)):
                    good = [b for b in BITS if all(ok(S[var][bb][meth]) for bb in BITS if bb >= b)]
                    res[f"{var}_{crit}"] = min(good) if good else None
            S["min_bits"][meth] = res
            if meth == "RZF":
                mac, act = counts_rzf()
            else:
                mac, act = counts_wmmse_gram(int(meth.split()[1]))
            e = {"mac": mac, "energy_uJ_16b": energy_uJ(mac, act, 16), "energy_uJ_32b": energy_uJ(mac, act, 32)}
            for crit in ("1pct", "0.1bps"):
                bm = res[f"peruser_{crit}"]
                if bm is not None:
                    e[f"bits_{crit}"] = bm
                    e[f"energy_uJ_{crit}"] = energy_uJ(mac, act, bm)
                    e[f"sr_{crit}"] = S["peruser"][bm][meth]
            S["energy"][meth] = e
            print(site, meth, res, e, flush=True)
        json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
