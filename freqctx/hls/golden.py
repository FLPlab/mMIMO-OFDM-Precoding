"""Reference « golden » du precodeur par paires quantifie, en numpy, operation par operation (meme ordre que
le C++ HLS).

Golden      : arithmetique flottante exacte (doit egaler le modele PyTorch quantifie).
GoldenFixed : arithmetique du materiel.
  - produits quantifies par ART : entiers exacts (entrees et poids sur b+1 bits, accumulateur entier) ;
  - chaque echelle (s_a * s_w, 1 / s_a, ...) devient un entier de `mant` bits fois une puissance de 2 ;
  - valeurs intermediaires (residuel, LayerNorm, GELU, racines, log) arrondies a `frac` bits fractionnaires ;
  - GELU lue dans une table de 257 valeurs (pas 1/16 sur [-8, 8]) avec interpolation lineaire ; racine, inverse de racine et log10 supposees exactes
    a `frac` bits pres (table + Newton en HLS, comme int_math.h du labo) ;
  - la normalisation par utilisateur (puissances sur ~50 dB a St-Catherine) est en virgule flottante par bloc :
    un exposant par utilisateur, mantisse de `mant` bits (compter les zeros de tete, decaler, table).
  Les maxima |x| de chaque grandeur sont releves pour dimensionner les entiers.

    python -m hls.golden          # controle contre PyTorch, puis balayage de `frac`
"""
import numpy as np
from scipy.special import erf

from hls.model import QuantPairPrecoder


def real(z):
    return np.concatenate([z.real, z.imag], -1)


def cplx(x):
    d = x.shape[-1] // 2
    return x[..., :d] + 1j * x[..., d:]


class Golden:
    """Arithmetique flottante exacte. Les operations elementaires sont des methodes, redefinies en virgule fixe."""

    def __init__(self, m: QuantPairPrecoder):
        self.m = m

    # operations elementaires -----------------------------------------------------------------------
    def r(self, x):                       # arrondi des valeurs intermediaires (aucun ici)
        return x

    def rel(self, x):                     # grandeur a grande dynamique (puissance par utilisateur) : mantisse + exposant
        return x

    def quant(self, q, x):                # entiers LSQ
        return q.q(x)

    def dense(self, L, x):
        y = (self.quant(L.a, x) @ L.w_int.T) * (L.a.s * L.w_scale.T)
        return y if L.bias is None else y + L.bias

    def qz(self, q, z):                   # operande complexe quantifie -> (entiers re, im)
        return self.quant(q, z.real) + 1j * self.quant(q, z.imag)

    def sqrt(self, x):
        return np.sqrt(x)

    def rsqrt(self, x):
        return 1 / np.sqrt(x)

    def log10(self, x):
        return np.log10(x)

    def gelu(self, x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def layernorm(self, x, w, b, eps=1e-6):
        mu = self.r(x.mean(-1, keepdims=True))
        xc = x - mu
        inv = self.rsqrt(self.r((xc ** 2).mean(-1, keepdims=True)) + eps)
        return self.r(self.r(xc * inv) * w + b)

    # precodeur ---------------------------------------------------------------------------------------
    def forward(self, h, snr_db):
        """h [N, K, M] complexe (convention freqctx : y = h W s), snr_db [N] -> W [N, M, K]."""
        m = self.m
        N, K, _ = h.shape
        n2 = self.rel((np.abs(h) ** 2).sum(-1))                                 # normalisation par utilisateur
        hp = self.r(h * self.rel(1 / np.sqrt(n2))[..., None])                    # (virgule flottante par bloc :
        pdb = self.r(np.log10(self.rel(n2 / n2.mean(-1, keepdims=True))))        #  un exposant par utilisateur)
        hi = self.qz(m.q_h, hp)                                                  # entiers
        g = (hi @ np.conj(hi).transpose(0, 2, 1)) * m.q_h.s ** 2                 # G = h h^H, [N, K, K]
        ef = np.stack([g.real, g.imag, self.r(self.sqrt(np.abs(g) ** 2))], -1)
        eye = np.broadcast_to(np.eye(K)[None, :, :, None], (N, K, K, 1))
        snr = np.broadcast_to(self.r(snr_db / 10.0)[:, None, None, None], (N, K, K, 1))
        extra = np.stack([np.broadcast_to(pdb[:, :, None], (N, K, K)), np.broadcast_to(pdb[:, None, :], (N, K, K))], -1)
        e = cplx(self.r(self.dense(m.init, np.concatenate([ef, eye, snr, extra], -1))))
        for blk in m.blocks:
            x = np.concatenate([self.layernorm(real(e), blk.ln_w, blk.ln_b), ef], -1)
            a = self.qz(blk.q_a, cplx(self.r(self.dense(blk.a, x))))
            b = self.qz(blk.q_b, cplx(self.r(self.dense(blk.b, x))))
            t = self.r(np.einsum("njlc,nlkc->njkc", a, b) * (blk.q_a.s * blk.q_b.s / K))   # produit triangulaire
            u = self.gelu(self.r(self.dense(blk.f1, np.concatenate([x, real(t)], -1))))
            e = e + cplx(self.r(self.dense(blk.f2, u)))
        ci = self.qz(m.q_c, cplx(self.r(self.dense(m.read, real(e))))[..., 0])  # C, entiers
        w = (np.conj(hi).transpose(0, 2, 1) @ ci) * (m.q_h.s * m.q_c.s)          # W = h^H C
        return self.r(w * self.rsqrt(self.r((np.abs(w) ** 2).sum((1, 2), keepdims=True)))[..., None, None][..., 0, 0])


class GoldenFixed(Golden):
    def __init__(self, m, frac=12, mant=16):
        super().__init__(m)
        self.frac, self.mant, self.peak = frac, mant, 0.0

    def r(self, x):
        self.peak = max(self.peak, float(np.max(np.abs(x))))
        return np.round(x * 2.0 ** self.frac) / 2.0 ** self.frac

    def rel(self, x):
        return self.dyadic(x)

    def dyadic(self, c):
        """c ~ entier de `mant` bits x 2^e (par element)."""
        c = np.asarray(c, dtype=np.float64)
        e = np.floor(np.log2(np.abs(c) + 1e-300)) - (self.mant - 1)
        return np.round(c / 2.0 ** e) * 2.0 ** e

    def quant(self, q, x):
        return np.round(np.clip(x * self.dyadic(1 / q.s), -q.lim, q.lim))

    def dense(self, L, x):
        y = (self.quant(L.a, x) @ L.w_int.T) * self.dyadic(L.a.s * L.w_scale.T)
        return y if L.bias is None else y + self.r(L.bias)

    def sqrt(self, x):
        return self.r(np.sqrt(x))

    def rsqrt(self, x):
        return self.r(1 / np.sqrt(np.maximum(x, 2.0 ** -self.frac)))

    def log10(self, x):
        return self.r(np.log10(x))

    def gelu(self, x):
        """Table de 257 valeurs (pas 1/16 sur [-8, 8]) avec interpolation lineaire entre deux entrees.
        (Une table brute au pas 1/32 coutait 0,18 bps/Hz.)"""
        grid = np.linspace(-8, 8, 257)
        tab = self.r(super().gelu(grid))
        y = np.interp(np.clip(x, -8, 8), grid, tab)
        return self.r(np.where(x > 8, x, np.where(x < -8, 0.0, y)))


def evaluate(n=5000, fracs=(16, 12, 10, 8)):
    import torch
    from torch_port.stecath import draw, lab_rate, ours_input, pool
    qm, net = QuantPairPrecoder.load()
    H = draw(pool(), n, seed=2026)
    h, snr = ours_input(H)
    hn, sn = h.numpy().astype(np.complex128), snr.numpy().astype(np.float64)
    with torch.no_grad():
        r_t = float(lab_rate(net(h, snr), H).mean())
    rate = lambda w: float(lab_rate(torch.tensor(w, dtype=torch.complex64), H).mean())
    print(f"PyTorch quantifie : {r_t:.4f} ; golden flottant : {rate(Golden(qm).forward(hn, sn)):.4f}", flush=True)
    res = {}
    for f in fracs:
        g = GoldenFixed(qm, frac=f)
        res[f] = rate(g.forward(hn, sn))
        print(f"golden virgule fixe, {f} bits fractionnaires : {res[f]:.4f}  (max |x| intermediaire {g.peak:.1f})", flush=True)
    return r_t, res


if __name__ == "__main__":
    evaluate()
