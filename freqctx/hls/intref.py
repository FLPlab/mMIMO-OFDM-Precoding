"""Reference ENTIERE du precodeur par paires quantifie : exactement les calculs du C++ HLS (hls/codegen.py lit
les memes constantes). Toutes les valeurs intermediaires sont des entiers en unites de 2^-F (F = 12).

Conventions (identiques en C++) :
  rnd_shift(x, s) = (x + 2^(s-1)) >> s          arrondi au plus proche, vers le haut (s > 0)
  echelle c ~ m * 2^-e, m entier signe sur 16 bits (Dyadic)
  quantification LSQ : q = clamp(rnd_shift(x * m_inv, e_inv + F), -(2^b - 1), 2^b - 1)
Version 2 : tout en entiers. Entree en virgule flottante par bloc (mantisses 16 bits, un exposant par
utilisateur) ; |G| par racine entiere exacte (isqrt) ; log2 par table interpolee (257 valeurs).
Version 3 : plus aucune division. Normalisation par utilisateur, LayerNorm et normalisation de W par une inverse
de racine entiere (irsqrt) : table de 193 valeurs interpolee puis un tour de Newton, multiplications seulement.

    python -m hls.intref [nom_du_modele]   # debit contre la reference golden en virgule fixe
"""
from dataclasses import dataclass

import numpy as np

from hls.model import QuantPairPrecoder

F = 12                                         # pair (LayerNorm : 2^(3F/2) / sqrt(var))
ONE = 1 << F


def rnd_shift(x, s):
    x = np.asarray(x, dtype=np.int64)
    if np.ndim(s) == 0:
        return (x + (1 << (s - 1))) >> s if s > 0 else x << (-s)
    s = np.asarray(s, dtype=np.int64)                                          # decalage variable par element
    return np.where(s > 0, (x + (np.int64(1) << np.maximum(s - 1, 0))) >> np.maximum(s, 0), x << np.maximum(-s, 0))


def to_fix(x):
    return np.floor(np.asarray(x, dtype=np.float64) * ONE + 0.5).astype(np.int64)


def isqrt(x):
    """Racine carree entiere exacte (partie entiere), elementwise."""
    x = np.asarray(x, dtype=np.int64)
    r = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    r = np.where((r + 1) * (r + 1) <= x, r + 1, r)
    return np.where(r * r > x, r - 1, r)


def msb_of(x):
    """Position du bit de tete (x >= 1), elementwise."""
    x = np.asarray(x, dtype=np.int64)
    msb = np.floor(np.log2(np.maximum(x, 1).astype(np.float64))).astype(np.int64)
    msb = np.where((np.int64(1) << (msb + 1)) <= x, msb + 1, msb)
    return np.where((np.int64(1) << msb) > x, msb - 1, msb)


RCP = 30                                       # inverse de racine de normalisation : ~ 2^(F + RCP) / sqrt(x)
RS_TAB = np.floor(2.0 ** 30 / np.sqrt((np.arange(193) + 64) / 256.0) + 0.5).astype(np.int64)   # 2^30 / sqrt(u), u in [1/4, 1]


def irsqrt(x, S):
    """~ 2^S / sqrt(x) pour x entier (x < 1 traite comme 1), sans division : t = x * 2^-e2 dans [2^30, 2^32)
    (e2 pair), table RS_TAB interpolee (u = t / 2^32, 8 bits d'index, 24 de fraction) puis un tour de Newton
    y <- y (3 - u y^2) / 2 en Q30 ; resultat = y * 2^(S - 46 - e2/2). Erreur relative ~ 1e-9."""
    x = np.maximum(np.asarray(x, dtype=np.int64), 1)
    msb = msb_of(x)
    e2 = msb - 30 - (msb & 1)
    t = np.where(e2 >= 0, x >> np.maximum(e2, 0), x << np.maximum(-e2, 0))
    i = (t >> 24) - 64
    fr = t - ((i + 64) << 24)
    y = RS_TAB[i] + rnd_shift((RS_TAB[i + 1] - RS_TAB[i]) * fr, 24)
    yy = (y * y) >> 30
    p = ((t >> 2) * yy) >> 30
    y = (y * ((np.int64(3) << 30) - p)) >> 31
    return rnd_shift(y, 46 + (e2 >> 1) - S)


LOG2_TAB = np.floor(np.log2(1 + np.arange(257) / 256.0) * ONE + 0.5).astype(np.int64)
LOG10_2 = int(np.floor(np.log10(2.0) * ONE + 0.5))


def ilog2_fix(x):
    """log2(x) en unites 2^-F pour x entier > 0 : position du bit de tete + table interpolee (257 valeurs)."""
    x = np.asarray(x, dtype=np.int64)
    msb = msb_of(x)
    t = np.where(msb >= 30, x >> np.maximum(msb - 30, 0), x << np.maximum(30 - msb, 0))   # [2^30, 2^31)
    u = t - (np.int64(1) << 30)
    i = u >> 22
    fr = u - (i << 22)
    return msb * ONE + LOG2_TAB[i] + rnd_shift((LOG2_TAB[i + 1] - LOG2_TAB[i]) * fr, 22)


@dataclass
class Dyadic:
    """c ~ m * 2^-e, m entier signe sur `mant` bits."""
    m: np.ndarray
    e: np.ndarray

    @classmethod
    def of(cls, c, mant=16):
        c = np.asarray(c, dtype=np.float64)
        e = (mant - 1) - np.floor(np.log2(np.abs(c) + 1e-300)).astype(np.int64)
        m = np.floor(c * 2.0 ** e + 0.5).astype(np.int64)
        return cls(m, e)


class IntLayer:
    """Couche dense : constantes entieres (poids, multiplicateurs, biais, quantification de l'entree)."""

    def __init__(self, L):
        self.name, self.w = L.name, L.w_int.astype(np.int64)                  # [out, in]
        self.lim = L.a.lim
        self.qin = Dyadic.of(1.0 / L.a.s)                                      # x -> entiers d'entree
        self.out = Dyadic.of((L.a.s * L.w_scale).reshape(-1))                  # acc -> valeur (par canal)
        self.bias = None if L.bias is None else to_fix(L.bias)

    def quant(self, x):
        return np.clip(rnd_shift(x * self.qin.m, int(self.qin.e) + F), -self.lim, self.lim)

    def __call__(self, x):
        acc = self.quant(x) @ self.w.T                                         # entiers exacts
        y = np.stack(
            [rnd_shift(acc[..., c] * self.out.m[c], int(self.out.e[c]) - F) for c in range(self.w.shape[0])], -1)
        return y if self.bias is None else y + self.bias


class IntQuant:
    """Quantificateur d'operande de produit (echelle scalaire)."""

    def __init__(self, q):
        self.lim, self.qin, self.s = q.lim, Dyadic.of(1.0 / q.s), q.s

    def __call__(self, x):
        return np.clip(rnd_shift(x * self.qin.m, int(self.qin.e) + F), -self.lim, self.lim)


class IntPairPrecoder:
    GELU_LO, GELU_STEP_LOG2 = -8, 4                                             # table sur [-8, 8], pas 1/16

    def __init__(self, m: QuantPairPrecoder):
        self.K, self.M, self.d = m.K, m.M, m.d
        self.peak = {}                                                         # max |x| releve par grandeur (largeurs HLS)
        self.init, self.read = IntLayer(m.init), IntLayer(m.read)
        self.q_h, self.q_c = IntQuant(m.q_h), IntQuant(m.q_c)
        self.g_out = Dyadic.of(m.q_h.s ** 2)
        self.blocks = []
        for b in m.blocks:
            self.blocks.append(dict(ln_w=to_fix(b.ln_w), ln_b=to_fix(b.ln_b), a=IntLayer(b.a), b=IntLayer(b.b),
                                    f1=IntLayer(b.f1), f2=IntLayer(b.f2), q_a=IntQuant(b.q_a), q_b=IntQuant(b.q_b),
                                    t_out=Dyadic.of(b.q_a.s * b.q_b.s / m.K)))
        grid = np.arange(257) / 16.0 - 8
        from scipy.special import erf
        self.gelu_tab = to_fix(0.5 * grid * (1 + erf(grid / np.sqrt(2))))

    def layers(self):
        """Toutes les couches denses (dimensionnement des types partages du C++)."""
        yield self.init
        for b in self.blocks:
            yield from (b["a"], b["b"], b["f1"], b["f2"])
        yield self.read

    # operations elementaires (identiques en C++) ------------------------------------------------------
    def gelu(self, x):
        lo, sh = self.GELU_LO * ONE, F - self.GELU_STEP_LOG2
        u = np.clip(x - lo, 0, 256 << sh)
        i = np.minimum(u >> sh, 255)
        fr = u - (i << sh)
        y = self.gelu_tab[i] + rnd_shift((self.gelu_tab[i + 1] - self.gelu_tab[i]) * fr, sh)
        return np.where(x >= 8 * ONE, x, np.where(x <= lo, 0, y))

    def track(self, name, x):
        """Releve max |x| (dimensionnement des ap_int du C++)."""
        self.peak[name] = max(self.peak.get(name, 0), int(np.max(np.abs(x))))
        return x

    def layernorm(self, x, w, b):
        n = x.shape[-1]                                                        # 32 = 2^5
        sh = int(np.log2(n))
        mu = rnd_shift(x.sum(-1, keepdims=True), sh)
        xc = x - mu
        var = self.track("var", rnd_shift((xc * xc).sum(-1, keepdims=True), sh + F))   # unites 2^-F
        inv = self.track("inv_ln", irsqrt(var, 3 * F // 2))                    # 2^F / sqrt(var / 2^F)
        return rnd_shift(rnd_shift(self.track("xc_inv", xc * inv), F) * w, F) + b

    def normalize(self, mr, mi, ex):
        """Normalisation par utilisateur en virgule flottante par bloc : h_k = mantisse_k * 2^ex_k.
        Renvoie hp (unites 2^-F) et pdb = log10 de la puissance relative (unites 2^-F)."""
        n2 = self.track("n2", (mr * mr + mi * mi).sum(-1))                    # [N, K], mantisses
        ri = self.track("ri_norm", irsqrt(n2, F + RCP))[..., None]
        hpr, hpi = rnd_shift(mr * ri, RCP), rnd_shift(mi * ri, RCP)
        self.track("hp", hpr); self.track("hp", hpi)
        emin = ex.min(-1, keepdims=True)
        n2a = n2 << (2 * (ex - emin))                                         # puissances alignees
        L = ilog2_fix(n2a)
        Ls = ilog2_fix(self.track("n2a_sum", n2a.sum(-1, keepdims=True))) - 2 * ONE   # log2 de la moyenne (K = 4)
        return hpr, hpi, self.track("pdb", rnd_shift((L - Ls) * LOG10_2, F))

    def forward(self, mr, mi, ex, snr_db):
        """mr, mi [N, K, M] mantisses entieres (16 bits), ex [N, K] exposants (h = mantisse * 2^ex),
        snr_db [N] -> (W_re, W_im) [N, M, K] entiers, normalises, unites 2^-F."""
        K, d = self.K, self.d
        N = mr.shape[0]
        hpr, hpi, p = self.normalize(mr, mi, ex)
        hr, him = self.q_h(hpr), self.q_h(hpi)
        gr = hr @ hr.transpose(0, 2, 1) + him @ him.transpose(0, 2, 1)        # G = h h^H (entiers)
        gi = him @ hr.transpose(0, 2, 1) - hr @ him.transpose(0, 2, 1)
        self.track("g_acc", gr); self.track("g_acc", gi)
        gr, gi = rnd_shift(gr * self.g_out.m, int(self.g_out.e) - F), rnd_shift(gi * self.g_out.m, int(self.g_out.e) - F)
        ga = isqrt(gr * gr + gi * gi)
        ef = self.track("ef", np.stack([gr, gi, ga], -1))                      # [N, K, K, 3]
        eye = np.broadcast_to((np.eye(K, dtype=np.int64) * ONE)[None, :, :, None], (N, K, K, 1))
        snr = np.broadcast_to(to_fix(self.track("snr10", snr_db / 10.0))[:, None, None, None], (N, K, K, 1))
        extra = np.stack([np.broadcast_to(p[:, :, None], (N, K, K)), np.broadcast_to(p[:, None, :], (N, K, K))], -1)
        e = self.track("act", self.init(np.concatenate([ef, eye, snr, extra], -1)))   # [N, K, K, 2d] (re puis im)
        for blk in self.blocks:
            x = self.track("act", np.concatenate([self.layernorm(e, blk["ln_w"], blk["ln_b"]), ef], -1))
            a, b = self.track("act", blk["a"](x)), self.track("act", blk["b"](x))
            ar, ai = blk["q_a"](a[..., :d]), blk["q_a"](a[..., d:])
            br, bi = blk["q_b"](b[..., :d]), blk["q_b"](b[..., d:])
            tr = np.einsum("njlc,nlkc->njkc", ar, br) - np.einsum("njlc,nlkc->njkc", ai, bi)
            ti = np.einsum("njlc,nlkc->njkc", ar, bi) + np.einsum("njlc,nlkc->njkc", ai, br)
            to = blk["t_out"]
            t = np.concatenate([rnd_shift(tr * to.m, int(to.e) - F), rnd_shift(ti * to.m, int(to.e) - F)], -1)
            h1 = self.track("act", blk["f1"](np.concatenate([x, self.track("act", t)], -1)))
            e = self.track("act", e + self.track("act", blk["f2"](self.gelu(h1))))
        c = self.track("act", self.read(e))                                    # [N, K, K, 2]
        cr, ci = self.q_c(c[..., 0]), self.q_c(c[..., 1])
        # W = h^H C (entiers) : W[m, k] = sum_j conj(h[j, m]) C[j, k]
        wr = hr.transpose(0, 2, 1) @ cr + him.transpose(0, 2, 1) @ ci
        wi = hr.transpose(0, 2, 1) @ ci - him.transpose(0, 2, 1) @ cr
        self.track("w_acc", wr); self.track("w_acc", wi)
        ri = self.track("ri_pow", irsqrt(self.track("p", (wr * wr + wi * wi).sum((1, 2), keepdims=True)), F + RCP))
        return rnd_shift(wr * ri, RCP), rnd_shift(wi * ri, RCP)              # W normalise, unites 2^-F


def block_float(h, bits=16):
    """Entree en virgule flottante par bloc : un exposant par utilisateur, mantisses sur `bits` bits signes."""
    peak = np.maximum(np.abs(h.real), np.abs(h.imag)).max(-1)                # [N, K]
    ex = (np.ceil(np.log2(peak + 1e-300)) - (bits - 1)).astype(np.int64)
    sc = 2.0 ** (-ex)[..., None]
    lim = 2 ** (bits - 1) - 1
    mr = np.clip(np.floor(h.real * sc + 0.5), -lim, lim).astype(np.int64)
    mi = np.clip(np.floor(h.imag * sc + 0.5), -lim, lim).astype(np.int64)
    return mr, mi, ex


def evaluate(name="art_pair_h16_L2_lam5_qprod_norm", n=5000):
    import torch
    from hls.golden import GoldenFixed
    from torch_port.stecath import draw, lab_rate, ours_input, pool
    qm, net = QuantPairPrecoder.load(name)
    H = draw(pool(), n, seed=2026)
    h, snr = ours_input(H)
    hn, sn = h.numpy().astype(np.complex128), snr.numpy().astype(np.float64)
    rate = lambda w: float(lab_rate(torch.tensor(w, dtype=torch.complex64), H).mean())
    with torch.no_grad():
        print(f"PyTorch quantifie : {float(lab_rate(net(h, snr), H).mean()):.4f}", flush=True)
    print(f"golden virgule fixe (F=12) : {rate(GoldenFixed(qm, frac=12).forward(hn, sn)):.4f}", flush=True)
    ip = IntPairPrecoder(qm)
    wr, wi = ip.forward(*block_float(hn), sn)
    print(f"reference entiere (F=12, tout en entiers) : {rate((wr + 1j * wi) / ONE):.4f}", flush=True)
    print("max |x| (bits signes) : " + ", ".join(f"{k} {v} ({v.bit_length() + 1})" for k, v in ip.peak.items()), flush=True)


if __name__ == "__main__":
    import sys
    evaluate(*sys.argv[1:2])
