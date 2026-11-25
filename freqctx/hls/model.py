"""Extraction du precodeur par paires quantifie (ART, quantificateurs LSQ du labo) vers des tableaux numpy :
poids entiers, echelles, bits. Point de depart de la reference « golden » (golden.py) et du generateur
HLS (codegen.py).

Convention LSQ du labo (quan/quantizer/lsq.py) : q = round(clamp(x / s, -(2^b - 1), 2^b - 1)), x ~ q * s.
Les valeurs occupent donc b + 1 bits signes. Poids : une echelle par canal de sortie ; activations et
operandes des produits : une echelle scalaire.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

WEIGHTS = "/export/tmp/sala/freqctx/weights/stecath"


@dataclass
class Quant:
    """Un quantificateur LSQ fige : echelle s (scalaire ou par canal) et bits b."""
    s: np.ndarray
    b: int

    @property
    def lim(self):
        return 2 ** self.b - 1

    def q(self, x):
        """Entiers (en flottants) : round(clamp(x / s, -lim, lim))."""
        return np.round(np.clip(x / self.s, -self.lim, self.lim))

    def __call__(self, x):
        return self.q(x) * self.s


@dataclass
class Dense:
    """Couche dense quantifiee : y = (q_w(W) * s_w) (q_a(x) * s_a) + biais."""
    name: str
    w_int: np.ndarray          # [out, in], entiers
    w_scale: np.ndarray        # [out, 1]
    w_bits: int
    a: Quant
    bias: np.ndarray = None    # [out] ou None

    def __call__(self, x):
        y = self.a(x) @ (self.w_int * self.w_scale).T
        return y if self.bias is None else y + self.bias


@dataclass
class Block:
    """Couche triangulaire : LayerNorm, projections a et b, produit triangulaire, FFN, residuel."""
    ln_w: np.ndarray
    ln_b: np.ndarray
    a: Dense
    b: Dense
    f1: Dense
    f2: Dense
    q_a: Quant
    q_b: Quant


@dataclass
class QuantPairPrecoder:
    K: int
    M: int
    d: int
    init: Dense
    blocks: list
    read: Dense
    q_h: Quant
    q_c: Quant
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _quant(q):
        return Quant(q.s.detach().cpu().numpy().astype(np.float64).reshape(-1)[0] if q.s.numel() == 1
                     else q.s.detach().cpu().numpy().astype(np.float64), int(round(float(q.bit))))

    @classmethod
    def _dense(cls, name, m):
        qw = cls._quant(m.quan_w_fn)
        w = m.weight.detach().cpu().numpy().astype(np.float64)
        sw = np.broadcast_to(np.asarray(qw.s).reshape(-1, 1), (w.shape[0], 1)).copy()
        w_int = np.round(np.clip(w / sw, -qw.lim, qw.lim))
        bias = None if m.bias is None else m.bias.detach().cpu().numpy().astype(np.float64)
        return Dense(name, w_int, sw, qw.b, cls._quant(m.quan_a_fn), bias)

    @classmethod
    def from_torch(cls, net, K=4, M=64):
        blocks = []
        for i, blk in enumerate(net.blocks):
            blocks.append(Block(blk.norm.weight.detach().cpu().numpy().astype(np.float64),
                                blk.norm.bias.detach().cpu().numpy().astype(np.float64),
                                cls._dense(f"blocks.{i}.a", blk.a), cls._dense(f"blocks.{i}.b", blk.b),
                                cls._dense(f"blocks.{i}.f1", blk.f1), cls._dense(f"blocks.{i}.f2", blk.f2),
                                cls._quant(blk.q_a), cls._quant(blk.q_b)))
        d = net.read.in_features // 2
        return cls(K, M, d, cls._dense("init", net.init), blocks, cls._dense("read", net.read),
                   cls._quant(net.q_h), cls._quant(net.q_c))

    @staticmethod
    def spec(name):
        """(hidden, layers) lus dans le nom : art_pair_h<hidden>[_L<layers>]_... (8 couches par defaut)."""
        import re
        h, l = re.search(r"_h(\d+)", name), re.search(r"_L(\d+)", name)
        return int(h.group(1)) if h else 32, int(l.group(1)) if l else 8

    @classmethod
    def load(cls, name="art_pair_h32_lam5_qprod_norm", hidden=None, layers=None):
        """Charge un modele ART sauvegarde par torch_port/art_stecath.py (entree normalisee, produits quantifies).
        hidden et layers sont deduits du nom s'ils ne sont pas donnes."""
        from torch_port.art_stecath import quantize, quantize_products
        from torch_port.models import PairPrecoder
        h, l = cls.spec(name)
        net = PairPrecoder(d=16, hidden=hidden or h, layers=layers or l, inp="norm")
        quantize(net)
        quantize_products(net)
        sd = torch.load(f"{WEIGHTS}/{name}.pt", map_location="cpu", weights_only=True)
        ref = net.state_dict()
        net.load_state_dict({k: v.reshape(ref[k].shape) for k, v in sd.items()})
        net.eval()
        return cls.from_torch(net), net

    def dense_layers(self):
        yield self.init
        for blk in self.blocks:
            yield from (blk.a, blk.b, blk.f1, blk.f2)
        yield self.read
