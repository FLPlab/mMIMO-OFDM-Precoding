"""Precodeur "axial" : un token par COEFFICIENT du canal (antenne m, utilisateur k).

Ecarte comme modele en phase 6 ; REMIS en phase 8 comme REFERENCE APPRISE de
la litterature (proche des "2D-Gformer" de Guo et Yang 2025 : tokens
antenne x utilisateur, attention sans softmax entre utilisateurs dont les
scores somment sur les antennes, sortie directe). Ni G ni W = h^H C imposes.
Chaque couche :
  - scores entre utilisateurs appris, sommes sur les antennes :
        s_h[j, k] = (1/M) sum_m sum_c conj(q_h[j, m, c]) kk_h[k, m, c]   (par tete h)
    (une "matrice de Gram apprise", calculee a chaque couche) ;
  - mise a jour lineaire signee (pas de softmax, phase 1) :
        x[j, m] += sum_k s_h[j, k] v_h[k, m]                 , puis FFN par token.
Equivariant aux permutations des utilisateurs ET des antennes ; parametres
independants de M et K. Litterature : couches "echangeables" sur matrices
(Hartford et al. 2018), GNN sur graphe biparti antennes-utilisateurs pour le
precodage (edge-GNN), attention axiale (Ho et al. 2019).
"""
import keras
import tensorflow as tf

from models.gram_precoder import _cplx, _real
from precoders.classical import normalize_power


class AxialBlock(keras.layers.Layer):
    def __init__(self, d, hidden, heads, **kw):
        super().__init__(**kw)
        self.d, self.H = d, heads
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.q = keras.layers.Dense(2 * d, use_bias=False)
        self.k = keras.layers.Dense(2 * d, use_bias=False)
        self.v = keras.layers.Dense(2 * d, use_bias=False)
        self.o = keras.layers.Dense(2 * d)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.f1 = keras.layers.Dense(hidden, activation="gelu")
        self.f2 = keras.layers.Dense(2 * d)

    def _heads(self, z):                                    # [N, K, M, d] -> [N, K, M, H, d/H]
        s = tf.shape(z)
        return tf.reshape(z, [s[0], s[1], s[2], self.H, self.d // self.H])

    def call(self, x):                                      # x [N, K, M, d] complexe
        y = self.norm(_real(x))
        q, k, v = (self._heads(_cplx(f(y))) for f in (self.q, self.k, self.v))
        M = tf.cast(tf.shape(x)[2], x.dtype)
        s = tf.einsum("njmhc,nkmhc->nhjk", tf.math.conj(q), k) / M     # Gram appris par tete
        K = tf.cast(tf.shape(x)[1], x.dtype)
        u = tf.einsum("nhjk,nkmhc->njmhc", s, v) / K
        u = tf.reshape(u, tf.shape(x))
        x = x + _cplx(self.o(_real(u)))
        return x + _cplx(self.f2(self.f1(self.norm2(_real(x)))))


class AxialPrecoder(keras.Model):
    def __init__(self, d=16, hidden=32, heads=4, layers=8, tied=True, **kw):
        super().__init__(**kw)
        self.d, self.hidden, self.H, self.L, self.tied = d, hidden, heads, layers, tied
        self.init = keras.layers.Dense(2 * d)
        self.blocks = [AxialBlock(d, hidden, heads)] if tied else \
            [AxialBlock(d, hidden, heads) for _ in range(layers)]
        self.read = keras.layers.Dense(2)

    def call(self, h, snr_db):
        """h [N, K, M] complex64, snr_db [N] -> W [N, M, K]."""
        snr = tf.broadcast_to(tf.reshape(tf.cast(snr_db, tf.float32) / 10.0, [-1, 1, 1, 1]),
                              tf.concat([tf.shape(tf.math.real(h)), [1]], 0))
        f = tf.concat([tf.stack([tf.math.real(h), tf.math.imag(h), tf.abs(h)], -1), snr], -1)
        x = _cplx(self.init(f))                             # [N, K, M, d]
        for i in range(self.L):
            x = self.blocks[0 if self.tied else i](x)
        w = _cplx(self.read(_real(x)))[..., 0]               # [N, K, M]
        return normalize_power(tf.transpose(w, [0, 2, 1]))   # [N, M, K]

    def macs_per_sc(self, K, M=8):
        """MAC reelles : par token q, k, v, o + FFN ; scores K^2 M d et mise a jour K^2 M d (complexes)."""
        d, hd, L = self.d, self.hidden, self.L
        per_tok = 4 * 2 * d * 2 * d + 2 * d * hd + hd * 2 * d
        return M * K * (4 * 2 * d + 2 * d * 2) + L * (M * K * per_tok + 2 * 4 * K * K * M * d)
