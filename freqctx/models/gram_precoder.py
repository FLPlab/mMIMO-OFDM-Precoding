"""Precodeur a "attention de Gram" (phase 5, massif).

Constat : en massif (M=64, K=8), le precodeur par tokens d'utilisateur doit
a la fois conserver 64 antennes et inverser l'interaction 8x8 entre
utilisateurs ; il plafonne a 50 % de RZF. Ici :
  - sortie structuree W = h^H C : le modele ne produit que C [K, K] (le
    precodeur optimal est toujours de cette forme) ;
  - C est calcule par un etat par PAIRE d'utilisateurs E[j, k] (d complexes),
    mis a jour par un bloc repete L fois : E <- E + FFN([E, G E]), ou G E est
    une "attention" dont les poids sont la matrice de Gram G = h h^H / M.
    Appliquer G de facon repetee = la forme d'une inversion iterative
    (Neumann, Richardson) ; aucune inversion explicite.
Equivariant (permuter les utilisateurs permute lignes et colonnes de G et E),
et le cout ne depend de M que via G et W = h^H C.
"""
import keras
import tensorflow as tf

from precoders.classical import normalize_power


def _cplx(x):
    """[..., 2d] reel -> [..., d] complexe."""
    d = x.shape[-1] // 2
    return tf.complex(x[..., :d], x[..., d:])


def _real(z):
    return tf.concat([tf.math.real(z), tf.math.imag(z)], axis=-1)


class GramBlock(keras.layers.Layer):
    def __init__(self, d, hidden, **kw):
        super().__init__(**kw)
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.f1 = keras.layers.Dense(hidden, activation="gelu")
        self.f2 = keras.layers.Dense(2 * d)

    def call(self, e, g):                                   # e [N, K, K, d] complexe, g [N, K, K]
        ge = tf.einsum("njl,nlkd->njkd", g, e)              # "attention" de poids G
        x = self.norm(tf.concat([_real(e), _real(ge)], axis=-1))
        return e + _cplx(self.f2(self.f1(x)))


class TriangularBlock(keras.layers.Layer):
    """Attention "triangulaire" lineaire entre paires (Edge Transformer,
    Bergen et al. 2021 ; AlphaFold) : la paire (j, k) recoit
    sum_l a(e_jl) * b(e_lk), avec a, b appris. Generalise G.E (si un canal de
    e porte G) et permet E.E (Newton-Schulz : degre double a chaque couche).
    Lineaire (sans softmax) : la phase 1 a montre que softmax ne sait pas
    soustraire. Pre-normalisation pour que les produits ne divergent pas."""

    def __init__(self, d, hidden, edge_g=False, **kw):
        super().__init__(**kw)
        self.edge_g = edge_g                                # G reinjecte a chaque couche (arete)
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.a = keras.layers.Dense(2 * d, use_bias=False)
        self.b = keras.layers.Dense(2 * d, use_bias=False)
        self.f1 = keras.layers.Dense(hidden, activation="gelu")
        self.f2 = keras.layers.Dense(2 * d)

    def call(self, e, g):                                   # g utilise seulement si edge_g
        x = self.norm(_real(e))
        if self.edge_g:                                     # caracteristique d'arete, pas un calcul impose
            x = tf.concat([x, tf.stack([tf.math.real(g), tf.math.imag(g), tf.abs(g)], -1)], -1)
        K = tf.cast(tf.shape(e)[1], tf.complex64)
        m = tf.einsum("njlc,nlkc->njkc", _cplx(self.a(x)), _cplx(self.b(x))) / K
        return e + _cplx(self.f2(self.f1(tf.concat([x, _real(m)], axis=-1))))


class GramPrecoder(keras.Model):
    """mix : "gram" (bloc G.E), "triangular", "triangular_g" (G reinjecte).
    tied : un seul bloc repete L fois (True) ou L blocs distincts."""

    def __init__(self, num_bs_ant, d=16, hidden=64, layers=4, mix="gram", tied=True, **kw):
        super().__init__(**kw)
        self.M, self.d, self.hidden, self.L, self.mix = num_bs_ant, d, hidden, layers, mix
        self.tied = tied
        self.init = keras.layers.Dense(2 * d)

        def make():
            if mix == "gram":
                return GramBlock(d, hidden)
            return TriangularBlock(d, hidden, edge_g=(mix == "triangular_g"))
        self.blocks = [make()] if tied else [make() for _ in range(layers)]
        self.read = keras.layers.Dense(2)

    def call(self, h, snr_db):
        """h [N, K, M] complex64, snr_db [N] -> W [N, M, K]."""
        return normalize_power(tf.matmul(h, self.coeffs(h, snr_db), adjoint_a=True))   # W = h^H C

    def coeffs(self, h, snr_db):
        """Les coefficients C [N, K, K] (W = h^H C avant normalisation)."""
        K = tf.shape(h)[1]
        M = tf.cast(tf.shape(h)[2], h.dtype)                 # M lu sur l'entree : un modele pour tout M
        g = tf.matmul(h, h, adjoint_b=True) / M             # [N, K, K]
        snr_db = tf.cast(snr_db, tf.float32)
        eye = tf.broadcast_to(tf.eye(K)[None], tf.shape(tf.math.real(g)))
        snr = tf.broadcast_to(tf.reshape(snr_db / 10.0, [-1, 1, 1]), tf.shape(eye))
        f = tf.stack([tf.math.real(g), tf.math.imag(g), tf.abs(g), eye, snr], axis=-1)
        e = _cplx(self.init(f))                             # [N, K, K, d]
        for i in range(self.L):
            e = self.blocks[0 if self.tied else i](e, g)
        return _cplx(self.read(_real(e)))[..., 0]            # [N, K, K]

    def macs_per_sc(self, K, M=None):
        """MAC reelles par SC : G, initialisation, L x (G.E + FFN), lecture, W = h^H C."""
        M = self.M if M is None else M
        d, hd, L = self.d, self.hidden, self.L
        per_edge = 5 * 2 * d + L * (4 * d * hd + hd * 2 * d) + 2 * d * 2
        if self.mix.startswith("triangular"):                        # projections a, b ; FFN sur [x, m]
            per_edge += L * 2 * (2 * d + 3 * (self.mix == "triangular_g")) * (2 * d)
        return 4 * K * K * M + K * K * per_edge + L * 4 * K ** 3 * d + 4 * M * K * K
