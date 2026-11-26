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
        self.d, self.hidden = d, hidden

    def build_sym(self):
        """FFN conjugaison-equivariant (version symetrique) : poids REELS appliques
        de la meme facon aux parties reelle et imaginaire, biais reels, porte sur le module."""
        if not hasattr(self, "g1"):
            self.g1 = keras.layers.Dense(self.hidden, use_bias=False)
            self.g2 = keras.layers.Dense(self.d, use_bias=False)
            self.b1 = self.add_weight(shape=(self.hidden,), initializer="zeros")
            self.b2 = self.add_weight(shape=(self.d,), initializer="zeros")
            self.gb = self.add_weight(shape=(self.hidden,), initializer="zeros")

    def _cffn(self, s):
        lin = lambda layer, z, b: tf.complex(layer(tf.math.real(z)) + b, layer(tf.math.imag(z)))
        h = lin(self.g1, s, self.b1)
        h = h * tf.cast(tf.sigmoid(tf.abs(h) + self.gb), h.dtype)          # porte sur le module
        return lin(self.g2, h, self.b2)

    def call(self, e, g, upper=None):                       # g utilise seulement si edge_g
        x = self.norm(_real(e))
        if self.edge_g:                                     # caracteristique d'arete, pas un calcul impose
            x = tf.concat([x, tf.stack([tf.math.real(g), tf.math.imag(g), tf.abs(g)], -1)], -1)
        K = tf.cast(tf.shape(e)[1], tf.complex64)
        m = tf.einsum("njlc,nlkc->njkc", _cplx(self.a(x)), _cplx(self.b(x))) / K
        if upper is None:
            return e + _cplx(self.f2(self.f1(tf.concat([x, _real(m)], axis=-1))))
        # symetrie hermitienne (phase 9) : entree de la paire {j, k} construite avec ses deux
        # orientations, s_jk = c_jk + conj(c_kj) (donc s_kj = conj(s_jk), s_jj reel) ; FFN
        # conjugaison-equivariant sur les seules paires j <= k ; la paire (k, j) recoit le conjugue.
        self.build_sym()
        c = tf.concat([e, m, g[..., None]], axis=-1)        # [N, K, K, 2d+1] complexe
        sc = c + tf.math.conj(tf.transpose(c, [0, 2, 1, 3]))
        sc = sc / tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(sc) ** 2, -1, keepdims=True) + 1e-6), sc.dtype)
        up_idx, full_idx, conj = upper
        su = tf.gather(tf.reshape(sc, [tf.shape(sc)[0], -1, sc.shape[-1]]), up_idx, axis=1)
        du = self._cffn(su)                                 # [N, P, d]
        df = tf.gather(du, full_idx, axis=1)                # [N, K*K, d]
        df = tf.where(conj[None, :, None] > 0, tf.math.conj(df), df)
        return e + tf.reshape(df, tf.shape(e))


class GramPrecoder(keras.Model):
    """mix : "gram" (bloc G.E), "triangular", "triangular_g" (G reinjecte).
    tied : un seul bloc repete L fois (True) ou L blocs distincts."""

    def __init__(self, num_bs_ant, d=16, hidden=64, layers=4, mix="gram", tied=True, sym=False, inp="raw",
                 err_in=False, **kw):
        """sym (phase 9, triangulaire seulement) : etat hermitien E[k,j] = conj(E[j,k]) ; le FFN
        ne tourne que sur les K(K+1)/2 paires j <= k ; sortie C = S diag(p), S hermitienne lue
        sur les paires, p > 0 une puissance par utilisateur lue sur la diagonale (forme de WMMSE).
        err_in (phase 14, CSI imparfait) : la variance d'erreur relative par utilisateur
        (10 log10(err_var_k / puissance_k), dB/10 comme pdb) est donnee en plus sur les tokens de
        paire (j, k), les deux scalaires (utilisateur j et utilisateur k) comme pour pdb. Ĝ=ĤĤ^H ne
        dit pas quel utilisateur est mal estime : ce canal supplementaire comble ce manque. Par
        defaut (err_in=False) le comportement est inchange ; `call`/`coeffs` prennent alors un
        `err_db` optionnel qui est ignore."""
        super().__init__(**kw)
        self.M, self.d, self.hidden, self.L, self.mix = num_bs_ant, d, hidden, layers, mix
        self.tied, self.sym, self.inp, self.err_in = tied, sym, inp, err_in
        # inp="norm" (phase 11, comme torch_port inp="norm") : canal de chaque utilisateur normalise
        # (G = correlations), puissance relative (dB/10) de chaque utilisateur donnee sur les paires,
        # W = h~^H C. Rend le precodeur robuste aux puissances inegales et a la quantification.
        if sym:
            assert mix.startswith("triangular")
            self.read_p = keras.layers.Dense(1)
        self.init = keras.layers.Dense(2 * d)

        def make():
            if mix == "gram":
                return GramBlock(d, hidden)
            return TriangularBlock(d, hidden, edge_g=(mix == "triangular_g"))
        self.blocks = [make()] if tied else [make() for _ in range(layers)]
        self.read = keras.layers.Dense(2)

    def _norm(self, h):
        n2 = tf.reduce_sum(tf.abs(h) ** 2, axis=-1)                      # [N, K]
        hp = h / tf.cast(tf.sqrt(n2), h.dtype)[..., None]
        pdb = tf.math.log(n2 / tf.reduce_mean(n2, -1, keepdims=True)) / tf.math.log(10.0)  # dB/10
        return hp, pdb

    def call(self, h, snr_db, err_db=None):
        """h [N, K, M] complex64, snr_db [N] -> W [N, M, K]. err_db [N, K] (option, err_in=True
        seulement) : variance d'erreur relative par utilisateur, 10 log10(err_var_k / puissance_k)."""
        hp = self._norm(h)[0] if self.inp == "norm" else h               # W = h~^H C (inp="norm")
        return normalize_power(tf.matmul(hp, self.coeffs(h, snr_db, err_db), adjoint_a=True))

    def coeffs(self, h, snr_db, err_db=None):
        """Les coefficients C [N, K, K] (W = h^H C avant normalisation)."""
        K = tf.shape(h)[1]
        M = tf.cast(tf.shape(h)[2], h.dtype)                 # M lu sur l'entree : un modele pour tout M
        if self.inp == "norm":
            hp, pdb = self._norm(h)
            g = tf.matmul(hp, hp, adjoint_b=True)           # correlations, |g| <= 1
        else:
            g = tf.matmul(h, h, adjoint_b=True) / M         # [N, K, K]
        snr_db = tf.cast(snr_db, tf.float32)
        eye = tf.broadcast_to(tf.eye(K)[None], tf.shape(tf.math.real(g)))
        snr = tf.broadcast_to(tf.reshape(snr_db / 10.0, [-1, 1, 1]), tf.shape(eye))
        parts = [tf.math.real(g), tf.math.imag(g), tf.abs(g), eye, snr]
        if self.inp == "norm":                              # puissance relative des deux utilisateurs
            parts += [tf.broadcast_to(pdb[:, :, None], tf.shape(eye)), tf.broadcast_to(pdb[:, None, :], tf.shape(eye))]
        if self.err_in:                                     # variance d'erreur relative des deux utilisateurs
            edb = tf.cast(err_db, tf.float32) / 10.0
            parts += [tf.broadcast_to(edb[:, :, None], tf.shape(eye)), tf.broadcast_to(edb[:, None, :], tf.shape(eye))]
        f = tf.stack(parts, axis=-1)
        e = _cplx(self.init(f))                             # [N, K, K, d]
        if not self.sym:
            for i in range(self.L):
                e = self.blocks[0 if self.tied else i](e, g)
            return _cplx(self.read(_real(e)))[..., 0]        # [N, K, K]
        Kp = h.shape[1]
        up = [(j, k) for j in range(Kp) for k in range(j, Kp)]
        pos = {jk: p for p, jk in enumerate(up)}
        up_idx = tf.constant([j * Kp + k for j, k in up], tf.int32)
        full_idx = tf.constant([pos[(min(j, k), max(j, k))] for j in range(Kp) for k in range(Kp)], tf.int32)
        conj = tf.constant([1 if j > k else (-1 if j == k else 0) for j in range(Kp) for k in range(Kp)], tf.int32)
        e = 0.5 * (e + tf.math.conj(tf.transpose(e, [0, 2, 1, 3])))          # etat initial hermitien
        for i in range(self.L):
            e = self.blocks[0 if self.tied else i](e, g, upper=(up_idx, full_idx, conj))
        s = _cplx(self.read(_real(e)))[..., 0]
        s = 0.5 * (s + tf.linalg.adjoint(s))                                 # S hermitienne
        diag = tf.einsum("nkkd->nkd", e)                                     # [N, K, d], reel
        p = tf.nn.softplus(self.read_p(tf.math.real(diag)))[..., 0]          # puissance par utilisateur
        return s * tf.cast(p, s.dtype)[:, None, :]                           # C = S diag(p)

    def macs_per_sc(self, K, M=None):
        """MAC reelles par SC : G, initialisation, L x (G.E + FFN), lecture, W = h^H C."""
        M = self.M if M is None else M
        d, hd, L = self.d, self.hidden, self.L
        per_edge = 5 * 2 * d + L * (4 * d * hd + hd * 2 * d) + 2 * d * 2
        if self.mix.startswith("triangular"):                        # projections a, b ; FFN sur [x, m]
            per_edge += L * 2 * (2 * d + 3 * (self.mix == "triangular_g")) * (2 * d)
        if self.sym:            # FFN (poids reels sur Re et Im) sur K(K+1)/2 paires seulement
            ffn = L * (4 * d * hd + hd * 2 * d)
            cffn = L * 2 * ((2 * d + 1) * hd + hd * d)
            return (4 * K * K * M + K * K * (per_edge - ffn) + K * (K + 1) // 2 * cffn
                    + L * 4 * K ** 3 * d + 4 * M * K * K)
        return 4 * K * K * M + K * K * per_edge + L * 4 * K ** 3 * d + 4 * M * K * K
