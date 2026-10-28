"""TA-RB du memoire, reecrit dans le cadre freqctx (reference apprise, phase 8).

Reecriture fidele de TransformerPrecoderCleanResidualSignedAttn
(mimo_precoding/precoders/signed_attention.py et base_residual.py, NON
modifies), configuration du memoire en M8K4 : D=128, 4 tetes, 4 couches,
T=4 tokens par RB.

  - tokens : chaque RB (12 SC) est coupe en T groupes de 12/T SC ; par
    (groupe, utilisateur) : moyenne du canal (2M), log(1 + variance) (M),
    standardises sur l'axe des features, + identite de l'utilisateur
    (one-hot K) + log(bruit) ; Dense(D, gelu) + RMSNorm + SNR + position ;
  - L blocs : attention en frequence (softmax, sur tous les groupes de la
    bande, par utilisateur) -> attention SIGNEE entre utilisateurs
    (softmax * tanh) -> FFN 4D ; normalisation RMS, residus ponderes (0,1) ;
  - decodeur residuel : Dense(2M) par token, interpolation bilineaire fixe
    vers les F SC, correction Conv1D (noyau 12) sur [base, Re/Im/|.|/phase
    du canal d'entree], W = base + alpha * correction.

Seules differences avec le memoire (pour une comparaison equitable) :
entree = LS des pilotes en peigne (interpolation lineaire, comme notre
modele) au lieu d'un canal bruite synthetique ; SNR par echantillon ;
puissance TOTALE normalisee par SC (le memoire normalisait chaque
utilisateur) ; meme perte (debit relatif a RZF) et memes donnees que notre
modele, sans pre-entrainement sur RZF.
"""
import keras
import numpy as np
import tensorflow as tf

from precoders.classical import normalize_power


class SignedGateMHA(keras.layers.Layer):
    """Attention signee du memoire : poids = softmax(s) * tanh(s)."""

    def __init__(self, num_heads, key_dim, **kw):
        super().__init__(**kw)
        self.h, self.kd = num_heads, key_dim
        d = num_heads * key_dim
        self.q, self.k, self.v, self.o = (keras.layers.Dense(d) for _ in range(4))

    def call(self, x):
        N, K = tf.shape(x)[0], tf.shape(x)[1]

        def heads(t):
            return tf.transpose(tf.reshape(t, [N, K, self.h, self.kd]), [0, 2, 1, 3])
        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        raw = tf.einsum("nhkd,nhld->nhkl", q, k) * self.kd ** -0.5
        w = tf.nn.softmax(raw, axis=-1) * tf.tanh(raw)
        out = tf.transpose(tf.einsum("nhkl,nhld->nhkd", w, v), [0, 2, 1, 3])
        return self.o(tf.reshape(out, [N, K, self.h * self.kd]))


class SeparableBlock(keras.layers.Layer):
    """Frequence (softmax) -> utilisateurs (signee) -> FFN, comme le memoire."""

    def __init__(self, D, heads, **kw):
        super().__init__(**kw)
        self.nf, self.nu, self.nffn = (keras.layers.RMSNormalization(epsilon=1e-6) for _ in range(3))
        self.freq = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=D // heads)
        self.user = SignedGateMHA(heads, D // heads)
        self.f1, self.f2 = keras.layers.Dense(4 * D, activation="gelu"), keras.layers.Dense(D)
        self.s = [self.add_weight(shape=(), initializer=keras.initializers.Constant(0.1)) for _ in range(3)]

    def call(self, x):                                          # x [B, T, K, D]
        B, T, K, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[3]
        xf = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * K, T, D])
        y = self.nf(xf)
        xf = xf + self.s[0] * self.freq(y, y)
        x = tf.transpose(tf.reshape(xf, [B, K, T, D]), [0, 2, 1, 3])
        xu = tf.reshape(x, [B * T, K, D])
        xu = xu + self.s[1] * self.user(self.nu(xu))
        x = tf.reshape(xu, [B, T, K, D])
        return x + self.s[2] * self.f2(self.f1(self.nffn(x)))


class TARBPrecoder(keras.Model):
    def __init__(self, num_sc, num_users, num_bs_ant, tokens_per_rb=4, D=128, heads=4, layers=4,
                 rb_size=12, alpha_init=0.1, **kw):
        super().__init__(**kw)
        self.F, self.K, self.M = num_sc, num_users, num_bs_ant
        self.S = rb_size // tokens_per_rb                        # SC par token
        self.T = num_sc // self.S                                # tokens par utilisateur
        self.D, self.L, self.rb = D, layers, rb_size
        self.inp, self.inorm = keras.layers.Dense(D, activation="gelu"), keras.layers.RMSNormalization(epsilon=1e-6)
        self.snr = keras.layers.Dense(D, activation="gelu")
        self.pos = keras.layers.Embedding(self.T, D)
        self.blocks = [SeparableBlock(D, heads) for _ in range(layers)]
        self.joint = keras.layers.Dense(D)
        self.tok2w = keras.layers.Dense(2 * num_bs_ant)
        self.r1 = keras.layers.Conv1D(4 * num_bs_ant, rb_size, padding="same", activation="gelu")
        self.r2 = keras.layers.Conv1D(2 * num_bs_ant, 1, padding="same")
        self.alpha = self.add_weight(shape=(), initializer=keras.initializers.Constant(alpha_init))

    def call(self, h, snr_db, aux=None):
        """h [B, F, K, M] (CSI connu de la BS), snr_db [B] -> W [B, F, M, K]."""
        B, F, K, M, T, S = tf.shape(h)[0], self.F, self.K, self.M, self.T, self.S
        log_no = -tf.cast(snr_db, tf.float32) / 10.0 * np.log(10.0)          # log(sigma^2)
        hu = tf.transpose(h, [0, 2, 3, 1])                                       # [B, K, M, F]
        ht = tf.reshape(hu, [B, K, M, T, S])
        mean = tf.reduce_mean(ht, -1)                                            # [B, K, M, T]
        var = tf.math.log1p(tf.reduce_mean(tf.abs(ht - mean[..., None]) ** 2, -1))
        p = lambda t: tf.transpose(t, [0, 3, 1, 2])                              # -> [B, T, K, M]
        feat = tf.concat([tf.math.real(p(mean)), tf.math.imag(p(mean)), p(var)], -1)
        feat = (feat - tf.reduce_mean(feat, -1, keepdims=True)) / (tf.math.reduce_std(feat, -1, keepdims=True) + 1e-6)
        onehot = tf.broadcast_to(tf.eye(K)[None, None], [B, T, K, K])
        lno = tf.broadcast_to(tf.reshape(log_no, [-1, 1, 1, 1]), [B, T, K, 1])
        x = self.inorm(self.inp(tf.concat([feat, onehot, lno], -1)))
        x = x + tf.reshape(self.snr(log_no[:, None]), [-1, 1, 1, self.D])
        x = x + tf.reshape(self.pos(tf.range(T)), [1, T, 1, self.D])
        for blk in self.blocks:
            x = blk(x)
        x = self.joint(x)                                                        # [B, T, K, D]
        xk = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * K, T, self.D])
        tok = self.tok2w(xk)                                                     # [B*K, T, 2M]
        base = tf.image.resize(tok[:, None], [1, F], method="bilinear")[:, 0]   # [B*K, F, 2M]
        hk = tf.reshape(tf.transpose(h, [0, 2, 1, 3]), [B * K, F, M])
        ref = tf.concat([base, tf.math.real(hk), tf.math.imag(hk), tf.abs(hk), tf.math.angle(hk)], -1)
        w = base + self.alpha * self.r2(self.r1(ref))                            # [B*K, F, 2M]
        w = tf.reshape(w, [B, K, F, 2 * M])
        w = tf.complex(w[..., :M], w[..., M:])                                   # [B, K, F, M]
        return normalize_power(tf.transpose(w, [0, 2, 3, 1]))                   # [B, F, M, K]

    def macs_per_sc(self, K):
        """MAC reelles par SC (formule de complexite du memoire, FLOPs / 2, sans x14 OFDM)."""
        M, T, D, F, RB = self.M, self.T, self.D, self.F, self.rb
        fd = 3 * M + K + 1
        mf = lambda seq, d: 8.0 * seq * d ** 2 + 4.0 * seq ** 2 * d
        f = 2.0 * T * K * fd * D
        f += self.L * (K * mf(T, D) + T * mf(K, D) + 2 * (2.0 * T * K * D * 4 * D))
        f += 2.0 * T * K * D * D + 2.0 * T * K * D * 2 * M
        f += K * (2.0 * F * RB * 6 * M * 4 * M + 2.0 * F * 4 * M * 2 * M)
        return int(f / 2 / F)
