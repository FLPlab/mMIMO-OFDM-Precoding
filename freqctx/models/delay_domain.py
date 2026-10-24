"""Estimation dans le domaine des retards + precodeur par SC (phase 4, variante a).

Un seul modele, appris de bout en bout sur le debit, en deux parties :

1. Estimation sur TOUTE la bande, pour chaque utilisateur k (memes poids) :
     y_k  = LS sur ses N = F/K SC pilotes (peigne kronecker de Sionna : k, k+K, ...)
     g_k  = IDFT(y_k)                  N retards x M antennes      (fixe, sans parametre)
     token par retard t : [Re g_t, Im g_t (2M), log puissance] + position t
       -> Dense(De) -> 2 x [attention entre retards (softmax) + FFN]
       -> Dense(1) -> sigmoide = gain_t dans [0, 1], commun aux M antennes
     h_k  = DFT(gain * g_k) sur les F SC                            (fixe)
   Un gain entre 0 et 1 par retard = la forme du filtre de Wiener ; un trajet
   arrive avec le meme retard sur toutes les antennes, d'ou un gain commun.
   Option history=True (phase 4f) : chaque token de retard recoit en plus le
   log de sa puissance moyenne sur les sondages PRECEDENTS (profil des trajets,
   qui varie lentement) -- l'information que la borne genie connait d'avance.
   Le modele recoit alors call(h_ls, snr_db, aux=pdp), pdp [B, K, N].
   Option angle_mask=True (variante b) : les antennes passent d'abord dans le
   domaine ANGULAIRE (DFT unitaire sur les M/2 antennes de chaque polarisation ;
   ordre Sionna : antennes 0..M/2-1 = polarisation 1, M/2..M-1 = polarisation 2),
   et le gain est appris par couple (retard, angle) : un trajet = un pic en
   (retard, direction), comme les estimateurs "beamspace".
2. Precodeur par SC : PerSCPrecoder (fenetre 1) applique a h_hat sur chaque SC.

Entree : h_ls [B, F, K, M] (LS lineaire du sondage), snr_db [B].
Sortie : W [B, F, M, K].
"""
import keras
import numpy as np
import tensorflow as tf

from models.per_sc import Block, PerSCPrecoder


class DelayEstimator(keras.layers.Layer):
    def __init__(self, num_sc, num_users, num_bs_ant, d=32, layers=2, heads=2, guard=4,
                 gain_bias_init=3.0, angle_mask=False, history=False, **kw):
        super().__init__(**kw)
        F, K, self.M = num_sc, num_users, num_bs_ant
        self.angle_mask, self.history = angle_mask, history
        half = num_bs_ant // 2
        u = np.exp(-2j * np.pi * np.outer(np.arange(half), np.arange(half)) / half) / np.sqrt(half)
        U = np.zeros((num_bs_ant, num_bs_ant), complex)
        U[:half, :half], U[half:, half:] = u, u                     # DFT par polarisation
        self.U = tf.constant(U, tf.complex64)
        self.N = N = F // K
        t = np.arange(N)
        t_signed = np.where(t < N - guard, t, t - N)             # quelques retards "negatifs"
        self.pilot_idx = tf.constant(np.stack([k + K * t for k in range(K)]), tf.int32)  # [K, N]
        idft = np.stack([np.exp(2j * np.pi * np.outer(t_signed, k + K * t) / F) / N
                         for k in range(K)])                       # [K, N retards, N pilotes]
        self.idft = tf.constant(idft, tf.complex64)
        self.dft = tf.constant(np.exp(-2j * np.pi * np.outer(np.arange(F), t_signed) / F),
                               tf.complex64)                       # [F, N]
        self.embed = keras.layers.Dense(d)
        self.pos = keras.layers.Embedding(N, d)
        self.blocks = [Block(d, heads, "attention", attn="softmax") for _ in range(layers)]
        self.gain = keras.layers.Dense(num_bs_ant if angle_mask else 1,
                                       bias_initializer=keras.initializers.Constant(gain_bias_init))
        # gain_bias_init=3 -> gains initiaux ~0.95 : le modele part de l'estimation LS

    def delays(self, h_ls):
        """IDFT du LS du peigne de chaque utilisateur : [B, F, K, M] -> [B, K, N, M]."""
        B = tf.shape(h_ls)[0]
        hk = tf.transpose(h_ls, [0, 2, 1, 3])
        y = tf.gather(hk, tf.tile(self.pilot_idx[None], [B, 1, 1]), axis=2, batch_dims=2)
        return tf.einsum("ktn,bknm->bktm", self.idft, y)

    def call(self, h_ls, pdp=None):
        B = tf.shape(h_ls)[0]
        K, N = self.pilot_idx.shape
        hk = tf.transpose(h_ls, [0, 2, 1, 3])                      # [B, K, F, M]
        y = tf.gather(hk, tf.tile(self.pilot_idx[None], [B, 1, 1]), axis=2, batch_dims=2)
        g = tf.einsum("ktn,bknm->bktm", self.idft, y)              # [B, K, N, M] retards
        if self.angle_mask:
            g = tf.matmul(g, self.U)                               # antennes -> angles
        p = tf.reduce_mean(tf.abs(g) ** 2, axis=-1, keepdims=True)
        parts = [tf.math.real(g), tf.math.imag(g), tf.math.log(p + 1e-9) / 10.0]
        if self.history:                                           # profil des sondages precedents
            parts.append(tf.math.log(pdp[..., None] + 1e-9) / 10.0)
        feats = tf.concat(parts, axis=-1)                          # [B, K, N, 2M+1(+1)]
        x = self.embed(tf.reshape(feats, [B * K, N, feats.shape[-1]])) + self.pos(tf.range(N))[None]
        for blk in self.blocks:
            x = blk(x)                                             # attention entre les N retards
        gain = tf.sigmoid(self.gain(x))                            # [B*K, N, 1 ou M]
        G = gain.shape[-1]
        self.last_gain = tf.reshape(gain, [B, K, N, G])
        g_hat = g * tf.cast(tf.reshape(gain, [B, K, N, G]), g.dtype)
        if self.angle_mask:
            g_hat = tf.matmul(g_hat, self.U, adjoint_b=True)       # angles -> antennes
        h_hat = tf.einsum("ft,bktm->bfkm", self.dft, g_hat)        # [B, F, K, M]
        return h_hat


class DelayDomainPrecoder(keras.Model):
    def __init__(self, num_sc, num_users, num_bs_ant, est_d=32, est_layers=2, angle_mask=False,
                 history=False, **precoder_kw):
        super().__init__()
        self.F, self.K, self.M = num_sc, num_users, num_bs_ant
        self.estimator = DelayEstimator(num_sc, num_users, num_bs_ant, d=est_d, layers=est_layers,
                                        angle_mask=angle_mask, history=history)
        self.precoder = PerSCPrecoder(num_bs_ant, **precoder_kw)
        self.est_d, self.est_layers = est_d, est_layers

    def call(self, h_ls, snr_db, aux=None):
        h_hat = self.estimator(h_ls, aux)                          # [B, F, K, M]
        B = tf.shape(h_hat)[0]
        w = self.precoder(tf.reshape(h_hat, [B * self.F, self.K, self.M]),
                          tf.repeat(snr_db, self.F))
        return tf.reshape(w, [B, self.F, self.M, self.K])

    def macs_per_sc(self, K):
        """Precodeur par SC + estimation (une fois par bande, ramenee par SC)."""
        N, d, M, L = self.F // self.K, self.est_d, self.M, self.est_layers
        fft = 4 * (N * N + self.F * N) * M                         # IDFT + DFT en matrices (borne haute)
        per_tok = (2 * M + 1) * d + L * (3 * d * d + 2 * N * d + d * d + 4 * d * d) + d
        est = K * (fft + N * per_tok)
        return self.precoder.macs_per_sc(K) + est // self.F
