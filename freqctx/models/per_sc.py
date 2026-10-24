"""Transformer minimal par sous-porteuse (phase 1).

Une SC = un ensemble de K tokens, un par utilisateur :
    token k = [Re h_k, Im h_k, snr_db/10]          (2M+1 valeurs)
    avec window=W (phase 3) : h_k des W SC centrees sur la SC a precoder ;
    la sortie reste celle de la SC centrale. `freq` = comment on les combine :
      "concat"    : concatenes dans le token (2MW+1 valeurs) -> filtre FIXE appris
      "attention" : la SC centrale interroge ses W voisines (softmax sur la
                    fenetre, position relative apprise) -> moyenne ponderee
                    ADAPTATIVE (depend du canal), ajoutee a l'embedding de la
                    SC centrale. Softmax voulu ici : en frequence on lisse
                    (moyenne positive), on ne soustrait pas.
    -> Dense(D) -> L x [melange entre utilisateurs + FFN] -> Dense(2M) = w_k
    -> W [M, K], ||W||_F = 1.
Pas d'encodage de position : permuter les utilisateurs permute les sorties.
Le meme modele (memes poids) est applique a chaque SC independamment.

`mixer` = la seule operation qui fait communiquer les utilisateurs :
    "attention" : self-attention softmax (le transformer)
    "mean"      : moyenne uniforme des valeurs (DeepSets) -- ablation qui
                  isole l'apport de l'attention elle-meme

`attn` = comment les scores s = q.k / sqrt(dh) deviennent des poids :
    "softmax" : poids >= 0, somme 1 (moyenne ponderee : ne sait pas soustraire)
    "signed"  : softmax(s) * tanh(s) -- coefficients signes, donc capacite de
                SOUSTRAIRE un utilisateur ; repris de SignedGateMHA
                (mimo_precoding/precoders/signed_attention.py)
    "linear"  : s / K directement -- ni exp, ni division, ni tanh (FPGA) ;
                empiler ces couches = polynomes des interactions entre
                utilisateurs, la forme d'une inversion approchee (Neumann)

Options (phase 1b/1c, 4), chacune testable separement :
    power_head=True : la sortie separe direction et puissance : chaque w_k est
                      ramene a une norme 1, puis multiplie par sqrt(p_k), avec
                      p = softmax des K scores de puissance (somme 1). Le modele
                      choisit explicitement la repartition de puissance entre
                      utilisateurs (ce que WMMSE fait au faible SNR), sans inversion.
    tied=True       : UN seul bloc applique L fois (iterations), parametres /L
    phase_norm=True : chaque h_k est tourne pour que sa composante la plus
                      forte soit reelle positive. Exact : le SINR ne depend que
                      de |h_k w_j|, donc W optimal est invariant a la phase
                      propre de chaque utilisateur -> variable inutile retiree.
    snr_every=True  : un vecteur appris a partir du SNR est ajoute aux tokens
                      avant CHAQUE iteration du bloc (pas seulement a l'entree)
    rzf_init=True   : W = RZF(h normalise) + correction apprise (tete initialisee
                      a zero : le modele demarre exactement sur RZF). Reintroduit
                      une inversion K x K par SC, comme RZF.
    gram=True   : le score de la paire (k, j) recoit un biais appris
                  f(Re G_kj, Im G_kj, |G_kj|), G = H H^H / M (produits
                  scalaires entre canaux, la premiere etape de RZF). Biais
                  par paire -> l'equivariance est conservee.
"""
import keras
import tensorflow as tf

from precoders.classical import normalize_power, rzf


class Block(keras.layers.Layer):
    def __init__(self, d, heads, mixer, ffn_mult=2, attn="softmax", gram=False, **kw):
        super().__init__(**kw)
        self.d, self.heads, self.mixer, self.attn = d, heads, mixer, attn
        self.edge = keras.layers.Dense(heads) if gram else None
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.qkv = keras.layers.Dense(3 * d if mixer == "attention" else d)
        self.out = keras.layers.Dense(d)
        self.ffn1 = keras.layers.Dense(ffn_mult * d, activation="gelu")
        self.ffn2 = keras.layers.Dense(d)

    def call(self, x, edge=None):              # x [N, K, D], edge [N, K, K, 3]
        y = self.qkv(self.ln1(x))
        if self.mixer == "attention":
            N, K = tf.shape(x)[0], tf.shape(x)[1]
            dh = self.d // self.heads
            q, k, v = [tf.transpose(tf.reshape(t, [N, K, self.heads, dh]), [0, 2, 1, 3])
                       for t in tf.split(y, 3, axis=-1)]
            s = tf.matmul(q, k, transpose_b=True) / dh ** 0.5      # [N, heads, K, K]
            if self.edge is not None:
                s = s + tf.transpose(self.edge(edge), [0, 3, 1, 2])
            if self.attn == "linear":
                a = s / tf.cast(K, s.dtype)
            else:
                a = tf.nn.softmax(s, axis=-1)
                if self.attn == "signed":
                    a = a * tf.tanh(s)
            self.last_attention = a                         # [N, heads, K, K], pour l'analyse
            m = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [N, K, self.d])
        else:
            m = tf.broadcast_to(tf.reduce_mean(y, axis=1, keepdims=True), tf.shape(y))
        x = x + self.out(m)
        return x + self.ffn2(self.ffn1(self.ln2(x)))


class PerSCPrecoder(keras.Model):
    def __init__(self, num_bs_ant, d=32, layers=2, heads=2, mixer="attention",
                 attn="softmax", gram=False, tied=False, phase_norm=False, window=1,
                 freq="concat", power_head=False, snr_every=False, rzf_init=False, **kw):
        super().__init__(**kw)
        self.window, self.freq, self.power_head = window, freq, power_head
        self.snr_every, self.rzf_init = snr_every, rzf_init
        if snr_every:
            self.snr_emb = keras.layers.Dense(d)
        if window > 1 and freq == "attention":
            self.f_q = keras.layers.Dense(d)
            self.f_kv = keras.layers.Dense(2 * d)
            self.f_pos = keras.layers.Embedding(window, d)   # position relative dans la fenetre
            self.f_out = keras.layers.Dense(d)
        self.M, self.d, self.L, self.heads, self.mixer = num_bs_ant, d, layers, heads, mixer
        self.attn, self.gram, self.tied, self.phase_norm = attn, gram, tied, phase_norm
        self.embed = keras.layers.Dense(d)
        n_blocks = 1 if tied else layers
        self.blocks = [Block(d, heads, mixer, attn=attn, gram=gram) for _ in range(n_blocks)]
        self.head = keras.layers.Dense(2 * num_bs_ant + (1 if power_head else 0),
                                       kernel_initializer="zeros" if rzf_init else "glorot_uniform")

    def call(self, h, snr_db):
        """h [N, K, M] (ou [N, W, K, M] avec window) complex64, snr_db [N]
        -> W [N, M, K] complex64 (pour la SC centrale)."""
        h_win = None
        if self.window > 1 and self.freq == "attention":
            h_win = tf.transpose(h, [0, 2, 1, 3])                # [N, K, W, M]
            h = h[:, self.window // 2]
            K = tf.shape(h)[1]
        elif self.window > 1:
            feats = tf.transpose(h, [0, 2, 1, 3])                # [N, K, W, M]
            K = tf.shape(h)[2]
            feats = tf.reshape(tf.concat([tf.math.real(feats), tf.math.imag(feats)], -1),
                               [tf.shape(h)[0], K, 2 * self.window * self.M])
            h = h[:, self.window // 2]                           # SC centrale
        else:
            K = tf.shape(h)[1]
        if self.phase_norm:
            ref = tf.gather(h, tf.argmax(tf.abs(h), axis=-1), axis=-1, batch_dims=2)  # [N, K]
            h = h * tf.math.conj(ref / tf.cast(tf.abs(ref), ref.dtype))[..., None]
        snr = tf.tile(tf.reshape(snr_db / 10.0, [-1, 1, 1]), [1, K, 1])
        if self.window == 1 or h_win is not None:
            feats = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        x = self.embed(tf.concat([feats, snr], axis=-1))
        if h_win is not None:
            x = x + self._freq_attention(h_win, feats)
        edge = None
        if self.gram:
            g = tf.matmul(h, h, adjoint_b=True) / self.M         # G_kj = h_k h_j^H / M
            edge = tf.stack([tf.math.real(g), tf.math.imag(g), tf.abs(g)], axis=-1)
        se = self.snr_emb(snr) if self.snr_every else 0.0
        for i in range(self.L):
            x = self.blocks[0 if self.tied else i](x + se, edge)
        o = self.head(x)
        w = tf.complex(o[..., :self.M], o[..., self.M:2 * self.M])   # [N, K, M]
        if self.rzf_init:                                          # depart sur RZF, correction apprise
            e = tf.reduce_mean(tf.abs(h) ** 2, axis=[1, 2], keepdims=True)
            hn = h / tf.cast(tf.sqrt(e + 1e-12), h.dtype)              # echelle du canal normalisee
            nv = tf.reshape(10.0 ** (-snr_db / 10.0), [-1, 1, 1])
            w = w + tf.transpose(rzf(hn, nv), [0, 2, 1])
        if self.power_head:
            w = w / tf.cast(tf.norm(w, axis=-1, keepdims=True), w.dtype)   # directions, norme 1
            p = tf.nn.softmax(o[..., 2 * self.M], axis=-1)                   # [N, K], somme 1
            w = w * tf.cast(tf.sqrt(p), w.dtype)[..., None]
        return normalize_power(tf.transpose(w, [0, 2, 1]))

    def _freq_attention(self, h_win, center_feats):
        """h_win [N, K, W, M], center_feats [N, K, 2M] -> [N, K, D]. Une requete
        (SC centrale) par utilisateur, W cles/valeurs (les voisines)."""
        N, K = tf.shape(h_win)[0], tf.shape(h_win)[1]
        W, d, nh = self.window, self.d, self.heads
        dh = d // nh
        fw = tf.concat([tf.math.real(h_win), tf.math.imag(h_win)], -1)          # [N, K, W, 2M]
        kv = self.f_kv(fw) + tf.tile(self.f_pos(tf.range(W)), [1, 2])[None, None]
        k, v = tf.split(kv, 2, axis=-1)                                         # [N, K, W, d]
        q = self.f_q(center_feats)                                              # [N, K, d]
        k = tf.reshape(k, [N, K, W, nh, dh])
        v = tf.reshape(v, [N, K, W, nh, dh])
        q = tf.reshape(q, [N, K, 1, nh, dh])
        a = tf.nn.softmax(tf.reduce_sum(q * k, -1) / dh ** 0.5, axis=2)         # [N, K, W, nh]
        self.last_freq_attention = a
        m = tf.reshape(tf.reduce_sum(a[..., None] * v, axis=2), [N, K, d])
        return self.f_out(m)

    def macs_per_sc(self, K):
        """Multiplications-accumulations reelles par SC (inference)."""
        d, M = self.d, self.M
        if self.window > 1 and self.freq == "attention":
            # projections cle/valeur calculees UNE fois par SC et reutilisees par
            # toutes les fenetres qui la contiennent (fenetre glissante)
            per_token = (2 * M + 1) * d + d * 2 * M                  # embed + head
            per_token += 2 * M * 2 * d + 2 * M * d + 2 * self.window * d + d * d
        else:
            per_token = (2 * M * self.window + 1) * d + d * 2 * M   # embed + head
        mix = 3 * d * d + 2 * K * d if self.mixer == "attention" else d * d
        per_token += self.L * (mix + d * d + 2 * d * 2 * d)  # melange + out + FFN (x2)
        extra = 4 * K * self.M if self.phase_norm else 0
        if self.gram:                                        # G (K^2 M complexes) + biais
            extra += 4 * K * K * self.M + self.L * K * K * 3 * self.heads
        if self.power_head:
            extra += K * (d + 4 * M + 2)                       # score + normalisation + racine
        if self.snr_every:
            extra += self.L * K * d
        if self.rzf_init:
            extra += 4 * (K * K * M + K ** 3 + M * K * K + M * K)   # RZF
        return K * per_token + extra


def rzf_macs_per_sc(M, K):
    """RZF en MAC reelles (1 MAC complexe = 4 reelles) : Gram K^2 M,
    inversion ~K^3, produit H^H (.) M K^2, normalisation M K."""
    return 4 * (K * K * M + K ** 3 + M * K * K + M * K)
