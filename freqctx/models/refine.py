"""Phase 15 -- raffinement itteratif appris (passe 2) pour le precodeur par paires 3GPP
(demande F. Leduc-Primeau, 26 nov. 2026). Constat (FINDINGS.md section Phase 15) : le precodeur
phase13 (GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm"))
a un plancher de fuite d'interference vers -45 dB (WMMSE descend a -79 dB) ; plus de couches
sature, entrainer a plus fort SNR deplace la precision au lieu d'en ajouter.

Idee : une 2e passe qui VOIT l'erreur de la 1re et la corrige, sans division, sans forme imposee.
  - Passe 1 = GramPrecoder existant, FIGE (poids charges, jamais entraine ici), donne C1 [N,K,K]
    (et W1 = h~^H C1, h~ le canal normalise par utilisateur comme la base inp="norm").
  - Passe 2 (ce module) = un second reseau, MEME FAMILLE DE BLOC que la base (TriangularBlock,
    mix="triangular_g" : G reinjecte comme arete, importe de models.gram_precoder SANS LE
    MODIFIER), 4 couches distinctes, d=16, FFN 32. Recoit par paire (j,k) les memes
    caracteristiques de G que la base (Re, Im, |.|, snr, puissance relative des 2 utilisateurs)
    PLUS celles du canal effectif normalise E = h~ W1n (W1n = W1 renormalise en puissance comme
    la sortie finale de la base ; E = G C1 / ||W1|| a un facteur d'echelle pres -- note dans la
    demande). Sort DELTA_C ; C2 = C1 + DELTA_C.
  - Couche de lecture (read) initialisee a zero (noyau ET biais) : au pas 0, DELTA_C = 0 partout,
    donc C2 = C1 EXACTEMENT -- le depart est la base seule (meme debit, verifie dans
    experiments/phase15_refine.py avant le vrai entrainement).
  - Renormalisation de puissance finale identique a la base (precoders.classical.normalize_power).

    from models.refine import RefineNet
"""
import keras
import tensorflow as tf

from models.gram_precoder import TriangularBlock, _cplx, _real
from precoders.classical import normalize_power


class RefineNet(keras.Model):
    """base : GramPrecoder deja construit et charge (poids figes -- ce module ne le re-cree pas,
    voir experiments/phase15_refine.py pour le gel des variables a l'entrainement)."""

    def __init__(self, base, d=16, hidden=32, layers=4, **kw):
        super().__init__(**kw)
        self.base = base
        self.d, self.hidden, self.L = d, hidden, layers
        self.init = keras.layers.Dense(2 * d)
        self.blocks = [TriangularBlock(d, hidden, edge_g=True) for _ in range(layers)]
        # lecture initialisee a zero (noyau + biais) : depart = base exactement (DELTA_C = 0)
        self.read = keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, h, snr_db, err_db=None):
        """h [N, K, M] complex64, snr_db [N] -> W [N, M, K] (C2 = C1 + DELTA_C, puis normalise_power)."""
        c2 = self.coeffs(h, snr_db, err_db)
        hp = self.base._norm(h)[0]
        return normalize_power(tf.matmul(hp, c2, adjoint_a=True))

    def delta(self, h, snr_db, err_db=None):
        """DELTA_C seul (pour verifier qu'il est nul a l'init)."""
        hp, pdb = self.base._norm(h)
        c1 = self.base.coeffs(h, snr_db, err_db)
        return self._delta_from(hp, pdb, c1, snr_db)

    def coeffs(self, h, snr_db, err_db=None):
        """C2 = C1 + DELTA_C [N, K, K] (avant normalisation de puissance finale)."""
        hp, pdb = self.base._norm(h)
        c1 = self.base.coeffs(h, snr_db, err_db)
        return c1 + self._delta_from(hp, pdb, c1, snr_db)

    def _delta_from(self, hp, pdb, c1, snr_db):
        K = tf.shape(hp)[1]
        g = tf.matmul(hp, hp, adjoint_b=True)                            # [N,K,K], |g|<=1 (base, inp="norm")
        w1 = tf.matmul(hp, c1, adjoint_a=True)                           # W1 = h~^H C1  [N,M,K]
        pw1 = tf.reduce_sum(tf.abs(w1) ** 2, axis=[-2, -1], keepdims=True)
        w1n = w1 / tf.cast(tf.sqrt(tf.maximum(pw1, 1e-30)), w1.dtype)    # W1 renormalise (comme la sortie base)
        e = tf.matmul(hp, w1n)                                           # canal effectif normalise [N,K,K]
        snr_db = tf.cast(snr_db, tf.float32)
        eye = tf.broadcast_to(tf.eye(K)[None], tf.shape(tf.math.real(g)))
        snr = tf.broadcast_to(tf.reshape(snr_db / 10.0, [-1, 1, 1]), tf.shape(eye))
        parts = [tf.math.real(g), tf.math.imag(g), tf.abs(g), eye, snr,
                 tf.broadcast_to(pdb[:, :, None], tf.shape(eye)), tf.broadcast_to(pdb[:, None, :], tf.shape(eye)),
                 tf.math.real(e), tf.math.imag(e), tf.abs(e)]
        f = tf.stack(parts, axis=-1)                                     # [N,K,K,10] : 7 (base) + 3 (E)
        state = _cplx(self.init(f))
        for blk in self.blocks:
            state = blk(state, g)
        return _cplx(self.read(_real(state)))[..., 0]

    def macs_per_sc(self, K, M=None):
        """MAC reelles/SC de la PASSE 2 SEULE (sans la base), meme convention que
        GramPrecoder.macs_per_sc : init Dense (10 features/arete), L x (triangulaire + FFN),
        lecture, plus le cout de construction de E (W1 = h~^H C1, E = h~ W1n -- G et C1 deja
        comptes dans la base)."""
        M = self.base.M if M is None else M
        d, hd, L = self.d, self.hidden, self.L
        n_feat = 10
        per_edge = n_feat * 2 * d + L * (4 * d * hd + hd * 2 * d) + 2 * d * 2
        per_edge += L * 2 * (2 * d + 3) * (2 * d)                        # projections a, b (edge_g=True)
        tri = L * 4 * K ** 3 * d
        e_build = 4 * M * K * K + 4 * K * K * K                          # W1 = h~^H C1, E = h~ W1n
        return K * K * per_edge + tri + e_build
