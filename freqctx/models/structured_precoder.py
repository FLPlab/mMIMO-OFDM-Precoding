"""Phase 16 -- variante STRUCTUREE du precodeur par paires (prix de la liberte, demande du
26 nov. 2026). Diagnostic phase 15 (FINDINGS.md, experiments/phase15_diag.py) : le reseau LIBRE
(models/gram_precoder.py) semble redecouvrir la forme optimale de Bjornson et al. 2014 (forme
Gram) C = (Lambda G + sigma^2 I)^-1 P^(1/2), avec (lambda, p) tires de WMMSE convergee et une
inversion de Newton-Schulz (X <- X(2I - A X), X0 = A^H/(||A||_1 ||A||_inf)) : 15 pas egalent le
reseau libre a 20 dB mais pas a 30 dB (fit EXACT, aucun entrainement de ce module-la).

Ici : la meme forme, mais ENTIEREMENT ENTRAINEE de bout en bout (pas de fit post-hoc) --
  - un petit transformer par PAIRES (memes blocs que models/gram_precoder.py : TriangularBlock,
    mix="triangular_g", G reinjecte comme arete), 2 couches DISTINCTES, d=16, FFN(hidden)=32,
    MEMES ENTREES NORMALISEES que la base GramPrecoder(inp="norm") (Re/Im/|g_norm|, snr, puissance
    relative des 2 utilisateurs -- g_norm = correlations, |g_norm| <= 1, PAS le G physique) ;
  - sur les tokens DIAGONAUX (j=j) seulement, deux scalaires POSITIFS par utilisateur (lambda_k,
    p_k), softplus ;
  - A = Lambda G_phys + sigma^2 I, G_phys = H H^H (canal PHYSIQUE, PAS normalise -- convention de
    precoders/classical.py et de experiments/phase15_diag.fit_structure_batch/run_freedom, deja
    validee : c'est dans cette normalisation-la, avec sigma^2 = bruit physique 10^(-SNR/10), que le
    fit au reseau libre et a WMMSE donne une faible erreur de reconstruction. Le petit transformer,
    lui, ne VOIT que des entrees normalisees comme la base (stabilite), mais la forme algebrique
    imposee reste dans l'echelle physique -- G_phys est reutilise pour deriver g_norm (pas de 2e
    produit de Gram : g_norm[j,k] = G_phys[j,k] / sqrt(G_phys[j,j] G_phys[k,k])) ;
  - inversion de A par Newton-Schulz DEROULEE a T pas, DIFFERENTIABLE (pas de division non lissee
    autre que X0, qui est un simple scalaire reciproque par echantillon -- meme convention que le
    rsqrt de la normalisation d'entree de la base, et que experiments/phase15_diag.newton_schulz) ;
  - C = X_T diag(sqrt(p)), W = H^H C, puis precoders.classical.normalize_power (meme convention
    que toute la base).

Equivariant aux permutations d'utilisateurs comme GramPrecoder (G, Lambda, NS commutent tous avec
une permutation simultanee des lignes/colonnes).

    from models.structured_precoder import StructuredPrecoder
"""
import keras
import tensorflow as tf

from models.gram_precoder import TriangularBlock, _cplx, _real
from precoders.classical import normalize_power


def newton_schulz(a, t_steps):
    """X <- X(2I - A X), X0 = A^H / (||A||_1 ||A||_inf), complex64, DIFFERENTIABLE (boucle Python
    deroulee en T matmuls, pas de tf.stop_gradient). Meme formule que
    experiments/phase15_diag.newton_schulz (complex128, EVALUATION SEULE, non reentraine ici),
    reecrite ici en complex64 pour l'entrainement."""
    k = tf.shape(a)[-1]
    n1 = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=-2), axis=-1)       # ||A||_1 : max somme colonne
    ninf = tf.reduce_max(tf.reduce_sum(tf.abs(a), axis=-1), axis=-1)     # ||A||_inf : max somme ligne
    scale = 1.0 / tf.maximum(n1 * ninf, 1e-12)
    x = tf.linalg.adjoint(a) * tf.cast(scale, a.dtype)[:, None, None]
    eye = tf.eye(k, dtype=a.dtype)[None]
    for _ in range(t_steps):
        x = tf.matmul(x, 2 * eye - tf.matmul(a, x))
    return x


class StructuredPrecoder(keras.Model):
    """num_bs_ant : M par defaut pour macs_per_sc seulement (M est lu sur l'entree, un modele pour
    tout M, comme GramPrecoder). ns_steps = T (pas de Newton-Schulz, DEROULES -- fixe a la
    construction, fait partie de l'architecture)."""

    def __init__(self, num_bs_ant, ns_steps=10, d=16, hidden=32, layers=2, lam_max=None, **kw):
        super().__init__(**kw)
        self.M, self.T, self.d, self.hidden, self.L = num_bs_ant, ns_steps, d, hidden, layers
        self.lam_max = lam_max                         # borne optionnelle sur lambda (stabilite NS a fort SNR)
        self.init = keras.layers.Dense(2 * d)
        self.blocks = [TriangularBlock(d, hidden, edge_g=True) for _ in range(layers)]
        self.read = keras.layers.Dense(2)              # -> (lambda_raw, p_raw), lu sur la diagonale SEULEMENT

    def lambda_p(self, g_phys, snr_db):
        """(lambda_k, p_k) > 0, [N, K] chacun. g_phys [N,K,K] complex64 = H H^H (PHYSIQUE). Les
        features du petit transformer sont NORMALISEES (g_norm, pdb), comme GramPrecoder(inp="norm")."""
        diag = tf.math.real(tf.linalg.diag_part(g_phys))                             # [N,K], = |h_k|^2
        pdb = tf.math.log(diag / tf.reduce_mean(diag, -1, keepdims=True)) / tf.math.log(10.0)
        rn = tf.cast(tf.sqrt(tf.maximum(diag, 1e-30)), g_phys.dtype)
        g_norm = g_phys / rn[:, :, None] / rn[:, None, :]                            # correlations, |.| <= 1
        K = tf.shape(g_phys)[1]
        snr_db = tf.cast(snr_db, tf.float32)
        eye = tf.broadcast_to(tf.eye(K)[None], tf.shape(tf.math.real(g_norm)))
        snr = tf.broadcast_to(tf.reshape(snr_db / 10.0, [-1, 1, 1]), tf.shape(eye))
        parts = [tf.math.real(g_norm), tf.math.imag(g_norm), tf.abs(g_norm), eye, snr,
                 tf.broadcast_to(pdb[:, :, None], tf.shape(eye)), tf.broadcast_to(pdb[:, None, :], tf.shape(eye))]
        f = tf.stack(parts, axis=-1)                                                 # [N,K,K,7]
        e = _cplx(self.init(f))
        for blk in self.blocks:
            e = blk(e, g_norm)
        diag_e = tf.einsum("nkkd->nkd", e)                                           # [N,K,d] complexe
        out = self.read(_real(diag_e))                                              # [N,K,2]
        lam = tf.nn.softplus(out[..., 0])
        if self.lam_max is not None:
            lam = tf.minimum(lam, self.lam_max)
        p = tf.nn.softplus(out[..., 1]) + 1e-6
        return lam, p

    def coeffs(self, h, snr_db):
        """C [N,K,K] AVANT normalisation de puissance finale (W = h^H C)."""
        h64 = tf.cast(h, tf.complex64)
        g_phys = tf.matmul(h64, h64, adjoint_b=True)                                 # [N,K,K], PHYSIQUE
        lam, p = self.lambda_p(g_phys, snr_db)
        nv = tf.cast(10.0 ** (-tf.cast(snr_db, tf.float32) / 10.0), tf.complex64)
        K = tf.shape(h)[1]
        eye = tf.eye(K, dtype=tf.complex64)[None]
        a = tf.cast(lam, tf.complex64)[:, :, None] * g_phys + nv[:, None, None] * eye
        x = newton_schulz(a, self.T)
        return x * tf.cast(tf.sqrt(p), tf.complex64)[:, None, :]

    def call(self, h, snr_db):
        """h [N,K,M] complex64, snr_db [N] -> W [N,M,K]."""
        c = self.coeffs(h, snr_db)
        w = tf.matmul(tf.cast(h, tf.complex64), c, adjoint_a=True)
        return normalize_power(w)

    def macs_per_sc(self, K, M=None):
        """MAC reelles/SC, convention experiments/phase15_macs.py :
          - G_phys = H H^H (reutilise pour g_norm, division negligeable) ;
          - petit transformer sur K*K paires (7 features, L couches TriangularBlock edge_g=True) ;
          - lecture (lambda,p) sur les K tokens diagonaux SEULEMENT ;
          - Newton-Schulz : T pas x 2 matmuls KxK (8 K^3 MAC/pas, comme phase15_diag) ;
          - W = H^H C."""
        M = self.M if M is None else M
        d, hd, L, T = self.d, self.hidden, self.L, self.T
        n_feat = 7
        per_edge = n_feat * 2 * d + L * (4 * d * hd + hd * 2 * d) + L * 2 * (2 * d + 3) * (2 * d)
        mac_g = 4 * K * K * M
        mac_edges = K * K * per_edge
        mac_tri = L * 4 * K ** 3 * d
        mac_read = K * (2 * d) * 2
        mac_ns = T * 8 * K ** 3
        mac_w = 4 * M * K * K
        return mac_g + mac_edges + mac_tri + mac_read + mac_ns + mac_w
