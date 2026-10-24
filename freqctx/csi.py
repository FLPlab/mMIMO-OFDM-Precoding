"""CSI a la BS : sondage uplink (pilotes Sionna) + estimation, et lissages classiques.

Sondage type SRS : un symbole OFDM, pilotes "kronecker" de Sionna, c.-a-d. un
peigne en frequence orthogonal entre les K utilisateurs (chaque utilisateur
n'est observe que sur 1 SC sur K). LS aux positions pilotes puis
interpolation en frequence (Sionna LSChannelEstimator).
"""
import numpy as np
import tensorflow as tf
from sionna.phy.channel import ApplyOFDMChannel
from sionna.phy.ofdm import (LMMSEInterpolator, LSChannelEstimator, ResourceGrid,
                             ResourceGridMapper)

from config import CSIConfig, SystemConfig


def estimate_freq_cov(gen, num_batches=32, batch_size=32):
    """Covariance frequentielle moyenne du scenario, E[h h^H] sur l'axe SC
    (moyenne sur tirages, utilisateurs, antennes ; slot 0). Statistique de
    long terme, comme dans le tutoriel Sionna sur l'interpolation LMMSE."""
    F = gen.cfg.num_sc
    cov = tf.zeros([F, F], tf.complex64)
    n = 0
    for _ in range(num_batches):
        h = tf.transpose(gen(batch_size)[:, 0], [0, 2, 3, 1])     # [B, K, M, F]
        h = tf.reshape(h, [-1, F])
        cov += tf.matmul(h, h, adjoint_a=True)
        n += h.shape[0]
    return tf.transpose(cov) / n                                 # E[h h^H], h colonne


class UplinkSounder:
    """interpolation : "lin" | "nn" (Sionna LS + interpolateur standard) ou
    "lmmse" (Sionna LMMSEInterpolator, necessite cov_freq [F, F])."""

    def __init__(self, cfg: SystemConfig, interpolation="lin", cov_freq=None):
        self.cfg = cfg
        self.rg = ResourceGrid(num_ofdm_symbols=1, fft_size=cfg.num_sc,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=cfg.num_users, num_streams_per_tx=1,
                               pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[0])
        self.mapper = ResourceGridMapper(self.rg)
        self.apply = ApplyOFDMChannel(add_awgn=True)
        # LMMSEInterpolator materialise des tenseurs [B, ..., F+1, F+1] (~3 Go a B=32) :
        # on estime par paquets de `chunk` tirages (le bruit est tire avant -> meme resultat).
        self.chunk = 4 if interpolation == "lmmse" else None
        if interpolation == "lmmse":
            interp = LMMSEInterpolator(self.rg.pilot_pattern,
                                       cov_mat_time=tf.ones([1, 1], tf.complex64),
                                       cov_mat_freq=cov_freq, order="f-t")
            self.estimator = LSChannelEstimator(self.rg, interpolator=interp)
        else:
            self.estimator = LSChannelEstimator(self.rg, interpolation_type=interpolation)

    def __call__(self, h, pilot_snr_db, return_err_var=False):
        """h : [B, F, K, M] canal vrai au slot de sondage -> h_hat meme forme.
        return_err_var : renvoie aussi la variance d'erreur que l'estimateur
        Sionna associe a chaque element (base de la reference robuste)."""
        B = tf.shape(h)[0]
        x = self.mapper(tf.zeros([B, self.cfg.num_users, 1, 0], tf.complex64))
        # [B, F, K, M] -> [B, 1, M, K, 1, 1, F]
        h_ul = tf.transpose(h, [0, 3, 2, 1])[:, None, :, :, None, None, :]
        no = tf.cast(10.0 ** (-pilot_snr_db / 10.0), tf.float32)
        y = self.apply(x, h_ul, no)
        step = self.chunk or y.shape[0]
        outs = [self.estimator(y[i:i + step], no) for i in range(0, y.shape[0], step)]
        h_ls = tf.concat([o[0] for o in outs], axis=0)
        h_hat = tf.transpose(h_ls[:, 0, :, :, 0, 0, :], [0, 3, 2, 1])
        if not return_err_var:
            return h_hat
        err = tf.concat([o[1] for o in outs], axis=0)
        return h_hat, tf.transpose(tf.math.real(tf.cast(err, tf.complex64))[:, 0, :, :, 0, 0, :],
                                   [0, 3, 2, 1])


def csi_at_bs(h, csi: CSIConfig, sounder: UplinkSounder):
    """h : [B, S, F, K, M]. Renvoie (h_bs, h_data) : CSI connu de la BS (sonde
    au slot 0) et canal vrai au slot de donnees `delay_slots`."""
    h_data = h[:, csi.delay_slots]
    if csi.estimator == "perfect":
        return h[:, 0], h_data
    return sounder(h[:, 0], csi.pilot_snr_db), h_data


def smooth_freq(h, width):
    """Moyenne glissante centree de largeur `width` (impair) sur l'axe SC,
    bords renormalises. h : [..., F, K, M]. Baseline "lissage classique"."""
    if width <= 1:
        return h
    half = width // 2
    F = tf.shape(h)[-3]
    c = tf.cumsum(tf.pad(h, [[0, 0]] * (len(h.shape) - 3) + [[1, 0], [0, 0], [0, 0]]), axis=-3)
    idx = tf.range(F)
    hi = tf.minimum(idx + half + 1, F)
    lo = tf.maximum(idx - half, 0)
    s = tf.gather(c, hi, axis=-3) - tf.gather(c, lo, axis=-3)
    n = tf.cast(hi - lo, h.dtype)[:, None, None]
    return s / n


def delay_domain_estimate(h_ls, pilot_snr_db, num_taps=None, kappa=None, guard=4):
    """Estimateur classique dans le domaine des retards (type "DFT-based",
    van de Beek et al. 1995 ; Edfors et al. 1998), a partir du LS du sondage.

    Utilisateur k : pilotes sur les SC k, k+K, ... (peigne kronecker de Sionna),
    N = F/K mesures. Le LS lineaire passe exactement par ces mesures, on les
    relit donc dans h_ls. IDFT a N points -> N retards (resolution 1/(F df)),
    puis on garde :
        num_taps=L : les L premiers retards (+ `guard` retards "negatifs"
                     pour la fuite du premier trajet) -- troncature fixe
        kappa=c    : les retards dont la puissance depasse c fois la variance
                     du bruit par retard (seuil adaptatif, la BS connait N0)
    puis DFT vers les F SC.  h_ls : [..., F, K, M] -> meme forme."""
    F, K = h_ls.shape[-3], h_ls.shape[-2]
    N = F // K
    t = np.arange(N)
    t_signed = np.where(t < N - guard, t, t - N)                # retards "negatifs" en fin
    sigma2_tap = (10.0 ** (-pilot_snr_db / 10.0) / K) / N        # bruit LS (|p|^2 = K) par retard
    out = []
    for k in range(K):
        f_k = k + K * np.arange(N)                               # SC pilotes de l'utilisateur k
        y = tf.gather(h_ls[..., k, :], f_k, axis=-2)             # [..., N, M]
        idft = np.exp(2j * np.pi * np.outer(t_signed, f_k) / F) / N          # [N, N]
        g = tf.einsum("tn,...nm->...tm", tf.constant(idft, tf.complex64), y)  # [..., N, M]
        if num_taps is not None:
            keep = ((t_signed >= 0) & (t_signed < num_taps)) | (t_signed < 0)
            g = g * tf.constant(keep.astype(np.complex64))[:, None]
        if kappa is not None:
            g = g * tf.cast(tf.abs(g) ** 2 > kappa * sigma2_tap, g.dtype)
        dft = np.exp(-2j * np.pi * np.outer(np.arange(F), t_signed) / F)       # [F, N]
        out.append(tf.einsum("ft,...tm->...fm", tf.constant(dft, tf.complex64), g))
    return tf.stack(out, axis=-2)                                # [..., F, K, M]


def genie_lmmse(h_ls, h_true, cir, frequencies, pilot_snr_db, t_idx=0):
    """Borne "genie" : LMMSE par lien avec la covariance frequentielle EXACTE
    du lien, R(f1, f2) = sum_l p_l exp(-j 2 pi (f1 - f2) tau_l), construite a
    partir des retards et puissances de ses trajets (CIR Sionna du slot 0),
    a l'echelle de l'energie reelle du lien. Irrealisable en pratique : c'est
    le plafond d'une estimation lineaire adaptee a chaque lien.
    h_ls, h_true : [B, F, K, M] ; cir = (a, tau) du ChannelGenerator."""
    a, tau = cir
    a = a[..., t_idx:t_idx + 1]                                     # instant du sondage
    F, K = h_ls.shape[1], h_ls.shape[2]
    N = F // K
    sigma2 = 10.0 ** (-pilot_snr_db / 10.0) / K                    # bruit LS (|p|^2 = K)
    pw = tf.reduce_mean(tf.abs(a[:, 0, :, :, 0, :, 0]) ** 2, axis=1)  # [B, K, L] puissance par trajet
    tk = tau[:, 0, :, :]                                            # [B, K, L]
    fr = tf.cast(frequencies, tf.float32)
    out = []
    for k in range(K):
        e = tf.exp(tf.complex(0.0, -2.0 * np.pi * fr[None, :, None] * tk[:, k, None, :]))   # [B, F, L]
        scale = tf.reduce_mean(tf.abs(h_true[:, :, k, :]) ** 2, axis=[1, 2]) \
            / tf.maximum(tf.reduce_sum(pw[:, k], -1), 1e-12)
        p = tf.cast(pw[:, k] * scale[:, None], tf.complex64)                               # [B, L]
        R = tf.matmul(e * p[:, None, :], e, adjoint_b=True)                                # [B, F, F]
        idx = k + K * np.arange(N)
        R_hy = tf.gather(R, idx, axis=2)                                                   # [B, F, N]
        R_yy = tf.gather(R_hy, idx, axis=1) + tf.cast(sigma2, tf.complex64) * tf.eye(N, dtype=tf.complex64)
        y = tf.gather(h_ls[:, :, k, :], idx, axis=1)                                       # [B, N, M]
        out.append(tf.matmul(R_hy, tf.linalg.solve(R_yy, y)))                             # [B, F, M]
    return tf.stack(out, axis=2)


def pdp_lmmse(h_ls, pdp_raw, pilot_snr_db, guard=4):
    """Contrepartie CLASSIQUE de l'historique : LMMSE par lien dont la covariance
    est construite a partir du profil des trajets mesure sur les sondages
    precedents, sur la grille des retards : R(f1, f2) = sum_t p_t exp(-j 2 pi
    (f1 - f2) t / F), avec p_t = max(pdp_raw_t - bruit par retard, 0).
    h_ls [B, F, K, M] ; pdp_raw [B, K, N] (puissance moyenne par retard, bruit
    compris, meme convention que DelayEstimator)."""
    F, K = h_ls.shape[1], h_ls.shape[2]
    N = F // K
    t = np.arange(N)
    t_signed = np.where(t < N - guard, t, t - N)
    sigma2 = 10.0 ** (-pilot_snr_db / 10.0) / K                    # bruit LS par pilote
    p = tf.nn.relu(tf.cast(pdp_raw, tf.float32) - sigma2 / N)       # puissance utile par retard
    e = tf.constant(np.exp(-2j * np.pi * np.outer(np.arange(F), t_signed) / F), tf.complex64)  # [F, N]
    out = []
    for k in range(K):
        pk = tf.cast(p[:, k], tf.complex64)                         # [B, N]
        R = tf.matmul(e[None] * pk[:, None, :], tf.tile(e[None], [tf.shape(pk)[0], 1, 1]),
                      adjoint_b=True)                               # [B, F, F]
        idx = k + K * np.arange(N)
        R_hy = tf.gather(R, idx, axis=2)
        R_yy = tf.gather(R_hy, idx, axis=1) + tf.cast(sigma2, tf.complex64) * tf.eye(N, dtype=tf.complex64)
        y = tf.gather(h_ls[:, :, k, :], idx, axis=1)
        out.append(tf.matmul(R_hy, tf.linalg.solve(R_yy, y)))
    return tf.stack(out, axis=2)
