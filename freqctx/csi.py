"""CSI a la BS : sondage uplink (pilotes Sionna) + estimation, et lissages classiques.

Sondage type SRS : un symbole OFDM, pilotes "kronecker" de Sionna, c.-a-d. un
peigne en frequence orthogonal entre les K utilisateurs (chaque utilisateur
n'est observe que sur 1 SC sur K). LS aux positions pilotes puis
interpolation en frequence (Sionna LSChannelEstimator).
"""
import tensorflow as tf
from sionna.phy.channel import ApplyOFDMChannel
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid, ResourceGridMapper

from config import CSIConfig, SystemConfig


class UplinkSounder:
    def __init__(self, cfg: SystemConfig, interpolation="lin"):
        self.cfg = cfg
        self.rg = ResourceGrid(num_ofdm_symbols=1, fft_size=cfg.num_sc,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=cfg.num_users, num_streams_per_tx=1,
                               pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[0])
        self.mapper = ResourceGridMapper(self.rg)
        self.apply = ApplyOFDMChannel(add_awgn=True)
        self.estimator = LSChannelEstimator(self.rg, interpolation_type=interpolation)

    def __call__(self, h, pilot_snr_db):
        """h : [B, F, K, M] canal vrai au slot de sondage -> h_hat meme forme."""
        B = tf.shape(h)[0]
        x = self.mapper(tf.zeros([B, self.cfg.num_users, 1, 0], tf.complex64))
        # [B, F, K, M] -> [B, 1, M, K, 1, 1, F]
        h_ul = tf.transpose(h, [0, 3, 2, 1])[:, None, :, :, None, None, :]
        no = tf.cast(10.0 ** (-pilot_snr_db / 10.0), tf.float32)
        y = self.apply(x, h_ul, no)
        h_ls, _ = self.estimator(y, no)
        return tf.transpose(h_ls[:, 0, :, :, 0, 0, :], [0, 3, 2, 1])


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
