"""Tests de coherence de la chaine phase 0. Lancer depuis freqctx/ :
    python -m tests.test_sanity
"""
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, csi_at_bs, smooth_freq
from metrics import sinr, sum_rate
from precoders.classical import mrt, rzf, wmmse, zf


def nmse_db(h_hat, h):
    return 10 * np.log10(float(tf.reduce_mean(tf.abs(h_hat - h) ** 2) / tf.reduce_mean(tf.abs(h) ** 2)))


def main():
    sionna_config.seed = 0
    cfg = SystemConfig()
    gen = ChannelGenerator(cfg)
    h = gen(16)
    K, M, F = cfg.num_users, cfg.num_bs_ant, cfg.num_sc
    assert h.shape == (16, cfg.num_slots, F, K, M), h.shape
    e = float(tf.reduce_mean(tf.abs(h) ** 2))
    assert abs(e - 1.0) < 0.05, f"energie canal {e}"
    print(f"[ok] canal {h.shape}, energie moyenne {e:.3f}, bande {cfg.bandwidth/1e6:.2f} MHz")

    h0 = h[:, 0]
    nv = 10 ** (-10 / 10)
    for name, fn in [("MRT", mrt), ("ZF", zf), ("RZF", rzf), ("WMMSE", wmmse)]:
        w = fn(h0, nv)
        p = tf.reduce_sum(tf.abs(w) ** 2, axis=[-2, -1])
        assert np.allclose(p.numpy(), 1.0, atol=1e-4), name
    print("[ok] puissance ||W||_F = 1 pour MRT/ZF/RZF/WMMSE")

    leak = tf.abs(tf.matmul(h0, zf(h0))) ** 2
    off = float(tf.reduce_max(leak - tf.linalg.diag(tf.linalg.diag_part(leak))))
    assert off < 1e-6, off
    print(f"[ok] ZF annule l'interference (max fuite {off:.1e})")

    hi = 10 ** (-40 / 10)
    d = float(tf.reduce_max(tf.abs(rzf(h0, hi) - zf(h0))))
    assert d < 1e-2, d
    print(f"[ok] RZF -> ZF a haut SNR (ecart max {d:.1e})")

    for snr_db in [0, 10, 20]:
        nv = 10 ** (-snr_db / 10)
        r = {n: float(sum_rate(h0, fn(h0, nv), nv)) for n, fn in
             [("MRT", mrt), ("ZF", zf), ("RZF", rzf), ("WMMSE", wmmse)]}
        assert r["WMMSE"] >= r["RZF"] - 1e-3 and r["RZF"] >= r["ZF"] - 1e-3, r
        print(f"[ok] {snr_db:2d} dB : " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()))

    sounder = UplinkSounder(cfg)
    prev = 0.0
    for p in [0, 10, 20, 40]:
        h_bs, _ = csi_at_bs(h, CSIConfig("ls_lin", p, 0), sounder)
        n = nmse_db(h_bs, h0)
        print(f"[..] LS+lin, pilote {p:2d} dB : NMSE {n:6.2f} dB "
              f"| lisse W=5 : {nmse_db(smooth_freq(h_bs, 5), h0):6.2f} dB")
        assert n < prev, "la NMSE doit baisser avec le SNR pilote"
        prev = n
    h_bs, h_data = csi_at_bs(h, CSIConfig("perfect", 20, 0), sounder)
    assert float(tf.reduce_max(tf.abs(h_bs - h_data))) == 0.0
    print("[ok] CSI parfait = canal vrai")

    for d in range(1, cfg.num_slots):
        print(f"[..] vieillissement {d} slot(s) : NMSE {nmse_db(h[:, 0], h[:, d]):6.2f} dB")
    print("TOUS LES TESTS PASSENT")


if __name__ == "__main__":
    main()
