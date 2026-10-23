"""Tests des modeles appris (sans entrainement). Lancer depuis freqctx/ :
    python -m tests.test_models
"""
import numpy as np
import tensorflow as tf

from models.per_sc import PerSCPrecoder


def main():
    tf.random.set_seed(0)
    N, K, M = 16, 4, 8
    h = tf.complex(tf.random.normal([N, K, M]), tf.random.normal([N, K, M])) / np.sqrt(2)
    snr = tf.random.uniform([N], 0, 25)
    perm = [2, 0, 3, 1]
    for kw in [{}, {"mixer": "mean"}, {"attn": "signed"}, {"attn": "linear"}, {"gram": True},
               {"attn": "signed", "gram": True}, {"attn": "signed", "tied": True},
               {"attn": "signed", "phase_norm": True}]:
        m = PerSCPrecoder(M, d=16, layers=2, **kw)
        w = m(h, snr)
        assert w.shape == (N, M, K)
        p = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 2]).numpy()
        assert np.allclose(p, 1.0, atol=1e-4), kw
        w_perm = m(tf.gather(h, perm, axis=1), snr)
        err = float(tf.reduce_max(tf.abs(w_perm - tf.gather(w, perm, axis=2))))
        assert err < 1e-5, (kw, err)
        if kw.get("phase_norm"):                       # invariance a la phase par utilisateur
            ph = tf.exp(tf.complex(0.0, tf.random.uniform([N, K, 1], 0, 6.28)))
            e2 = float(tf.reduce_max(tf.abs(m(h * ph, snr) - w)))
            assert e2 < 1e-4, e2
        print(f"[ok] {kw or 'base'} : forme, puissance, equivariance (ecart {err:.1e}), "
              f"{m.count_params()} params, {m.macs_per_sc(K)} MAC/SC")
    print("TOUS LES TESTS PASSENT")


if __name__ == "__main__":
    main()
