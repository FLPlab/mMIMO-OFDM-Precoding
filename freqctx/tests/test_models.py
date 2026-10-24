"""Tests des modeles appris (sans entrainement). Lancer depuis freqctx/ :
    python -m tests.test_models
"""
import numpy as np
import tensorflow as tf

from models.per_sc import PerSCPrecoder
from train import DropPool, WindowedPool, evaluate
from models.delay_domain import DelayDomainPrecoder


def main():
    tf.random.set_seed(0)
    N, K, M = 16, 4, 8
    h = tf.complex(tf.random.normal([N, K, M]), tf.random.normal([N, K, M])) / np.sqrt(2)
    snr = tf.random.uniform([N], 0, 25)
    perm = [2, 0, 3, 1]
    for kw in [{}, {"mixer": "mean"}, {"attn": "signed"}, {"attn": "linear"}, {"gram": True},
               {"attn": "linear", "snr_every": True}, {"attn": "linear", "rzf_init": True},
               {"attn": "linear", "mrt_init": True},
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
    # fenetre (phase 3) : equivariance, et la SC centrale doit compter
    hw = tf.complex(tf.random.normal([N, 5, K, M]), tf.random.normal([N, 5, K, M])) / np.sqrt(2)
    m = PerSCPrecoder(M, d=16, layers=2, attn="linear", window=5)
    w = m(hw, snr)
    err = float(tf.reduce_max(tf.abs(m(tf.gather(hw, perm, axis=2), snr) - tf.gather(w, perm, axis=2))))
    assert w.shape == (N, M, K) and err < 1e-5, err
    print(f"[ok] window=5 : forme, equivariance (ecart {err:.1e}), {m.count_params()} params, "
          f"{m.macs_per_sc(K)} MAC/SC")
    m = PerSCPrecoder(M, d=16, layers=2, attn="linear", window=5, freq="attention")
    w = m(hw, snr)
    err = float(tf.reduce_max(tf.abs(m(tf.gather(hw, perm, axis=2), snr) - tf.gather(w, perm, axis=2))))
    a = m.last_freq_attention
    assert w.shape == (N, M, K) and err < 1e-5, err
    assert np.allclose(tf.reduce_sum(a, axis=2).numpy(), 1.0, atol=1e-5)
    print(f"[ok] window=5, attention en frequence : equivariance (ecart {err:.1e}), poids somme 1, "
          f"{m.count_params()} params, {m.macs_per_sc(K)} MAC/SC")
    m = PerSCPrecoder(M, d=16, layers=2, attn="linear", window=5, freq="attention", power_head=True)
    w = m(hw, snr)
    err = float(tf.reduce_max(tf.abs(m(tf.gather(hw, perm, axis=2), snr) - tf.gather(w, perm, axis=2))))
    assert err < 1e-5 and np.allclose(tf.reduce_sum(tf.abs(w) ** 2, [1, 2]).numpy(), 1.0, atol=1e-4)
    print(f"[ok] tete de puissance : equivariance (ecart {err:.1e}), puissance 1, "
          f"{m.count_params()} params, {m.macs_per_sc(K)} MAC/SC")
    from precoders.classical import rzf as rzf_ref
    m = PerSCPrecoder(M, d=16, layers=2, attn="linear", rzf_init=True)
    m(h, snr)
    m.head.bias.assign(tf.zeros_like(m.head.bias))
    hn = h / tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(h) ** 2, axis=[1, 2], keepdims=True)), h.dtype)
    err = float(tf.reduce_max(tf.abs(m(h, snr) - rzf_ref(hn, tf.reshape(10.0 ** (-snr / 10.0), [-1, 1, 1])))))
    assert err < 1e-5, err
    print(f"[ok] rzf_init : demarre exactement sur RZF (ecart {err:.1e})")
    test_windowed_pool()
    test_delay_domain()
    print("TOUS LES TESTS PASSENT")


def test_windowed_pool():
    h_in = tf.complex(tf.random.normal([3, 10, 4, 8]), 0.0)
    h_t = h_in * 2
    p = WindowedPool(h_in, h_t, 5)
    x = p._windows(tf.constant([1, 2]), tf.constant([0, 9]))
    assert x.shape == (2, 5, 4, 8)
    assert tf.reduce_all(x[0, 2] == h_in[1, 0]) and tf.reduce_all(x[0, 0] == h_in[1, 0])  # bord g.
    assert tf.reduce_all(x[1, 1] == h_in[2, 8]) and tf.reduce_all(x[1, 4] == h_in[2, 9])  # bord d.
    xs, ys = next(p.chunks(2))
    assert xs.shape == (20, 5, 4, 8) and tf.reduce_all(ys[3] == h_t[0, 3])
    xi, yi = p.sample(6)
    assert xi.shape == (6, 5, 4, 8) and yi.shape == (6, 4, 8)
    print("[ok] WindowedPool : centre, bords bornes, paquets, tirage")



def test_delay_domain():
    F, K, M, B = 48, 4, 8, 3
    h = tf.complex(tf.random.normal([B, F, K, M]), tf.random.normal([B, F, K, M]))
    m = DelayDomainPrecoder(F, K, M, d=16, layers=2, tied=True, attn="linear")
    w = m(h, tf.fill([B], 10.0))
    assert w.shape == (B, F, M, K)
    assert np.allclose(tf.reduce_sum(tf.abs(w) ** 2, axis=[2, 3]).numpy(), 1.0, atol=1e-4)
    est = m.estimator
    est.gain.bias.assign([30.0])                      # gains = 1 partout
    est.gain.kernel.assign(tf.zeros_like(est.gain.kernel))
    h_hat = est(h)
    for k in range(K):                                # aux SC pilotes : exactement le LS
        idx = k + K * np.arange(F // K)
        err = float(tf.reduce_max(tf.abs(tf.gather(h_hat[:, :, k], idx, axis=1)
                                          - tf.gather(h[:, :, k], idx, axis=1))))
        assert err < 1e-4, (k, err)
    ma = DelayDomainPrecoder(F, K, M, angle_mask=True, d=16, layers=2, tied=True, attn="linear")
    ma(h, tf.fill([B], 10.0))
    ea = ma.estimator
    ea.gain.bias.assign(tf.fill([M], 30.0))
    ea.gain.kernel.assign(tf.zeros_like(ea.gain.kernel))
    ha = ea(h)                                         # gains = 1 : la DFT angulaire est unitaire
    for k in range(K):
        idx = k + K * np.arange(F // K)
        assert float(tf.reduce_max(tf.abs(tf.gather(ha[:, :, k], idx, axis=1)
                                          - tf.gather(h[:, :, k], idx, axis=1)))) < 1e-4
    mb = DelayDomainPrecoder(F, K, M, banded=True, d=16, layers=2, tied=True, attn="linear")
    mb(h, tf.fill([B], 10.0))
    eb = mb.estimator
    eb.gain.bias.assign([30.0, 0, 0, 0, 0, 0])         # centre 1, voisins 0 : LS exact
    hb = eb(h)
    for k in range(K):
        idx = k + K * np.arange(F // K)
        assert float(tf.reduce_max(tf.abs(tf.gather(hb[:, :, k], idx, axis=1)
                                          - tf.gather(h[:, :, k], idx, axis=1)))) < 1e-4
    mo = DelayDomainPrecoder(F, K, M, banded=True, oversample=2, d=16, layers=2, tied=True,
                             attn="linear")
    mo(h, tf.fill([B], 10.0))
    eo = mo.estimator
    eo.gain.bias.assign([30.0, 0, 0, 0, 0, 0])         # grille fine, gains = 1 : LS exact
    ho = eo(h)
    for k in range(K):
        idx = k + K * np.arange(F // K)
        assert float(tf.reduce_max(tf.abs(tf.gather(ho[:, :, k], idx, axis=1)
                                          - tf.gather(h[:, :, k], idx, axis=1)))) < 1e-3
    r = evaluate(m, DropPool(h, h), [10])
    print(f"[ok] domaine des retards : forme, puissance, gain=1 -> LS exact aux pilotes, "
          f"masque retard x angle, filtre a bande et grille fine idem, evaluation par tirage ({r[0]:.2f}), {m.count_params()} params")


if __name__ == "__main__":
    main()
