"""Precodeurs de reference, par SC. H [..., K, M] -> W [..., M, K], ||W||_F = 1."""
import tensorflow as tf


def normalize_power(w):
    p = tf.reduce_sum(tf.abs(w) ** 2, axis=[-2, -1], keepdims=True)
    return w / tf.cast(tf.sqrt(p), w.dtype)


def mrt(h, noise_var=None):
    return normalize_power(tf.linalg.adjoint(h))


def rzf(h, noise_var, alpha=None):
    """W = H^H (H H^H + alpha I)^-1, alpha = K sigma^2 / P par defaut
    (regularisation optimale en canal i.i.d., Peel et al. 2005)."""
    K = h.shape[-2]
    if alpha is None:
        alpha = K * noise_var
    gram = tf.matmul(h, h, adjoint_b=True)
    reg = tf.cast(alpha, h.dtype) * tf.eye(K, dtype=h.dtype)
    return normalize_power(tf.matmul(h, tf.linalg.inv(gram + reg), adjoint_a=True))


def zf(h, noise_var=None):
    return rzf(h, None, alpha=0.0)


def wmmse(h, noise_var, num_iter=30):
    """WMMSE somme-debit, puissance totale par SC (Shi et al. 2011), avec la
    mise a jour a multiplicateur ferme mu = sigma^2 * sum(omega |g|^2)
    (Christensen et al. 2008) suivie d'une renormalisation. Initialise par
    RZF, calcule en complex128 (conditionnement a haut SNR)."""
    h64 = tf.cast(h, tf.complex128)
    nv = tf.cast(noise_var, tf.float64)
    M = h.shape[-1]
    w = tf.cast(rzf(h, noise_var), tf.complex128)
    eye = tf.eye(M, dtype=tf.complex128)
    for _ in range(num_iter):
        hw = tf.matmul(h64, w)                                   # [..., K, K]
        sig = tf.linalg.diag_part(hw)                            # h_k w_k
        tot = tf.reduce_sum(tf.abs(hw) ** 2, axis=-1) + nv       # [..., K]
        g = tf.math.conj(sig) / tf.cast(tot, tf.complex128)      # recepteur MMSE
        omega = tot / (tot - tf.abs(sig) ** 2)                   # 1 / MSE
        a = omega * tf.abs(g) ** 2                               # [..., K]
        A = tf.matmul(h64 * tf.cast(a[..., None], tf.complex128), h64, adjoint_a=True)
        mu = nv * tf.reduce_sum(a, axis=-1)
        A = A + tf.cast(mu[..., None, None], tf.complex128) * eye
        Bm = tf.linalg.adjoint(h64) * tf.cast(omega, tf.complex128)[..., None, :] \
            * tf.math.conj(g)[..., None, :]
        w = normalize_power(tf.linalg.solve(A, Bm))
    return tf.cast(w, h.dtype)


PRECODERS = {"MRT": mrt, "ZF": zf, "RZF": rzf, "WMMSE": wmmse}
