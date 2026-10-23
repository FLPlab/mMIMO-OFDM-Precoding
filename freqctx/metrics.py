"""Metriques de debit. Le SINR est toujours evalue sur le canal VRAI."""
import tensorflow as tf


def sinr(h, w, noise_var):
    """h [..., K, M], w [..., M, K] -> SINR [..., K]."""
    p = tf.abs(tf.matmul(h, w)) ** 2                    # |h_k w_j|^2
    sig = tf.linalg.diag_part(p)
    return sig / (tf.reduce_sum(p, axis=-1) - sig + noise_var)


def sum_rate(h, w, noise_var):
    """Debit somme moyen par SC [bps/Hz] (moyenne sur batch et SC)."""
    r = tf.reduce_sum(tf.math.log(1.0 + sinr(h, w, noise_var)), axis=-1) / tf.math.log(2.0)
    return tf.reduce_mean(r)
