"""Donnees par SC, entrainement et evaluation des modeles appris.

Echantillon = le canal d'UNE sous-porteuse, H [K, M]. Les SC de tous les
tirages sont melangees : le modele ne voit jamais la frequence (phases 1-2).
Ensemble d'evaluation = memes tirages que la phase 0 (graine 2026).

Donnees = soit un tenseur h (CSI parfait : entree = vrai canal), soit une
paire (h_in, h_true) : le precodeur voit h_in (CSI connu de la BS), le debit
est TOUJOURS calcule sur h_true.
"""
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from csi import UplinkSounder, csi_at_bs
from metrics import sinr
from precoders.classical import rzf, wmmse

EVAL_SEED, EVAL_BATCHES, EVAL_BATCH = 2026, 16, 32


def channel_pool(cfg, num_drops, seed, batch=32):
    """Canal vrai au slot 0, aplati en SC : [num_drops * F, K, M]."""
    sionna_config.seed = seed
    gen = ChannelGenerator(cfg)
    h = [gen(batch)[:, 0] for _ in range(num_drops // batch)]
    return tf.reshape(tf.concat(h, 0), [-1, cfg.num_users, cfg.num_bs_ant])


def eval_pool(cfg):
    return channel_pool(cfg, EVAL_BATCHES * EVAL_BATCH, EVAL_SEED, EVAL_BATCH)


def csi_pool(cfg, csi, num_drops, seed, batch=32, cov_freq=None, flat=True):
    """Paire (h_in, h_true) : CSI connu de la BS (sondage au slot 0, estimateur
    csi.estimator) et vrai canal au slot de donnees. flat=True : aplatie en SC
    [N*F, K, M] ; sinon bande complete [N, F, K, M] (pour les fenetres)."""
    sionna_config.seed = seed
    gen = ChannelGenerator(cfg)
    interp = csi.estimator[3:] if csi.estimator.startswith("ls_") else "lin"
    sounder = UplinkSounder(cfg, interp, cov_freq)
    h_in, h_true = [], []
    for _ in range(num_drops // batch):
        a, b = csi_at_bs(gen(batch), csi, sounder)
        h_in.append(a)
        h_true.append(b)
    h_in, h_true = tf.concat(h_in, 0), tf.concat(h_true, 0)
    if not flat:
        return h_in, h_true
    shape = [-1, cfg.num_users, cfg.num_bs_ant]
    return tf.reshape(h_in, shape), tf.reshape(h_true, shape)


def eval_csi_pool(cfg, csi, cov_freq=None, flat=True):
    return csi_pool(cfg, csi, EVAL_BATCHES * EVAL_BATCH, EVAL_SEED, EVAL_BATCH, cov_freq, flat)


class WindowedPool:
    """Bande complete (h_in, h_true) [N, F, K, M] ; un echantillon = la SC f
    d'un tirage, avec en entree les W SC centrees sur f (indices bornes aux
    bords de la bande) : h_in [W, K, M], h_true [K, M] de la SC centrale."""

    def __init__(self, h_in, h_true, window):
        self.h_in, self.h_true, self.W = h_in, h_true, window
        self.offsets = tf.range(window) - window // 2
        self.F = h_in.shape[1]

    def _windows(self, d, f):
        idx = tf.clip_by_value(f[:, None] + self.offsets[None, :], 0, self.F - 1)  # [N, W]
        x = tf.gather_nd(self.h_in, tf.stack([tf.broadcast_to(d[:, None], tf.shape(idx)),
                                              idx], -1))
        return x[:, 0] if self.W == 1 else x                    # W=1 : [N, K, M] (phase 2)

    def sample(self, batch):
        d = tf.random.uniform([batch], 0, self.h_in.shape[0], dtype=tf.int32)
        f = tf.random.uniform([batch], 0, self.F, dtype=tf.int32)
        return self._windows(d, f), tf.gather_nd(self.h_true, tf.stack([d, f], -1))

    def chunks(self, drops_per_chunk=32):
        """Toutes les SC de tous les tirages, par paquets (evaluation)."""
        for d0 in range(0, self.h_in.shape[0], drops_per_chunk):
            d = tf.range(d0, min(d0 + drops_per_chunk, self.h_in.shape[0]))
            dd, ff = [tf.reshape(t, [-1]) for t in tf.meshgrid(d, tf.range(self.F), indexing="ij")]
            yield self._windows(dd, ff), tf.gather_nd(self.h_true, tf.stack([dd, ff], -1))

    def subset(self, step):
        return WindowedPool(self.h_in[::step], self.h_true[::step], self.W)


class DropPool:
    """Tirages entiers (h_in, h_true) [D, F, K, M] : pour les modeles qui
    traitent toute la bande d'un coup (domaine des retards). Un lot = des
    tirages complets ; le debit est calcule sur toutes leurs SC."""

    def __init__(self, h_in, h_true):
        self.h_in, self.h_true = h_in, h_true
        self.F = h_in.shape[1]

    def sample(self, batch_drops):
        d = tf.random.uniform([batch_drops], 0, self.h_in.shape[0], dtype=tf.int32)
        return tf.gather(self.h_in, d), tf.gather(self.h_true, d)

    def chunks(self, drops_per_chunk=16):
        for d0 in range(0, self.h_in.shape[0], drops_per_chunk):
            yield self.h_in[d0:d0 + drops_per_chunk], self.h_true[d0:d0 + drops_per_chunk]

    def subset(self, step):
        return DropPool(self.h_in[::step], self.h_true[::step])


def _flat_rate(h_true, w, snr_db):
    """Debit par SC pour des sorties par tirage : h_true [B, F, K, M], w [B, F, M, K]."""
    K, M = h_true.shape[-2], h_true.shape[-1]
    F = h_true.shape[1]
    nv = tf.repeat(10.0 ** (-snr_db / 10.0), F)
    return rate_per_sample(tf.reshape(h_true, [-1, K, M]), tf.reshape(w, [-1, M, K]), nv)


def pair(data):
    """(h_in, h_true) depuis un tenseur (CSI parfait) ou une paire."""
    return data if isinstance(data, tuple) else (data, data)


def subsample(data, step):
    if isinstance(data, (WindowedPool, DropPool)):
        return data.subset(step)
    h_in, h_true = pair(data)
    return (h_in[::step], h_true[::step])


def _center(h_in):
    """SC centrale d'une fenetre [N, W, K, M] (ou h_in tel quel si [N, K, M])."""
    return h_in[:, h_in.shape[1] // 2] if len(h_in.shape) == 4 else h_in


def rate_per_sample(h, w, nv):
    """Debit somme [bps/Hz] par echantillon. nv : scalaire ou [N]."""
    nv = tf.reshape(tf.cast(nv, tf.float32), [-1, 1])
    return tf.reduce_sum(tf.math.log(1.0 + sinr(h, w, nv)), axis=-1) / tf.math.log(2.0)


def evaluate(precoder, data, snrs_db, chunk=65536):
    """precoder(h_in, snr_db [N]) -> W, debit sur h_true. Debit moyen par SNR."""
    if isinstance(data, DropPool):
        out = []
        for snr in snrs_db:
            r = [_flat_rate(ht, precoder(hc, tf.fill([hc.shape[0]], float(snr))),
                            tf.fill([hc.shape[0]], float(snr))) for hc, ht in data.chunks()]
            out.append(float(tf.reduce_mean(tf.concat(r, 0))))
        return out
    if isinstance(data, WindowedPool):
        parts = list(data.chunks())
    else:
        h_in, h_true = pair(data)
        parts = [(h_in[i:i + chunk], h_true[i:i + chunk]) for i in range(0, h_in.shape[0], chunk)]
    out = []
    for snr in snrs_db:
        r = []
        for hc, ht in parts:
            snr_v = tf.fill([hc.shape[0]], float(snr))
            r.append(rate_per_sample(ht, precoder(hc, snr_v), 10.0 ** (-snr / 10.0)))
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def train(model, pool, steps=20000, batch=2048, lr=3e-3, snr_range=(0.0, 25.0), seed=0,
          log_every=1000, val=None, loss="rate"):
    """SNR tire uniformement par echantillon. pool = h ou (h_in, h_true).
    val = (donnees, snrs) optionnel.
    loss = "rate"    : -debit moyen (les forts SNR, aux debits plus grands, dominent)
           "rel_rzf" : -moyenne de debit / debit RZF sur le meme echantillon
                       (chaque SNR pese pareil)."""
    tf.random.set_seed(seed)
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, steps)
    opt = tf.keras.optimizers.Adam(sched)
    windowed = isinstance(pool, WindowedPool)
    by_drop = isinstance(pool, DropPool)
    drops_per_batch = max(1, -(-batch // pool.F)) if by_drop else None
    if not windowed and not by_drop:
        pool_in, pool_true = pair(pool)
        n = pool_in.shape[0]

    @tf.function
    def step_drop():
        h_in, h = pool.sample(drops_per_batch)                    # [Bd, F, K, M]
        snr = tf.random.uniform([drops_per_batch], *snr_range)
        with tf.GradientTape() as tape:
            r = _flat_rate(h, model(h_in, snr), snr)
            if loss == "rel_rzf":
                K, M = h.shape[-2], h.shape[-1]
                nv = tf.repeat(10.0 ** (-snr / 10.0), pool.F)
                r_ref = rate_per_sample(tf.reshape(h, [-1, K, M]),
                                        rzf(tf.reshape(h_in, [-1, K, M]), tf.reshape(nv, [-1, 1, 1])), nv)
                r = r / tf.stop_gradient(r_ref)
            loss_v = -tf.reduce_mean(r)
        opt.apply_gradients(zip(tape.gradient(loss_v, model.trainable_variables),
                                model.trainable_variables))
        return loss_v

    @tf.function
    def step():
        if windowed:
            h_in, h = pool.sample(batch)
        else:
            idx = tf.random.uniform([batch], 0, n, dtype=tf.int32)
            h_in, h = tf.gather(pool_in, idx), tf.gather(pool_true, idx)
        snr = tf.random.uniform([batch], *snr_range)
        nv = 10.0 ** (-snr / 10.0)
        with tf.GradientTape() as tape:
            r = rate_per_sample(h, model(h_in, snr), nv)
            if loss == "rel_rzf":                       # reference : RZF sur le meme CSI
                r_ref = rate_per_sample(h, rzf(_center(h_in), tf.reshape(nv, [-1, 1, 1])), nv)
                r = r / tf.stop_gradient(r_ref)
            loss_v = -tf.reduce_mean(r)
        opt.apply_gradients(zip(tape.gradient(loss_v, model.trainable_variables),
                                model.trainable_variables))
        return loss_v

    hist, t0 = [], time.time()
    for s in range(1, steps + 1):
        loss_v = step_drop() if by_drop else step()
        if s % log_every == 0 or s == steps:
            entry = {"step": s, "train_objective": -float(loss_v),
                     "time_s": round(time.time() - t0, 1)}
            if val is not None:
                entry["val_rate"] = evaluate(model, val[0], val[1])
            hist.append(entry)
            print(f"    pas {s:6d}  objectif {entry['train_objective']:.3f}"
                  + (f"  val {np.round(entry['val_rate'], 2)}" if val is not None else "")
                  + f"  ({entry['time_s']} s)")
    return hist


def references(data, snrs_db):
    """Debits RZF et WMMSE (sur le CSI connu, evalues sur le vrai canal)."""
    return {"RZF": evaluate(lambda x, s: rzf(x, 10.0 ** (-s[0] / 10.0)), data, snrs_db),
            "WMMSE": evaluate(lambda x, s: wmmse(x, 10.0 ** (-s[0] / 10.0)), data, snrs_db,
                              chunk=16384)}


def run_variant(name, model, pool, h_eval, snrs_db, ref, steps, weights_dir=None, lr=1e-3,
                seed=0, loss="rate"):
    """Entraine, evalue et resume un modele ; sauve les poids si weights_dir.
    La graine fixe l'initialisation des poids ET le tirage des donnees."""
    if isinstance(pool, DropPool):
        sample_in = next(pool.chunks(1))[0]
    elif isinstance(pool, WindowedPool):
        sample_in = next(pool.chunks(1))[0][:2]
    else:
        sample_in = pair(pool)[0][:2]
    K = sample_in.shape[-2]
    print(f"== {name}")
    keras.utils.set_random_seed(seed)
    model(sample_in, tf.zeros([sample_in.shape[0]]))  # construit les poids (init graine)
    val = (subsample(h_eval, 20), [0, 10, 20])
    hist = train(model, pool, steps=steps, lr=lr, seed=seed, log_every=max(steps // 4, 1),
                 val=val, loss=loss)
    rate = evaluate(model, h_eval, snrs_db)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, f"{name}.weights.h5"))
    run = {"name": name, "seed": seed, "loss": loss, "params": int(model.count_params()),
           "macs_per_sc": model.macs_per_sc(K),
           "rate_bps_hz": rate, "pct_rzf": [r / z * 100 for r, z in zip(rate, ref["RZF"])],
           "history": hist}
    i = snrs_db.index(20) if 20 in snrs_db else -1
    print(f"   {name}: {run['params']} params, {run['macs_per_sc']} MAC/SC, "
          f"@{snrs_db[i]}dB {rate[i]:.2f} ({run['pct_rzf'][i]:.1f}% RZF)")
    return run
