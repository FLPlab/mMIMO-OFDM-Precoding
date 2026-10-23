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


def csi_pool(cfg, csi, num_drops, seed, batch=32, cov_freq=None):
    """Paire (h_in, h_true) aplatie en SC : CSI connu de la BS (sondage au
    slot 0, estimateur csi.estimator) et vrai canal au slot de donnees."""
    sionna_config.seed = seed
    gen = ChannelGenerator(cfg)
    interp = csi.estimator[3:] if csi.estimator.startswith("ls_") else "lin"
    sounder = UplinkSounder(cfg, interp, cov_freq)
    h_in, h_true = [], []
    for _ in range(num_drops // batch):
        a, b = csi_at_bs(gen(batch), csi, sounder)
        h_in.append(a)
        h_true.append(b)
    shape = [-1, cfg.num_users, cfg.num_bs_ant]
    return tf.reshape(tf.concat(h_in, 0), shape), tf.reshape(tf.concat(h_true, 0), shape)


def eval_csi_pool(cfg, csi, cov_freq=None):
    return csi_pool(cfg, csi, EVAL_BATCHES * EVAL_BATCH, EVAL_SEED, EVAL_BATCH, cov_freq)


def pair(data):
    """(h_in, h_true) depuis un tenseur (CSI parfait) ou une paire."""
    return data if isinstance(data, tuple) else (data, data)


def subsample(data, step):
    h_in, h_true = pair(data)
    return (h_in[::step], h_true[::step])


def rate_per_sample(h, w, nv):
    """Debit somme [bps/Hz] par echantillon. nv : scalaire ou [N]."""
    nv = tf.reshape(tf.cast(nv, tf.float32), [-1, 1])
    return tf.reduce_sum(tf.math.log(1.0 + sinr(h, w, nv)), axis=-1) / tf.math.log(2.0)


def evaluate(precoder, data, snrs_db, chunk=65536):
    """precoder(h_in, snr_db [N]) -> W, debit sur h_true. Debit moyen par SNR."""
    h_in, h_true = pair(data)
    out = []
    for snr in snrs_db:
        r = []
        for i in range(0, h_in.shape[0], chunk):
            hc = h_in[i:i + chunk]
            snr_v = tf.fill([hc.shape[0]], float(snr))
            r.append(rate_per_sample(h_true[i:i + chunk], precoder(hc, snr_v),
                                     10.0 ** (-snr / 10.0)))
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
    pool_in, pool_true = pair(pool)
    n = pool_in.shape[0]

    @tf.function
    def step():
        idx = tf.random.uniform([batch], 0, n, dtype=tf.int32)
        h_in, h = tf.gather(pool_in, idx), tf.gather(pool_true, idx)
        snr = tf.random.uniform([batch], *snr_range)
        nv = 10.0 ** (-snr / 10.0)
        with tf.GradientTape() as tape:
            r = rate_per_sample(h, model(h_in, snr), nv)
            if loss == "rel_rzf":                       # reference : RZF sur le meme CSI
                r_ref = rate_per_sample(h, rzf(h_in, tf.reshape(nv, [-1, 1, 1])), nv)
                r = r / tf.stop_gradient(r_ref)
            loss_v = -tf.reduce_mean(r)
        opt.apply_gradients(zip(tape.gradient(loss_v, model.trainable_variables),
                                model.trainable_variables))
        return loss_v

    hist, t0 = [], time.time()
    for s in range(1, steps + 1):
        loss_v = step()
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
    K = pair(h_eval)[0].shape[1]
    print(f"== {name}")
    keras.utils.set_random_seed(seed)
    model(pair(pool)[0][:2], tf.zeros([2]))            # construit les poids (init graine)
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
