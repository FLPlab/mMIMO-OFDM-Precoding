"""Donnees par SC, entrainement et evaluation des modeles appris.

Echantillon = le canal d'UNE sous-porteuse, H [K, M]. Les SC de tous les
tirages sont melangees : le modele ne voit jamais la frequence (phase 1).
Ensemble d'evaluation = memes tirages que la phase 0 (graine 2026).
"""
import os
import time

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
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


def rate_per_sample(h, w, nv):
    """Debit somme [bps/Hz] par echantillon. nv : scalaire ou [N]."""
    nv = tf.reshape(tf.cast(nv, tf.float32), [-1, 1])
    return tf.reduce_sum(tf.math.log(1.0 + sinr(h, w, nv)), axis=-1) / tf.math.log(2.0)


def evaluate(precoder, h, snrs_db, chunk=65536):
    """precoder(h, snr_db [N]) -> W. Renvoie le debit moyen par SNR."""
    out = []
    for snr in snrs_db:
        r = []
        for i in range(0, h.shape[0], chunk):
            hc = h[i:i + chunk]
            snr_v = tf.fill([hc.shape[0]], float(snr))
            r.append(rate_per_sample(hc, precoder(hc, snr_v), 10.0 ** (-snr / 10.0)))
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def train(model, pool, steps=20000, batch=2048, lr=3e-3, snr_range=(0.0, 25.0), seed=0,
          log_every=1000, val=None):
    """Maximise le debit somme moyen, SNR tire uniformement par echantillon.
    val = (h_val, snrs) optionnel, evalue tous les log_every pas."""
    tf.random.set_seed(seed)
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, steps)
    opt = tf.keras.optimizers.Adam(sched)
    n = pool.shape[0]

    @tf.function
    def step():
        idx = tf.random.uniform([batch], 0, n, dtype=tf.int32)
        h = tf.gather(pool, idx)
        snr = tf.random.uniform([batch], *snr_range)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, model(h, snr), 10.0 ** (-snr / 10.0)))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return loss

    hist, t0 = [], time.time()
    for s in range(1, steps + 1):
        loss = step()
        if s % log_every == 0 or s == steps:
            entry = {"step": s, "train_rate": -float(loss), "time_s": round(time.time() - t0, 1)}
            if val is not None:
                entry["val_rate"] = evaluate(model, val[0], val[1])
            hist.append(entry)
            print(f"    pas {s:6d}  debit train {entry['train_rate']:.2f}"
                  + (f"  val {np.round(entry['val_rate'], 2)}" if val is not None else "")
                  + f"  ({entry['time_s']} s)")
    return hist


def references(h, snrs_db):
    """Debits RZF et WMMSE sur le meme ensemble (evaluation appariee)."""
    return {"RZF": evaluate(lambda x, s: rzf(x, 10.0 ** (-s[0] / 10.0)), h, snrs_db),
            "WMMSE": evaluate(lambda x, s: wmmse(x, 10.0 ** (-s[0] / 10.0)), h, snrs_db,
                              chunk=16384)}


def run_variant(name, model, pool, h_eval, snrs_db, ref, steps, weights_dir=None, lr=1e-3):
    """Entraine, evalue et resume un modele ; sauve les poids si weights_dir."""
    K = h_eval.shape[1]
    print(f"== {name}")
    model(pool[:2], tf.zeros([2]))
    val = (h_eval[::20], [0, 10, 20])
    hist = train(model, pool, steps=steps, lr=lr, seed=0, log_every=max(steps // 4, 1), val=val)
    rate = evaluate(model, h_eval, snrs_db)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, f"{name}.weights.h5"))
    run = {"name": name, "params": int(model.count_params()), "macs_per_sc": model.macs_per_sc(K),
           "rate_bps_hz": rate, "pct_rzf": [r / z * 100 for r, z in zip(rate, ref["RZF"])],
           "history": hist}
    i = snrs_db.index(20) if 20 in snrs_db else -1
    print(f"   {name}: {run['params']} params, {run['macs_per_sc']} MAC/SC, "
          f"@{snrs_db[i]}dB {rate[i]:.2f} ({run['pct_rzf'][i]:.1f}% RZF)")
    return run
