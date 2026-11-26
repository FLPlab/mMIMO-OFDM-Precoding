"""Phase 14 -- CSI imparfait, variance d'erreur en entree (suite de phase14_csi, demande de F. Leduc-Primeau,
2026-11-25) : notre precodeur lit Ĝ = Ĥ Ĥ^H par sous-porteuse et sort W = Ĥ^H C ; Ĝ ne dit pas quel
utilisateur est mal estime, et en massif l'erreur LMMSE est structuree spatialement (l'optimum robuste sort
de span(Ĥ^H)). Test : donner au reseau la variance d'erreur RELATIVE par utilisateur,
10 log10(err_var_k / puissance_k), ajoutee aux tokens de paire (j, k) (les deux scalaires j et k, comme la
puissance relative deja donnee par inp="norm") -- nouvelle option GramPrecoder(err_in=True), comportement
par defaut inchange (err_in=False).

Entrainement (comme phase14_csi, mais erreur TIREE PAR UTILISATEUR -- pas par tirage -- pour que la
variance en entree soit informative, et sur le melange M8K4 + M64K8 de phase13_train_massive puisque le
precodeur de base gere les deux tailles) : precodeur initialise par le precodeur CSI parfait
(phase13_train_massive), 20 000 pas (alternes entre les deux tailles), h_est_k = h_k + e_k,
e_k ~ CN(0, s2_k * puissance_k), s2_k tire en dB UNIFORMEMENT PAR UTILISATEUR dans [-25, -5], err_db_k =
10 log10(s2_k) donne au reseau ; debit calcule sur le vrai canal h. 3 modeles 3GPP melanges.
Baseline "sans entree d'erreur" : poids EXISTANTS de phase14_csi (M8K4 seul, erreur tiree par TIRAGE, pas
par utilisateur), reutilises tels quels (zero-shot en massif pour ce baseline).

Deploiement (comme phase14_csi) : Sionna, estimation LS + interpolation LMMSE (covariance du melange) ou
LS + interpolation lineaire, pilotes 0 / 10 / 20 dB. M8K4 : UMi, 192 tirages (phase 11). M64K8 : UMi, 64
tirages (phase13_massive/phase13_massive_pilots). err_db au deploiement = 10 log10(err_var_k / puissance_k),
err_var_k = moyenne sur les antennes de err_var renvoye par sound_power_controlled(..., return_err_var=True),
puissance_k = moyenne sur les antennes de |h_est_k|^2 (le vrai canal n'est pas connu du recepteur).
Precodeurs compares : CSI parfait, synthetique sans entree d'erreur (existant), synthetique AVEC entree
d'erreur, WMMSE 100 it. (robuste pour LMMSE), RZF.

    python -u -m experiments.phase14_csi_errin
"""
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder, estimate_freq_cov
from experiments.phase7_context import cov_mixture, sound_power_controlled
from experiments.phase13_massive_pilots import KW as KW_MASSIVE
from experiments.phase13_train_massive import pool as pool_sizes
from experiments.phase14_csi import B, DROPS, PILOTS, SNRS, STEPS, W14, load_perfect, metrics
from models.gram_precoder import GramPrecoder
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

BATCH = 2048
B64, DROPS64 = 8, 64                                 # memes tirages que phase13_massive(_pilots)
PILOTS_MASSIVE = [0, 10, 20]


def err_db_from(err, h_est):
    """err, h_est : [..., K, M]. 10 log10(variance d'erreur relative par utilisateur), moyenne sur
    les antennes des deux membres (le vrai canal n'est pas connu du recepteur)."""
    if err is None:
        return None
    ep = tf.reduce_mean(err, axis=-1)
    pw = tf.reduce_mean(tf.abs(h_est) ** 2, axis=-1)
    return 10.0 * tf.math.log(tf.maximum(ep, 1e-12) / tf.maximum(pw, 1e-12)) / tf.math.log(10.0)


def load_existing_no_err():
    """Baseline "synthetique sans entree d'erreur" = poids existants de phase14_csi (M8K4 seul)."""
    net = GramPrecoder(8, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W14, "precoder_synth_csi.weights.h5"))
    return net


def warm_start(net_se, net_p):
    """net_se (err_in=True) <- net_p (CSI parfait, err_in=False) : blocs et lecture copies a
    l'identique (meme forme), init (Dense d'entree) copiee avec les 2 colonnes d'erreur a zero
    (ignorees au depart -- le modele part du comportement du precodeur CSI parfait)."""
    for b_se, b_p in zip(net_se.blocks, net_p.blocks):
        b_se.set_weights(b_p.get_weights())
    net_se.read.set_weights(net_p.read.get_weights())
    k_p, bias_p = net_p.init.get_weights()
    k_se, _ = net_se.init.get_weights()
    k_se[...] = 0.0
    k_se[:k_p.shape[0]] = k_p
    net_se.init.set_weights([k_se, bias_p])


def train_errin(net_p, steps=None, batch=None, seed0=1):
    """Precodeur avec entree de variance d'erreur, entraine sur le melange M8K4 + M64K8 (phase13_train_massive
    gere les deux tailles), erreur tiree PAR UTILISATEUR, initialise par net_p (CSI parfait)."""
    steps = STEPS if steps is None else steps
    batch = BATCH if batch is None else batch
    keras.utils.set_random_seed(0)
    pools = {k: pool_sizes(*k, seed0=seed0) for k in ((8, 4), (64, 8))}
    print({f"M{m}K{k}": int(p.shape[0]) for (m, k), p in pools.items()}, flush=True)
    net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm", err_in=True)
    for p in pools.values():
        net(p[:2], tf.zeros([2]), tf.zeros([2, p.shape[1]]))
    warm_start(net, net_p)
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(3e-4, steps))

    def make_step(p):
        n, K = p.shape[0], p.shape[1]

        @tf.function
        def step():
            h = tf.gather(p, tf.random.uniform([batch], 0, n, dtype=tf.int32))
            pw = tf.reduce_mean(tf.abs(h) ** 2, axis=-1, keepdims=True)                 # [n, K, 1]
            err_db = tf.random.uniform([batch, K], -25.0, -5.0)                         # PAR UTILISATEUR
            s2 = 10.0 ** (err_db[:, :, None] / 10.0)
            sd = tf.cast(tf.sqrt(s2 * pw / 2), tf.complex64)
            e = tf.complex(tf.random.normal(tf.shape(h, out_type=tf.int32)),
                           tf.random.normal(tf.shape(h, out_type=tf.int32))) * sd
            snr = tf.random.uniform([batch], 0.0, 30.0)
            nv = 10.0 ** (-snr / 10.0)
            ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(rate_per_sample(h, net(h + e, snr, err_db), nv) / tf.stop_gradient(ref))
            g, _ = tf.clip_by_global_norm(tape.gradient(loss, net.trainable_variables), 1.0)
            opt.apply_gradients(zip(g, net.trainable_variables))
            return loss
        return step

    keys = list(pools)
    steps_fn = {k: make_step(pools[k]) for k in keys}
    t0 = time.time()
    for s in range(1, steps + 1):
        loss = steps_fn[keys[s % 2]]()
        if s % 5000 == 0:
            print(f"   pas {s} objectif {-float(loss):.3f} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(W14, exist_ok=True)
    net.save_weights(os.path.join(W14, "precoder_synth_csi_errin.weights.h5"))
    return net


def evaluate(fns, h_est, h, err=None, chunk=8192):
    """fns : name -> fn(h_est_chunk, nv, snr, err_chunk) -> W. h_est, h : [..., K, M] (K, M lus
    dynamiquement -- generalisation de phase14_csi.evaluate au massif). err : [..., K, M] ou [..., K, 1]
    (l'estimateur Sionna LS + interpolation lineaire renvoie une variance d'erreur identique sur les
    antennes -- dernier axe de taille 1, cf. csi.UplinkSounder) : reshape avec SA propre derniere
    dimension, pas celle de h."""
    K, M = h.shape[-2], h.shape[-1]
    hf, he = tf.reshape(h, [-1, K, M]), tf.reshape(h_est, [-1, K, M])
    ef = None if err is None else tf.reshape(err, [-1, K, err.shape[-1]])
    out = {}
    for name, fn in fns.items():
        res = {"rate": [], "ber_qpsk": [], "ber_16qam": []}
        for snr in SNRS:
            nv = 10.0 ** (-snr / 10.0)
            m = [metrics(hf[i:i + chunk], fn(he[i:i + chunk], nv, snr, None if ef is None else ef[i:i + chunk]), nv)
                 for i in range(0, hf.shape[0], chunk)]
            for k, j in (("rate", 0), ("ber_qpsk", 1), ("ber_16qam", 2)):
                res[k].append(float(np.mean([x[j] for x in m])))
        out[name] = res
    return out


def main():
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")}, "snrs_db": SNRS}

    # --- M8K4 (comme phase14_csi) ---
    net_p = load_perfect()
    net_s = load_existing_no_err()
    net_se = train_errin(net_p)

    cfg8 = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen8 = ChannelGenerator(cfg8)
    h8 = tf.concat([gen8(B)[:, 0] for _ in range(DROPS // B)], 0)
    cov8 = cov_mixture()

    out8 = {"csi_parfait": {}, "estime": {}}
    perfect8 = {"notre precodeur (CSI parfait)": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                "WMMSE 100 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=100),
                "RZF": lambda x, nv, s, e: rzf(x, nv)}
    out8["csi_parfait"] = evaluate(perfect8, h8, h8)
    for est in ("lmmse", "lin"):
        snd = UplinkSounder(cfg8, est, cov8 if est == "lmmse" else None)
        for p in PILOTS:
            pr = [sound_power_controlled(snd, h8[j:j + B], float(p), return_err_var=True) for j in range(0, DROPS, B)]
            he, ev = tf.concat([a for a, _ in pr], 0), tf.concat([b for _, b in pr], 0)
            fns = {
                "precodeur entraine CSI parfait": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                "precodeur entraine erreur synthetique (existant)":
                    lambda x, nv, s, e: net_s(x, tf.fill([x.shape[0]], float(s))),
                "precodeur entraine erreur synthetique + variance entree":
                    lambda x, nv, s, e: net_se(x, tf.fill([x.shape[0]], float(s)), err_db_from(e, x)),
                "WMMSE 100 it." + (" robuste" if est == "lmmse" else ""):
                    (lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e)) if est == "lmmse"
                    else (lambda x, nv, s, e: wmmse(x, nv, num_iter=100)),
                "RZF": lambda x, nv, s, e: rzf(x, nv)}
            r = evaluate(fns, he, h8, ev)
            out8["estime"][f"{est}_pilote{p}"] = r
            print(f"M8K4  {est} pilote {p} dB @20dB : " + " | ".join(
                f"{k} {v['rate'][4]:.2f} (BER16 {v['ber_16qam'][4]:.3f})" for k, v in r.items()), flush=True)
    out["m8k4"] = out8
    os.makedirs(os.path.join(RESULTS, "phase14"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase14", "csi_errin.json"), "w"), indent=1)  # sauvegarde partielle

    # --- M64K8 (massif, "le precodeur de base phase13_train_massive gere les deux tailles") ---
    cfg64 = SystemConfig(scenario="umi", num_slots=1, relative_power=True, **KW_MASSIVE)
    covs = []
    for i, scen in enumerate(["umi", "uma", "rma"]):
        sionna_config.seed = 2027 + 10 * i
        covs.append(estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen, **KW_MASSIVE))))
    cov64 = tf.add_n(covs) / 3
    sionna_config.seed = 4026                                     # memes tirages que phase13_massive(_pilots)
    gen64 = ChannelGenerator(cfg64)
    h64 = tf.concat([gen64(B64)[:, 0] for _ in range(DROPS64 // B64)], 0)

    out64 = {"csi_parfait": {}, "estime": {}}
    perfect64 = {"notre precodeur (CSI parfait)": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                 "WMMSE 100 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=100),
                 "RZF": lambda x, nv, s, e: rzf(x, nv)}
    out64["csi_parfait"] = evaluate(perfect64, h64, h64, chunk=2048)
    for est in ("lmmse", "lin"):
        snd = UplinkSounder(cfg64, est, cov64 if est == "lmmse" else None)
        for p in PILOTS_MASSIVE:
            pr = [sound_power_controlled(snd, h64[j:j + B64], float(p), return_err_var=True)
                  for j in range(0, DROPS64, B64)]
            he, ev = tf.concat([a for a, _ in pr], 0), tf.concat([b for _, b in pr], 0)
            fns = {
                "precodeur entraine CSI parfait": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                "precodeur entraine erreur synthetique (existant, M8K4)":
                    lambda x, nv, s, e: net_s(x, tf.fill([x.shape[0]], float(s))),
                "precodeur entraine erreur synthetique + variance entree":
                    lambda x, nv, s, e: net_se(x, tf.fill([x.shape[0]], float(s)), err_db_from(e, x)),
                "WMMSE 100 it." + (" robuste" if est == "lmmse" else ""):
                    (lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e)) if est == "lmmse"
                    else (lambda x, nv, s, e: wmmse(x, nv, num_iter=100)),
                "RZF": lambda x, nv, s, e: rzf(x, nv)}
            r = evaluate(fns, he, h64, ev, chunk=2048)
            out64["estime"][f"{est}_pilote{p}"] = r
            print(f"M64K8 {est} pilote {p} dB @20dB : " + " | ".join(
                f"{k} {v['rate'][4]:.2f} (BER16 {v['ber_16qam'][4]:.3f})" for k, v in r.items()), flush=True)
            out["m64k8"] = out64
            json.dump(out, open(os.path.join(RESULTS, "phase14", "csi_errin.json"), "w"), indent=1)  # sauvegarde incrementale

    out["m64k8"] = out64
    json.dump(out, open(os.path.join(RESULTS, "phase14", "csi_errin.json"), "w"), indent=1)
    print("-> results/phase14/csi_errin.json", flush=True)


if __name__ == "__main__":
    main()
