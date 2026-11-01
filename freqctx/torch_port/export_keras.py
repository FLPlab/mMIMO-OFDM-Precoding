"""Exporte les poids Keras de la base figee vers un .npz aux noms du modele PyTorch
(torch_port/models.py), plus un jeu d'entrees / sorties de reference pour verifier
l'equivalence. A lancer dans l'environnement TensorFlow (tf-gpu) :

    python -m torch_port.export_keras [--weights pilot_rand_ph32]
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from config import SystemConfig
from experiments.phase7_context import PRECODER
from models.delay_domain import DelayDomainPrecoder

W7 = "/export/tmp/sala/freqctx/weights/phase7"
OUT = "/export/tmp/sala/freqctx/torch_port"


def dense(sd, name, layer, bias=True):
    sd[f"{name}.weight"] = layer.kernel.numpy()             # [in, out] : transpose a l'import
    if bias:
        sd[f"{name}.bias"] = layer.bias.numpy()


def ln(sd, name, layer):
    sd[f"{name}.weight"], sd[f"{name}.bias"] = layer.gamma.numpy(), layer.beta.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="pilot_rand_ph32")
    ap.add_argument("--hidden", type=int, default=32)
    a = ap.parse_args()
    cfg = SystemConfig()
    net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                              precoder_arch="pair", **dict(PRECODER, hidden=a.hidden))
    h0 = tf.zeros([1, cfg.num_sc, cfg.num_users, cfg.num_bs_ant], tf.complex64)
    net(h0, tf.zeros([1]))
    net.load_weights(os.path.join(W7, f"{a.weights}.weights.h5"))
    sd, est, pre = {}, net.estimator, net.precoder
    dense(sd, "estimator.embed", est.embed)
    sd["estimator.pos.weight"] = est.pos.embeddings.numpy()
    for i, b in enumerate(est.blocks):
        p = f"estimator.blocks.{i}"
        ln(sd, f"{p}.ln1", b.ln1), ln(sd, f"{p}.ln2", b.ln2)
        dense(sd, f"{p}.qkv", b.qkv), dense(sd, f"{p}.out", b.out)
        dense(sd, f"{p}.ffn1", b.ffn1), dense(sd, f"{p}.ffn2", b.ffn2)
    dense(sd, "estimator.gain", est.gain)
    dense(sd, "precoder.init", pre.init)
    for i, b in enumerate(pre.blocks):
        p = f"precoder.blocks.{i}"
        ln(sd, f"{p}.norm", b.norm)
        dense(sd, f"{p}.a", b.a, bias=False), dense(sd, f"{p}.b", b.b, bias=False)
        dense(sd, f"{p}.f1", b.f1), dense(sd, f"{p}.f2", b.f2)
    dense(sd, "precoder.read", pre.read)
    n_keras = sum(int(np.prod(v.shape)) for v in net.trainable_weights)
    n_export = sum(int(np.prod(v.shape)) for v in sd.values())
    assert n_keras == n_export, (n_keras, n_export)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{a.weights}.npz"), **sd)
    rng = np.random.default_rng(0)                          # entrees de reference (canal LS synthetique)
    h = (rng.standard_normal((2, cfg.num_sc, cfg.num_users, cfg.num_bs_ant))
         + 1j * rng.standard_normal((2, cfg.num_sc, cfg.num_users, cfg.num_bs_ant))).astype(np.complex64)
    snr = np.array([5.0, 20.0], np.float32)
    w = net(tf.constant(h), tf.constant(snr)).numpy()
    np.savez(os.path.join(OUT, f"{a.weights}_ref.npz"), h=h, snr=snr, w=w,
             h_hat=net.encode(tf.constant(h)).numpy())
    print(f"-> {OUT}/{a.weights}.npz ({n_export} parametres) et _ref.npz")


if __name__ == "__main__":
    main()
