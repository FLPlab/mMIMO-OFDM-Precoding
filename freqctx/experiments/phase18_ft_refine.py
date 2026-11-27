"""Phase 18 -- adaptation au site de la SEULE passe 2 (RefineNet, base figee) avec peu
d'echantillons de canal du site (demande F. Leduc-Primeau, 27 nov. 2026).

Constat (FINDINGS.md phases 14-18) : fine-tuner la BASE sur un site n'aide pas (ecart = limite de
precision de la base, pas un probleme de site, voir phase14_finetune/phase14_ft_diag). La passe 2
(models/refine.py, poids weights/phase15/refine_L4.weights.h5, base figee
weights/phase13/precoder_relpow_mix.weights.h5) comble l'essentiel en zero-shot mais reste sous
WMMSE 300 it. a haut SNR sur Ericsson (30 dB 35.88 vs 36.15, 40 dB 46.43 vs 47.84). Hypothese :
la passe 2, qui voit l'erreur de la base, peut s'adapter au regime de SINR eleve d'un site avec
peu de donnees (base TOUJOURS figee -- aucun compromis sur la base).

Protocole -- repris EXACTEMENT de experiments/phase14_finetune.py (import SANS MODIFIER :
load_base, site_drops, NS, TEST_DROPS), SEULES differences :
  - depart = base + passe 2 zero-shot (poids phase15/refine_L4.weights.h5, RefineNet, meme
    construction que experiments.phase15_refine_eval.load_refine, importe SANS MODIFIER) ;
  - seule la passe 2 est adaptee (base.trainable=False, deja fixe par load_refine) :
      (a) complet : toutes les variables propres a la passe 2 (experiments.phase15_refine.own_vars,
          importe SANS MODIFIER), pas 1e-4, 2000 pas (budget fixe, "jusqu'a plateau" non automatise) ;
      (b) lora    : rang 4 sur les 6 couches Dense de la passe 2 seulement (init, read, et a/b/f1/f2
          de chacun des 4 TriangularBlock -- PAS les couches de la base), pas 1e-3, meme nombre de
          pas que experiments.phase14_finetune.STEPS (40->500, 400->1000, 4000->2000) ;
  - SNR d'ADAPTATION U(0, 40) dB (au lieu de U(0, 30)) : la base reste figee, donc pas de compromis
    sur sa plage d'entrainement a imposer a l'adaptation ;
  - evaluation (test, 64 tirages seed 77, moitie disjointe de l'adaptation) a 10/20/30/40 dB contre
    WMMSE 300 it., base seule, base+passe2 zero-shot, base+passe2 appliquee deux fois zero-shot
    (experiments.phase15_refine_more.two_pass_coeffs, importe SANS MODIFIER), et chaque variante
    adaptee (a)/(b) x N in {40, 400, 4000}.

Critere fixe a l'avance (voir prompt) : RETENU si, avec N <= 400, base+passe2 adaptee est
>= WMMSE 300 - 0.05 a 30 dB sur les 3 sites ET ne perd pas plus de 0.05 face au zero-shot a
10-20 dB. L'ecart restant a 40 dB est indique a part (hors critere).

    python -u -m experiments.phase18_ft_refine [--smoke] [--sites stecath decarie ericsson]
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

from config import SystemConfig
from experiments.phase13_bench import CHUNK
from experiments.phase14_finetune import NS, TEST_DROPS, load_base, site_drops
from experiments.phase15_refine import own_vars
from experiments.phase15_refine_eval import load_refine
from experiments.phase15_refine_more import two_pass_coeffs
from precoders.classical import rzf, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

SITES = ["stecath", "decarie", "ericsson"]
SNRS_EVAL = [10, 20, 30, 40]
SNR_ADAPT_LO, SNR_ADAPT_HI = 0.0, 40.0
LR_COMPLET, STEPS_COMPLET = 1e-4, 2000
LR_LORA = 1e-3
STEPS_LORA = {40: 500, 400: 1000, 4000: 2000}                  # memes pas que phase14_finetune
OUT_DIR = os.path.join(RESULTS, "phase18")
OUT_PATH = os.path.join(OUT_DIR, "ft_refine.json")


def rates_at(fn, h, snrs):
    """Comme experiments.phase13_bench.rates, mais pour une liste de SNR arbitraire (ici 10-40 dB,
    la liste SNRS de phase13_bench s'arrete a 30)."""
    h = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])
    out = []
    for snr in snrs:
        nv = 10.0 ** (-snr / 10.0)
        r = [rate_per_sample(h[i:i + CHUNK], fn(h[i:i + CHUNK], nv, snr), nv) for i in range(0, h.shape[0], CHUNK)]
        out.append(float(tf.reduce_mean(tf.concat(r, 0))))
    return out


def passe2_dense_layers(net):
    """Les couches Dense PROPRES a la passe 2 (init, read, a/b/f1/f2 de chaque TriangularBlock) --
    jamais celles de net.base (figee)."""
    layers = [net.init, net.read]
    for blk in net.blocks:
        layers += [blk.a, blk.b, blk.f1, blk.f2]
    return layers


def peft_lora_passe2(net, rank=4):
    """Meme recette que experiments.phase14_finetune.peft(method="lora"), mais restreinte aux
    couches de la passe 2 (passe2_dense_layers) -- la base ne doit jamais etre touchee."""
    train_vars = []
    for layer in passe2_dense_layers(net):
        w0 = tf.constant(layer.kernel.numpy())
        b = layer.bias
        act = layer.activation
        a = tf.Variable(tf.random.normal([w0.shape[0], rank], stddev=0.01))
        bb = tf.Variable(tf.zeros([rank, w0.shape[1]]))
        train_vars += [a, bb]
        kern = lambda w0=w0, a=a, bb=bb: w0 + a @ bb

        def call(x, *args, kern=kern, b=b, act=act, **kw):
            y = tf.tensordot(x, kern(), axes=[[x.shape.rank - 1], [0]])
            return act(y + b if b is not None else y)
        layer.call = call
    return train_vars


def finetune_passe2(net, method, data, steps, lr):
    """Adapte SEULEMENT la passe 2 (net.base reste figee, cf. load_refine) sur `data`
    [n, K, M] (1 SC par echantillon, site_drops per_sc=True), SNR U(0, 40) dB."""
    variables = own_vars(net) if method == "complet" else peft_lora_passe2(net)
    opt = keras.optimizers.Adam(lr)
    n = data.shape[0]
    bs = min(n, 256)

    @tf.function
    def step():
        h = tf.gather(data, tf.random.uniform([bs], 0, n, dtype=tf.int32))
        perm = tf.argsort(tf.random.uniform([bs, 4]), axis=-1)             # permutation des utilisateurs
        h = tf.gather(h, perm, batch_dims=1)
        snr = tf.random.uniform([bs], SNR_ADAPT_LO, SNR_ADAPT_HI)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
        g = tape.gradient(loss, variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, variables))
        return loss

    last = None
    for _ in range(steps):
        last = step()
    return sum(int(np.prod(v.shape)) for v in variables), float(last)


def run_site(site, ns, steps_complet, steps_lora, test_drops):
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    cells = read_cells_cached(site)
    adapt, test = cells[0::2], cells[1::2]                                  # moities disjointes
    h_test = site_drops(test, test_drops, freqs, seed=77, per_sc=False)

    res = {"cells_adapt": len(adapt), "cells_test": len(test), "rates": {}, "params": {}}
    t0 = time.time()

    res["rates"]["WMMSE 300 it."] = rates_at(lambda x, nv, s: wmmse(x, nv, num_iter=300), h_test, SNRS_EVAL)
    base = load_base()
    res["rates"]["base (zero-shot)"] = rates_at(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), h_test, SNRS_EVAL)
    net0 = load_refine()
    res["rates"]["base+passe2 (zero-shot)"] = rates_at(
        lambda x, nv, s: net0(x, tf.fill([x.shape[0]], float(s))), h_test, SNRS_EVAL)
    res["rates"]["base+passe2 x2 (zero-shot)"] = rates_at(
        lambda x, nv, s: two_pass_coeffs(net0, x, tf.fill([x.shape[0]], float(s))), h_test, SNRS_EVAL)
    print(f"{site} references ({time.time() - t0:.0f} s) : WMMSE300 {dict(zip(SNRS_EVAL, [round(v, 2) for v in res['rates']['WMMSE 300 it.']]))}  "
          f"base {[round(v, 2) for v in res['rates']['base (zero-shot)']]}  "
          f"passe2 {[round(v, 2) for v in res['rates']['base+passe2 (zero-shot)']]}  "
          f"passe2x2 {[round(v, 2) for v in res['rates']['base+passe2 x2 (zero-shot)']]}", flush=True)

    for n in ns:
        data = site_drops(adapt, n, freqs, seed=100 + n, per_sc=True)
        for method, lr, steps in (("complet", LR_COMPLET, steps_complet), ("lora", LR_LORA, steps_lora[n])):
            net = load_refine()
            npar, last_loss = finetune_passe2(net, method, data, steps, lr)
            key = f"{method} N={n}"
            res["rates"][key] = rates_at(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h_test, SNRS_EVAL)
            res["params"][key] = npar
            print(f"{site} {key} ({npar} param. appris, {steps} pas, perte finale {last_loss:.4f}) : "
                  f"{dict(zip(SNRS_EVAL, [round(v, 2) for v in res['rates'][key]]))}", flush=True)
    return res


_CELLS_CACHE = {}


def read_cells_cached(site):
    from experiments.phase13_lab import read_cells
    if site not in _CELLS_CACHE:
        _CELLS_CACHE[site] = read_cells(site, "psi2az")
    return _CELLS_CACHE[site]


def verdict(out):
    sites3 = SITES
    ok_30, ok_low = [], []
    detail_40 = {}
    for s in sites3:
        r = out["sites"][s]["rates"]
        wmmse300 = dict(zip(SNRS_EVAL, r["WMMSE 300 it."]))
        best_adapted = {}
        for n in (40, 400):
            for method in ("complet", "lora"):
                key = f"{method} N={n}"
                if key in r:
                    best_adapted[key] = dict(zip(SNRS_EVAL, r[key]))
        zero_shot = dict(zip(SNRS_EVAL, r["base+passe2 (zero-shot)"]))
        for key, by_snr in best_adapted.items():
            ok_30.append((s, key, by_snr[30] >= wmmse300[30] - 0.05))
            for snr in (10, 20):
                ok_low.append((s, key, snr, by_snr[snr] - zero_shot[snr] >= -0.05))
        detail_40[s] = {"wmmse300_40db": wmmse300[40],
                         **{k: v[40] for k, v in best_adapted.items()},
                         "ecart_restant_40db": {k: wmmse300[40] - v[40] for k, v in best_adapted.items()}}
    retenu = all(v for _, _, v in ok_30) and all(v for _, _, _, v in ok_low)
    return {"retenu": retenu, "detail_30db_vs_wmmse300_N<=400": ok_30,
            "detail_10_20db_vs_zeroshot_N<=400": ok_low, "ecart_40db": detail_40}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="test court (peu de pas/tirages) pour valider le script")
    ap.add_argument("--sites", nargs="+", default=SITES, choices=SITES)
    a = ap.parse_args()
    keras.utils.set_random_seed(0)

    ns = [40] if a.smoke else NS
    steps_complet = 20 if a.smoke else STEPS_COMPLET
    steps_lora = {n: 10 for n in ns} if a.smoke else STEPS_LORA
    test_drops = 4 if a.smoke else TEST_DROPS

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "smoke": a.smoke,
                    "snrs_eval_db": SNRS_EVAL, "snr_adapt_range_db": [SNR_ADAPT_LO, SNR_ADAPT_HI],
                    "steps_complet": steps_complet, "steps_lora": steps_lora, "ns": ns},
           "sites": {}}
    t0 = time.time()
    for site in a.sites:
        out["sites"][site] = run_site(site, ns, steps_complet, steps_lora, test_drops)
        json.dump(out, open(OUT_PATH, "w"), indent=1)
        print(f"-> {site} termine ({time.time() - t0:.0f} s cumule), sauvegarde {OUT_PATH}", flush=True)

    if not a.smoke and set(a.sites) == set(SITES):
        out["verdict"] = verdict(out)
        json.dump(out, open(OUT_PATH, "w"), indent=1)
        print("verdict :", out["verdict"], flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
