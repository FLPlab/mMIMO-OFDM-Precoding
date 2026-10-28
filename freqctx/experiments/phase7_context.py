"""Phase 7 -- ce qu'apporte le contexte frequentiel, a precodeur fixe.

Precodeur : transformer par paires d'utilisateurs retenu en phase 6 (attention
triangulaire, G reinjecte, 8 couches distinctes, W = h^H C). M8K4, pilote
10 dB, melange UMi + UMa + RMa, 1 056 tirages / scenario, memes tirages
d'evaluation que les phases 3-6 (192 / scenario, graines 2026 + 10 i).

Variantes (--variant) :
    none      : aucun contexte -- le precodeur voit le LS (interpolation
                lineaire) de chaque SC, seule.
    lmmse     : contexte classique -- le precodeur voit l'estimation LMMSE
                (covariance frequentielle moyenne des 3 scenarios), par SC.
    sub       : encodeur frequentiel (domaine des retards) restreint a des
                sous-bandes de --rb RB (1, 3, 17 ; 51 = toute la bande = phase 6).
    separate  : toute la bande, mais entrainement SEPARE : (1) l'encodeur
                apprend a reproduire le vrai canal (erreur quadratique
                normalisee), (2) il est fige et le precodeur apprend le debit.
    simple    : toute la bande, encodeur SANS attention entre retards
                (un MLP par retard, --est_layers 0).
    lmmse_axial : reference apprise (phase 8) -- LMMSE (melange) + precodeur de
                la litterature type 2D-Gformer (models/axial_precoder.py),
                4 couches distinctes, meme calcul que notre precodeur en M8K4.
    tarb      : reference apprise (phase 8) -- TA-RB du memoire reecrit
                (models/tarb.py), sur le LS des pilotes, memes donnees et perte.
    interp    : toute la bande ; le precodeur n'est calcule qu'a 1 SC sur
                --step (+ la derniere), W est interpole lineairement sur les
                autres SC, et le debit est appris sur TOUTES les SC.

    python -u -m experiments.phase7_context --variant none|lmmse|sub --rb 3|separate|simple
"""
import argparse
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import estimate_freq_cov
from models.delay_domain import DelayDomainPrecoder
from models.gram_precoder import GramPrecoder
from train import DropPool, csi_pool, evaluate, run_variant
from utils import RESULTS, git_commit

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT = ["umi", "uma", "rma"], 10.0
TRAIN_DROPS, EVAL_DROPS, STEPS = 1056, 192, 80000
W7 = "/export/tmp/sala/freqctx/weights/phase7"
PRECODER = dict(d=16, hidden=64, layers=8, mix="triangular_g", tied=False)


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def cov_mixture():
    covs = []
    for i, scen in enumerate(SCENARIOS):                       # memes graines que la phase 3
        sionna_config.seed = 2027 + 10 * i
        covs.append(estimate_freq_cov(ChannelGenerator(SystemConfig(scenario=scen))))
    return tf.add_n(covs) / len(covs)


def data(estimator, cov=None):
    """(train_in, train_true), {scen: (eval_in, eval_true)} ; bande complete [N, F, K, M]."""
    csi = CSIConfig(estimator, PILOT, 0)
    tr_in, tr_true, ev = [], [], {}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        a, b = csi_pool(cfg, csi, TRAIN_DROPS, seed=1 + i, cov_freq=cov, flat=False)
        tr_in.append(a), tr_true.append(b)
        ev[scen] = csi_pool(cfg, csi, EVAL_DROPS, 2026 + 10 * i, cov_freq=cov, flat=False)
    return (tf.concat(tr_in, 0), tf.concat(tr_true, 0)), ev


def train_encoder_mse(model, pool, steps, lr=1e-3):
    """Etape 1 de --variant separate : l'encodeur seul, erreur quadratique normalisee."""
    enc = model.estimator
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(lr, steps))

    @tf.function
    def step():
        h_in, h, _ = pool.sample(4)
        with tf.GradientTape() as tape:
            e = model.encode(h_in) - h
            loss = tf.reduce_mean(tf.abs(e) ** 2) / tf.reduce_mean(tf.abs(h) ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, enc.trainable_variables), enc.trainable_variables))
        return loss

    hist = []
    for s in range(1, steps + 1):
        loss = step()
        if s % (steps // 4) == 0:
            hist.append({"step": s, "nmse_db": float(10 * np.log10(float(loss)))})
            print("    encodeur seul", hist[-1], flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["none", "lmmse", "sub", "separate", "simple", "interp", "lmmse_axial", "tarb"])
    ap.add_argument("--rb", type=int, default=None)
    ap.add_argument("--step", type=int, default=None, help="interp : precodeur a 1 SC sur step, W interpole")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--quick", action="store_true", help="test de la chaine (32 tirages, 200 pas)")
    a = ap.parse_args()
    global TRAIN_DROPS, EVAL_DROPS
    if a.quick:
        TRAIN_DROPS, EVAL_DROPS, a.steps = 32, 32, 200
    name = a.variant + (f"_{a.rb}rb" if a.variant == "sub" else "") + (f"_s{a.step}" if a.variant == "interp" else "")
    print("==", name, flush=True)
    t0 = time.time()
    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M")},
           "variant": name, "pilot_snr_db": PILOT, "snrs_db": SNRS_DB, "precoder": PRECODER,
           "train_drops_per_scenario": TRAIN_DROPS, "steps": a.steps}
    os.makedirs(W7, exist_ok=True)

    if a.variant in ("none", "lmmse", "lmmse_axial"):          # precodeur seul, SC par SC
        if a.variant == "none":
            (ti, tt), ev = data("ls_lin")
        else:
            (ti, tt), ev = data("ls_lmmse", cov_mixture())    # lmmse et lmmse_axial
        if a.variant == "lmmse_axial":
            from models.axial_precoder import AxialPrecoder
            net = AxialPrecoder(d=16, hidden=64, heads=4, layers=4, tied=False)
            out["precoder"] = {"arch": "axial (type 2D-Gformer)", "d": 16, "hidden": 64, "heads": 4,
                               "layers": 4, "tied": False}
        else:
            net = GramPrecoder(8, **PRECODER)
        run = run_variant(name, net, (flat(ti), flat(tt)), (flat(ev["umi"][0][:16]), flat(ev["umi"][1][:16])),
                          SNRS_DB, {"RZF": [1.0] * len(SNRS_DB)}, a.steps, seed=0, loss="rel_rzf",
                          weights_dir=W7)
        rates = {s: evaluate(net, (flat(ev[s][0]), flat(ev[s][1])), SNRS_DB) for s in SCENARIOS}
    else:
        (ti, tt), ev = data("ls_lin")
        pool = DropPool(ti, tt)
        cfg = SystemConfig()
        if a.variant == "tarb":
            from models.tarb import TARBPrecoder
            net = TARBPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant)
            out["precoder"] = {"arch": "TA-RB du memoire", "D": 128, "heads": 4, "layers": 4,
                               "tokens_per_rb": 4}
        else:
            net = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, banded=True, oversample=2,
                                    est_layers=0 if a.variant == "simple" else 2,
                                    subband_rb=a.rb if a.variant == "sub" else None,
                                    anchor_step=a.step if a.variant == "interp" else None,
                                    precoder_arch="pair", **PRECODER)
        if a.variant == "separate":
            net(ti[:1], tf.zeros([1]))
            keras.utils.set_random_seed(0)
            out["encoder_history"] = train_encoder_mse(net, pool, a.steps // 2)
            net.estimator.trainable = False                    # fige : seul le precodeur apprend
        run = run_variant(name, net, pool, DropPool(ev["umi"][0][:16], ev["umi"][1][:16]), SNRS_DB,
                          {"RZF": [1.0] * len(SNRS_DB)}, a.steps, seed=0, loss="rel_rzf", weights_dir=W7)
        rates = {s: evaluate(net, DropPool(*ev[s]), SNRS_DB) for s in SCENARIOS}
    for s in SCENARIOS:
        print(f"{s} : " + " ".join(f"{x:.2f}" for x in rates[s]), flush=True)
    out.update({"params": run["params"], "macs_per_sc": run["macs_per_sc"], "rates": rates,
                "training_history": run["history"], "time_s": round(time.time() - t0)})
    phase = "phase8" if a.variant in ("lmmse_axial", "tarb") else "phase7"
    os.makedirs(os.path.join(RESULTS, phase), exist_ok=True)
    path = os.path.join(RESULTS, phase, f"{name}{'_quick' if a.quick else ''}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
