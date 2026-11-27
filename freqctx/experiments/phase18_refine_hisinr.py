"""Phase 18 -- entrainement de la PASSE 2 (RefineNet, models/refine.py) sur un melange enrichi en
regime de SINR eleve (variante de experiments/phase15_refine.py, demande F. Leduc-Primeau,
2026-11-27 ; voir FINDINGS.md phase18 "Adapter la 2e passe au site").

Constat : en zero-shot, la passe 2 actuelle (poids weights/phase15/refine_L4.weights.h5) egale ou
depasse WMMSE 300 it. sur St-Catherine/Decarie a tous les SNR et sur les 3 sites a 20 dB, mais
reste sous WMMSE a Ericsson 30/40 dB (35,93/36,06 x2 contre 36,19 ; 46,47/46,97 x2 contre 47,89).
L'adaptation au site (phase18_ft_refine) n'aide pas : ce n'est pas un probleme de SITE mais de
REGIME (SINR tres eleve, canaux bien conditionnes, peu de dispersion), vraisemblablement sous-
represente dans le melange 3GPP M8K4+M64K8 (umi/uma/rma, SNR U(0,30)) utilise par phase15_refine.

Choix des donnees "haut SINR" (documente ici, pas de donnee de site) : SOUS-ECHANTILLONNAGE PAR
cond(G) du MEME melange 3GPP (umi/uma/rma, meme fonction experiments.phase13_train_massive.pool,
import SANS MODIFIER) plutot qu'un forcage LOS/RMa cote Sionna. Deux raisons : (1) ce critere est
deja valide dans ce depot comme proxy du regime "facile" (experiments/phase15_diag.py,
"umi_quartile_bas_condG", quantile 0.25) ; (2) gen_single_sector_topology (Sionna) n'expose pas de
parametre pour forcer LOS -- seul le scenario RMa a lui seul ne garantit pas un cond(G) bas (sa
dispersion depend aussi de la distance tiree). On garde donc, par taille (M,K), le quart des
tirages du melange SPLIT au cond(G) le plus bas (hisinr_pool ci-dessous) : canaux ou les
utilisateurs sont bien separes spatialement -- le regime qui produit un SINR eleve et peu de
fuite structurelle, comme Ericsson.

Difference avec phase15_refine.py (SEULE difference ; architecture, base figee, sortie
initialisee a zero, verification de depart, budget 60 000 pas / batch 2048, identiques) :
  - a chaque pas, un lot sur deux (s%4 : 2 lots sur 4) vient du melange 3GPP HABITUEL (meme pool,
    seed0=1, SNR U(0,30), comme phase15_refine) et un lot sur deux du melange HISINR (meme
    scenarios, seed0=5, cond(G) quartile bas, SNR U(15,40) -- la zone ou l'ecart a WMMSE est
    observe a Ericsson) ; dans chaque moitie, (8,4)/(64,8) restent equilibres 50/50 comme avant.
  - le reste (perte, reference RZF, clip de gradient, cosine decay, 60000 pas) est identique.

Verification de depart reutilisee de phase15_refine (check_start_equals_base, own_vars -- IMPORT
SANS MODIFIER) : au pas 0 DELTA_C = 0 -> C2 = C1 -> meme debit que la base seule.

    python -u -m experiments.phase18_refine_hisinr [--steps 60000]
"""
import argparse
import os
import time

import json
import keras
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from experiments.phase13_bench import SNRS, rates
from experiments.phase13_train_massive import SPLIT, pool
from experiments.phase14_finetune import load_base
from experiments.phase15_refine import check_start_equals_base, own_vars
from models.refine import RefineNet
from precoders.classical import rzf
from train import rate_per_sample
from utils import RESULTS, git_commit

W18 = "/export/tmp/sala/freqctx/weights/phase18"
STEPS, BATCH = 60000, 2048
HISINR_FRAC = 0.25                       # quantile bas de cond(G), comme phase15_diag (quartile_bas_condG)
HISINR_SEED0 = 5                         # distinct du seed0=1 des lots "habituels"
NORMAL_SNR = (0.0, 30.0)                 # comme phase15_refine.py
HISINR_SNR = (15.0, 40.0)                # zone ou l'ecart a WMMSE est observe (Ericsson 30/40 dB)


def hisinr_pool(M, K, seed0=HISINR_SEED0, frac=HISINR_FRAC):
    """Memes scenarios 3GPP (umi/uma/rma) et meme convention que pool() (SPLIT, import sans
    modifier), mais ne garde que la fraction `frac` des tirages au cond(G) le plus bas -- canaux
    bien conditionnes / utilisateurs bien separes, substitut 3GPP (sans donnee de site) du
    regime Ericsson (SINR tres eleve, peu de fuite structurelle)."""
    h = pool(M, K, seed0)
    g = tf.matmul(tf.cast(h, tf.complex128), tf.cast(h, tf.complex128), adjoint_b=True)
    cond = np.linalg.cond(g.numpy())
    thresh = float(np.quantile(cond, frac))
    h_sel = tf.gather(h, np.where(cond <= thresh)[0])
    return h_sel, thresh, cond


def make_step(net, p, opt, tv, snr_lo, snr_hi):
    """p : pool [n, K, M] (normal ou hisinr) ; snr_lo/snr_hi : plage de SNR d'entrainement propre
    a ce lot (identique a phase15_refine.py pour le lot normal)."""
    n = p.shape[0]

    @tf.function
    def step():
        h = tf.gather(p, tf.random.uniform([BATCH], 0, n, dtype=tf.int32))
        snr = tf.random.uniform([BATCH], snr_lo, snr_hi)
        nv = 10.0 ** (-snr / 10.0)
        ref = rate_per_sample(h, rzf(h, tf.reshape(nv, [-1, 1, 1])), nv)
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(rate_per_sample(h, net(h, snr), nv) / tf.stop_gradient(ref))
        g = tape.gradient(loss, tv)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, tv))
        return loss
    return step


def main(steps=STEPS):
    keras.utils.set_random_seed(0)
    t0 = time.time()
    pools_normal = {k: pool(*k, seed0=1) for k in SPLIT}
    pools_hisinr, thresh_by_key = {}, {}
    for k in SPLIT:
        h_sel, thresh, cond = hisinr_pool(*k)
        pools_hisinr[k] = h_sel
        thresh_by_key[k] = {"cond_seuil": thresh, "cond_median_avant": float(np.median(cond)),
                            "cond_median_apres": float(np.median(np.linalg.cond(
                                tf.matmul(tf.cast(h_sel, tf.complex128), tf.cast(h_sel, tf.complex128),
                                          adjoint_b=True).numpy())))}
    print({f"M{m}K{k}_normal": int(pools_normal[(m, k)].shape[0]) for (m, k) in SPLIT}, flush=True)
    print({f"M{m}K{k}_hisinr": int(pools_hisinr[(m, k)].shape[0]) for (m, k) in SPLIT}, thresh_by_key,
          f"donnees {time.time() - t0:.0f} s", flush=True)

    base = load_base()
    base.trainable = False
    net = RefineNet(base, d=16, hidden=32, layers=4)
    for p in pools_normal.values():
        net(p[:2], tf.zeros([2]))

    check_start_equals_base(net, pools_normal[(8, 4)][:256])

    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps))
    tv = own_vars(net)
    print(f"variables entrainees (passe 2 seule) : {sum(int(tf.size(v)) for v in tv)}", flush=True)

    keys = list(SPLIT)                                           # [(64,8), (8,4)] ou [(8,4), (64,8)]
    # 4 lots : (normal, keys[0]), (normal, keys[1]), (hisinr, keys[0]), (hisinr, keys[1])
    stepfns = [make_step(net, pools_normal[keys[0]], opt, tv, *NORMAL_SNR),
               make_step(net, pools_normal[keys[1]], opt, tv, *NORMAL_SNR),
               make_step(net, pools_hisinr[keys[0]], opt, tv, *HISINR_SNR),
               make_step(net, pools_hisinr[keys[1]], opt, tv, *HISINR_SNR)]
    tags = [f"normal_{keys[0]}", f"normal_{keys[1]}", f"hisinr_{keys[0]}", f"hisinr_{keys[1]}"]
    hist = []
    for s in range(1, steps + 1):
        i = s % 4
        loss = stepfns[i]()
        if s % 5000 < 2 or s == steps:
            hist.append({"step": s, "obj": -float(loss), "config": tags[i], "time_s": round(time.time() - t0)})
            print("   ", hist[-1], flush=True)

    os.makedirs(W18, exist_ok=True)
    wpath = os.path.join(W18, "refine_L4_hisinr.weights.h5")
    net.save_weights(wpath)

    # sanity rapide (memes tirages que phase15_refine.py, pour comparer au depart)
    sionna_config.seed = 4026                                           # memes tirages que phase13_massive
    gen = ChannelGenerator(SystemConfig(num_bs_ant=64, num_users=8, num_rb=50, scenario="umi", num_slots=1,
                                        relative_power=True))
    hm = tf.concat([gen(8)[:, 0] for _ in range(8)], 0)
    sionna_config.seed = 4026                                           # memes tirages que phase13_bench (umi)
    gen = ChannelGenerator(SystemConfig(scenario="umi", num_slots=1, relative_power=True))
    h4 = tf.concat([gen(16)[:, 0] for _ in range(12)], 0)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"), "steps": steps,
                    "hisinr_frac_condG": HISINR_FRAC, "hisinr_seed0": HISINR_SEED0,
                    "normal_snr_db": list(NORMAL_SNR), "hisinr_snr_db": list(HISINR_SNR),
                    "cond_G_seuils": {str(k): v for k, v in thresh_by_key.items()}},
           "snrs_db": SNRS, "history": hist, "params_passe2": sum(int(tf.size(v)) for v in tv),
           "M64K8_base+passe2hisinr": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4_base+passe2hisinr": rates(lambda x, nv, s: net(x, tf.fill([x.shape[0]], float(s))), h4),
           "M64K8_base": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), hm),
           "M8K4_base": rates(lambda x, nv, s: base(x, tf.fill([x.shape[0]], float(s))), h4)}
    print({k: [round(v, 2) for v in out[k]] for k in
           ("M64K8_base+passe2hisinr", "M8K4_base+passe2hisinr", "M64K8_base", "M8K4_base")}, flush=True)
    os.makedirs(os.path.join(RESULTS, "phase18"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "phase18", "refine_hisinr_train.json"), "w"), indent=1)
    print(f"-> poids {wpath}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    main(steps=a.steps)
