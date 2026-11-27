"""Phase 16 -- la forme structuree (models/structured_precoder.py, poids
weights/phase16/structured_T20.weights.h5) predit (lambda_k, p_k) par un petit transformer de 2
couches (~171 k MAC/SC hors Newton-Schulz), puis Newton-Schulz (20 pas, ~10 k MAC) inverse
A = Lambda G + sigma^2 I, C = X P^(1/2). (lambda, p) sont des parametres de regularisation/puissance
par utilisateur : ils devraient varier LENTEMENT en frequence (FINDINGS.md phase 15, erreur
d'interpolation = fuite d'interference). Ici : transformer SEULEMENT aux sous-porteuses ancres (1 sur
s), (lambda, p) interpoles (lineaire ou log-lineaire) entre ancres, puis G, Newton-Schulz, C, W
EXACTS a CHAQUE sous-porteuse (le transformer ne voit que les ancres, le reste de la chaine est
recalcule partout). Compare a l'interpolation de W (ce qu'on fait aujourd'hui, experiments/
phase15_adaptive_step.rates_per_drop) et au calcul a chaque SC.

EVALUATION SEULE, aucun entrainement. CSI parfait, M8K4, puissances reelles, SNR 20/30/40 dB. Jeux :
memes generateurs large bande que experiments.phase15_adaptive_step (= experiments.phase13_bench.
channels : Sionna UMi/UMa/RMa + St-Catherine/Decarie/Ericsson, psi2az). s in {1,3,6,12,24}.

Variantes, par s :
  (a) structure complet (transformer + Newton-Schulz) aux ancres, W interpole LINEAIREMENT (identique
      a phase15_adaptive_step.rates_per_drop, reutilise sans modification, net = structure T20).
  (b) (lambda, p) PREDITS AUX ANCRES SEULEMENT par le transformer, interpoles (lineaire OU
      log-lineaire) sur toute la bande, puis a CHAQUE SC : G = H H^H (exact), A = Lambda G + sigma^2 I,
      Newton-Schulz 20 pas (exact), C = X P^(1/2) (exact), W = H^H C, normalize_power.
References (independantes de s) :
  - structure a chaque SC (s=1, (a) et (b) coincident alors exactement : anchors = toutes les SC).
  - reseau libre 3GPP (weights/phase13/precoder_relpow_mix.weights.h5, experiments.phase14_finetune.
    load_base) avec W interpole au meme s (phase15_adaptive_step.rates_per_drop).
  - WMMSE 300 it. a chaque SC (precoders.classical.wmmse) : 20/30 dB REPRIS de results/phase13/
    bench.json (memes tirages, memes SNR, non recalcule) ; 40 dB calcule ici (CHUNK batches).

MAC/SC (convention experiments/phase15_macs.py = models.*.macs_per_sc) :
  - structure complet (T20) a chaque SC : 193 280 (= net.macs_per_sc(4,8), FINDINGS phase 16).
  - decompose en : transformer+lecture (K*K paires, 2 couches TriangularBlock) = ne depend que des
    ancres, donc /s ; G + Newton-Schulz(20 pas) + W = EXACT a chaque SC, independant de s.
  - (a) MAC/SC(s) = 193280 / s (+ interpolation de W, negligeable, meme convention que
    phase15_adaptive_step/phase16_anchor.py qui l'ignorent aussi).
  - (b) MAC/SC(s) = transformer_macs / s + (G + NS + W)_macs  (independant du mode lin/log).
  - reseau libre interpole : 717312 / s (MAC_BASE, phase15_adaptive_step).
  - WMMSE 300 : reference hors ligne, non quantifiee en MAC comparable (phase16_eval.run_macs).

Conclusion recherchee : plus grand s tel que (b) reste a <= 0.1 bps/Hz de s=1, et son cout MAC/SC.

    python -u -m experiments.phase16_struct_anchor [--ndrops N] [--sets ...]
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from config import SystemConfig
from experiments.phase13_bench import CHUNK, OUT as PHASE13_OUT, SETS, channels
from experiments.phase14_finetune import load_base
from experiments.phase15_adaptive_step import rates_per_drop
from models.structured_precoder import StructuredPrecoder, newton_schulz
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

W16 = "/export/tmp/sala/freqctx/weights/phase16"
BENCH_PATH = os.path.join(PHASE13_OUT, "bench.json")
OUT_PATH = os.path.join(RESULTS, "phase16", "struct_anchor.json")
SNRS = [20.0, 30.0, 40.0]
STEPS = [1, 3, 6, 12, 24]
T_STEPS = 20
BATCH = 16
WMMSE_CHUNK = CHUNK
RATE_TOL = 0.1          # bps/Hz, critere "reste a <= 0.1 bps/Hz de s=1"


def load_structured(t_steps=T_STEPS):
    net = StructuredPrecoder(8, ns_steps=t_steps, d=16, hidden=32, layers=2)
    net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
    net.load_weights(os.path.join(W16, f"structured_T{t_steps}.weights.h5"))
    return net


def struct_macs_breakdown(net, K=4, M=8):
    """Decompose net.macs_per_sc (meme formule, SANS MODIFIER models/structured_precoder.py) en
    transformer+lecture (ancres seulement, /s) et G+Newton-Schulz+W (exact, chaque SC)."""
    d, hd, L, T = net.d, net.hidden, net.L, net.T
    n_feat = 7
    per_edge = n_feat * 2 * d + L * (4 * d * hd + hd * 2 * d) + L * 2 * (2 * d + 3) * (2 * d)
    mac_g = 4 * K * K * M
    mac_edges = K * K * per_edge
    mac_tri = L * 4 * K ** 3 * d
    mac_read = K * (2 * d) * 2
    mac_ns = T * 8 * K ** 3
    mac_w = 4 * M * K * K
    transformer = mac_edges + mac_tri + mac_read
    exact = mac_g + mac_ns + mac_w
    total = transformer + exact
    assert total == net.macs_per_sc(K, M), (total, net.macs_per_sc(K, M))
    return transformer, exact, total


def interp_scalar(x_a, anchors, F, mode):
    """x_a [N,A,K] (valeurs aux ancres) -> [N,F,K] interpole lineairement (mode="lin") ou
    log-lineairement (mode="log", x_a > 0) sur toute la bande F. anchors : indices SC tries (dont 0 et
    F-1), meme convention j/t que phase15_adaptive_step.rates_per_drop."""
    A = len(anchors)
    j = np.clip(np.searchsorted(anchors, np.arange(F), side="right") - 1, 0, A - 2)
    t_np = ((np.arange(F) - anchors[j]) / (anchors[j + 1] - anchors[j])).astype(np.float32)
    t = tf.constant(t_np[None, :, None])
    xj, xj1 = tf.gather(x_a, j, axis=1), tf.gather(x_a, j + 1, axis=1)
    if mode == "lin":
        return (1 - t) * xj + t * xj1
    floor = tf.cast(1e-8, xj.dtype)       # softplus(lambda) peut sous-flotter a 0.0 (pas de +eps comme p) :
    lxj, lxj1 = tf.math.log(tf.maximum(xj, floor)), tf.math.log(tf.maximum(xj1, floor))   # evite log(0)=-inf -> nan
    return tf.exp((1 - t) * lxj + t * lxj1)


def rates_variant_b(h, net, s, snr, mode, batch=BATCH):
    """(lambda, p) predits aux ancres seulement, interpoles (mode), puis G/Newton-Schulz/C/W EXACTS a
    chaque SC. h [B,F,K,M]. Renvoie debit moyen sur F, par tirage [B]."""
    B, F, K, M = h.shape
    anchors = np.unique(np.append(np.arange(0, F, s), F - 1))
    A = len(anchors)
    nv = 10.0 ** (-snr / 10.0)
    nvc = tf.cast(nv, tf.complex64)
    eye = tf.eye(K, dtype=tf.complex64)[None]
    out = []
    for b in range(0, B, batch):
        hb = h[b:b + batch]
        nb = hb.shape[0]
        ha = tf.cast(tf.reshape(tf.gather(hb, anchors, axis=1), [-1, K, M]), tf.complex64)
        g_a = tf.matmul(ha, ha, adjoint_b=True)
        snr_a = tf.fill([nb * A], float(snr))
        lam_a, p_a = net.lambda_p(g_a, snr_a)
        lam_a = tf.reshape(lam_a, [nb, A, K])
        p_a = tf.reshape(p_a, [nb, A, K])
        lam_full = interp_scalar(lam_a, anchors, F, mode)
        p_full = interp_scalar(p_a, anchors, F, mode)
        hflat = tf.cast(tf.reshape(hb, [-1, K, M]), tf.complex64)
        g_full = tf.matmul(hflat, hflat, adjoint_b=True)
        lam_flat = tf.cast(tf.reshape(lam_full, [-1, K]), tf.complex64)
        p_flat = tf.cast(tf.reshape(p_full, [-1, K]), tf.complex64)
        a_mat = lam_flat[:, :, None] * g_full + nvc * eye
        x = newton_schulz(a_mat, net.T)
        c = x * tf.sqrt(p_flat)[:, None, :]
        w = normalize_power(tf.matmul(hflat, c, adjoint_a=True))
        r = rate_per_sample(hflat, w, nv)
        out.append(tf.reduce_mean(tf.reshape(r, [nb, F]), axis=1))
    return tf.concat(out, 0).numpy()


def wmmse300_compute(h, snr, batch=WMMSE_CHUNK):
    """WMMSE 300 it. a `snr` dB, a chaque SC (calcule directement sur h -- utilise pour 40 dB, absent
    de bench.json qui s'arrete a 30 dB, et pour --ndrops ou le sous-ensemble ne correspond pas au tirage
    complet de bench.json)."""
    hflat = tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])
    nv = 10.0 ** (-snr / 10.0)
    out = []
    for i in range(0, hflat.shape[0], batch):
        w = wmmse(hflat[i:i + batch], nv, num_iter=300)
        out.append(rate_per_sample(hflat[i:i + batch], w, nv))
    return float(tf.reduce_mean(tf.concat(out, 0)))


def wmmse300_from_bench(name, snr):
    bench = json.load(open(BENCH_PATH))
    idx = bench["snrs_db"].index(int(snr))
    return float(bench["sets"][name]["WMMSE 300 it."][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--ndrops", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    cfg = SystemConfig()
    net_struct = load_structured(T_STEPS)
    net_free = load_base()
    mac_struct_transformer, mac_struct_exact, mac_struct_total = struct_macs_breakdown(net_struct)
    mac_free_total = int(net_free.macs_per_sc(4, 8))
    print(f"MAC/SC structure T{T_STEPS} total {mac_struct_total} "
          f"(transformer+lecture {mac_struct_transformer} aux ancres, "
          f"G+NS{T_STEPS}+W {mac_struct_exact} exact chaque SC) ; reseau libre {mac_free_total}", flush=True)

    out = {"meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                    "steps": STEPS, "snrs_db": SNRS, "t_steps_ns": T_STEPS, "rate_tol_bps_hz": RATE_TOL,
                    "mac_struct_total": mac_struct_total, "mac_struct_transformer": mac_struct_transformer,
                    "mac_struct_exact_g_ns_w": mac_struct_exact, "mac_free_total": mac_free_total},
           "sets": {}}

    for name in a.sets:
        t1 = time.time()
        h = channels(name)
        if a.ndrops:
            h = h[:a.ndrops]
        entry = {"n": int(h.shape[0]), "snrs": {}}
        for snr in SNRS:
            s_entry = {"a_struct_interpW": {}, "b_lin": {}, "b_log": {}, "free_interpW": {}}
            for s in STEPS:
                ra = float(np.mean(rates_per_drop(h, net_struct, s, snr)))
                rb_lin = float(np.mean(rates_variant_b(h, net_struct, s, snr, "lin")))
                rb_log = float(np.mean(rates_variant_b(h, net_struct, s, snr, "log")))
                rf = float(np.mean(rates_per_drop(h, net_free, s, snr)))
                s_entry["a_struct_interpW"][str(s)] = {"rate": ra, "mac": mac_struct_total / s}
                s_entry["b_lin"][str(s)] = {"rate": rb_lin,
                                             "mac": mac_struct_transformer / s + mac_struct_exact}
                s_entry["b_log"][str(s)] = {"rate": rb_log,
                                             "mac": mac_struct_transformer / s + mac_struct_exact}
                s_entry["free_interpW"][str(s)] = {"rate": rf, "mac": mac_free_total / s}
            if snr in (20.0, 30.0) and not a.ndrops:
                try:
                    s_entry["wmmse300"] = {"rate": wmmse300_from_bench(name, snr), "source": "bench.json"}
                except (FileNotFoundError, KeyError, ValueError) as e:
                    s_entry["wmmse300"] = {"rate": None, "error": str(e)}
            else:
                s_entry["wmmse300"] = {"rate": wmmse300_compute(h, snr), "source": f"calcule ({snr:.0f} dB)"}
            r1 = s_entry["b_lin"]["1"]["rate"]
            best_s, best_mac = 1, mac_struct_total
            for s in STEPS:
                loss_lin = r1 - s_entry["b_lin"][str(s)]["rate"]
                loss_log = r1 - s_entry["b_log"][str(s)]["rate"]
                if max(loss_lin, loss_log) <= RATE_TOL:
                    best_s, best_mac = s, s_entry["b_lin"][str(s)]["mac"]
            s_entry["b_best_s_le_0p1"] = {"s": best_s, "mac": best_mac}
            entry["snrs"][str(snr)] = s_entry
            print(f"{name} {snr:.0f}dB : struct(s=1) {r1:.2f} | "
                  + " | ".join(f"s={s} a={s_entry['a_struct_interpW'][str(s)]['rate']:.2f} "
                              f"blin={s_entry['b_lin'][str(s)]['rate']:.2f} "
                              f"blog={s_entry['b_log'][str(s)]['rate']:.2f} "
                              f"libre={s_entry['free_interpW'][str(s)]['rate']:.2f}" for s in STEPS)
                  + f" | wmmse300 {s_entry['wmmse300']['rate']:.2f} | b: s<=0.1 jusqu'a s={best_s} "
                    f"({best_mac / 1e3:.0f}k MAC)", flush=True)
        out["sets"][name] = entry
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        json.dump(out, open(OUT_PATH, "w"), indent=1)
        print(f"{name} termine ({time.time() - t1:.0f} s)", flush=True)

    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
