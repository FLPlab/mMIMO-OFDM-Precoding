"""Phase 14 -- BER reel sous CSI imparfait (suite de phase14_csi, demande de F. Leduc-Primeau : "BER avec
une modulation"). Reutilise phase14_csi tel quel : memes precodeurs (notre precodeur CSI parfait, notre
precodeur entraine erreur synthetique, WMMSE 100 it., RZF, + MRT), meme deploiement (Sionna UMi,
puissances reelles, M8K4, estimation LS + LMMSE ou LS + lineaire, pilote 0/10/20 dB).

Pourquoi un nouveau script : le BER de phase14_csi est une approximation gaussienne a partir du SINR de
chaque flux, moyennee sur les flux. Les precodeurs qui maximisent le debit somme concentrent la puissance
sur les bons utilisateurs ; les flux faibles ont un SINR tres bas, donc un BER moyen PLUS ELEVE que RZF
(qui etale la puissance) -- trompeur si presente seul comme "le BER du precodeur".

Methode :
1. BER REEL par simulation symbole (pas l'approximation gaussienne) : QAM carre Gray-code (QPSK/16/64/
   256-QAM, energie moyenne 1 par flux actif), transmis a travers le canal precode h^T W, bruit
   thermique, PUIS egalisation par le gain effectif CONNU du recepteur (genie : h_k^T w_k exact). Choix
   documente : on isole ainsi l'effet du precodeur / de l'erreur de CSI a la BS, sans ajouter le bruit
   d'une estimation du gain effectif cote recepteur (pilote precode) qui est une question separee.
   Demodulation dure (plus proche voisin). Auto-test au demarrage (`_selftest`) : cas SISO sans
   interference, compare le BER simule a l'approximation gaussienne (doit coller a quelques % pres).
2. Deux metriques non trompeuses (la 3e, pire utilisateur, est omise : les deux suffisent et evitent la
   sur-ingenierie) :
   a. MODULATION ADAPTATIVE par flux : ordre QAM choisi par flux a partir du SINR (log-SINR -> approx.
      gaussienne inversee) pour une cible BER 1e-3 (table MCS a 4 points QPSK/16/64/256-QAM) ; un flux
      dont le SINR ne permet meme pas QPSK a 1e-3 n'est pas transmis ce slot (0 bit). Debit EFFECTIF =
      somme des bits/symbole reellement alloues ; BER = celui mesure par simulation a cet ordre (verifie
      si la cible est tenue en pratique). C'est la comparaison la plus juste entre precodeurs.
   b. BER a MODULATION FIXE (QPSK et 16-QAM), en excluant les flux "eteints" par le precodeur (puissance
      du flux < 10 % de la puissance moyenne par flux, 1/K) ; le nombre moyen de flux actifs est reporte.
3. Sortie : results/phase14/ber.json (meta git_commit comme les autres scripts, via utils.save) et
   results/phase14/ber.pdf /.png (debit effectif + BER 16-QAM, LMMSE pilote 10 dB).

    python -u -m experiments.phase14_ber
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tensorflow as tf
from scipy.special import erfc, erfcinv
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from csi import UplinkSounder
from experiments.phase14_csi import B, DROPS, PILOTS, SNRS, W14, load_perfect, train_synthetic
from experiments.phase7_context import cov_mixture, sound_power_controlled
from metrics import sinr
from models.gram_precoder import GramPrecoder
from precoders.classical import mrt, rzf, wmmse
from utils import BLUES, COLORS, save

MCS_BITS = [2, 4, 6, 8]            # QPSK, 16-QAM, 64-QAM, 256-QAM
TARGET_BER = 1e-3
N_REP = 8                          # symboles QAM simules par echantillon de canal et par flux
POWER_ACTIVE = 0.1                 # flux "actif" (modulation fixe) : puissance > 10 % de la moyenne 1/K


# ---------------------------------------------------------------- QAM Gray, reel/simule ----------
def _gray_index(bits):
    """bits [..., b] int32 (0/1, code Gray, MSB en premier) -> indice naturel [...] int32."""
    b = bits.shape[-1]
    nat = [bits[..., 0]]
    for i in range(1, b):
        nat.append(tf.bitwise.bitwise_xor(nat[-1], bits[..., i]))
    nat = tf.stack(nat, axis=-1)
    w = tf.constant([2 ** (b - 1 - i) for i in range(b)], tf.int32)
    return tf.reduce_sum(nat * w, axis=-1)


def _index_gray(idx, b):
    """idx [...] int32 (0..2^b-1) -> bits Gray [..., b] int32."""
    nat = tf.stack([tf.bitwise.bitwise_and(tf.bitwise.right_shift(idx, b - 1 - i), 1) for i in range(b)], -1)
    gray = [nat[..., 0]]
    for i in range(1, b):
        gray.append(tf.bitwise.bitwise_xor(nat[..., i - 1], nat[..., i]))
    return tf.stack(gray, axis=-1)


def _qam_params(bits):
    """bits par symbole (pair) -> (bits par axe I/Q, nb de niveaux, normalisation pour E|x|^2 = 1)."""
    bpd = bits // 2
    L = 2 ** bpd
    return bpd, L, float(np.sqrt(3.0 / (2.0 * (L ** 2 - 1))))


def real_ber(h, w, nv, bits, n_rep=N_REP, seed=0):
    """BER REEL par simulation symbole. h [N, K, M] (canal vrai), w [N, M, K] (precodeur, ||w||_F = 1),
    nv : variance de bruit (scalaire), bits [N, K] int32 dans {0, 2, 4, 6, 8} (0 = flux non transmis ce
    slot). Egalisation par le gain effectif CONNU (genie, cf. docstring module). Retourne (erreurs,
    bits totaux) sommes sur tous les flux actifs x repetitions."""
    tf.random.set_seed(seed)
    N, K = h.shape[0], bits.shape[1]
    hw = tf.matmul(tf.cast(h, tf.complex64), tf.cast(w, tf.complex64))        # [N, K, K]
    g = tf.linalg.diag_part(hw)
    g = tf.where(tf.abs(g) < 1e-12, tf.cast(1e-12, g.dtype), g)
    x = tf.zeros([N, K, n_rep], tf.complex64)
    stash, used = {}, [b for b in MCS_BITS if bool(tf.reduce_any(tf.equal(bits, b)))]
    for b in used:                                            # ignore les ordres jamais choisis (cout evite)
        bpd, L, norm = _qam_params(b)
        raw = tf.random.uniform([N, K, n_rep, 2 * bpd], 0, 2, dtype=tf.int32)
        idx_i, idx_q = _gray_index(raw[..., :bpd]), _gray_index(raw[..., bpd:])
        sym = tf.complex((2.0 * tf.cast(idx_i, tf.float32) - (L - 1)) * norm,
                          (2.0 * tf.cast(idx_q, tf.float32) - (L - 1)) * norm)
        x = x + sym * tf.cast(tf.equal(bits, b), tf.complex64)[..., None]
        stash[b] = raw
    if not used:
        return 0.0, 0.0
    noise = tf.complex(tf.random.normal([N, K, n_rep]), tf.random.normal([N, K, n_rep])) \
        * tf.cast(tf.sqrt(tf.cast(nv, tf.float32) / 2.0), tf.complex64)
    y = tf.einsum("nkj,njr->nkr", hw, x) + noise
    xhat = y / g[..., None]
    err, tot = 0.0, 0.0
    for b in used:
        bpd, L, norm = _qam_params(b)
        ri, rq = tf.math.real(xhat) / norm, tf.math.imag(xhat) / norm
        ii = tf.cast(tf.clip_by_value(tf.round((ri + (L - 1)) / 2.0), 0, L - 1), tf.int32)
        iq = tf.cast(tf.clip_by_value(tf.round((rq + (L - 1)) / 2.0), 0, L - 1), tf.int32)
        hat = tf.concat([_index_gray(ii, bpd), _index_gray(iq, bpd)], axis=-1)
        mask = tf.cast(tf.equal(bits, b), tf.float32)[..., None, None]
        err += float(tf.reduce_sum(tf.cast(tf.not_equal(hat, stash[b]), tf.float32) * mask))
        tot += float(tf.reduce_sum(mask)) * n_rep * 2 * bpd
    return err, tot


def _ber_approx(sinr_lin, bits):
    """Approximation gaussienne usuelle du BER M-QAM carre Gray (M = 2**bits), a partir du SINR lineaire
    (meme convention SINR = Es/N0 que metrics.sinr : redonne Q(sqrt(SINR)) en QPSK, 0.75 Q(sqrt(SINR/5))
    en 16-QAM, comme phase14_csi.metrics)."""
    M = 2.0 ** bits
    A = (4.0 / bits) * (1.0 - 1.0 / np.sqrt(M))
    c = 3.0 / (M - 1.0)
    return A * 0.5 * erfc(np.sqrt(c * sinr_lin / 2.0))


def _mcs_thresholds(target=TARGET_BER):
    th = {}
    for b in MCS_BITS:
        M = 2.0 ** b
        A = (4.0 / b) * (1.0 - 1.0 / np.sqrt(M))
        c = 3.0 / (M - 1.0)
        z = np.sqrt(2.0) * erfcinv(2.0 * target / A)
        th[b] = float(z ** 2 / c)
    return th


MCS_THRESH = _mcs_thresholds()


def select_mcs(sinr_lin):
    """SINR lineaire [..., K] -> ordre QAM par flux (bits/symbole, 0 si meme QPSK a 1e-3 n'est pas tenu)."""
    bits = tf.zeros_like(sinr_lin, dtype=tf.int32)
    for b in MCS_BITS:
        bits = tf.where(sinr_lin >= MCS_THRESH[b], tf.fill(tf.shape(sinr_lin), b), bits)
    return bits


def _selftest():
    """Sanite : cas SISO sans interference (K=1), BER simule vs approximation gaussienne a partir du
    SINR. Doit coller a quelques % pres (ni precodeur ni interference, seule la chaine QAM est testee)."""
    print("autotest : BER simule vs approximation gaussienne (SISO, sans interference)", flush=True)
    for b in (2, 4):
        for s in (2.0, 8.0, 20.0):
            h = tf.ones([6000, 1, 1], tf.complex64)
            w = tf.ones([6000, 1, 1], tf.complex64)
            bits = tf.fill([6000, 1], b)
            err, tot = real_ber(h, w, 1.0 / s, bits, n_rep=300, seed=1)
            sim, approx = err / tot, float(_ber_approx(np.array([s]), b)[0])
            flag = "OK" if abs(sim - approx) < max(0.25 * approx, 0.01) else "ECART"
            print(f"   {b} bits/symb, SINR {s:.0f} : simule {sim:.4f} / approx {approx:.4f}  [{flag}]",
                  flush=True)


# ---------------------------------------------------------------- precodeurs et evaluation ----------
def load_synthetic():
    """Charge le precodeur entraine (erreur synthetique) de phase14_csi s'il existe deja ; sinon le
    (re)entraine (meme fonction, meme comportement -- aucune modification de phase14_csi)."""
    path = os.path.join(W14, "precoder_synth_csi.weights.h5")
    if os.path.exists(path):
        net = GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm")
        net(tf.zeros([1, 4, 8], tf.complex64), tf.zeros([1]))
        net.load_weights(path)
        return net
    return train_synthetic()


def evaluate_ber(fns, h_est, h, err=None):
    """Pour chaque precodeur (meme signature fn(h_est, nv, snr, err_var) -> w que phase14_csi) et chaque
    SNR de donnees : debit effectif + BER reel en modulation adaptative, BER reel + flux actifs en QPSK
    et 16-QAM fixes, et le debit Shannon (sum log2(1+SINR), pour comparaison avec phase14_csi)."""
    hf, he = tf.reshape(h, [-1, 4, 8]), tf.reshape(h_est, [-1, 4, 8])
    ef = None if err is None else tf.reshape(err, [-1, 4, 8])
    out = {}
    for name, fn in fns.items():
        res = {"debit_shannon": [], "debit_adapt": [], "ber_adapt": [],
               "ber_qpsk": [], "actifs_qpsk": [], "ber_16qam": [], "actifs_16qam": []}
        for si, snr in enumerate(SNRS):
            nv = 10.0 ** (-snr / 10.0)
            w = fn(he, nv, snr, ef)
            K = w.shape[-1]
            sg = sinr(hf, w, nv)                                        # [N, K], canal vrai
            res["debit_shannon"].append(float(tf.reduce_mean(
                tf.reduce_sum(tf.math.log1p(sg) / tf.math.log(2.0), -1))))

            bits_a = select_mcs(sg)
            e, t = real_ber(hf, w, nv, bits_a, seed=1000 * si + hash(name) % 997)
            res["debit_adapt"].append(float(tf.reduce_sum(tf.cast(bits_a, tf.float32))) / hf.shape[0])
            res["ber_adapt"].append(e / max(t, 1.0))

            pw = tf.reduce_sum(tf.abs(w) ** 2, axis=-2)                  # [N, K]
            act = tf.cast(pw > POWER_ACTIVE / K, tf.int32)
            for b, key in ((2, "qpsk"), (4, "16qam")):
                e2, t2 = real_ber(hf, w, nv, act * b, seed=2000 * si + b + hash(name) % 997)
                res[f"ber_{key}"].append(e2 / max(t2, 1.0))
                res[f"actifs_{key}"].append(float(tf.reduce_mean(tf.cast(act, tf.float32))) * K)
        out[name] = res
    return out


def main():
    _selftest()
    net_s, net_p = load_synthetic(), load_perfect()
    mrt_net = mrt  # trivial, pas d'entrainement
    cfg = SystemConfig(scenario="umi", num_slots=1, relative_power=True)
    sionna_config.seed = 4026
    gen = ChannelGenerator(cfg)
    h = tf.concat([gen(B)[:, 0] for _ in range(DROPS // B)], 0)
    cov = cov_mixture()
    out = {"snrs_db": SNRS, "note": "debit_adapt en bits/symbole (= bps/Hz) ; BER par simulation reelle "
                                    "(pas l'approximation gaussienne) ; flux actifs sur K=4",
           "csi_parfait": {}, "estime": {}}

    perfect = {"precodeur entraine (CSI parfait)": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
               "WMMSE 100 it.": lambda x, nv, s, e: wmmse(x, nv, num_iter=100),
               "RZF": lambda x, nv, s, e: rzf(x, nv),
               "MRT": lambda x, nv, s, e: mrt_net(x, nv)}
    t0 = time.time()
    out["csi_parfait"] = evaluate_ber(perfect, h, h)
    print(f"CSI parfait fait ({time.time() - t0:.0f} s)", flush=True)

    for est in ("lmmse", "lin"):
        snd = UplinkSounder(cfg, est, cov if est == "lmmse" else None)
        for p in PILOTS:
            t0 = time.time()
            pr = [sound_power_controlled(snd, h[j:j + B], float(p), return_err_var=True)
                  for j in range(0, DROPS, B)]
            he, ev = tf.concat([a for a, _ in pr], 0), tf.concat([b for _, b in pr], 0)
            fns = {"precodeur entraine CSI parfait": lambda x, nv, s, e: net_p(x, tf.fill([x.shape[0]], float(s))),
                   "precodeur entraine erreur synthetique": lambda x, nv, s, e: net_s(x, tf.fill([x.shape[0]], float(s))),
                   "WMMSE 100 it." + (" robuste" if est == "lmmse" else ""):
                       (lambda x, nv, s, e: wmmse(x, nv, num_iter=100, err_var=e)) if est == "lmmse"
                       else (lambda x, nv, s, e: wmmse(x, nv, num_iter=100)),
                   "RZF": lambda x, nv, s, e: rzf(x, nv),
                   "MRT": lambda x, nv, s, e: mrt_net(x, nv)}
            r = evaluate_ber(fns, he, h, ev if est == "lmmse" else None)
            out["estime"][f"{est}_pilote{p}"] = r
            print(f"{est} pilote {p} dB @20dB ({time.time() - t0:.0f} s) : " + " | ".join(
                f"{k} debit {v['debit_adapt'][4]:.2f} (BER adapt {v['ber_adapt'][4]:.1e}, "
                f"BER16 {v['ber_16qam'][4]:.3f}/{v['actifs_16qam'][4]:.1f} flux)"
                for k, v in r.items()), flush=True)

    fig = make_figure(out)
    save("phase14", "ber", out, fig)


def make_figure(out):
    cond = out["estime"]["lmmse_pilote10"]
    pal = {"precodeur entraine CSI parfait": BLUES[5], "precodeur entraine erreur synthetique": BLUES[2],
           "WMMSE 100 it. robuste": COLORS["WMMSE"], "RZF": COLORS["RZF"], "MRT": COLORS["MRT"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, c in pal.items():
        v = cond[name]
        ax1.plot(out["snrs_db"], v["debit_adapt"], color=c, marker="o", ms=4, label=name)
        ax2.semilogy(out["snrs_db"], np.maximum(v["ber_16qam"], 1e-6), color=c, marker="o", ms=4, label=name)
    ax1.set_xlabel("SNR de données (dB)")
    ax1.set_ylabel("débit effectif (bits/symbole = bps/Hz)")
    ax1.set_title("modulation adaptative par flux (cible BER 1e-3)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    ax2.set_xlabel("SNR de données (dB)")
    ax2.set_ylabel("BER (16-QAM, flux actifs)")
    ax2.set_title("BER réel, modulation fixe 16-QAM", fontsize=9)
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle("BER réel sous CSI imparfait -- UMi, M=8, K=4, pilote LMMSE 10 dB", fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
