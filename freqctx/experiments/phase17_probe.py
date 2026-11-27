"""Phase 17 -- que calcule chaque couche du precodeur libre ? EVALUATION SEULE (aucun entrainement ;
seules des sondes lineaires ajustees en forme fermee par moindres carres).

Reseau : GramPrecoder(64, d=16, hidden=32, layers=8, mix="triangular_g", tied=False, inp="norm"),
poids phase13 (experiments.phase14_finetune.load_base). Activations intermediaires obtenues en appelant
les blocs a la main (meme calcul que GramPrecoder.coeffs, verifie), sans modifier le modele.

Donnees : UMi M8K4 (tirages phase13_bench) et Ericsson (moitie test, seed 77, comme phase15_diag),
CSI parfait, SNR 10 / 20 / 30 dB. Separation apprentissage / test des sondes PAR TIRAGE (1re moitie des
tirages -> N_TR SC, 2e moitie -> N_TE SC).

Conventions (domaine normalise du modele) : hp = h normalise par utilisateur, D = diag(||h_k||),
G_n = hp hp^H, W = hp^H C (puis normalisation de puissance). Cibles de type C (inverse regularisee,
C de RZF, C de WMMSE 300) exprimees dans la JAUGE du reseau : phase par colonne alignee sur le C final
du reseau et meme norme de Frobenius par echantillon (le debit est invariant a la phase par colonne et a
l'echelle globale ; ce n'est qu'un choix de representant).

  1. layers   : lecture finale de C appliquee a l'etat apres chaque couche ; lecture lineaire ajustee
                par couche (vers le C final du reseau et vers le C de WMMSE) ; debit et fuite mediane.
  2. probes   : sondes lineaires (par paire, diagonale / hors diagonale separees) vers G^n, inverse,
                C RZF, C WMMSE, (lambda, p) ; R^2 hors diagonale et erreur relative.
  3. pca      : dimension de l'etat (90 / 99 % de variance), sondes restreintes aux r premieres
                composantes ; decomposition de l'etat en parties invariante / covariante sous rotation de
                phase par utilisateur.
  4. scale    : normes avant LayerNorm, information portee par la norme, residu de structure
                ||I - B C_l||, correlations fuite finale / dynamique.
  5. ablation : couche retiree (identite) ou produit triangulaire annule, debit a 20 / 30 dB.

    python -u -m experiments.phase17_probe
"""
import json
import logging
import os
import time

import keras
import numpy as np
import tensorflow as tf
from sionna.phy.channel import subcarrier_frequencies

tf.get_logger().setLevel(logging.ERROR)

from config import SystemConfig
from experiments.phase13_bench import channels as bench_channels
from experiments.phase13_lab import read_cells
from experiments.phase14_finetune import TEST_DROPS, load_base, site_drops
from experiments.phase15_diag import fit_structure_batch, leak_db_sig_intf
from models.gram_precoder import _cplx, _real
from precoders.classical import normalize_power, wmmse
from train import rate_per_sample
from utils import RESULTS, git_commit

OUT_DIR = os.path.join(RESULTS, "phase17")
OUT_PATH = os.path.join(OUT_DIR, "probe.json")
SNRS = [10, 20, 30]
N_TR, N_TE = 3072, 2048
CHUNK = 2048
GPOW = [1, 2, 3, 4, 8]
N_PHASE, R_PHASE = 512, 32


# ---------------------------------------------------------------------- donnees et passage avant


def datasets():
    cfg = SystemConfig()
    freqs = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing).numpy()
    raw = {"umi": bench_channels("umi"),
           "ericsson": site_drops(read_cells("ericsson", "psi2az")[1::2], TEST_DROPS, freqs, seed=77, per_sc=False)}
    rng = np.random.default_rng(17)
    out = {}
    for name, h in raw.items():
        half = h.shape[0] // 2

        def pick(hh, m):
            f = tf.reshape(hh, [-1, hh.shape[-2], hh.shape[-1]])
            return tf.gather(f, np.sort(rng.choice(f.shape[0], m, replace=False)))
        out[name] = (pick(h[:half], N_TR), pick(h[half:], N_TE))
        print(f"donnees {name}: {tuple(h.shape)} -> {N_TR} SC app. / {N_TE} SC test", flush=True)
    return out


def embed(net, h, snr):
    """Entree du modele (copie de GramPrecoder.coeffs, inp="norm") -> hp, g, e0."""
    hp, pdb = net._norm(h)
    g = tf.matmul(hp, hp, adjoint_b=True)
    K = h.shape[1]
    eye = tf.broadcast_to(tf.eye(K)[None], tf.shape(tf.math.real(g)))
    s = tf.fill(tf.shape(eye), float(snr) / 10.0)
    parts = [tf.math.real(g), tf.math.imag(g), tf.abs(g), eye, s,
             tf.broadcast_to(pdb[:, :, None], tf.shape(eye)), tf.broadcast_to(pdb[:, None, :], tf.shape(eye))]
    return hp, pdb, g, _cplx(net.init(tf.stack(parts, -1)))


def block_call(blk, e, g, zero_m=False):
    """TriangularBlock.call (edge_g, sans sym) ; zero_m : produit triangulaire annule."""
    x = blk.norm(_real(e))
    x = tf.concat([x, tf.stack([tf.math.real(g), tf.math.imag(g), tf.abs(g)], -1)], -1)
    if zero_m:
        m = tf.zeros(e.shape, e.dtype)
    else:
        K = tf.cast(tf.shape(e)[1], tf.complex64)
        m = tf.einsum("njlc,nlkc->njkc", _cplx(blk.a(x)), _cplx(blk.b(x))) / K
    return e + _cplx(blk.f2(blk.f1(tf.concat([x, _real(m)], axis=-1))))


def read_c(net, e):
    return _cplx(net.read(_real(e)))[..., 0]


def states(net, h, snr, skip=None, zero_m=None):
    """-> hp [N,K,M], g [N,K,K], liste de L+1 etats [N,K,K,d] (numpy complex64)."""
    hps, gs, es = [], [], []
    for i in range(0, h.shape[0], CHUNK):
        hp, _, g, e = embed(net, h[i:i + CHUNK], snr)
        cur = [e.numpy()]
        for li, blk in enumerate(net.blocks):
            if li != skip:
                e = block_call(blk, e, g, zero_m=(zero_m == li or zero_m == "all"))
            cur.append(e.numpy())
        hps.append(hp.numpy()); gs.append(g.numpy()); es.append(np.stack(cur, 1))
    return np.concatenate(hps), np.concatenate(gs), np.concatenate(es)     # es [N, L+1, K, K, d]


def w_from_c(hp, c):
    return normalize_power(tf.matmul(tf.constant(hp, tf.complex64), tf.constant(c.astype(np.complex64)), adjoint_a=True))


def rate_leak(h, w, nv):
    r = rate_per_sample(h, w, nv).numpy()
    leak, _, _ = leak_db_sig_intf(h, w)
    return r, leak                                                          # [N], [N,K] dB


def c_of_states(net, es_l):
    return np.concatenate([read_c(net, tf.constant(es_l[i:i + CHUNK])).numpy() for i in range(0, len(es_l), CHUNK)])


# ---------------------------------------------------------------------- cibles


def align(t, cref, hp):
    """Jauge du reseau, mesuree dans le domaine W = hp^H C : phase par colonne alignee sur celle de cref,
    meme norme de W par echantillon."""
    hpc = np.conj(np.transpose(hp, (0, 2, 1)))
    wt, wr = hpc @ t, hpc @ cref
    inner = np.sum(np.conj(wt) * wr, axis=1)                               # [N,K]
    t = t * np.exp(1j * np.angle(inner))[:, None, :]
    s = np.linalg.norm(wr, axis=(1, 2)) / np.maximum(np.linalg.norm(wt, axis=(1, 2)), 1e-30)
    return t * s[:, None, None]


def targets(h, hp, gn, c_net, w_wm, nv, lam_net, p_net, lam_wm, p_wm, w_struct):
    h = h.astype(np.complex128); hp = hp.astype(np.complex128); gn = gn.astype(np.complex128)
    K = h.shape[1]
    eye = np.eye(K)[None]
    n2 = np.sum(np.abs(h) ** 2, -1)                                        # ||h_k||^2
    D = np.sqrt(n2)
    t = {}
    for n in GPOW:
        gp = np.linalg.matrix_power(gn, n)
        t[f"G^{n}"] = gp * (K / np.linalg.norm(gp, axis=(1, 2)))[:, None, None]
    alpha_n = K * nv / np.mean(n2, -1)                                     # alpha de RZF en domaine normalise
    t["inv"] = align(np.linalg.inv(gn + alpha_n[:, None, None] * eye), c_net, hp)
    gphys = h @ np.conj(np.transpose(h, (0, 2, 1)))
    t["C_rzf"] = align(D[:, :, None] * np.linalg.inv(gphys + K * nv * eye), c_net, hp)
    t["C_wmmse"] = align(np.linalg.solve(gn, hp @ w_wm.astype(np.complex128)), c_net, hp)
    # forme de Bjornson EXACTE avec les (lambda, p) ajustes au reseau : ce que le reseau "vise"
    t["C_struct"] = align(np.linalg.solve(gn, hp @ w_struct.astype(np.complex128)), c_net, hp)
    t["C_net"] = c_net.astype(np.complex128)
    sc = {"loglam_net": np.log(lam_net * n2 / nv), "logp_net": np.log(p_net) - np.mean(np.log(p_net), -1, keepdims=True),
          "loglam_wmmse": np.log(lam_wm * n2 / nv), "logp_wmmse": np.log(p_wm) - np.mean(np.log(p_wm), -1, keepdims=True)}
    return t, sc


# ---------------------------------------------------------------------- sondes lineaires


def feats(e):                                                              # [..., d] complexe -> [..., 2d] reel
    return np.concatenate([e.real, e.imag], -1).astype(np.float64)


def lstsq(x, y, w=None):
    """Moindres carres (biais inclus), poids par ligne optionnels."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
    if w is not None:
        xb, y = xb * np.sqrt(w)[:, None], y * np.sqrt(w)[:, None]
    a = xb.T @ xb
    a += 1e-8 * np.trace(a) / a.shape[0] * np.eye(a.shape[0])
    if w is None:
        return np.linalg.solve(a, xb.T @ y)
    wts = np.linalg.solve(a, xb.T @ y)
    return wts


def apply(wts, x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], 1) @ wts


def probe_matrix(xtr, ttr, xte):
    """xtr [N,K,K,F] reel, ttr [N,K,K] complexe -> prediction test [N',K,K] complexe.
    Sondes separees diagonale / hors diagonale (lineaires, avec biais). Moindres carres PONDERES par
    1 / ||T_n||^2 (chaque echantillon compte autant ; sinon les canaux mal conditionnes, a C enorme, dominent)."""
    K = xtr.shape[1]
    dm = np.eye(K, dtype=bool)
    wn = 1.0 / np.maximum(np.mean(np.abs(ttr) ** 2, axis=(1, 2)), 1e-30)
    out = np.zeros(xte.shape[:3], np.complex128)
    for mask in (dm, ~dm):
        y = np.stack([ttr[:, mask].real.ravel(), ttr[:, mask].imag.ravel()], 1)
        wr = np.repeat(wn, mask.sum())
        wts = lstsq(xtr[:, mask].reshape(-1, xtr.shape[-1]), y, wr)
        p = apply(wts, xte[:, mask].reshape(-1, xte.shape[-1]))
        out[:, mask] = (p[:, 0] + 1j * p[:, 1]).reshape(xte.shape[0], -1)
    return out


HP_TE = None                                                               # hp du test (erreur dans le domaine W)


def metrics(pred, t):
    """r2_off : R^2 hors diagonale (domaine C) ; relerr : erreur relative globale (domaine C) ;
    relerr_W : erreur relative de W = hp^H C (domaine qui compte pour le debit), moyenne sur echantillons."""
    K = t.shape[1]
    off = ~np.eye(K, dtype=bool)
    err = np.abs(pred - t) ** 2
    to = t[:, off]
    sst = np.sum(np.abs(to - to.mean()) ** 2)
    out = {"r2_off": float(1 - err[:, off].sum() / sst),
           "relerr": float(np.sqrt(err.sum() / np.sum(np.abs(t) ** 2))),
           "relerr_off": float(np.sqrt(err[:, off].sum() / np.sum(np.abs(to) ** 2)))}
    if HP_TE is not None and HP_TE.shape[0] == t.shape[0]:
        hpc = np.conj(np.transpose(HP_TE, (0, 2, 1)))
        ew = np.linalg.norm(hpc @ (pred - t), axis=(1, 2)) / np.linalg.norm(hpc @ t, axis=(1, 2))
        out["relerr_W"] = float(np.mean(ew))
        out["relerr_W_med"] = float(np.median(ew))
    return out


def probe_scalar(xtr, ytr, xte, yte):
    """Cible reelle par utilisateur, lue sur le jeton diagonal."""
    K = xtr.shape[1]
    idx = np.arange(K)
    wts = lstsq(xtr[:, idx, idx].reshape(-1, xtr.shape[-1]), ytr.reshape(-1, 1))
    p = apply(wts, xte[:, idx, idx].reshape(-1, xte.shape[-1]))[:, 0]
    y = yte.ravel()
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def pca_basis(x):
    """x [N,K,K,F] -> moyenne, vecteurs propres (colonnes, par variance decroissante), variances."""
    f = x.reshape(-1, x.shape[-1])
    mu = f.mean(0)
    ev, vec = np.linalg.eigh(np.cov((f - mu).T))
    o = np.argsort(ev)[::-1]
    return mu, vec[:, o], np.maximum(ev[o], 0)


def ncomp(ev, frac):
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, frac) + 1)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------- analyses par (jeu, SNR)


def run_one(net, name, h_tr, h_te, snr):
    nv = 10.0 ** (-snr / 10.0)
    res = {}
    L = len(net.blocks)

    # passage avant, verification contre net(...)
    hp_tr, g_tr, es_tr = states(net, h_tr, snr)
    hp_te, g_te, es_te = states(net, h_te, snr)
    global HP_TE
    HP_TE = hp_te.astype(np.complex128)
    w_ref = net(h_te[:256], tf.fill([256], float(snr))).numpy()
    w_man = w_from_c(hp_te[:256], c_of_states(net, es_te[:256, L])).numpy()
    res["verif_max_abs_diff_W"] = float(np.max(np.abs(w_ref - w_man)))

    c_net_tr, c_net_te = c_of_states(net, es_tr[:, L]), c_of_states(net, es_te[:, L])
    w_net_tr, w_net_te = w_from_c(hp_tr, c_net_tr), w_from_c(hp_te, c_net_te)
    w_wm_tr, w_wm_te = wmmse(h_tr, nv, num_iter=300), wmmse(h_te, nv, num_iter=300)
    r_net, leak_net = rate_leak(h_te, w_net_te, nv)
    r_wm, leak_wm = rate_leak(h_te, w_wm_te, nv)
    res["ref"] = {"debit_reseau": float(r_net.mean()), "debit_wmmse300": float(r_wm.mean()),
                  "fuite_med_reseau": float(np.median(leak_net)), "fuite_med_wmmse300": float(np.median(leak_wm))}
    # phase des colonnes du reseau (diagonale de C) : jauge naturelle ?
    res["ref"]["phase_diag_C_net_med_deg"] = float(np.degrees(np.median(np.abs(np.angle(np.einsum("nkk->nk", c_net_te))))))

    # (lambda, p) ajustes (structure de Bjornson, phase15_diag) sur le reseau et sur WMMSE
    fits = {}
    for tag, wtr, wte in (("net", w_net_tr, w_net_te), ("wmmse", w_wm_tr, w_wm_te)):
        wstr, ltr, ptr, etr = fit_structure_batch(h_tr, wtr, nv)
        wste, lte, pte, ete = fit_structure_batch(h_te, wte, nv)
        fits[tag] = (ltr, ptr, lte, pte, wstr.numpy(), wste.numpy())
        res["ref"][f"debit_struct_exact_{tag}"] = float(tf.reduce_mean(rate_per_sample(h_te, wste, nv)))
        res["ref"][f"fit_struct_{tag}_erreur_rel_med"] = float(np.median(ete))
    t_tr, s_tr = targets(h_tr.numpy(), hp_tr, g_tr, c_net_tr, w_wm_tr.numpy(), nv, fits["net"][0], fits["net"][1],
                         fits["wmmse"][0], fits["wmmse"][1], fits["net"][4])
    t_te, s_te = targets(h_te.numpy(), hp_te, g_te, c_net_te, w_wm_te.numpy(), nv, fits["net"][2], fits["net"][3],
                         fits["wmmse"][2], fits["wmmse"][3], fits["net"][5])
    # parties "nouvelles" : cible moins sa meilleure prediction lineaire depuis l'etat initial e0 (la part de
    # G^n deja lineairement contenue dans G n'est pas un calcul du reseau)
    x0_tr, x0_te = feats(es_tr[:, 0]), feats(es_te[:, 0])
    for tg in [f"G^{n}" for n in GPOW if n > 1] + ["inv", "C_wmmse", "C_struct"]:
        t_tr[f"{tg}_perp"] = t_tr[tg] - probe_matrix(x0_tr, t_tr[tg], x0_tr)
        t_te[f"{tg}_perp"] = t_te[tg] - probe_matrix(x0_tr, t_tr[tg], x0_te)
    res["ref"]["ecart_rel_C_net_vs_C_wmmse"] = float(np.linalg.norm(t_te["C_wmmse"] - t_te["C_net"]) / np.linalg.norm(t_te["C_net"]))
    res["ref"]["ecart_W_C_net_vs_C_wmmse"] = metrics(t_te["C_net"], t_te["C_wmmse"])
    res["ref"]["ecart_W_C_net_vs_C_struct"] = metrics(t_te["C_net"], t_te["C_struct"])

    # ---- 1 + 2 : par couche
    lay, prb, pcs, scl = [], [], [], []
    for l in range(L + 1):
        e_tr, e_te = es_tr[:, l], es_te[:, l]
        x_tr, x_te = feats(e_tr), feats(e_te)
        # 1. lecture finale et lecture ajustee
        c_l = c_of_states(net, e_te)
        r, lk = rate_leak(h_te, w_from_c(hp_te, c_l), nv)
        ent = {"couche": l, "debit_lecture_finale": float(r.mean()), "fuite_med_lecture_finale": float(np.median(lk))}
        for tg in ("C_net", "C_struct", "C_wmmse"):
            pred = probe_matrix(x_tr, t_tr[tg], x_te)
            r2, lk2 = rate_leak(h_te, w_from_c(hp_te, pred), nv)
            ent[f"debit_lecture_ajustee_{tg}"] = float(r2.mean())
            ent[f"fuite_med_lecture_ajustee_{tg}"] = float(np.median(lk2))
        lay.append(ent)
        # 2. sondes
        pe = {"couche": l}
        for tg in t_tr:
            pe[tg] = metrics(probe_matrix(x_tr, t_tr[tg], x_te), t_te[tg])
        for tg in s_tr:
            pe[tg] = probe_scalar(x_tr, s_tr[tg], x_te, s_te[tg])
        # sondes depuis LN(e) (ce que voient les produits et le FFN) -- couches 0..L-1
        if l < L:
            ln = lambda e: net.blocks[l].norm(tf.constant(feats(e).astype(np.float32))).numpy().astype(np.float64)
            ln_tr, ln_te = ln(e_tr), ln(e_te)
            for tg in ("C_wmmse", "C_struct", "inv", "C_net"):
                pe[f"LN:{tg}"] = metrics(probe_matrix(ln_tr, t_tr[tg], ln_te), t_te[tg])
        prb.append(pe)
        # 3. ACP + sondes restreintes
        mu, vec, ev = pca_basis(x_tr)
        pc = {"couche": l, "n90": ncomp(ev, 0.90), "n99": ncomp(ev, 0.99), "n999": ncomp(ev, 0.999),
              "part_var_1re": float(ev[0] / ev.sum())}
        if l in (2, 4, 6, L):
            for rr in (1, 2, 4, 8, 16, 32):
                ptr_, pte_ = (x_tr - mu) @ vec[:, :rr], (x_te - mu) @ vec[:, :rr]
                for tg in ("C_net", "C_struct", "C_wmmse", "inv"):
                    pc[f"r{rr}:{tg}"] = metrics(probe_matrix(ptr_, t_tr[tg], pte_), t_te[tg])
        if l < L:
            mu2, vec2, ev2 = pca_basis(ln_tr)
            pc["LN_n90"], pc["LN_n99"] = ncomp(ev2, 0.90), ncomp(ev2, 0.99)
        pcs.append(pc)
        # 4. echelle
        nrm = np.linalg.norm(e_te, axis=-1)                                 # [N,K,K]
        K = nrm.shape[1]
        dm = np.eye(K, dtype=bool)
        cv = np.std(nrm[:, ~dm], 1) / np.mean(nrm[:, ~dm], 1)
        lo = np.log(nrm[:, ~dm]).ravel()
        corr2 = lambda y: float(np.corrcoef(lo, y)[0, 1] ** 2)
        se = {"couche": l, "norme_moy_diag": float(nrm[:, dm].mean()), "norme_moy_hors": float(nrm[:, ~dm].mean()),
              "cv_norme_hors_par_echantillon": float(cv.mean()),
              "R2_lognorme_vs_log|G|": corr2(np.log(np.abs(g_te[:, ~dm]) + 1e-12).ravel()),
              "R2_lognorme_vs_log|Cwmmse|": corr2(np.log(np.abs(t_te["C_wmmse"][:, ~dm]) + 1e-12).ravel())}
        if l < L:
            lnn = np.linalg.norm(ln_te, axis=-1)
            se["norme_apres_LN_moy"], se["norme_apres_LN_cv"] = float(lnn.mean()), float(lnn.std() / lnn.mean())
        # residu de structure ||s B C_l Theta - P^1/2|| / ||P^1/2|| avec (lambda, p) ajustes au reseau final :
        # C* = D (Lambda G + s2 I)^-1 P^1/2 (domaine normalise) <=> B C* = P^1/2, B = Lambda D G_n + s2 D^-1
        lam, p = fits["net"][2], fits["net"][3]
        hh = h_te.numpy().astype(np.complex128)
        Dn = np.linalg.norm(hh, axis=-1)
        B = lam[:, :, None] * Dn[:, :, None] * g_te.astype(np.complex128) + nv * np.eye(K)[None] / Dn[:, :, None]
        Mm = B @ c_l.astype(np.complex128)
        Mm = Mm * np.exp(-1j * np.angle(np.einsum("nkk->nk", Mm)))[:, None, :]
        sp = np.sqrt(p)
        s = np.sum(np.real(np.einsum("nkk->nk", Mm)) * sp, -1) / np.sum(np.abs(Mm) ** 2, (1, 2))
        resid = np.linalg.norm(s[:, None, None] * Mm - sp[:, None, :] * np.eye(K)[None], axis=(1, 2)) / np.linalg.norm(sp, axis=-1)
        se["residu_struct_med"] = float(np.median(resid))
        scl.append(se)
        print(f"  {name} {snr}dB couche {l}: debit lect. finale {ent['debit_lecture_finale']:6.2f} "
              f"(fuite {ent['fuite_med_lecture_finale']:6.1f})  ajustee->Cnet {ent['debit_lecture_ajustee_C_net']:6.2f}  "
              f"R2off G^1 {pe['G^1']['r2_off']:.3f} G^2 {pe['G^2']['r2_off']:.3f} inv {pe['inv']['r2_off']:.3f} "
              f"Cwm {pe['C_wmmse']['r2_off']:.3f} (W {pe['C_wmmse']['relerr_W']:.3f}) Cst W {pe['C_struct']['relerr_W']:.3f}  n99 {pc['n99']}  resid {se['residu_struct_med']:.3f}", flush=True)
    res["couches"], res["sondes"], res["acp"], res["echelle"] = lay, prb, pcs, scl

    # ---- 4 bis : fuite finale par echantillon vs dynamique
    gp = h_te.numpy().astype(np.complex128)
    gphys = gp @ np.conj(np.transpose(gp, (0, 2, 1)))
    cond_n = np.log10(np.linalg.cond(g_te.astype(np.complex128)))
    cond_p = np.log10(np.linalg.cond(gphys))
    n2 = np.sum(np.abs(gp) ** 2, -1)
    spread = 10 * np.log10(n2.max(-1) / n2.min(-1))
    # fuite par echantillon : mediane sur les utilisateurs ACTIFS (||w_k||^2 > 1 % de la puissance totale ;
    # WMMSE eteint des utilisateurs au fort SNR, leur "fuite" n'a pas de sens)
    pw_net = np.sum(np.abs(w_net_te.numpy()) ** 2, 1)                         # [N,K]
    pw_wm = np.sum(np.abs(w_wm_te.numpy()) ** 2, 1)
    act_n, act_w = pw_net > 0.01, pw_wm > 0.01
    ln_ = np.array([np.median(a[m]) if m.any() else np.nan for a, m in zip(leak_net, act_n)])
    lw_ = np.array([np.median(a[m]) if m.any() else np.nan for a, m in zip(leak_wm, act_w)])
    ok = np.isfinite(ln_) & np.isfinite(lw_)
    q = np.quantile(cond_n, [0.25, 0.5, 0.75])
    bins = np.digitize(cond_n, q)
    qs = np.quantile(spread, [0.25, 0.5, 0.75])
    bins_s = np.digitize(spread, qs)
    # par utilisateur : la fuite suit-elle la puissance du flux (plancher d'erreur ABSOLU sur W) ?
    _, sig_n, intf_n = leak_db_sig_intf(h_te, w_net_te)
    _, sig_w, intf_w = leak_db_sig_intf(h_te, w_wm_te)
    sig_rel_n = 10 * np.log10(np.maximum(sig_n / n2, 1e-30))                 # |hp_k w_k|^2 (dB)
    intf_rel_n = 10 * np.log10(np.maximum(intf_n / n2, 1e-30))               # interference / ||h_k||^2 (dB)
    a_ = act_n.ravel()
    slope = np.polyfit(sig_rel_n.ravel()[a_], (intf_rel_n - sig_rel_n).ravel()[a_], 1)[0]
    pdb_k = 10 * np.log10(n2 / n2.mean(-1, keepdims=True))
    res["fuite_vs_dynamique"] = {
        "frac_utilisateurs_actifs": {"reseau": float(act_n.mean()), "wmmse300": float(act_w.mean())},
        "spearman_fuite_reseau": {"log_cond_Gn": spearman(ln_[ok], cond_n[ok]), "log_cond_Gphys": spearman(ln_[ok], cond_p[ok]),
                                  "ecart_puissance_dB": spearman(ln_[ok], spread[ok])},
        "spearman_exces_fuite_vs_wmmse": {"log_cond_Gn": spearman((ln_ - lw_)[ok], cond_n[ok]),
                                          "log_cond_Gphys": spearman((ln_ - lw_)[ok], cond_p[ok]),
                                          "ecart_puissance_dB": spearman((ln_ - lw_)[ok], spread[ok])},
        "fuite_med_reseau_par_quartile_condGn": [float(np.nanmedian(ln_[bins == b])) for b in range(4)],
        "fuite_med_wmmse_par_quartile_condGn": [float(np.nanmedian(lw_[bins == b])) for b in range(4)],
        "fuite_med_reseau_par_quartile_ecart_puissance": [float(np.nanmedian(ln_[bins_s == b])) for b in range(4)],
        "fuite_med_wmmse_par_quartile_ecart_puissance": [float(np.nanmedian(lw_[bins_s == b])) for b in range(4)],
        "par_utilisateur_reseau": {
            "pente_fuite_dB_vs_signal_dB": float(slope),
            "spearman_fuite_vs_puissance_flux": spearman(leak_net[act_n], 10 * np.log10(pw_net[act_n])),
            "spearman_fuite_vs_pdb_utilisateur": spearman(leak_net[act_n], pdb_k[act_n]),
            "interference_abs_med_dB": float(np.median(intf_rel_n[act_n])),
            "interference_abs_med_dB_wmmse": float(np.median(10 * np.log10(np.maximum(intf_w / n2, 1e-30))[act_w]))}}

    # ---- 3 bis : invariance / covariance de phase par utilisateur (hors diagonale)
    if snr == 30:
        res["phase"] = phase_decomp(net, h_te[:N_PHASE], snr, nv)

    # ---- 5 : ablations
    if snr in (20, 30):
        abl = {}
        for li in range(L):
            for kind in ("couche", "produit"):
                hp_a, _, es_a = states(net, h_te, snr, skip=li if kind == "couche" else None,
                                       zero_m=li if kind == "produit" else None)
                r, lk = rate_leak(h_te, w_from_c(hp_a, c_of_states(net, es_a[:, L])), nv)
                abl[f"{kind}_{li + 1}"] = {"debit": float(r.mean()), "fuite_med": float(np.median(lk))}
        hp_a, _, es_a = states(net, h_te, snr, zero_m="all")
        r, lk = rate_leak(h_te, w_from_c(hp_a, c_of_states(net, es_a[:, L])), nv)
        abl["produit_tous"] = {"debit": float(r.mean()), "fuite_med": float(np.median(lk))}
        res["ablation"] = abl
        print(f"  {name} {snr}dB ablation : " + " ".join(f"{k}:{v['debit']:.2f}" for k, v in abl.items()), flush=True)
    return res


def phase_decomp(net, h, snr, nv):
    """Rotation de phase par utilisateur h_k -> e^{i phi_k} h_k (G_jk -> e^{i(phi_j - phi_k)} G_jk). Parts
    de la VARIANCE de l'etat hors diagonale (etat centre sur le jeton moyen) : invariante (E_phi[e]),
    covariante (E_phi[e conj(u)], tourne comme G_jk), conjuguee (E_phi[e u], comme G_kj), reste (harmoniques
    superieures) ; u = e^{i(phi_j - phi_k)}. Estimateur sans biais de |E x|^2 : (R |moy x|^2 - moy |x|^2)/(R-1).
    Plus l'ecart d'equivariance de C et la variation de debit sous rotation."""
    rng = np.random.default_rng(5)
    N, K = h.shape[0], h.shape[1]
    off = ~np.eye(K, dtype=bool)
    eqv, drate, all_es, all_u = [], [], [], []
    _, _, es0 = states(net, h, snr)
    L = es0.shape[1] - 1
    c0 = c_of_states(net, es0[:, L])
    r0 = rate_per_sample(h, w_from_c(net._norm(h)[0].numpy(), c0), nv).numpy()
    for _ in range(R_PHASE):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, [N, K]))
        hr = tf.constant((h.numpy() * ph[:, :, None]).astype(np.complex64))
        hp_r, _, es = states(net, hr, snr)
        u = ph[:, :, None] * np.conj(ph[:, None, :])                            # [N,K,K]
        all_es.append(es[:, :, off]); all_u.append(u[:, off])                    # [N,L+1,P,d], [N,P]
        cr = c_of_states(net, es[:, L])
        eqv.append(np.linalg.norm(cr - u * c0, axis=(1, 2)) / np.linalg.norm(c0, axis=(1, 2)))
        drate.append(rate_per_sample(hr, w_from_c(hp_r, cr), nv).numpy() - r0)
    E = np.stack(all_es)                                                         # [R,N,L+1,P,d]
    U = np.stack(all_u)[:, :, None, :, None]                                     # [R,N,1,P,1]
    R = E.shape[0]
    out = []
    for l in range(L + 1):
        el = E[:, :, l:l + 1].astype(np.complex128)
        el = el - el.mean(axis=(0, 1, 3), keepdims=True)                        # centre (jeton moyen)
        m2 = np.mean(np.abs(el) ** 2, 0)                                         # [N,1,P,d]
        tot = m2.sum()
        parts = []
        for x in (el, el * np.conj(U), el * U):
            parts.append(float(np.sum((R * np.abs(x.mean(0)) ** 2 - m2) / (R - 1)) / tot))
        out.append({"couche": l, "invariante": parts[0], "covariante": parts[1], "conjuguee": parts[2],
                    "reste": float(1 - sum(parts))})
    return {"parts_variance_hors_diag": out, "ecart_equivariance_C_med": float(np.median(np.concatenate(eqv))),
            "variation_debit_moy_abs": float(np.mean(np.abs(np.concatenate(drate)))), "R": R_PHASE}


def main():
    keras.utils.set_random_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {
        "meta": {"git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M"),
                 "N_tr": N_TR, "N_te": N_TE, "snrs": SNRS}, "res": {}}
    net = load_base()
    data = datasets()
    t0 = time.time()
    for name, (h_tr, h_te) in data.items():
        for snr in SNRS:
            key = f"{name}_{snr}"
            if key in out["res"]:
                continue
            print(f"== {key}", flush=True)
            out["res"][key] = run_one(net, name, h_tr, h_te, snr)
            json.dump(out, open(OUT_PATH, "w"), indent=1)
            print(f"   ref {out['res'][key]['ref']}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"-> {OUT_PATH} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
