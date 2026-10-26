"""Phase 4c -- estimation dans le domaine des retards (variante a : un gain
entre 0 et 1 par retard, commun aux antennes) + precodeur par SC SANS fenetre,
appris de bout en bout. Seul changement par rapport au candidat : l'attention
en frequence (fenetre 49) est remplacee par ce module sur toute la bande.

Memes tirages que le melange UMi + UMa + RMa (pilote 10 dB) ; references
relues dans results/phase3/mixed_p10.json (verifie : RZF(LS) identique) et
borne genie dans results/phase4/genie_bound_p10.json.

    python -m experiments.phase4_delay_model --d 64 [--quick]
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from config import CSIConfig, SystemConfig
from models.delay_domain import DelayDomainPrecoder
from precoders.classical import rzf
from train import DropPool, csi_pool, evaluate, run_variant
from utils import RESULTS, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT = ["umi", "uma", "rma"], 10.0
W4 = "/export/tmp/sala/freqctx/weights/phase4"


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64, help="largeur du precodeur par SC")
    ap.add_argument("--est_d", type=int, default=32)
    ap.add_argument("--angle", action="store_true", help="gain par (retard, angle) (variante b)")
    ap.add_argument("--banded", action="store_true", help="filtre complexe sur 3 retards voisins")
    ap.add_argument("--oversample", type=int, default=1, help="grille de retards O fois plus fine")
    ap.add_argument("--max_taps", type=int, default=None, help="ne garder que les L premiers retards (54 ns)")
    ap.add_argument("--snr_every", action="store_true", help="SNR ajoute avant chaque iteration du precodeur")
    ap.add_argument("--rzf_init", action="store_true", help="precodeur = RZF + correction apprise")
    ap.add_argument("--mrt_init", action="store_true", help="precodeur = MRT + correction apprise (sans inversion)")
    ap.add_argument("--precoder", default="persc", choices=["persc", "gram", "trig"],
                    help="precodeur par SC : tokens d'utilisateur (persc) ou etat par paire (phase 6)")
    ap.add_argument("--p_layers", type=int, default=8, help="couches du precodeur par paire")
    ap.add_argument("--p_untied", action="store_true", help="couches distinctes (transformer standard)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--load", action="store_true", help="recharger les poids deja entraines (pas d'entrainement)")
    ap.add_argument("--train_drops", type=int, default=352, help="tirages d'entrainement par scenario")
    ap.add_argument("--steps", type=int, default=40000)
    args = ap.parse_args()
    steps, train_drops, eval_drops = ((2000, 64, 32) if args.quick
                                      else (args.steps, args.train_drops, 192))
    ref_file = json.load(open(os.path.join(RESULTS, "phase3", "mixed_p10.json")))
    genie = json.load(open(os.path.join(RESULTS, "phase4", "genie_bound_p10.json")))["results"]
    cand = json.load(open(os.path.join(RESULTS, "phase3", "mixed_p10_fattn_D64.json")))
    nv = lambda s: 10.0 ** (-s[0] / 10.0)

    ev, tr_in, tr_true = {}, [], []
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        csi = CSIConfig("ls_lin", PILOT, 0)
        ev[scen] = csi_pool(cfg, csi, eval_drops, 2026 + 10 * i, flat=False)
        a, b = csi_pool(cfg, csi, train_drops, seed=1 + i, flat=False)
        tr_in.append(a)
        tr_true.append(b)
        if not args.quick:                                     # memes tirages que les references ?
            r = evaluate(lambda x, s: rzf(x, nv(s)), (flat(ev[scen][0]), flat(ev[scen][1])), [20])[0]
            stored = ref_file["references"][scen]["RZF (LS)"][4]
            assert abs(r - stored) < 1e-3, (scen, r, stored)
    train_pool = DropPool(tf.concat(tr_in, 0), tf.concat(tr_true, 0))

    cfg = SystemConfig()
    if args.precoder == "persc":
        pkw = dict(snr_every=args.snr_every, rzf_init=args.rzf_init, mrt_init=args.mrt_init, d=args.d,
                   layers=4, tied=True, attn="linear")
    else:                                                      # precodeur par paire (phase 6)
        pkw = dict(precoder_arch="pair", d=16, hidden=64, layers=args.p_layers,
                   mix={"gram": "gram", "trig": "triangular_g"}[args.precoder], tied=not args.p_untied)
    model = DelayDomainPrecoder(cfg.num_sc, cfg.num_users, cfg.num_bs_ant, est_d=args.est_d,
                                angle_mask=args.angle, banded=args.banded,
                                oversample=args.oversample, max_taps=args.max_taps, **pkw)
    tag = (("_angle" if args.angle else "") + ("_bande" if args.banded else "")
           + (f"_os{args.oversample}" if args.oversample > 1 else "")
           + (f"_L{args.max_taps}" if args.max_taps else "")
           + ("_snrall" if args.snr_every else "") + ("_rzf0" if args.rzf_init else "")
           + ("_mrt0" if args.mrt_init else "")
           + ("" if args.precoder == "persc" else f"_{args.precoder}L{args.p_layers}")
           + ("_untied" if args.p_untied else ""))
    if not args.quick and (train_drops != 352 or steps != 40000):
        tag += f"_n{train_drops}_s{steps // 1000}k"                # donnees / entrainement non standard
    name = f"delay{tag}_est{args.est_d}_D{args.d}_T4_linear_melange_p10"
    if args.load:
        model(ev["umi"][0][:1], tf.zeros([1]))
        model.load_weights(os.path.join(W4, f"{name}.weights.h5"))
        run = {"params": int(model.count_params()), "macs_per_sc": model.macs_per_sc(cfg.num_users),
               "history": [], "reloaded": True}
    else:
        run = run_variant(name, model, train_pool, DropPool(*ev["umi"]), SNRS_DB,
                          {"RZF": ref_file["references"]["umi"]["RZF (LS)"]}, steps, seed=0,
                          loss="rel_rzf", weights_dir=None if args.quick else W4)
    rates, gains = {}, {}
    for scen in SCENARIOS:
        pool = DropPool(*ev[scen])
        rates[scen] = evaluate(model, pool, SNRS_DB)
        hc, _, _ = next(pool.chunks(16))
        model(hc, tf.fill([hc.shape[0]], 20.0))
        g = model.estimator.last_gain.numpy()                  # [B, K, N, 1 ou M]
        gains[scen] = {"profile": g.mean(axis=(0, 1, 3)).tolist(),
                       "kept_taps_mean": float(g.sum(axis=2).mean()),
                       "kept_bins_mean": float(g.sum(axis=(2, 3)).mean())}
        refs = ref_file["references"][scen]
        print(f"{scen} @20dB : modele {rates[scen][4]:.2f} | candidat (freq. 49) "
              f"{list(cand['models'].values())[0][scen][4]:.2f} | WMMSE(LMMSE melange) "
              f"{refs['WMMSE (LMMSE melange)'][4]:.2f} | genie {genie[scen]['WMMSE (genie par lien)'][4]:.2f}"
              f" | retards gardes {gains[scen]['kept_taps_mean']:.1f}")

    payload = {"scenarios": SCENARIOS, "pilot_snr_db": PILOT, "snrs_db": SNRS_DB,
               "model": {"precoder": args.precoder, "p_layers": args.p_layers, "p_untied": args.p_untied,
                         "est_d": args.est_d, "d": args.d, "angle_mask": args.angle,
                         "banded": args.banded, "oversample": args.oversample,
                         "max_taps": args.max_taps, "snr_every": args.snr_every,
                         "rzf_init": args.rzf_init, "mrt_init": args.mrt_init, "layers": 4, "tied": True,
                         "attn": "linear", "params": run["params"],
                         "macs_per_sc": run["macs_per_sc"]},
               "steps": steps, "train_drops_per_scenario": train_drops, "rates": rates, "delay_gains": gains,
               "training_history": run["history"],
               "compare": {s: {"WMMSE (LMMSE melange)": ref_file["references"][s]["WMMSE (LMMSE melange)"],
                               "WMMSE (LMMSE oracle)": ref_file["references"][s]["WMMSE (LMMSE oracle)"],
                               "WMMSE (genie par lien)": genie[s]["WMMSE (genie par lien)"],
                               "candidat freq. W49": list(cand["models"].values())[0][s]}
                           for s in SCENARIOS}}
    est = f"_est{args.est_d}" if args.est_d != 32 else ""
    save("phase4", f"delay_model{tag}{est}_D{args.d}" + ("_quick" if args.quick else ""), payload,
         plot(payload))


def plot(p):
    i20 = p["snrs_db"].index(20)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    names = ["WMMSE (LMMSE melange)", "WMMSE (LMMSE oracle)", "candidat freq. W49", "modele retards",
             "WMMSE (genie par lien)"]
    colors = ["#f2a27e", "#eb6834", "#6da7ec", "#1baf7a", "#9085e9"]
    x = np.arange(len(p["scenarios"]))
    wd = 0.8 / len(names)
    for j, n in enumerate(names):
        v = [(p["rates"][s] if n == "modele retards" else p["compare"][s][n])[i20] for s in p["scenarios"]]
        ax.bar(x + (j - len(names) / 2 + 0.5) * wd, v, wd * 0.92, color=colors[j], label=n)
    ax.set(xticks=x, xticklabels=[s.upper() for s in p["scenarios"]],
           ylabel="Debit somme a 20 dB [bps/Hz]", title="Estimation dans le domaine des retards, pilote 10 dB")
    lo = min(p["compare"][s]["WMMSE (LMMSE melange)"][i20] for s in p["scenarios"])
    ax.set_ylim(lo - 1.5, None)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for s, c in zip(p["scenarios"], ["#2a78d6", "#eb6834", "#1baf7a"]):
        prof = p["delay_gains"][s]["profile"]
        ax2.plot(np.arange(len(prof)), prof, color=c, lw=1.4,
                 label=f"{s.upper()} ({p['delay_gains'][s]['kept_taps_mean']:.0f} retards gardes)")
    ax2.set(xlabel="Retard [indice, 54 ns]", ylabel="Gain moyen", title="Gains appris par retard (20 dB)",
            xlim=(0, 60))
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
