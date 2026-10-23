"""Phase 1a -- transformer minimal par SC, CSI parfait : taille (D, L) et
role de l'attention (ablation "mean" = moyenne uniforme entre utilisateurs).

Entrainement : tirages graine 1 (1024 drops, toutes les SC melangees),
SNR uniforme 0-25 dB. Evaluation : tirages de la phase 0 (graine 2026,
512 drops x 612 SC), appariee avec RZF/WMMSE sur le meme ensemble.

    python -m experiments.phase1_size_sweep [--quick]
    python -m experiments.phase1_size_sweep --signed --grid 16x1,16x2 --tag a   # une partie
    python -m experiments.phase1_size_sweep --signed --merge                    # fusion des parties
"""
import argparse
import dataclasses

from config import SystemConfig
from models.per_sc import PerSCPrecoder, rzf_macs_per_sc
from train import channel_pool, eval_pool, references, run_variant
from utils import BLUES, COLORS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
VARIANTS = [(d, L, "attention") for L in (1, 2, 4) for d in (16, 32, 64, 128)] \
    + [(64, 2, "mean")]
WEIGHTS = "/export/tmp/sala/freqctx/weights/phase1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--signed", action="store_true", help="attention signee pour tous")
    ap.add_argument("--grid", help="sous-ensemble 'DxL,DxL' (attention seulement)")
    ap.add_argument("--tag", default="", help="suffixe du fichier (une partie du balayage)")
    ap.add_argument("--merge", action="store_true", help="fusionne les parties --tag")
    args = ap.parse_args()
    base = "size_sweep" + ("_signed" if args.signed else "")
    if args.merge:
        return merge_parts("phase1", base, plot, sort_key=lambda r: (r["layers"], r["d"]))
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    variants = VARIANTS[:2] if args.quick else VARIANTS
    if args.grid:
        variants = [(int(d), int(L), "attention")
                    for d, L in (g.split("x") for g in args.grid.split(","))]
    cfg = SystemConfig()
    K, M = cfg.num_users, cfg.num_bs_ant
    pool = channel_pool(cfg, drops, seed=1)
    h_eval = eval_pool(cfg)
    if args.quick:
        h_eval = h_eval[::50]
    ref = references(h_eval, SNRS_DB)
    print("references @20dB :", {k: round(v[4], 2) for k, v in ref.items()})

    runs = []
    for d, L, mixer in variants:
        name = f"D{d}_L{L}_{mixer}" + ("_signed" if args.signed else "")
        run = run_variant(name, PerSCPrecoder(M, d=d, layers=L, mixer=mixer,
                                        attn="signed" if args.signed else "softmax"),
                          pool, h_eval, SNRS_DB, ref, steps,
                          weights_dir=None if args.quick else WEIGHTS)
        runs.append({"d": d, "layers": L, "mixer": mixer, "signed": args.signed, **run})

    payload = {"config": dataclasses.asdict(cfg), "snrs_db": SNRS_DB, "steps": steps,
               "train_drops": drops, "lr": 1e-3, "batch": 2048, "references": ref,
               "rzf_macs_per_sc": rzf_macs_per_sc(M, K), "runs": runs}
    name = base + ("_quick" if args.quick else "") + (f"_part_{args.tag}" if args.tag else "")
    save("phase1", name, payload, plot(payload))


def plot(p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    i20 = p["snrs_db"].index(20)
    for j, L in enumerate((1, 2, 4)):
        rr = [r for r in p["runs"] if r["layers"] == L and r["mixer"] == "attention"]
        if rr:
            ax1.plot([r["macs_per_sc"] for r in rr], [r["pct_rzf"][i20] for r in rr], "o-",
                     color=BLUES[1 + 2 * j], label=f"transformer L={L}")
            for r in rr:
                ax1.annotate(f"D={r['d']}", (r["macs_per_sc"], r["pct_rzf"][i20]),
                             fontsize=7, xytext=(3, -9), textcoords="offset points")
    for r in p["runs"]:
        if r["mixer"] == "mean":
            ax1.plot(r["macs_per_sc"], r["pct_rzf"][i20], "s", color=COLORS["MRT"],
                     label=f"moyenne (sans attention) D={r['d']} L={r['layers']}")
    ax1.axvline(p["rzf_macs_per_sc"], color="0.5", ls=":", lw=1)
    ax1.text(p["rzf_macs_per_sc"], 101, " cout RZF", fontsize=8, color="0.4")
    ax1.axhline(100, color=COLORS["RZF"], lw=1, ls="--")
    ax1.set(xscale="log", xlabel="MAC reelles par SC", ylabel="% du debit RZF (20 dB)",
            title="Debit vs cout, CSI parfait")
    ax1.legend(fontsize=8)
    best = max(p["runs"], key=lambda r: r["rate_bps_hz"][i20])
    for n in ("RZF", "WMMSE"):
        ax2.plot(p["snrs_db"], p["references"][n], "o-", color=COLORS[n], label=n)
    ax2.plot(p["snrs_db"], best["rate_bps_hz"], "s-", color=BLUES[4], label=f"meilleur : {best['name']}")
    ax2.set(xlabel="SNR [dB]", ylabel="Debit somme [bps/Hz]", title="Meilleur modele vs references")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
