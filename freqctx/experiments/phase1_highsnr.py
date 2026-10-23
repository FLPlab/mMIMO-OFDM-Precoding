"""Phase 1c -- trois idees pour le fort SNR, chacune seule, comparees au
modele signe de reference (meme taille, memes donnees, meme evaluation) :
    tied   : UN bloc applique T fois (iterations) au lieu de L blocs distincts
    phase  : normalisation de la phase propre de chaque utilisateur (exacte)
    linear : attention sans softmax (poids = scores / K), FPGA-friendly

    python -m experiments.phase1_highsnr --d 32 --layers 2 [--variants signed,phase] [--tag a]
    python -m experiments.phase1_highsnr --merge
"""
import argparse
import dataclasses

from config import SystemConfig
from models.per_sc import PerSCPrecoder, rzf_macs_per_sc
from train import channel_pool, eval_pool, references, run_variant
from utils import BLUES, COLORS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
WEIGHTS = "/export/tmp/sala/freqctx/weights/phase1"


def variants(L):
    return {"signed":   dict(attn="signed", layers=L),
            "tied_T2":  dict(attn="signed", tied=True, layers=2),
            "tied_T4":  dict(attn="signed", tied=True, layers=4),
            "tied_T8":  dict(attn="signed", tied=True, layers=8),
            "phase":    dict(attn="signed", phase_norm=True, layers=L),
            "linear":   dict(attn="linear", layers=L)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--variants", help="sous-ensemble, separe par des virgules")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge_parts("phase1", "highsnr", plot, sort_key=lambda r: r["name"])
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    cfg = SystemConfig()
    pool = channel_pool(cfg, drops, seed=1)
    h_eval = eval_pool(cfg)
    if args.quick:
        h_eval = h_eval[::50]
    ref = references(h_eval, SNRS_DB)

    todo = variants(args.layers)
    if args.variants:
        todo = {k: todo[k] for k in args.variants.split(",")}
    runs = []
    for tag, kw in todo.items():
        name = f"D{args.d}_{tag}" + ("" if tag.startswith("tied") else f"_L{kw['layers']}")
        run = run_variant(name, PerSCPrecoder(cfg.num_bs_ant, d=args.d, **kw), pool, h_eval,
                          SNRS_DB, ref, steps, weights_dir=None if args.quick else WEIGHTS)
        runs.append({"variant": tag, **kw, **run})

    payload = {"config": dataclasses.asdict(cfg), "snrs_db": SNRS_DB, "d": args.d,
               "layers": args.layers, "steps": steps, "train_drops": drops, "lr": 1e-3,
               "batch": 2048, "references": ref,
               "rzf_macs_per_sc": rzf_macs_per_sc(cfg.num_bs_ant, cfg.num_users), "runs": runs}
    name = "highsnr" + ("_quick" if args.quick else "") + (f"_part_{args.tag}" if args.tag else "")
    save("phase1", name, payload, plot(payload))


def plot(p):
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in ("RZF", "WMMSE"):
        ax.plot(p["snrs_db"], [100 * v / r for v, r in zip(p["references"][n], p["references"]["RZF"])],
                "--", color=COLORS[n], lw=1.2, label=n)
    for j, r in enumerate(p["runs"]):
        ax.plot(p["snrs_db"], r["pct_rzf"], "o-", color=BLUES[j % len(BLUES)], lw=1.5,
                label=f"{r['name']} ({r['params']} p., {r['macs_per_sc'] // 1000}k MAC)")
    ax.set(xlabel="SNR [dB]", ylabel="% du debit RZF", title=f"Idees fort SNR (D={p['d']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
