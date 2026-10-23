"""Phase 1b -- deux ajouts a l'attention entre utilisateurs, seuls puis
ensemble, a taille fixe (choisie d'apres phase1_size_sweep) :
    signed : poids signes softmax(s) * tanh(s)  -> pouvoir SOUSTRAIRE
    gram   : biais de score f(G_kj), G = H H^H  -> savoir QUI interfere
Memes donnees, graines et evaluation que phase1_size_sweep ; la variante
de base (ni l'un ni l'autre) est re-entrainee ici pour un appariement exact.

    python -m experiments.phase1_signed_gram [--d 64] [--layers 2] [--quick]
"""
import argparse
import dataclasses

from config import SystemConfig
from models.per_sc import PerSCPrecoder, rzf_macs_per_sc
from train import channel_pool, eval_pool, references, run_variant
from utils import BLUES, COLORS, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
VARIANTS = {"base": {}, "signed": {"attn": "signed"}, "gram": {"gram": True},
            "signed+gram": {"attn": "signed", "gram": True}}
WEIGHTS = "/export/tmp/sala/freqctx/weights/phase1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    cfg = SystemConfig()
    pool = channel_pool(cfg, drops, seed=1)
    h_eval = eval_pool(cfg)
    if args.quick:
        h_eval = h_eval[::50]
    ref = references(h_eval, SNRS_DB)

    runs = []
    for tag, kw in VARIANTS.items():
        name = f"D{args.d}_L{args.layers}_{tag}"
        run = run_variant(name, PerSCPrecoder(cfg.num_bs_ant, d=args.d, layers=args.layers, **kw),
                          pool, h_eval, SNRS_DB, ref, steps,
                          weights_dir=None if args.quick else WEIGHTS)
        runs.append({"variant": tag, **kw, **run})

    payload = {"config": dataclasses.asdict(cfg), "snrs_db": SNRS_DB, "d": args.d,
               "layers": args.layers, "steps": steps, "train_drops": drops, "lr": 1e-3,
               "batch": 2048, "references": ref,
               "rzf_macs_per_sc": rzf_macs_per_sc(cfg.num_bs_ant, cfg.num_users), "runs": runs}
    save("phase1", "signed_gram" + ("_quick" if args.quick else ""), payload, plot(payload))


def plot(p):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for n in ("RZF", "WMMSE"):
        ax.plot(p["snrs_db"], p["references"][n], "o-", color=COLORS[n], label=n)
    for j, r in enumerate(p["runs"]):
        ax.plot(p["snrs_db"], r["rate_bps_hz"], "s-", color=BLUES[1 + j], lw=1.5,
                label=f"{r['variant']} ({r['params']} params)")
    ax.set(xlabel="SNR [dB]", ylabel="Debit somme [bps/Hz]",
           title=f"Attention signee / biais Gram (D={p['d']}, L={p['layers']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
