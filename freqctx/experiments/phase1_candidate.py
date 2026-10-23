"""Phase 1d -- modele candidat : D=32, UN bloc repete 4 fois, attention
lineaire (sans exp/division/tanh). Deux questions :
    1. la combinaison des deux idees validees tient-elle (perte "rate") ?
    2. la perte relative a RZF corrige-t-elle le faible SNR ? (3 graines,
       pour mesurer la variabilite d'un entrainement a l'autre)

    python -m experiments.phase1_candidate [--runs rate_s0,rel_s0] [--tag a]
    python -m experiments.phase1_candidate --merge
"""
import argparse
import dataclasses

from config import SystemConfig
from models.per_sc import PerSCPrecoder, rzf_macs_per_sc
from train import channel_pool, eval_pool, references, run_variant
from utils import BLUES, COLORS, merge_parts, plt, save

SNRS_DB = [0, 5, 10, 15, 20, 25]
WEIGHTS = "/export/tmp/sala/freqctx/weights/phase1"
MODEL = dict(d=32, layers=4, tied=True, attn="linear")
RUNS = {"rate_s0": ("rate", 0), "rel_s0": ("rel_rzf", 0), "rel_s1": ("rel_rzf", 1),
        "rel_s2": ("rel_rzf", 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="sous-ensemble de " + ",".join(RUNS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge_parts("phase1", "candidate", plot, sort_key=lambda r: r["name"])
    steps, drops = (2000, 128) if args.quick else (40000, 1024)
    cfg = SystemConfig()
    pool = channel_pool(cfg, drops, seed=1)
    h_eval = eval_pool(cfg)
    if args.quick:
        h_eval = h_eval[::50]
    ref = references(h_eval, SNRS_DB)
    todo = args.runs.split(",") if args.runs else list(RUNS)
    runs = []
    for key in todo:
        loss, seed = RUNS[key]
        run = run_variant(f"D32_T4_linear_{key}", PerSCPrecoder(cfg.num_bs_ant, **MODEL), pool,
                          h_eval, SNRS_DB, ref, steps, seed=seed, loss=loss,
                          weights_dir=None if args.quick else WEIGHTS)
        runs.append(run)
    payload = {"config": dataclasses.asdict(cfg), "snrs_db": SNRS_DB, "model": MODEL,
               "steps": steps, "train_drops": drops, "lr": 1e-3, "batch": 2048,
               "references": ref,
               "rzf_macs_per_sc": rzf_macs_per_sc(cfg.num_bs_ant, cfg.num_users), "runs": runs}
    name = "candidate" + ("_quick" if args.quick else "") + (f"_part_{args.tag}" if args.tag else "")
    save("phase1", name, payload, plot(payload))


def plot(p):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(100, color=COLORS["RZF"], ls="--", lw=1.2, label="RZF")
    ax.plot(p["snrs_db"], [100 * w / r for w, r in zip(p["references"]["WMMSE"], p["references"]["RZF"])],
            "--", color=COLORS["WMMSE"], lw=1.2, label="WMMSE")
    for j, r in enumerate(p["runs"]):
        ax.plot(p["snrs_db"], r["pct_rzf"], "o-", color=BLUES[1 + j % 5], lw=1.5,
                label=f"{r['name']} ({r['params']} p.)")
    ax.set(xlabel="SNR [dB]", ylabel="% du debit RZF",
           title="Candidat : D32, 1 bloc x4, attention lineaire")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
