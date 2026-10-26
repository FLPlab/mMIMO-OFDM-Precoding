"""Phase 6 -- tableau de synthese de tous les runs (results/phase6/*.json)
-> results/phase6/summary.md

    python -m experiments.phase6_summary
"""
import glob
import json
import os

from utils import RESULTS

SEEN = ["M64K8", "M8K4", "M64K8_rho0.8", "M64K8_rho0.95", "M8K4_rho0.8"]
UNSEEN = ["M32K8", "M16K8", "M8K8", "M64K4", "M32K4", "M16K4"]


def score(ev, names, snr, key="pct_rzf"):
    return sum(ev[n][str(snr)][key] for n in names) / len(names)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "phase6", "*.json"))):
        d = json.load(open(p))
        if "eval" not in d:
            continue
        ev = d["eval"]
        rows.append((d["name"], d["params"], d["macs_per_sc"], ev))
    out = ["# Phase 6 -- synthese (CSI parfait, % de RZF sauf mention)", "",
           "Moyennes : 'vues' = " + ", ".join(SEEN) + " ; 'non vues' = " + ", ".join(UNSEEN) + ".", "",
           "| run | params | MAC/SC M64K8 | MAC/SC M8K4 | vues 20 dB | vues 30 dB | non vues 20 dB "
           "| non vues 30 dB | pire config 20 dB | M8K4 %WMMSE 0/20 dB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, params, macs, ev in sorted(rows, key=lambda r: -score(r[3], UNSEEN, 20)):
        worst = min(SEEN + UNSEEN, key=lambda n: ev[n]["20"]["pct_rzf"])
        out.append(f"| {name} | {params} | {macs['M64K8'] / 1e6:.2f} M | {macs['M8K4'] / 1e6:.2f} M "
                   f"| {score(ev, SEEN, 20):.1f} | {score(ev, SEEN, 30):.1f} | {score(ev, UNSEEN, 20):.1f} "
                   f"| {score(ev, UNSEEN, 30):.1f} | {worst} {ev[worst]['20']['pct_rzf']:.1f} "
                   f"| {ev['M8K4']['0']['pct_wmmse']:.1f} / {ev['M8K4']['20']['pct_wmmse']:.1f} |")
    out += ["", "## Detail a 20 dB (% RZF)", "",
            "| run | " + " | ".join(SEEN + UNSEEN) + " |", "|---|" + "---|" * len(SEEN + UNSEEN)]
    for name, _, _, ev in rows:
        out.append(f"| {name} | " + " | ".join(f"{ev[n]['20']['pct_rzf']:.1f}" for n in SEEN + UNSEEN) + " |")
    out += ["", "## Detail a 30 dB (% RZF)", "",
            "| run | " + " | ".join(SEEN + UNSEEN) + " |", "|---|" + "---|" * len(SEEN + UNSEEN)]
    for name, _, _, ev in rows:
        out.append(f"| {name} | " + " | ".join(f"{ev[n]['30']['pct_rzf']:.1f}" for n in SEEN + UNSEEN) + " |")
    path = os.path.join(RESULTS, "phase6", "summary.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
