"""Sauvegarde des resultats et style des figures (communs a toutes les phases)."""
import dataclasses
import glob
import json
import os
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

# Une couleur fixe par methode (jamais recyclee d'une figure a l'autre).
COLORS = {"MRT": "#eda100", "ZF": "#e87ba4", "RZF": "#2a78d6", "WMMSE": "#eb6834",
          "RZF+lissage": "#1baf7a", "umi": "#2a78d6", "uma": "#eb6834"}
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "lines.markersize": 5, "font.size": 10,
                     "legend.frameon": False})


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


def save(phase, name, payload, fig=None):
    """Ecrit results/<phase>/<name>.json (+ .png/.pdf), ecrase la version
    precedente (l'historique est dans git). Ne garder ici que les resultats
    retenus ; tout le reste se resume dans FINDINGS.md."""
    os.makedirs(os.path.join(RESULTS, phase), exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    base = os.path.join(RESULTS, phase, name)
    payload = {"meta": {"created": stamp, "git_commit": git_commit()}, **payload}
    with open(base + ".json", "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: dataclasses.asdict(o)
                  if dataclasses.is_dataclass(o) else str(o))
    if fig is not None:
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"-> {base}.json")
    return base


def merge_parts(phase, base, plot_fn, sort_key=None):
    """Reunit results/<phase>/<base>_part_*.json (une partie par serveur/GPU)
    en <base>.json (+ figure). Les parties restent sur disque."""
    parts = sorted(glob.glob(os.path.join(RESULTS, phase, f"{base}_part_*.json")))
    ps = [json.load(open(f)) for f in parts]
    payload = {k: v for k, v in ps[0].items() if k not in ("meta", "runs")}
    payload["parts"] = [os.path.basename(f) for f in parts]
    payload["runs"] = sorted((r for p in ps for r in p["runs"]), key=sort_key)
    return save(phase, base, payload, plot_fn(payload))
