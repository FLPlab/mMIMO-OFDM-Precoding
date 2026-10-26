"""Phase 6 -- pour un modele a bloc partage (tied), que se passe-t-il si on
change le nombre de passages A L'INFERENCE, sans reentrainer ?
(transformers en boucle : Yang et al. ICLR 2024 ; Fan et al. 2024.)
Si le debit s'ameliore avec plus de passages sur les cas difficiles, le
modele a appris une regle iterative convergente ; sinon, il est specialise
a sa profondeur d'entrainement.

    python -u -m experiments.phase6_loops --name trig_L8_d16_h64
"""
import argparse
import json
import os

import tensorflow as tf

from experiments.phase6_bench import eval_sets, evaluate_all
from experiments.phase6_train import W6, build
from utils import RESULTS

CONFIGS = ["M64K8", "M64K8_rho0.95", "M8K4", "M8K4_rho0.8", "M16K8", "M8K8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--loops", default="4,8,12,16")
    a = ap.parse_args()
    d = json.load(open(os.path.join(RESULTS, "phase6", f"{a.name}.json")))
    sets = {k: v for k, v in eval_sets().items() if k in CONFIGS}
    out = {"name": a.name, "trained_layers": d["args"]["layers"], "by_loops": {}}
    for L in [int(x) for x in a.loops.split(",")]:
        args = argparse.Namespace(**{**d["args"], "layers": L})
        net = build(args)
        h = sets["M64K8"][0]
        net(h[:2], tf.zeros([2]))
        net.load_weights(os.path.join(W6, f"{a.name}.weights.h5"))
        print(f"== {a.name} avec {L} passages", flush=True)
        out["by_loops"][L] = evaluate_all(net, sets)
    path = os.path.join(RESULTS, "phase6", f"loops_{a.name}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
