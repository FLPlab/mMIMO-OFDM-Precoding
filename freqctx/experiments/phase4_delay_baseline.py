"""Phase 4a -- estimateur classique dans le domaine des retards (LS sur le
peigne -> IDFT -> troncature ou seuil -> DFT), suivi de RZF / WMMSE, sur les
memes tirages que le melange UMi + UMa + RMa (pilote 10 dB). Sans entrainement.
Question : la marge restante est-elle dans l'estimation (ce que ferait
mieux un traitement dans le domaine des retards) ?

    python -m experiments.phase4_delay_baseline
"""
import numpy as np
import tensorflow as tf

from config import CSIConfig, SystemConfig
from csi import delay_domain_estimate
from precoders.classical import rzf, wmmse
from train import csi_pool, evaluate
from utils import save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, EVAL_DROPS = ["umi", "uma", "rma"], 10.0, 192
VARIANTS = {"L=16": dict(num_taps=16), "L=32": dict(num_taps=32), "L=48": dict(num_taps=48),
            "seuil k=4": dict(kappa=4), "seuil k=8": dict(kappa=8)}


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def main():
    nv = lambda s: 10.0 ** (-s[0] / 10.0)
    res = {}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        ls, true = csi_pool(cfg, CSIConfig("ls_lin", PILOT, 0), EVAL_DROPS, 2026 + 10 * i, flat=False)
        t = flat(true)
        nmse = lambda e: float(10 * np.log10(tf.reduce_mean(tf.abs(e - true) ** 2)
                                             / tf.reduce_mean(tf.abs(true) ** 2)))
        r = {"LS": {"nmse_db": nmse(ls),
                    "RZF": evaluate(lambda x, s: rzf(x, nv(s)), (flat(ls), t), SNRS_DB)}}
        for name, kw in VARIANTS.items():
            est = delay_domain_estimate(ls, PILOT, **kw)
            r[name] = {"nmse_db": nmse(est),
                       "RZF": evaluate(lambda x, s: rzf(x, nv(s)), (flat(est), t), SNRS_DB)}
        best = max(VARIANTS, key=lambda n: r[n]["RZF"][4])
        est = delay_domain_estimate(ls, PILOT, **VARIANTS[best])
        r[best]["WMMSE"] = evaluate(lambda x, s: wmmse(x, nv(s)), (flat(est), t), SNRS_DB, 16384)
        r["sans bruit L=64 (plancher de fuite)"] = {"nmse_db": nmse(delay_domain_estimate(true, PILOT, num_taps=64))}
        res[scen] = r
        print(scen, {k: (round(v["nmse_db"], 1), round(v["RZF"][4], 2) if "RZF" in v else None)
                     for k, v in r.items()}, "| WMMSE", best, round(r[best]["WMMSE"][4], 2))
    save("phase4", "delay_baseline_p10", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT,
                                         "eval_drops_per_scenario": EVAL_DROPS, "results": res})


if __name__ == "__main__":
    main()
