"""Phase 3 -- reference classique la plus forte : WMMSE sur l'estimation
LMMSE (en plus de RZF), memes tirages et meme bruit pilote que la phase 3.

    python -m experiments.phase3_wmmse_ref --pilot 10
"""
import argparse

import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import estimate_freq_cov
from precoders.classical import rzf, wmmse
from train import eval_csi_pool, evaluate
from utils import save

SNRS_DB = [0, 5, 10, 15, 20, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=float, default=10.0)
    args = ap.parse_args()
    cfg = SystemConfig()
    sionna_config.seed = 2027
    cov = estimate_freq_cov(ChannelGenerator(cfg))
    lmmse = eval_csi_pool(cfg, CSIConfig("ls_lmmse", args.pilot, 0), cov)
    true = lmmse[1]
    nv = lambda s: 10.0 ** (-s[0] / 10.0)
    refs = {"RZF (LMMSE)": evaluate(lambda x, s: rzf(x, nv(s)), lmmse, SNRS_DB),
            "WMMSE (LMMSE)": evaluate(lambda x, s: wmmse(x, nv(s)), lmmse, SNRS_DB, chunk=16384),
            "RZF (CSI parfait)": evaluate(lambda x, s: rzf(x, nv(s)), true, SNRS_DB),
            "WMMSE (CSI parfait)": evaluate(lambda x, s: wmmse(x, nv(s)), true, SNRS_DB, chunk=16384)}
    for k, v in refs.items():
        print(f"{k:20s}", [round(x, 2) for x in v])
    save("phase3", f"wmmse_ref_p{int(args.pilot)}", {"snrs_db": SNRS_DB, "pilot_snr_db": args.pilot,
                                                     "references": refs})


if __name__ == "__main__":
    main()
