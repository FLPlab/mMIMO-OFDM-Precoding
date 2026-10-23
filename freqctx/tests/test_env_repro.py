"""Reproductibilite entre machines : meme graine -> meme canal et meme debit.

    python -m tests.test_env_repro --write   # sur la machine de reference (srvgrm12)
    python -m tests.test_env_repro           # ailleurs : compare a la reference
"""
import argparse
import json
import os
import socket

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from metrics import sum_rate
from precoders.classical import rzf

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_repro_reference.json")


def fingerprint():
    sionna_config.seed = 2026
    h = ChannelGenerator(SystemConfig())(4).numpy()            # [4, 3, 612, 4, 8]
    nv = 0.01
    return {"host": socket.gethostname(), "tf": tf.__version__,
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "h_first": [float(h.real.ravel()[0]), float(h.imag.ravel()[0])],
            "h_energy": float(np.mean(np.abs(h) ** 2)),
            "h_sample": np.round(h[:, 0, ::51, 0, 0], 5).real.ravel().tolist(),
            "rzf_rate_20db": float(sum_rate(h[:, 0], rzf(tf.constant(h[:, 0]), nv), nv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    fp = fingerprint()
    if args.write:
        json.dump(fp, open(REF, "w"), indent=1)
        print(f"reference ecrite ({fp['host']}) : debit RZF {fp['rzf_rate_20db']:.6f}")
        return
    ref = json.load(open(REF))
    d_h = np.max(np.abs(np.array(fp["h_sample"]) - np.array(ref["h_sample"])))
    d_r = abs(fp["rzf_rate_20db"] - ref["rzf_rate_20db"])
    print(f"{fp['host']} vs {ref['host']} : ecart canal max {d_h:.2e}, ecart debit {d_r:.2e}")
    assert d_h < 1e-3 and d_r < 1e-3, "canaux differents : resultats NON comparables"
    print("REPRODUCTIBLE")


if __name__ == "__main__":
    main()
