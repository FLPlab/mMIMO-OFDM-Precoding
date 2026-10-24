"""Phase 4b -- borne "genie" de l'estimation : LMMSE par lien avec la
covariance exacte de chaque lien (trajets connus), puis RZF / WMMSE. Memes
tirages et meme bruit pilote que le melange (pilote 10 dB). Sans entrainement.
Question : reste-t-il une vraie marge dans l'estimation, au-dela de notre modele ?

    python -m experiments.phase4_genie_bound
"""
import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import CSIConfig, SystemConfig
from csi import UplinkSounder, csi_at_bs, genie_lmmse
from precoders.classical import rzf, wmmse
from train import evaluate
from utils import save

SNRS_DB = [0, 5, 10, 15, 20, 25]
SCENARIOS, PILOT, EVAL_DROPS, BATCH = ["umi", "uma", "rma"], 10.0, 192, 32


def flat(h):
    return tf.reshape(h, [-1, h.shape[-2], h.shape[-1]])


def main():
    nv = lambda s: 10.0 ** (-s[0] / 10.0)
    res = {}
    for i, scen in enumerate(SCENARIOS):
        cfg = SystemConfig(scenario=scen)
        csi = CSIConfig("ls_lin", PILOT, 0)
        sionna_config.seed = 2026 + 10 * i                      # memes tirages que csi_pool
        gen, sounder = ChannelGenerator(cfg), UplinkSounder(cfg, "lin")
        ls, true, gen_est = [], [], []
        for _ in range(EVAL_DROPS // BATCH):
            h_bs, h_data = csi_at_bs(gen(BATCH), csi, sounder)
            ls.append(h_bs)
            true.append(h_data)
            gen_est.append(genie_lmmse(h_bs, h_data, gen.last_cir, gen.frequencies, PILOT))
        ls, true, gen_est = (tf.concat(x, 0) for x in (ls, true, gen_est))
        nmse = lambda e: float(10 * np.log10(tf.reduce_mean(tf.abs(e - true) ** 2)
                                             / tf.reduce_mean(tf.abs(true) ** 2)))
        t = flat(true)
        r = {"nmse_ls_db": nmse(ls), "nmse_genie_db": nmse(gen_est),
             "RZF (LS)": evaluate(lambda x, s: rzf(x, nv(s)), (flat(ls), t), SNRS_DB),
             "RZF (genie par lien)": evaluate(lambda x, s: rzf(x, nv(s)), (flat(gen_est), t), SNRS_DB),
             "WMMSE (genie par lien)": evaluate(lambda x, s: wmmse(x, nv(s)), (flat(gen_est), t),
                                                SNRS_DB, 16384)}
        res[scen] = r
        print(scen, f"NMSE LS {r['nmse_ls_db']:.1f} / genie {r['nmse_genie_db']:.1f} dB |",
              {k: round(v[4], 2) for k, v in r.items() if isinstance(v, list)})
    save("phase4", "genie_bound_p10", {"snrs_db": SNRS_DB, "pilot_snr_db": PILOT,
                                       "eval_drops_per_scenario": EVAL_DROPS, "results": res})


if __name__ == "__main__":
    main()
