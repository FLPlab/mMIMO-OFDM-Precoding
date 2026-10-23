"""Phase 0a -- statistiques du canal 20 MHz : correlation en frequence
(bande de coherence), correlation en temps (vieillissement) et etalement
des retards, par scenario. Sert a dimensionner les fenetres des phases 3/4.

    python -m experiments.phase0_channel_stats
"""
import dataclasses

import numpy as np
import tensorflow as tf
from sionna.phy.config import config as sionna_config

from channel import ChannelGenerator
from config import SystemConfig
from utils import COLORS, plt, save

SEED, BATCH, NUM_BATCHES, NUM_SLOTS, MAX_LAG_SC = 2026, 32, 10, 5, 144


def stats(cfg):
    gen = ChannelGenerator(cfg)
    fcorr = np.zeros(MAX_LAG_SC + 1)
    tcorr = np.zeros(NUM_SLOTS)
    ds = []
    for _ in range(NUM_BATCHES):
        h = gen(BATCH).numpy()                                   # [B,S,F,K,M]
        p = np.mean(np.abs(h[:, 0]) ** 2)
        for lag in range(MAX_LAG_SC + 1):
            fcorr[lag] += np.abs(np.mean(h[:, 0, lag:] * np.conj(h[:, 0, :h.shape[2] - lag]))) / p
        for d in range(NUM_SLOTS):
            tcorr[d] += np.abs(np.mean(h[:, d] * np.conj(h[:, 0]))) / p
        ds.append(rms_delay_spread(h[:, 0], cfg))
    return fcorr / NUM_BATCHES, tcorr / NUM_BATCHES, np.concatenate(ds)


def rms_delay_spread(hf, cfg):
    """Etalement RMS par lien (b, k) depuis le profil puissance-retard obtenu
    par IFFT sur la bande (resolution 1/bande ~ 54 ns)."""
    pdp = np.mean(np.abs(np.fft.ifft(hf, axis=1)) ** 2, axis=-1)     # [B, F, K]
    F = hf.shape[1]
    t = np.arange(F) / cfg.bandwidth
    t = np.where(np.arange(F) < F // 2, t, t - F / cfg.bandwidth)    # retards negatifs repliés
    w = pdp / pdp.sum(axis=1, keepdims=True)
    mean = np.sum(w * t[None, :, None], axis=1)
    return np.sqrt(np.sum(w * (t[None, :, None] - mean[:, None]) ** 2, axis=1)).ravel()


def main():
    sionna_config.seed = SEED
    out, fig = {}, plt.figure(figsize=(10, 3.4))
    ax_f, ax_t, ax_d = fig.subplots(1, 3)
    for scen in ["umi", "uma"]:
        cfg = SystemConfig(scenario=scen, num_slots=NUM_SLOTS)
        fc, tc, ds = stats(cfg)
        bc = {f"{thr}": int(np.argmax(fc < thr)) * cfg.subcarrier_spacing / 1e3
              for thr in (0.9, 0.7, 0.5)}
        out[scen] = {"config": dataclasses.asdict(cfg), "freq_corr": fc.tolist(),
                     "time_corr_per_slot": tc.tolist(),
                     "coherence_bw_khz": bc,
                     "delay_spread_ns": {"median": float(np.median(ds) * 1e9),
                                         "p90": float(np.percentile(ds, 90) * 1e9)}}
        print(f"{scen}: Bc(0.9/0.7/0.5) = {bc} kHz | correlation temps {np.round(tc, 3)} "
              f"| DS median {np.median(ds)*1e9:.0f} ns")
        lag_khz = np.arange(MAX_LAG_SC + 1) * cfg.subcarrier_spacing / 1e3
        ax_f.plot(lag_khz, fc, color=COLORS[scen], label=scen.upper())
        ax_t.plot(np.arange(NUM_SLOTS) * cfg.slot_duration * 1e3, tc, "o-", color=COLORS[scen],
                  label=scen.upper())
        ax_d.hist(ds * 1e9, bins=40, histtype="step", color=COLORS[scen], label=scen.upper(),
                  density=True)
    ax_f.axvline(12 * 30, color="0.5", lw=0.8, ls=":")
    ax_f.text(12 * 30, 0.02, " 1 RB", color="0.4", fontsize=8)
    ax_f.set(xlabel="Ecart en frequence [kHz]", ylabel="|correlation|", ylim=(0, 1.02),
             title="Correlation en frequence")
    ax_t.set(xlabel="Retard sondage -> donnees [ms]", ylabel="|correlation|",
             title="Correlation en temps (3-30 km/h)")
    ax_d.set(xlabel="Etalement des retards RMS [ns]", ylabel="densite", title="Etalement des retards")
    ax_f.legend()
    fig.tight_layout()
    save("phase0", "channel_stats", out, fig)


if __name__ == "__main__":
    main()
