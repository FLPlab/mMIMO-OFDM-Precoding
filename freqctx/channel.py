"""Generation du canal 3GPP (Sionna TR 38.901) sur toute la bande, par slot."""
import tensorflow as tf
from sionna.phy.channel import (cir_to_ofdm_channel, gen_single_sector_topology,
                                subcarrier_frequencies)
from sionna.phy.channel.tr38901 import UMa, UMi, PanelArray

from config import SystemConfig

_MODELS = {"umi": UMi, "uma": UMa}


class ChannelGenerator:
    """Un appel = un nouveau tirage de topologie + canal sur `num_slots` slots.

    Le canal est echantillonne une fois par slot (constant dans le slot :
    temps de coherence >> 0.5 ms aux vitesses considerees). Le modele Sionna
    est instancie en uplink (sondage par la BS) ; la sortie est le canal
    downlink par reciprocite.

    Sortie : h [B, num_slots, num_sc, K, M] complex64
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        fc = cfg.carrier_frequency
        ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                              polarization="single", polarization_type="V",
                              antenna_pattern="omni", carrier_frequency=fc)
        bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=cfg.num_bs_ant // 2,
                              polarization="dual", polarization_type="cross",
                              antenna_pattern="38.901", carrier_frequency=fc)
        self.model = _MODELS[cfg.scenario](
            carrier_frequency=fc, o2i_model="low", ut_array=ut_array, bs_array=bs_array,
            direction="uplink", enable_pathloss=False, enable_shadow_fading=False)
        self.frequencies = subcarrier_frequencies(cfg.num_sc, cfg.subcarrier_spacing)

    def __call__(self, batch_size):
        cfg = self.cfg
        topology = gen_single_sector_topology(
            batch_size, cfg.num_users, cfg.scenario,
            indoor_probability=cfg.indoor_probability,
            min_ut_velocity=cfg.min_speed_kmh / 3.6,
            max_ut_velocity=cfg.max_speed_kmh / 3.6)
        self.model.set_topology(*topology)
        a, tau = self.model(batch_size, cfg.num_slots, 1.0 / cfg.slot_duration)
        # [B, 1, M, K, 1, S, F] -> [B, S, F, K, M]
        h = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        return tf.transpose(h[:, 0, :, :, 0], [0, 3, 4, 2, 1])
