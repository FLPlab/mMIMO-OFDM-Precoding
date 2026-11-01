"""Configuration du systeme : une seule source de verite pour toutes les phases.

Conventions (valables partout dans freqctx/) :
  - Canal downlink par SC : H [..., K, M], y = H W s + n, reciprocite TDD
    (H_dl = transposee du canal uplink sonde par la BS).
  - Precodeur par SC : W [..., M, K], puissance totale ||W||_F^2 = 1 par SC.
  - SNR de donnees = 1 / sigma^2 (puissance d'emission par SC / bruit par RE),
    canal normalise a une energie moyenne unitaire par element.
  - Axes des tenseurs de canal : [batch, slot, sc, K, M].
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    num_bs_ant: int = 8
    num_users: int = 4
    carrier_frequency: float = 3.5e9
    subcarrier_spacing: float = 30e3
    num_rb: int = 51                    # 20 MHz a 30 kHz (TS 38.101-1, table 5.3.2-1)
    scenario: str = "umi"               # "umi" | "uma" | "rma"
    min_speed_kmh: float = 3.0
    max_speed_kmh: float = 30.0
    indoor_probability: float = None    # None = valeur 3GPP du scenario
    num_slots: int = 3                  # slot 0 = sondage, slots 1.. = donnees (vieillissement)
    slot_step: int = 1                  # ecart entre deux instants generes, en slots
    relative_power: bool = False        # (phase 11) affaiblissement de parcours + masque 3GPP gardes :
                                        # canal normalise par tirage (puissance moyenne 1 sur les
                                        # utilisateurs), ecarts de puissance reels entre utilisateurs

    @property
    def num_sc(self):
        return 12 * self.num_rb

    @property
    def bandwidth(self):
        return self.num_sc * self.subcarrier_spacing

    @property
    def slot_duration(self):
        return 1e-3 * 15e3 / self.subcarrier_spacing


@dataclass(frozen=True)
class CSIConfig:
    estimator: str = "ls_lin"           # "perfect" | "ls_nn" | "ls_lin" | "ls_lmmse"
    pilot_snr_db: float = 20.0          # SNR par RE pilote a l'antenne BS
    delay_slots: int = 0                # ecart sondage -> slot de donnees
