# freqctx — petit transformer par sous-porteuse et contexte fréquentiel

Nouveau départ, indépendant de `../mimo_precoding/` (rien n'est importé de l'ancien code).
Question du paper : **un petit transformer réutilisé sur chaque sous-porteuse (SC)
suffit-il, et que doit apporter la fréquence (fenêtre, RB/PRG, contexte global), à quel coût ?**
Cible matérielle : FPGA (un seul bloc multiplexé dans le temps sur toutes les SC).

## Structure

```
freqctx/
  config.py              SystemConfig (M, K, bande, scénario, vitesses) + CSIConfig
  channel.py             ChannelGenerator : Sionna TR 38.901 UMi/UMa -> h [B, slot, SC, K, M]
  csi.py                 UplinkSounder (pilotes kronecker + LS Sionna), csi_at_bs, smooth_freq
  metrics.py             sinr, sum_rate (toujours sur le canal VRAI)
  precoders/classical.py MRT, ZF, RZF, WMMSE (par SC)
  models/                (phase 1+) modèles appris
  experiments/           un script par expérience : phaseN_<nom>.py
  tests/test_sanity.py   tests de cohérence de la chaîne
  utils.py               sauvegarde résultats (json + png/pdf + commit git), style figures
  results/<phase>/       UNIQUEMENT les résultats retenus : <expérience>.{json,png,pdf}
  FINDINGS.md            journal : question -> chiffres -> verdict, pour chaque expérience
```

Règles :
- une expérience = un script `experiments/phaseN_<nom>.py`, qui écrit `results/phaseN/<nom>.*`
  (config, graine, commit dans le JSON ; une relance écrase, l'historique est dans git) ;
- évaluations **appariées** (un tirage de canal réutilisé pour toutes les méthodes/conditions) ;
- après chaque expérience : une entrée dans `FINDINGS.md`. Si ce n'est pas intéressant, on
  supprime les fichiers et on retire la condition du script ; seule l'entrée « écarté » reste ;
- pas d'abstraction tant qu'on n'en a pas besoin deux fois.

## Conventions

- Canal downlink par SC `H [..., K, M]`, `y = H W s + n`, réciprocité TDD (sondage uplink).
- Précodeur `W [..., M, K]`, puissance totale `||W||_F^2 = 1` par SC ; SNR = 1/σ².
- Canal normalisé (énergie moyenne 1 par élément), pas de pathloss ni shadowing.
- Défaut : M=8, K=4, 3,5 GHz, 30 kHz, 51 RB = 612 SC (canal 20 MHz), UMi, 3–30 km/h,
  1 slot = 0,5 ms ; CSI sondé au slot 0, données au slot `delay_slots`.
- CSI estimé : un symbole de sondage, peigne orthogonal entre utilisateurs (1 SC sur K),
  LS puis interpolation linéaire (`ls_lin`) ou plus proche voisin (`ls_nn`).

## Lancer (srvgrm12)

```
source /export/tmp/sala/miniconda/bin/activate tf-gpu
export LD_LIBRARY_PATH=/export/tmp/sala/miniconda/envs/tf-gpu/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3
cd freqctx
python -m tests.test_sanity
python -m experiments.phase0_channel_stats
python -m experiments.phase0_baselines [--quick]
```

## Plan

| Phase | Contenu | Question |
|---|---|---|
| 0 | canal 20 MHz, CSI réaliste (pilotes, vieillissement), références MRT/ZF/RZF/WMMSE, RZF robuste, RZF + lissage | quelle marge reste-t-il à gagner, et où ? |
| 1 | transformer minimal par SC (token = utilisateur, sans encodage de position, perte = −débit), CSI parfait ; balayage D/L ; MLP vs GNN vs transformer ; interprétabilité (attention vs Gram, similarité à RZF) | plus petit modèle ≈ RZF ? |
| 2 | même modèle entraîné sur le CSI estimé | l'apprentissage seul bat-il RZF(ĥ) / RZF robuste ? |
| 3 | fenêtre glissante de W SC (concaténation, puis attention en fréquence) ; scénarios mélangés | gain du contexte fréquentiel vs lissage classique ; fenêtre adaptative ? |
| 4 | (option) contexte temporel contre le vieillissement | |
| 5 | sortie par PRG, effet DMRS côté UE, coût FPGA (MAC, énergie/RB), quantification | |
| 6 | contrôle du débit par utilisateur (extension OFDM d'Emery et al., arXiv 2507.18587) | |
