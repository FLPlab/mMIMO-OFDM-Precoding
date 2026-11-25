# freqctx — petit transformer par sous-porteuse et contexte fréquentiel

Nouveau départ, indépendant de `../mimo_precoding/` (rien n'est importé de l'ancien code).
Question du paper : **un petit transformer réutilisé sur chaque sous-porteuse (SC)
suffit-il, et que doit apporter la fréquence (fenêtre, RB/PRG, contexte global), à quel coût ?**
Cible matérielle : FPGA (un seul bloc multiplexé dans le temps sur toutes les SC).

## Structure

```
freqctx/
  config.py              SystemConfig (M, K, bande, scénario, puissances réelles) + CSIConfig
  channel.py             ChannelGenerator : Sionna TR 38.901 UMi/UMa/RMa -> h [B, slot, SC, K, M]
  csi.py                 UplinkSounder (pilotes en peigne + LS / LMMSE Sionna), estimateurs de référence
  precoders/classical.py MRT, ZF, RZF, WMMSE (par SC ; WMMSE robuste)
  models/                delay_domain.py (encodeur fréquentiel, Wiener en formule fermée), gram_precoder.py
                         (précodeur par paires d'utilisateurs), per_sc.py (blocs de base)
  train.py               données, boucle d'entraînement, évaluation
  experiments/           un script par expérience : phaseN_<nom>.py (phases 0 à 14) + presentation_figs.py
  torch_port/            port PyTorch, St-Catherine, quantification ART (quantificateurs du labo), énergie
  hls/                   modèle quantifié -> référence entière -> C++ HLS (voir hls/README.md)
  tests/                 test_sanity, test_models (équivariance), test_env_repro (machines)
  results/<phase>/       un fichier par expérience : <expérience>.{json,png,pdf}
  results/presentation/  figures de synthèse et architecture.pdf
  FINDINGS.md            journal : question -> chiffres -> verdict (retenu / négatif / écarté)
  docs/                  RESUME.md (point de reprise), architecture.html, make_architecture_pdf.py, TRI_DEPOT.md
  requirements.txt       versions exactes de l'environnement
```

Règles :
- une expérience = un script `experiments/phaseN_<nom>.py`, qui écrit `results/phaseN/<nom>.*`
  (config, graine, commit dans le JSON ; une relance écrase, l'historique est dans git) ;
- évaluations **appariées** (un tirage de canal réutilisé pour toutes les méthodes/conditions) ;
- après chaque expérience : une entrée dans `FINDINGS.md` (question, chiffres, verdict),
  **y compris les résultats négatifs ou inutiles**, avec leurs chiffres. Une option écartée est
  retirée du code pour le garder simple ; elle reste décrite dans le journal ;
- pas d'abstraction tant qu'on n'en a pas besoin deux fois.

## Conventions

- Canal downlink par SC `H [..., K, M]`, `y = H W s + n`, réciprocité TDD (sondage uplink).
- Précodeur `W [..., M, K]`, puissance totale `||W||_F^2 = 1` par SC ; SNR = 1/σ².
- Canal normalisé (énergie moyenne 1 par élément), pas de pathloss ni shadowing.
- Défaut : M=8, K=4, 3,5 GHz, 30 kHz, 51 RB = 612 SC (canal 20 MHz), UMi, 3–30 km/h,
  1 slot = 0,5 ms ; CSI sondé au slot 0, données au slot `delay_slots`.
- CSI estimé : un symbole de sondage, peigne orthogonal entre utilisateurs (1 SC sur K),
  LS puis interpolation linéaire (`ls_lin`) ou plus proche voisin (`ls_nn`).

## Lancer

Environnement figé dans `requirements.txt` (Python 3.11.13, TF 2.20.0, Keras 3.11.3, Sionna 1.2.1).

**srvgrm12** (L40S, env `tf-gpu`) :
```
source /export/tmp/sala/miniconda/bin/activate tf-gpu
export LD_LIBRARY_PATH=/export/tmp/sala/miniconda/envs/tf-gpu/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true  # plusieurs jobs par GPU
```

**srvgrm08** (P100 16 Go, env `freqctx`, créé le 2026-10-23 ; ne pas confondre avec son `tf-gpu` en TF 2.13) :
```
source /export/tmp/sala/anaconda3/bin/activate /export/tmp/sala/anaconda3/envs/freqctx
export LD_LIBRARY_PATH=/export/tmp/sala/anaconda3/envs/freqctx/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true
```

Puis, depuis `freqctx/` :
```
python -m tests.test_sanity          # chaîne canal / CSI / précodeurs
python -m tests.test_models          # modèles : forme, puissance, équivariance
python -m tests.test_env_repro       # même canal que la référence srvgrm12 ? (obligatoire sur une nouvelle machine)
python -m experiments.phase0_channel_stats
python -m experiments.phase0_baselines [--quick]
python -m experiments.phase1_size_sweep [--quick]
python -m experiments.phase1_signed_gram [--d 64 --layers 2] [--quick]
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
