# freqctx : résumé pour reprendre (état au 1er novembre 2026)

Le journal complet est dans `FINDINGS.md`, phases 0 à 11, avec un résultat par section et les fichiers associés. La page d'architecture est https://claude.ai/artifact/MY5LBPGqDPuKfhQcL5dYgB (version 11, source `docs/architecture.html`). La page de vulgarisation est https://claude.ai/artifact/Jc2U4UTi2Dh2cbjRPaq7Cg (source `docs/explainer.html`).

## Le sujet

Précoder un système MU-MIMO OFDM en voie descendante **à partir des pilotes de sondage montant** (TDD, réciprocité), avec un petit transformer appliqué à chaque sous-porteuse. On veut comprendre ce qu'apporte le contexte fréquentiel. La cible matérielle est un FPGA.
- Tout est jugé sur le débit somme.
- Il n'y a aucune inversion de matrice dans le modèle.
- Simulation avec Sionna 1.2.1 (TF 2.20), modèles TR 38.901 UMi/UMa/RMa en mélange.
- M8K4 : 8 antennes, 4 utilisateurs, 612 SC.
- Massif : M64K8, 600 SC.

## La base figée (version 2)

1. **Encodeur fréquentiel** (`models/delay_domain.py`), par utilisateur, sur toute la bande. Les pilotes LS en peigne passent par une IDFT vers 306 retards (grille 2× plus fine), puis par 2 couches d'attention softmax entre les retards (largeur 32), puis par un filtre complexe sur les retards t−1, t, t+1, puis par une FFT. Sa sortie ĥ(f) donne surtout la **direction** du canal : 5 à 7,5 dB plus précise que LMMSE. Ce n'est pas un estimateur au sens de l'erreur quadratique.
2. **Transformer par paires d'utilisateurs** (`models/gram_precoder.py`, `GramPrecoder(mix="triangular_g", tied=False, layers=8, d=16, hidden=32, inp="norm")`), sur chaque SC, avec les mêmes poids pour toutes :
   - chaque canal d'utilisateur est normalisé, et sa puissance relative est donnée à part ;
   - G = h̃h̃ᴴ, avec un token par paire (j,k) ;
   - 8 couches d'attention triangulaire linéaire, où la paire (j,k) reçoit Σ_l a(j,l)·b(l,k), et G est redonné à chaque couche ;
   - on lit C (K×K), et W = h̃ᴴ C.
3. **Entraînement** : perte −débit / débit(RZF sur LS), 80 000 pas, 1 056 tirages par scénario, SNR de données U(0,25), **SNR pilote U(0,20)**, **puissances réelles entre utilisateurs** (`SystemConfig(relative_power=True)` : affaiblissement de parcours et masque 3GPP, sondage avec contrôle de puissance montant).
4. **Taille** : 72 k paramètres, 0,88 M MAC/SC (M8K4). **Poids** : `/export/tmp/sala/freqctx/weights/phase7/pilot_rand_ph32_norm_relpow.weights.h5` (srvgrm12).
5. **Variante de coût** : le précodeur est calculé sur 1 SC sur 3, et W est interpolé linéairement (`anchor_step=3`). Cela coûte −0,1 bps/Hz au plus, pour un calcul divisé par 2,4.

## Résultats clés

Pilote 10 dB, données à 20 dB, UMi / UMa / RMa, en bps/Hz.

| | notre modèle | meilleur classique réaliste (WMMSE robuste sur LMMSE) |
|---|---|---|
| M8K4, puissances égales | 25,9 / 25,5 / 26,3 | 24,2 / 23,7 / 24,4 |
| M8K4, puissances réelles (base v2) | 17,1 / 17,2 / 17,4 | 16,2 / 16,3 / 16,4 (100 it.) |
| Massif M64K8, puissances égales | 72,9 / 72,0 / 74,1 | 69,0 / 68,0 / 69,5 |
| M8K4, pilote 0 dB (pilote aléatoire) | 22,4 / 22,2 / 24,2 | 19,0 / 18,9 / 20,3 |

- **Sans encodeur, à partir des pilotes, ça ne marche pas** (phase 7). Le même transformer, sur le LS de chaque SC seule, fait 22,4. Avec LMMSE classique en entrée, il fait 24,3. Avec l'encodeur sur 1 / 3 / 17 / 51 RB, il fait 23,2 / 24,5 / 25,5 / 25,9. **Tout le gain à partir des pilotes vient de l'encodeur fréquentiel.** Entraîner encodeur et précodeur séparément perd 0,2.
- **Le précodeur seul, en CSI parfait**, égale ou dépasse RZF et WMMSE (banc de la phase 6, tailles jamais vues, utilisateurs corrélés).
- **Références apprises, dans le même cadre** : TA-RB du mémoire réécrit (1,1 M paramètres) fait 24,2 / 23,6 / 25,4. LMMSE + un précodeur de type 2D-Gformer (Guo et Yang) fait 24,2 / 23,8 / 24,5.
- **Robustesse** : le vieillissement jusqu'à 5 ms passe, ainsi qu'un pilote entre 0 et 20 dB (grâce à l'entraînement avec pilote aléatoire).
- **Écarté** (raisons dans `FINDINGS`) : un token par utilisateur, Gram déroulé, le modèle axial (antenne × utilisateur), la symétrie hermitienne, la réduction de l'encodeur, l'interpolation des coefficients C, et le précodeur par RB (qui perd 3,6 en UMi).

## Ce qui a été fait en PyTorch, et pourquoi

Sionna (TensorFlow) reste la référence pour **toute la chaîne OFDM** : canal, pilotes, encodeur, entraînements de bout en bout. Le dossier `torch_port/` sert à brancher **la chaîne BitAdapt du labo** (PyTorch), clonée dans `/users/sala/Documents/test_projet/external/mMIMO-ART-FPGA` :
- `models.py` : la base réécrite en PyTorch. L'équivalence avec Keras est vérifiée par `check_equivalence.py` (écart 4·10⁻⁵).
- `stecath.py` : **le précodeur seul** (sans encodeur, sans OFDM), entraîné de zéro sur le jeu St-Catherine du labo. C'est du lancer de rayons mono-porteuse, M=64, K=4, avec des fichiers `.mat` dans `/usr/datasets/deepMIMO/HH_channels/Dataset_matlab/`. Le débit est calculé exactement comme dans le labo. **Il n'y a pas de Sionna ici.**
- `art_stecath.py` : la quantification ART avec les quantificateurs LSQ du labo, et l'option `--qprod` qui quantifie aussi G, le produit triangulaire et W.
- `energy.py` : le modèle d'énergie de Moons et al. avec les constantes de Horowitz, **corrigé**.

Résultats St-Catherine, précodeur seul :
- En 32 bits : 27,6 (entrée normalisée), contre WMMSE convergé 26,9, et BitAdapt medium 24,5 avec 1,8 M paramètres.
- **Tout quantifié** : **26,6 bps/Hz à 0,11 µJ**, contre BitAdapt medium 24,1 à 0,81 µJ (énergie corrigée).
- WMMSE converge mal quand les puissances sont inégales : 30 itérations donnent 24,4, il en faut 300 pour 26,9.

**Correction du modèle d'énergie du labo** : la branche `fix/energy-model` a été poussée sur FLPlab/mMIMO-ART-FPGA, sans pull request (`gh` absent).
- la mémoire est linéaire en Q ;
- √p = √(64·16/Q) ;
- les MAC des couches denses sont multipliés par le nombre de tokens ;
- la longueur de séquence de l'attention est corrigée ;
- les appels utilisent (1, K, 2N).

## Questions ouvertes et suite proposée

1. **Que rédige-t-on ?** C'est à décider. Deux articles possibles :
   - **A. OFDM à partir des pilotes** (Sionna) : l'encodeur fréquentiel et le transformer par paires, les résultats M8K4 et massif, la robustesse, les puissances réelles, les références apprises. C'est la contribution principale. **L'encodeur y est indispensable.**
   - **B. Précodeur quantifié pour FPGA** (St-Catherine, chaîne BitAdapt) : le transformer par paires contre un token par utilisateur, ART, l'énergie corrigée, puis la synthèse HLS. C'est le prolongement de l'article SiPS 2026.
2. **Pas encore fait** :
   - la quantification de la base OFDM complète (l'encodeur compris) ;
   - le massif avec la base v2 ;
   - le générateur HLS : bloc triangulaire, produits complexes, normalisation par utilisateur avec `rsqrt` ;
   - le débit codé (LDPC) ;
   - un canal jamais vu (Sionna RT, par exemple Munich).
3. **Suite scientifique** : le contrôle de débit en OFDM (débit cible par utilisateur sur toute la bande, dans le prolongement d'Emery et al. 2507.18587), avec un token par utilisateur à l'échelle de la bande.

## Pratique

- **srvgrm12** (3× L40S, environnement `tf-gpu` pour TensorFlow et `bitadapt` pour PyTorch) est à préférer, même avec les jobs de l'utilisateur, qui accepte le partage avec `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- **srvgrm08** (4× P100, environnement `freqctx`) est 4 fois plus lent, et les convolutions de TA-RB n'y passent pas.
- **Sauvegardes** : le code, les résultats (`results/`, JSON) et le journal sont commités sur la branche `freqctx`, poussée sur GitHub (GlodiSala/Sionna, branche freqctx). Les poids (35 Mo) sont dans `/export/tmp/sala/freqctx/weights/` (srvgrm12, disque local) avec une copie dans `/users/sala/Documents/test_projet/freqctx_weights_backup/`. Les données d'entraînement (4 Go dans `/export/tmp/sala/freqctx/data/`) se régénèrent avec les scripts.
- Toujours lancer avec `python -u`. Les scripts de lancement sont dans `scripts/run6.sh` et `scripts/run7.sh`.
- **Règles de l'utilisateur** :
  - ne pas toucher à `mimo_precoding/` ;
  - une modification à la fois ;
  - noter tous les résultats, négatifs compris, dans `FINDINGS`, mais retirer du dépôt le code et les fichiers non retenus ;
  - mettre à jour la page d'architecture après chaque résultat validé ;
  - ne pas sur-ingénierer ;
  - répondre en français, de façon claire et concrète.
