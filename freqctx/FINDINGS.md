# Journal des résultats

## SYNTHÈSE (triée le 2026-10-24, à relire avant tout)

Réglages par défaut : M=8, K=4, 612 SC (20 MHz), pilotes Sionna en peigne + LS, pilote 10 dB, débit à 20 dB. Chaque ligne renvoie à son entrée dans le journal détaillé et à son fichier `results/`.

### A. L'histoire du paper (retenu, dans l'ordre logique)

| # | Constat | Fichier |
|---|---|---|
| A1 | **En CSI parfait, il n'y a presque rien à gagner par SC** : WMMSE = RZF + 1,7 % | `phase0/baselines` |
| A2 | **La marge est dans l'estimation** : RZF passe de 26,2 (parfait) à 21,7 (LS) et 23,5 (LMMSE) ; WMMSE(LMMSE) = 23,9 | `phase0/baselines`, `phase3/wmmse_ref_p10` |
| A3 | **Entre utilisateurs, l'attention doit pouvoir soustraire** : softmax 74 % de RZF, signée 93 %, linéaire équivalente à la signée | `phase1/signed_gram`, `phase1/highsnr` |
| A4 | **Ce sont les itérations qui comptent** : un seul bloc répété 4 fois = 4 blocs distincts, 3,7 fois moins de paramètres | `phase1/highsnr` |
| A5 | **Sans contexte fréquentiel, apprendre sur du CSI estimé n'apporte rien** (= RZF(LS)) | `phase2/estimated_csi_p10`, `_p20` |
| A6 | **Avec les SC voisines, le modèle bat le lissage classique** ; il faut une largeur D=64 | `phase3/window_p10`, `phase3/diagnostic_p10` |
| A7 | **Quand la covariance est mal connue (mélange UMi/UMa/RMa), LMMSE perd 1 bps/Hz sur RMa (canal plat)** | `phase3/mixed_p10` |
| A8 | **L'attention en fréquence (adaptative) bat WMMSE(LMMSE réaliste) sur RMa** (+0,64 à 20 dB, +1,32 à 25 dB), égalité en ville ; fenêtre effective plus large sur RMa (10,9 contre 8,5 SC) | `phase3/mixed_p10_fattn_D64` |
| A9 | **La marge restante est dans l'estimation** : borne génie par lien ≈ 2 bps/Hz au-dessus ; élargir la fenêtre (153 SC) n'y change rien (lissage à poids positifs) | `phase4/genie_bound_p10`, `mixed_p10_fattn_D64_W153` |
| A10 | **Estimer dans le domaine des retards (toute la bande) bat WMMSE(LMMSE réaliste) dans les TROIS scénarios** : +3 à 3,5 % à 20 dB, +6 à 7 % à 25 dB ; garde 14 retards sur RMa contre 23-24 en ville (adaptatif, interprétable) ; 40 % de l'écart au génie comblé | `phase4/delay_model_D64` |
| A11 | **Le gain tient face à un WMMSE robuste** (erreur d'estimation prise en compte) ; la version robuste n'apporte que +0,1 au classique | `phase4/robust_ref_p10` |
| A12 | **L'écart restant venait de l'estimation (attribution), et surtout de la fuite** : filtre à bande (+0,2), 3× plus de données (+0,25), grille de retards 2× plus fine (+0,5). Modèle final : **+1,6 à +1,7 bps/Hz (+6,5 à 6,9 %) à 20 dB et +3,0 à +3,5 à 25 dB sur WMMSE robuste réaliste**, au-dessus dès 10 dB ; à 0,2-0,4 bps/Hz du génie | `phase4/attribution_p10`, `delay_model_bande_os2_n1056_s80k_D64` |

### B. Résultats d'appui (retenu, utile pour justifier des choix)

| # | Constat | Fichier |
|---|---|---|
| B1 | Bande de cohérence : 9 SC à corrélation 0,9 ; RMa bien plus plat (0,98 sur 1 RB) | `phase0/channel_stats`, journal phase 3 |
| B2 | Le transformer simple a besoin de D=128 et L=4 pour égaler RZF en CSI parfait ; la profondeur compte plus que la largeur ; l'attention bat une moyenne de 18 points | `phase1/size_sweep`, `size_sweep_signed` |
| B3 | La perte relative à RZF corrige le faible SNR | `phase1/candidate` |
| B4 | Fenêtre concaténée W=49 : +0,2 / +0,44 bps/Hz sur RMa face à WMMSE(LMMSE réaliste), confirmé sur 3 graines (écart-type 0,01) | `phase3/mixed_p10_W49*` |
| B5 | La fenêtre large et la spécialisation par scénario comptent autant l'une que l'autre | `phase3/mixed_p10_specialiste_rma` |

### C. Négatifs (à citer brièvement, ne pas refaire)
Sortie structurée W = Hᴴ C ; biais Gram ; normalisation de phase ; attention signée avec fenêtre ; attention en fréquence à D=32 ; RZF robuste ; interpolation au plus proche voisin ; tête de puissance (+0,03 bps/Hz) ; estimateur classique dans le domaine des retards (limité par la fuite, battu par notre modèle).

### D. Limites connues du résultat actuel
- Le gain ne tient que là où la covariance est mal adaptée (RMa) et à SNR moyen/fort ; au faible SNR, environ −0,2 bps/Hz partout (limite du précodeur).
- Une seule configuration : M8K4, pilote 10 dB **seulement** (le modèle ne reçoit pas le SNR pilote), canal sans vieillissement.
- Une seule graine pour l'attention en fréquence ; pas de bootstrap sur les tirages.
- Coût : 595 k MAC/SC. À comparer à la **chaîne complète** LMMSE + WMMSE (estimée à 150-200 k MAC/SC avec inversions en double précision, à calculer précisément), pas à RZF seul.

### E. Plan (simple, sans s'éparpiller)
0. **(2026-10-24) Candidat actuel : estimation dans le domaine des retards sur une grille 2× plus fine (306 retards) avec filtre complexe sur 3 retards voisins + précodeur D64 par SC, entraîné sur 1 056 tirages par scénario et 80 000 pas** (`phase4/delay_model_bande_os2_n1056_s80k_D64`). +1,6 à +1,7 bps/Hz à 20 dB sur la meilleure référence classique ; reste 0,2-0,4 jusqu'au génie. Estimation terminée. Faible SNR : écart accepté (variante RZF documentée). Prochain : massif.
1. **Massif M=64, K=8** (après le point 0) : même modèle, mêmes références, coût de la chaîne classique complète.
2. **Carte de fonctionnement** (sans réentraîner) : étalement des retards en continu, SNR pilote.
3. **À la fin** : graines, robustesse, coûts précis.
**Mis de côté** : vieillissement du CSI, ray-tracing, BLER LDPC, contrôle du débit, PRG, quantification.

### F. Positionnement (revue de littérature du 2026-10-24, papiers vérifiés)
- **Concurrent le plus proche** : Park, Sohrabi, Ghosh, Andrews, TWC 2025 (arXiv:2402.01033). Bout en bout TDD, battent LMMSE + WMMSE, mais en **apprenant les pilotes** et **par RB** (sans exploiter la fréquence). Nous : pilotes standard (peigne), toute la bande dans le domaine des retards.
- **Autres proches** : Jha, Guo, Lau 2025 (arXiv:2504.20777 : parcimonie en retard, mais objectif EVM et précodeur déroulé) ; Jang et al. TCOM 2022 ; Attiah/Sohrabi/Yu TWC 2022 ; Guo et al. 2024-2025 (FDD).
- **À citer comme ancêtre de notre masque** : Neumann, Wiese, Utschick, « Learning the MMSE Channel Estimator », TSP 2018 (arXiv:1707.05674) ; et Ha et al. 2025 (arXiv:2506.00452 : un transformer produit un filtre MMSE linéaire). Les deux sont entraînés sur l'erreur d'estimation, pas sur le débit.
- **Théorie à l'appui du précodeur** : attention linéaire = itérations de gradient préconditionné (Ahn et al. NeurIPS 2023 ; Gatmiry et al. ICML 2024 ; Lutz et al. NeurIPS 2025) ; transformers en boucle (Yang et al. ICLR 2024 ; Giannou et al. ICML 2023).
- **Notre groupe** (travaux compagnons, pas concurrents) : Emery 2025, Kasalaee/Sala Mangituka et al. SiPS 2026, Hasanzadeh Karkan et al. 2026 (PaPP). Aucun ne fait l'estimation.
- **Nouveauté** : estimation sur toute la bande dans le domaine des retards (masque appris) + précodeur à attention linéaire en boucle, **appris ensemble sur le débit**, évalués dans Sionna contre LMMSE + WMMSE et une borne génie. Aucun papier de précodage appris trouvé n'utilise Sionna.

**Objections qu'un relecteur fera (à traiter) :**
1. **Référence robuste** : un WMMSE qui tient compte de l'erreur d'estimation (covariance d'erreur de LMMSE ; WMMSE stochastique / SAA, Razaviyayn 2016, Joudeh & Clerckx 2016 ; GPIP, Choi et al. 2020). Elle réduira probablement une partie de notre gain au fort SNR.
2. **Attribution** : notre estimateur + WMMSE classique, et LMMSE + notre précodeur, pour séparer le gain d'estimation du gain de précodage.
3. **Équité** : dire exactement ce que chaque référence connaît ; tester sur un scénario non vu à l'entraînement.
4. **Faible SNR**, **coût** (montrer la croissance avec M), **réalisme** (BLER LDPC, vieillissement, plusieurs SNR pilote).

**Idées simples proposées par la littérature** : (a) grille de retards sur-échantillonnée ou masque à bande étroite contre la fuite (DDA-Net, arXiv:2604.05389) ; (b) donner au précodeur une estimation de l'erreur résiduelle ; (c) initialiser la boucle du précodeur avec RZF et n'apprendre que la correction (SALLO-M, arXiv:2510.13077), pour le faible SNR ; (d) plus d'itérations à l'inférence.

### G. Coût (MAC réelles par SC, estimation + précodage)
| | M8K4 | M64K8 |
|---|---|---|
| LMMSE + RZF | 21 k | 193 k |
| LMMSE + WMMSE (30 it., double précision) | 122 k | 19 100 k |
| notre modèle (retards + précodeur D64) | 602 k (dont 541 k de précodeur) | 1 418 k |

En M8K4, le modèle coûte environ 5 fois LMMSE + WMMSE ; en M64K8, environ **13 fois moins** (WMMSE croît en M³, le modèle quasi linéairement). WMMSE exige aussi des inversions en double précision ; le modèle n'a que des multiplications et additions, quantifiables.

---

## Journal détaillé

Une entrée par expérience : question → réglages → chiffres → verdict.
- **Retenu** : le fichier est dans `results/<phase>/`, il peut servir dans le paper.
- **Écarté** : les fichiers sont supprimés et seule l'entrée ci-dessous reste, pour s'en souvenir et ne pas refaire l'expérience.

Réglages communs, sauf mention contraire : M=8, K=4, 612 SC à 30 kHz (canal de 20 MHz), 3,5 GHz, drops 3GPP standard, 3 à 30 km/h. Débit somme en bps/Hz, à 20 dB de SNR de données.

---

## Phase 0 — 2026-10-23

### Statistiques du canal — RETENU (`results/phase0/channel_stats.*`)
- **Question** : quelle sélectivité en fréquence et en temps a-t-on à 20 MHz ?
- **Bande de cohérence** (corrélation 0,9 / 0,5) : UMi 270 kHz (9 SC) / 1,6 MHz ; UMa 210 kHz / 1,35 MHz. **Un RB (360 kHz) dépasse déjà la bande de cohérence à 0,9.**
- **Corrélation en temps** : 0,991 / 0,966 / 0,87 à 0,5 / 1 / 2 ms.
- **Étalement des retards médian** : UMi 384 ns, UMa 402 ns.

### Références classiques — RETENU (`results/phase0/baselines.*`)
- **Question** : quelle marge reste-t-il pour un modèle appris, et où ?
- **Réglages** : 512 tirages par scénario, évaluation appariée.
- **LMMSE** : `LSChannelEstimator` + `LMMSEInterpolator` de Sionna. La covariance en fréquence (612×612) est la moyenne du scénario, estimée sur des tirages séparés.

| CSI connu de la BS (UMi), débit à 20 dB (25 dB) | Erreur CSI | RZF | WMMSE | RZF + meilleur lissage |
|---|---|---|---|---|
| Parfait | — | 26,11 (32,65) | 26,56 | 26,11 (W=1) |
| LS lin, pilote 20 dB | −23 dB | 25,12 (30,23) | 25,57 | 25,14 (W=3) |
| **LS LMMSE, pilote 20 dB** | −26 dB | **25,49 (31,02)** | 25,94 | 25,52 (W=3) |
| LS lin, pilote 10 dB | −17 dB | 21,59 (23,97) | 21,99 | 22,51 (W=9) |
| **LS LMMSE, pilote 10 dB** | −19,5 dB | **23,38 (26,79)** | 23,81 | 23,53 (W=5) |
| Parfait, vieilli de 1 slot | −18 dB | 25,57 | 26,03 | — |
| Parfait, vieilli de 2 slots | −12 dB | 24,55 | 25,00 | — |

UMa donne les mêmes conclusions, à ±0,2 bps/Hz près.

**Conclusions :**
- **CSI parfait** : WMMSE ne fait que +0,45 sur RZF (1,7 %). Il n'y a presque aucune marge par SC.
- **La marge est dans l'estimation.** À pilote 10 dB, on perd 4,5 bps/Hz à 20 dB de SNR de données.
  - Le lissage simple en récupère 0,9.
  - LMMSE (la meilleure méthode classique) en récupère 1,8.
  - **Il reste 2,7 bps/Hz (10 %) à 20 dB, et 5,9 bps/Hz (18 %) à 25 dB, même après LMMSE.** C'est la marge cible du transformer.
  - À pilote 20 dB, l'écart restant n'est que de 0,6 (1,6 à 25 dB).
- **Pourquoi LMMSE ne comble pas tout** : sa covariance est la moyenne du scénario, pas celle du lien. Or chaque réalisation n'a qu'une vingtaine de chemins (elle est parcimonieuse dans le domaine des retards). Un estimateur qui s'adapte à chaque réalisation peut donc mieux faire : c'est l'argument d'un transformer adaptatif.
- **La fenêtre optimale est petite** et de l'ordre de la bande de cohérence à 0,9 : W=3 (p20), W=9 (p10). Le lissage large détruit le canal : W=49 donne 17 bps/Hz.
- **Le vieillissement** coûte 1,6 bps/Hz après 2 slots, et le lissage en fréquence n'y peut rien.

**Écartés de cette expérience** (retirés du script, ne pas refaire) :
- *RZF robuste* (α = K(σ² + erreur CSI)) : pas mieux que RZF, et pire en vieillissement (23,99 contre 24,67 après 2 slots).
- *Interpolation LS au plus proche voisin* : moins bonne que l'interpolation linéaire (24,45 contre 25,26 à p20), et un lissage W=5 la ramène au niveau de la linéaire. Ça n'apporte rien de plus.
- *LS p20 + 1 slot de vieillissement* : redondant avec les deux effets pris séparément.

### Point de comparaison avec le mémoire
Dans le mémoire (NLOS forcé, utilisateurs regroupés dans un disque de rayon R, 2,88 MHz — cf. `../README.md` ; corrigé le 2026-10-23, la version précédente de cette ligne disait à tort « LOS forcé, 15° »), un pilote à 20 dB faisait tomber RZF de 40 à 27,6. En drops standard, la perte n'est que de 26,1 → 25,1. **Le gros gain de TA-RB tenait en grande partie à ce régime difficile.**

---

## Phase 1 — 2026-10-23

Modèle (`models/per_sc.py`) : un token par utilisateur `[Re h_k, Im h_k, SNR/10]` → Dense(D) → L × [attention entre utilisateurs + FFN] → Dense(2M) = w_k → normalisation de puissance. Perte = −débit somme. Même modèle pour toutes les SC, CSI parfait.

### Sondages exploratoires (avant le balayage) — NÉGATIF, chiffres seulement
Entraînement court (3 000 à 20 000 pas, pas d'apprentissage 3e-3, 256 à 512 drops), validation sur 32 drops (RZF : 5,21 / 14,29 / 26,62 à 0 / 10 / 20 dB).

| Modèle | Paramètres | 0 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| D=8, L=1, sortie directe, 3k pas | 0,9 k | 3,43 | 6,88 | 8,18 (≈ MRT) |
| D=32, L=2, sortie directe, 3k pas | 18 k | 4,86 | 11,50 | 14,93 |
| D=64, L=2, sortie directe, 20k pas | 69 k | 5,20 | 13,73 | 20,97 (79 %) |
| D=128, L=2, sortie directe, 20k pas | 269 k | 5,14 | 11,87 | 15,57 |
| D=8, L=1, **sortie W = Hᴴ C**, 10k pas | 1,0 k | 4,01 | 7,34 | 9,44 |
| D=16, L=1, **sortie W = Hᴴ C**, 10k pas | 3,1 k | 4,19 | 7,59 | 9,47 |
| D=32, L=2, **sortie W = Hᴴ C**, 10k pas | 19 k | 4,95 | 11,50 | 14,74 |

- **Au SNR faible, c'est facile** : on atteint RZF dès 18 k paramètres. **Au SNR fort, c'est difficile** : il faut annuler l'interférence, donc l'équivalent d'une inversion de matrice, et le modèle plafonne bien en dessous de RZF.
- **D=128 fait moins bien que D=64** à pas 3e-3 : probablement un pas trop grand, d'où le passage à 1e-3 dans le balayage.
- **Écarté : la sortie structurée W = Hᴴ C** (C[j,k] = ⟨a_j, b_k⟩, une forme qui contient RZF ; Björnson et al. 2014). Elle ne fait pas mieux que la sortie directe, à taille égale. **Le blocage n'est donc pas la forme de la sortie, mais le calcul des interactions entre utilisateurs** (Gram, inversion) à partir des valeurs brutes. Option retirée du code.
- *Note technique* : l'avertissement TF « casting complex64 to float32 » vient du gradient de la conversion réel → complexe dans `normalize_power`. Il est sans danger : la partie imaginaire du canal arrive bien au modèle (vérifié).

### Balayage de taille — RETENU (`results/phase1/size_sweep.*`)
- **Question** : quelle taille faut-il au transformer simple (softmax, sortie directe) pour égaler RZF en CSI parfait ? L'attention sert-elle ?
- **Réglages** : D ∈ {16, 32, 64, 128} × L ∈ {1, 2, 4}, plus « moyenne au lieu de l'attention » (D=64, L=2). 40 000 pas, pas 1e-3, batch 2 048, 1 024 drops d'entraînement (graine 1). Évaluation : 512 drops de la phase 0 × 612 SC. RZF = 26,18 et WMMSE = 26,63 à 20 dB. RZF coûte environ 1 400 MAC par SC.

% du débit RZF (MAC réelles par SC) :

| | 0 dB | 10 dB | 20 dB | 25 dB | MAC/SC |
|---|---|---|---|---|---|
| D16 L1 | 87 | 60 | 37 | 30 | 11 k |
| D64 L1 | 92 | 68 | 43 | 35 | 142 k |
| D128 L1 | 89 | 62 | 40 | 33 | 545 k |
| D32 L2 | 96 | 85 | 59 | 49 | 72 k |
| D64 L2 | 97 | 91 | 71 | 61 | 275 k |
| D128 L2 | 100 | 97 | 81 | 70 | 1,07 M |
| D16 L4 | 89 | 72 | 47 | 38 | 37 k |
| D32 L4 | 95 | 96 | 79 | 68 | 139 k |
| **D64 L4** | 101 | 102 | **97** | 92 | 541 k |
| **D128 L4** | 103 | 103 | **100** | 97 | 2,13 M |
| D64 L2, **moyenne** (sans attention) | 96 | 77 | **53** | 44 | 205 k |

**Conclusions :**
- **Un transformer simple par SC PEUT égaler RZF**, et dépasser légèrement RZF à faible SNR (≈ WMMSE), mais il lui faut D=128 et L=4 : 534 k paramètres, **environ 1 500 fois le coût de RZF**. Pour le FPGA, ce n'est pas défendable.
- **La profondeur compte plus que la largeur.** Avec une seule couche, on plafonne vers 40 % à 20 dB quelle que soit D. Passer de 2 à 4 couches à D=64 fait passer de 71 % à 97 %. C'est cohérent avec un calcul d'inversion qui demande plusieurs étapes successives d'échange entre utilisateurs.
- **L'attention sert vraiment.** À taille comparable, la remplacer par une moyenne fait perdre 18 points à 20 dB (71 → 53 %).
- **La difficulté monte avec le SNR** : tous les modèles sont proches de RZF à 0 dB. C'est l'annulation d'interférence, à fort SNR, qui coûte.
- **Variabilité d'un entraînement à l'autre** : D64 L2 donne 70,9 % ici et 73,8 % dans l'expérience 2 × 2 (même configuration et même graine, mais un autre GPU, où le calcul n'est pas déterministe). **Un écart de moins d'environ 3 points n'est donc pas significatif.**

### Attention signée × biais Gram (2 × 2) — RETENU (`results/phase1/signed_gram.*`)
- **Question** : qu'apportent, seuls puis ensemble, (a) des poids d'attention **signés**, softmax(s)·tanh(s), qui permettent de soustraire un utilisateur, et (b) un **biais Gram** f(Re, Im, |G_kj|) ajouté au score, qui dit qui interfère avec qui ?
- **Réglages** : D=64, L=2, mêmes données, graines et évaluation que le balayage. Calculé sur srvgrm08 (environnement vérifié identique).

| D64 L2 | 0 dB | 10 dB | 15 dB | 20 dB | 25 dB | MAC/SC |
|---|---|---|---|---|---|---|
| base (softmax) | 98,2 % | 93,8 % | 84,7 % | 73,8 % | 63,1 % | 275 k |
| + Gram | 98,4 % | 95,1 % | 86,7 % | 76,3 % | 65,7 % | 275 k |
| **+ signé** | 100,4 % | 101,1 % | 98,1 % | **93,1 %** | **86,3 %** | 275 k |
| + signé + Gram | 100,7 % | 101,2 % | 98,2 % | 93,2 % | 86,6 % | 275 k |

**Conclusions :**
- **L'attention signée est l'ingrédient clé : +19 points à 20 dB et +23 à 25 dB, pour un coût nul** (mêmes paramètres, un tanh en plus). Le blocage à fort SNR venait bien de l'impossibilité de **soustraire** avec un softmax.
- **Avec le signe, D64 L2 (275 k MAC) fait aussi bien que D64 L4 sans signe (541 k MAC)** : deux fois moins de calcul pour le même débit.
- **Négatif : le biais Gram n'apporte rien de significatif** (+2,5 points, dans la variabilité d'environ 3 points), ni seul ni avec le signe. Le modèle sait reconstruire lui-même qui interfère avec qui. Ce qui lui manquait, c'était l'opération, pas l'information. *Limite* : Gram n'a été testé que comme biais du score ; il ne modifie que les poids, pas les valeurs transmises. Option retirée du modèle par défaut, mais gardée dans le code pour l'instant.

### Balayage de taille avec attention signée — RETENU (`results/phase1/size_sweep_signed.*`, parties `_part_a/b/c` : srvgrm12 ×2, srvgrm08)
Mêmes réglages que le balayage sans signe. % de RZF :

| signé | 0 dB | 10 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|---|
| D16 L1 / L2 / L4 | 90 / 92 / 86 | 70 / 76 / 82 | 45 / 50 / 63 | 36 / 41 / 53 | 2,8 k / 5,0 k / 9,5 k | 11 k / 20 k / 37 k |
| D32 L1 / L2 / L4 | 97 / 95 / 101 | 83 / 100 / 102 | 58 / 87 / **100** | 48 / 78 / **98** | 9,6 k / 18 k / 35 k | 38 k / 72 k / 139 k |
| D64 L1 / L2 / L4 | 99 / 100 / 103 | 85 / 101 / 103 | 60 / 93 / **102** | 50 / 87 / **101** | 36 k / 69 k / 136 k | 142 k / 275 k / 541 k |

- **Avec le signe, D32 L4 atteint RZF à 20 dB (100,4 %)** avec 35 k paramètres, 4 fois moins que sans signe (D64 L4 : 136 k pour 97 %). **D64 L4 signé atteint le niveau de WMMSE** (101,9 % de RZF ; WMMSE = 101,7 %).
- **D=16 ne suffit pas** (63 % au mieux) : une largeur minimale d'environ 32 est nécessaire.
- La profondeur reste décisive, même avec le signe : avec L=1, on plafonne vers 60 %.

### Trois idées pour le fort SNR — RETENU (`results/phase1/highsnr.*`, parties `_part_a/b/c`)
- **Question** : par rapport au modèle signé D32 L2 (88 % à 20 dB), que changent (a) un seul bloc répété T fois, (b) la normalisation de phase par utilisateur, (c) l'attention linéaire (sans softmax ni tanh) ?

| D32 | 0 dB | 10 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|---|
| signé L2 (référence) | 95 | 100 | 88,0 | 79 | 18 k | 72 k |
| **bloc répété T=2** | 95 | 100 | 87,1 | 78 | **9,6 k** | 72 k |
| **bloc répété T=4** | 93 | 101 | **99,4** | **96** | **9,6 k** | 139 k |
| bloc répété T=8 | 99 | 100 | 99,2 | 97 | 9,6 k | 275 k |
| normalisation de phase | 100 | 100 | 87,7 | 79 | 18 k | 72 k |
| **attention linéaire** | 95 | 100 | 88,9 | 80 | 18 k | 72 k |

**Conclusions :**
- **Répéter un seul bloc ne coûte rien en débit.** T=2 égale 2 blocs distincts, T=4 (99,4 %) égale 4 blocs distincts (100,4 %), **avec 3,7 fois moins de paramètres** (9,6 k au lieu de 35 k). **C'est le nombre d'itérations qui compte, pas le nombre de paramètres.** Pour le FPGA : un seul petit bloc, réutilisé 4 fois.
- **Au-delà de 4 itérations, on ne gagne plus rien** (T=8 : 99,2 %) : le modèle est saturé à cette largeur.
- **L'attention linéaire fait aussi bien que l'attention signée** (88,9 contre 88,0 %, dans le bruit), **sans exponentielle, sans division ni tanh**. Pour le matériel, c'est un gain net à précision égale.
- **Négatif : la normalisation de phase n'apporte rien** (87,7 contre 88,0 %). Le réseau apprend déjà seul cette invariance. Option gardée dans le code, mais pas utilisée.
- **Point à surveiller : à 0 dB, plusieurs modèles sont à 93-95 % de RZF** alors qu'ils dépassent 100 % à 5-10 dB. Probable cause : la perte moyenne le débit sur des SNR tirés de 0 à 25 dB, et les débits à fort SNR, bien plus grands, dominent la perte. Le faible SNR est donc négligé. Ça se corrigerait en normalisant la perte par SNR (par exemple, débit / débit RZF).

### Modèle candidat (D32, 1 bloc × 4, attention linéaire) et perte relative — RETENU (`results/phase1/candidate.*`, parties a–c srvgrm12, d srvgrm08)
- **Questions** : (1) la combinaison « bloc répété + attention linéaire » tient-elle ? (2) une perte relative (débit / débit RZF sur le même échantillon) corrige-t-elle le faible SNR ? (3) quelle variabilité d'une graine à l'autre ? La graine fixe désormais aussi l'initialisation des poids.

% de RZF (WMMSE = 105 / 104 / 102 / 101 % aux mêmes SNR) :

| D32, 1 bloc × 4, linéaire | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---|---|---|---|---|---|
| perte « débit », graine 0 | **92,7** | 99,5 | 101,2 | 99,7 | 97,2 | 92,1 |
| perte relative, graine 0 | **100,4** | 101,7 | 101,6 | 100,0 | 97,2 | 92,3 |
| perte relative, graine 1 | 100,2 | 101,8 | 101,6 | 100,3 | 97,7 | 93,1 |
| perte relative, graine 2 | 99,6 | 101,7 | 101,6 | 100,5 | 98,4 | 94,4 |

**Conclusions :**
- **La perte relative corrige le faible SNR** : à 0 dB, on passe de 92,7 à 100,4 % de RZF, sans rien perdre ailleurs. On la garde par défaut.
- **La variabilité entre graines est faible** maintenant que l'initialisation est fixée : 97,2 à 98,4 % à 20 dB (écart de 1,2 point) et 92 à 94 % à 25 dB.
- **La combinaison coûte un peu au fort SNR.** Bloc répété + attention **signée** donnait 99,4 % (20 dB) et 96 % (25 dB). Bloc répété + attention **linéaire** donne environ 97,8 % et 93 %, soit environ 2 points de moins à 20 dB et 3 à 25 dB. C'est le prix de l'attention sans exponentielle, sans division ni tanh. **C'est un compromis pour le matériel, à trancher.**
- **Bilan du candidat** : 9 648 paramètres, environ 139 k MAC par SC. Il égale RZF jusqu'à 15 dB, reste à 97-98 % à 20 dB et 92-94 % à 25 dB.

---

## Phase 2 — 2026-10-23

### Le candidat entraîné sur le CSI estimé (par SC) — RETENU, résultat NÉGATIF (`results/phase2/estimated_csi_p10.*`, `_p20.*`)
- **Question** : sans contexte fréquentiel, entraîner le modèle sur le CSI estimé (LS + interpolation linéaire, pilotes Sionna) lui permet-il de faire mieux que RZF(LS), voire que RZF(LMMSE) ?
- **Réglages** : D32, 1 bloc × 4, perte relative à RZF(LS), graine 0, 40 000 pas, 1 024 drops (graine 1). Évaluation sur les tirages de la phase 0, avec le même bruit pilote pour toutes les méthodes. Référence supplémentaire : les modèles de la phase 1 (entraînés en CSI parfait) appliqués tels quels au LS. Parties a/b : srvgrm12 et srvgrm08.

Débit somme à 20 dB (25 dB) :

| | pilote 10 dB | pilote 20 dB |
|---|---|---|
| RZF (CSI parfait) | 26,18 (32,72) | 26,18 (32,72) |
| **RZF (LMMSE)** | **23,51 (26,94)** | **25,58 (31,12)** |
| RZF (LS) | 21,69 (24,07) | 25,22 (30,35) |
| WMMSE (LS) | 22,09 (24,27) | 25,67 (30,59) |
| phase 1, linéaire (CSI parfait) sur LS | 21,62 (23,68) | 24,70 (28,68) |
| phase 1, signé (CSI parfait) sur LS | 21,89 | 25,19 |
| **phase 2, linéaire (entraîné sur LS)** | 21,64 (23,63) | 24,60 (27,90) |
| **phase 2, signé (entraîné sur LS)** | 21,45 (23,28) | 24,86 (29,05) |

**Conclusions :**
- **Entraîner sur le CSI estimé n'apporte rien quand on traite chaque SC seule.** Le modèle de la phase 2 fait comme RZF(LS) et comme le modèle de la phase 1, et reste en dessous de RZF(LMMSE) : 92 % de LMMSE à 20 dB avec le pilote à 10 dB.
- **C'est logique.** Sur une seule SC, le modèle n'a qu'une observation bruitée du canal, et le bruit est indépendant de la SC voisine. Il ne peut donc rien débruiter. Il peut seulement ajuster la régularisation, et RZF le fait déjà presque optimalement.
- **Pour le paper, c'est l'argument direct en faveur du contexte fréquentiel** : la marge (environ 1,8 bps/Hz jusqu'à LMMSE, 4,5 jusqu'au CSI parfait, pilote 10 dB, à 20 dB) n'est atteignable **qu'en regardant les SC voisines**. C'est l'objet de la phase 3.
- *Au passage* : le petit écart du candidat au fort SNR (97 % de RZF en CSI parfait) se retrouve ici (pilote 20 dB, 25 dB de SNR de données : 27,9 à 29,1 contre 30,4 pour RZF(LS)).

---

## Phase 3 — 2026-10-23

### Fenêtre de SC voisines concaténée dans le token, pilote 10 dB — RETENU (`results/phase3/window_p10.*`, parties a–c srvgrm12, d srvgrm08)
- **Question** : en donnant au modèle le CSI estimé (LS) des W SC centrées sur la SC à précoder, récupère-t-on la marge mesurée en phase 0 ? Bat-on la contrepartie classique (RZF sur le LS moyenné sur W), puis RZF(LMMSE) ?
- **Réglages** : même modèle que la phase 2 (D32, 1 bloc × 4, attention linéaire, perte relative à RZF(LS)) ; seul changement, l'entrée de chaque token = 2·M·W valeurs (fenêtre bornée aux bords de la bande). Mêmes tirages et même bruit pilote pour tout.

Débit somme à 20 dB (25 dB), bps/Hz :

| W | transformer, fenêtre W | RZF + moyenne sur W | params | MAC/SC |
|---|---|---|---|---|
| 1 | 21,71 (23,75) | 21,69 (24,07) | 9,6 k | 139 k |
| 3 | 22,11 (24,11) | 21,93 (24,41) | 10,7 k | 143 k |
| 5 | 22,77 (25,37) | 22,27 (24,93) | 11,7 k | 148 k |
| 9 | 23,05 (25,83) | 22,67 (25,65) | 13,7 k | 156 k |
| **13** | **23,15 (25,98)** | 22,56 (25,65) | 15,8 k | 164 k |
| *RZF(LMMSE)* | *23,51 (26,94)* | | | |
| *RZF(CSI parfait)* | *26,18 (32,72)* | | | |

**Conclusions :**
- **Le contexte fréquentiel marche** : de W=1 à W=13, on gagne +1,44 bps/Hz à 20 dB (+6,6 %), pour seulement +6 k paramètres et +18 % de calcul.
- **Le transformer fait mieux que la moyenne classique à fenêtre égale** (+0,5 à 0,6 bps/Hz à W=9 et 13), et il continue de progresser à W=13, là où la moyenne commence à redescendre (elle mélange des SC trop différentes).
- **Mais il ne bat pas encore RZF(LMMSE)** : 98,5 % à 20 dB et 96,4 % à 25 dB. Le gain plafonne entre W=9 et 13 (+0,1).
- **Deux causes possibles, à séparer avant d'aller plus loin** :
  1. *L'estimation* : LMMSE utilise la covariance sur toute la bande (612 SC) et la structure exacte du peigne de pilotes, alors que le modèle ne voit que 13 SC déjà interpolées.
  2. *Le précodage* : en CSI parfait, ce modèle ne fait déjà que 97-98 % de RZF à 20 dB et 92-94 % à 25 dB. Une partie de l'écart à LMMSE est donc probablement la limite du précodeur lui-même, pas de l'estimation.
- *Technique* : la fusion des parties ne gardait que les références de la première partie (le lissage W=3, 5 et 9 manquait) ; corrigé dans `utils.merge_parts`.

### Diagnostic : estimation ou précodeur ? — RETENU (`results/phase3/diagnostic_p10.*`, parties a–c srvgrm12, d srvgrm08)
- **Question** : l'écart à RZF(LMMSE) de la fenêtre W=13 vient-il de l'estimation, du précodeur, ou de la compression de l'entrée (208 valeurs → D=32) ?
- **Réglages** : pilote 10 dB, 1 bloc × 4, perte relative, mêmes tirages. RZF(LMMSE) = 23,51 (20 dB) / 26,94 (25 dB).

| Modèle | Entrée | 0 dB | 10 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|---|---|
| D32 linéaire, W=1 | **LMMSE** | 100,8 % | 102,0 % | 99,2 % | 97,1 % | 9,6 k | 139 k |
| D32 signé, W=1 | **LMMSE** | 100,5 % | 101,8 % | 98,4 % | 95,9 % | 9,6 k | 139 k |
| D32 linéaire, W=13 (phase 3) | LS | 100,6 % | 101,6 % | 98,5 % | 96,4 % | 15,8 k | 164 k |
| D32 **signé**, W=13 | LS | 100,6 % | 101,1 % | 97,8 % | 95,5 % | 15,8 k | 164 k |
| **D64** linéaire, W=13 | LS | **102,7 %** | **102,4 %** | **100,6 %** (23,65) | **99,4 %** (26,78) | 48 k | 590 k |

(% de RZF(LMMSE), même SNR.)

**Conclusions :**
- **Le précodeur D32 coûte environ 1 point à 20 dB et 3 à 25 dB** : même avec l'estimation LMMSE en entrée, il n'atteint pas tout à fait RZF(LMMSE). C'est cohérent avec sa limite en CSI parfait.
- **L'attention signée ne change rien ici** (W=13 : 97,8 % contre 98,5 % en linéaire ; LMMSE en entrée : 98,4 % contre 99,2 %). Le linéaire reste le bon choix.
- **Le vrai goulot était la largeur.** Avec D=64, la fenêtre W=13 **dépasse RZF(LMMSE)** de 0 à 20 dB (100,6 à 102,7 %), sans connaître la covariance du canal, et égale presque LMMSE à 25 dB (99,4 %). Comprimer 208 valeurs d'entrée en 32 dimensions perdait de l'information.
- **Le prix est élevé** : 48 k paramètres et 590 k MAC par SC, soit 3,6 fois le modèle D32 et environ 420 fois RZF. Il faudra trouver un moyen moins coûteux de traiter la fenêtre (par exemple une première projection plus large seulement à l'entrée, ou une attention en fréquence).
- **Réserve importante** : RZF(LMMSE) n'est pas la meilleure référence classique. Au faible SNR, WMMSE fait environ 5 % de mieux que RZF (0 dB), et environ 1,7 % à 20 dB. Les 102-103 % du modèle D64 à 0-10 dB sont donc probablement **en dessous de WMMSE(LMMSE)**, qui n'a pas encore été calculé. Il faut ajouter cette référence avant d'affirmer quoi que ce soit.

### Référence WMMSE(LMMSE) — RETENU (`results/phase3/wmmse_ref_p10.json`)
Pilote 10 dB, mêmes tirages. Débit somme (0 / 5 / 10 / 15 / 20 / 25 dB) :
- RZF (LMMSE) : 5,07 / 8,85 / 13,61 / 18,79 / 23,51 / 26,94
- **WMMSE (LMMSE) : 5,35 / 9,28 / 14,13 / 19,33 / 23,93 / 27,15**
- RZF (CSI parfait) : 5,12 / 8,98 / 13,98 / 19,85 / 26,18 / 32,72
- WMMSE (CSI parfait) : 5,39 / 9,40 / 14,49 / 20,38 / 26,63 / 32,97

**Conclusion : le meilleur modèle (D64, W=13 : 5,21 / 9,10 / 13,94 / 19,10 / 23,65 / 26,78) reste à 97,4-98,8 % de WMMSE(LMMSE) à tous les SNR.** Il bat RZF(LMMSE), mais pas la meilleure méthode classique. Ma réserve du diagnostic était fondée.

### Scénarios mélangés UMi + UMa + RMa — RETENU (`results/phase3/mixed_p10.*`)
- **Question** : une covariance fixe pour LMMSE perd-elle quand les scénarios sont mélangés ? Le modèle (D64, W=13, linéaire) fait-il mieux ?
- **Contraste entre scénarios** : corrélation sur 1 RB de 0,85 (UMi), 0,81 (UMa) et **0,98 (RMa, canal bien plus plat)**.
- **Réglages** : pilote 10 dB. Entraînement sur 352 tirages par scénario (graines 1, 2, 3), mélangés, sans que le modèle connaisse le scénario. Évaluation sur 192 tirages par scénario (graines 2026, 2036, 2046), avec les mêmes tirages et le même bruit pilote pour toutes les méthodes. LMMSE « mélange » = covariance moyenne des trois scénarios ; LMMSE « oracle » = covariance du bon scénario.

Débit somme à 20 dB (bps/Hz) :

| | UMi | UMa | **RMa** |
|---|---|---|---|
| RZF (CSI parfait) | 26,31 | 25,98 | 25,95 |
| WMMSE (LMMSE oracle) | 24,02 | 23,45 | **25,24** |
| **WMMSE (LMMSE mélange)** | **24,07** | **23,62** | **24,20** |
| RZF (LMMSE oracle) | 23,60 | 23,00 | 24,64 |
| RZF (LMMSE mélange) | 23,66 | 23,16 | 23,62 |
| RZF (LS) | 21,81 | 21,37 | 21,68 |
| **modèle entraîné sur le mélange** | 23,70 | 23,16 | **24,09** |
| modèle entraîné sur UMi seul | 23,71 | 23,20 | 24,01 |

**Conclusions :**
- **Une covariance mal adaptée coûte cher là où le canal est différent** : sur RMa, LMMSE avec la covariance du mélange perd **1,0 bps/Hz** par rapport à l'oracle (WMMSE : 25,24 → 24,20). Sur UMi et UMa, la covariance du mélange suffit, parce qu'ils se ressemblent.
- **Mais notre modèle ne récupère pas cette perte** : sur RMa, il fait 24,09, donc au-dessus de RZF(LMMSE mélange) mais **sous WMMSE(LMMSE mélange)** (99,5 %), et loin de l'oracle (25,24). Il reste en dessous de WMMSE(LMMSE mélange) à tous les SNR, dans les trois scénarios.
- **Le modèle ne s'adapte pas** : entraîné sur le mélange ou sur UMi seul, il fait pareil sur RMa (24,09 contre 24,01). C'est cohérent avec ce qu'il est, un **filtre fréquentiel fixe** suivi d'un précodeur. (Ça règle aussi la question de la quantité de données : le mélange n'a rien coûté sur UMi et UMa.)
- **Deux raisons possibles, à séparer** : (1) le filtre est fixe, alors que RMa demanderait de moyenner plus largement ; (2) **W=13 est trop étroit pour RMa**, où le canal reste corrélé sur des dizaines de SC, alors que LMMSE utilise toute la bande.
- **Pour le paper**, c'est la situation qui justifie un traitement fréquentiel **adaptatif** : il y a 1 bps/Hz à gagner sur RMa par rapport à la meilleure méthode classique réaliste, et la concaténation ne le prend pas.

### Spécialiste RMa et fenêtre large W=49 — RETENU (`results/phase3/mixed_p10_specialiste_rma.*`, `mixed_p10_W49.*`)
- **Question** : sur RMa, est-ce la fenêtre trop étroite (W=13) ou le filtre fixe (pas d'adaptation) qui empêche notre modèle de récupérer le 1 bps/Hz perdu par LMMSE « mélange » ?
- **Réglages** : identiques au test mélangé (D64, 1 bloc × 4, linéaire, pilote 10 dB, 1 056 tirages d'entraînement au total, mêmes tirages d'évaluation). Spécialiste : W=13, entraîné sur RMa seul. Généraliste large : W=49 (4 RB), entraîné sur le mélange.

Débit somme à 20 dB (25 dB), bps/Hz :

| | UMi | UMa | RMa |
|---|---|---|---|
| WMMSE (LMMSE oracle) | 24,02 (27,23) | 23,45 (26,47) | 25,24 (29,61) |
| **WMMSE (LMMSE mélange)**, référence réaliste | **24,07 (27,33)** | **23,62 (26,79)** | **24,20 (27,54)** |
| généraliste W=13 (test précédent) | 23,70 (26,85) | 23,16 (26,20) | 24,09 (27,45) |
| **généraliste W=49** | **23,99 (27,34)** | **23,45 (26,67)** | **24,38 (27,97)** |
| spécialiste RMa W=13 | 22,83 (25,80) | 22,02 (24,79) | **24,47 (28,17)** |

**Conclusions :**
- **Les deux causes comptent, à parts à peu près égales.** Élargir la fenêtre (W=13 → 49) fait gagner +0,3 bps/Hz **dans les trois scénarios**. Se spécialiser sur RMa fait gagner +0,4 sur RMa. Aucun des deux n'atteint l'oracle (25,24) : il reste 0,8 à 0,9 bps/Hz sur RMa.
- **Première fois que le modèle dépasse WMMSE(LMMSE mélange)** : le généraliste W=49 sur RMa à 20 dB (24,38 contre 24,20, +0,7 %) et à 25 dB (27,97 contre 27,54, +1,6 %). Il l'égale sur UMi à fort SNR, et reste légèrement en dessous sur UMa (99,3 % à 20 dB). **Attention** : +0,18 bps/Hz est **du même ordre que la variabilité entre graines** (environ 0,3 bps/Hz). Il faut 3 graines avant de l'affirmer ; à 25 dB, l'écart (+0,43) est plus solide.
- **Au faible SNR, le modèle reste sous WMMSE partout** (RMa à 0 dB : 5,26 contre 5,45). C'est la limite du précodeur, déjà vue en CSI parfait (environ RZF + 2 %, contre RZF + 5 % pour WMMSE), et non de l'estimation.
- **Le spécialiste n'est pas une solution** : il s'effondre hors de RMa (UMi : 22,83). Il sert de borne : avec un filtre fixe adapté à RMa, on atteindrait 24,47.
- **Le coût monte** : W=49 = 85 k paramètres et 738 k MAC/SC (environ 530 fois RZF).

### Fenêtre W=49 sur le mélange, 3 graines — RETENU (`results/phase3/mixed_p10_W49.*`, `_W49_s1.*`, `_W49_s2.*`)
- **Question** : le dépassement de WMMSE(LMMSE mélange) sur RMa tient-il d'une graine à l'autre ?
- **Réglages** : identiques à W49 graine 0 ; graines 1 et 2 (initialisation des poids et tirage des lots). Données d'entraînement et d'évaluation identiques.

Moyenne sur 3 graines (écart-type) et écart à WMMSE(LMMSE mélange), bps/Hz :

| | 0 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---|---|---|---|---|
| UMi | 5,22 (0,01) / −0,15 | 14,05 (0,01) / −0,16 | 19,30 / −0,14 | 23,99 (0,01) / −0,08 | 27,33 (0,01) / **0,00** |
| UMa | 5,15 (0,02) / −0,17 | 13,78 / −0,20 | 18,89 / −0,20 | 23,45 / −0,17 | 26,67 / −0,12 |
| **RMa** | 5,27 (0,02) / −0,17 | 14,15 / −0,19 | 19,49 / −0,05 | **24,39 (0,01) / +0,20** | **27,98 (0,01) / +0,44** |

**Conclusions :**
- **La variabilité entre graines est très faible** à cette taille (écart-type d'environ 0,01 bps/Hz). Mon estimation précédente d'environ 0,3 bps/Hz venait des petits modèles D32 ; elle était trop pessimiste ici.
- **Le dépassement sur RMa est confirmé** : +0,20 bps/Hz à 20 dB et +0,44 à 25 dB, pour les 3 graines (au minimum +0,18 et +0,43), soit environ 20 fois l'écart-type.
- **Ailleurs, le modèle reste légèrement sous WMMSE(LMMSE mélange)** : −0,15 à −0,20 bps/Hz au faible et moyen SNR dans les trois scénarios, et sur UMa à tous les SNR. Il égale WMMSE sur UMi à 25 dB.
- **Bilan honnête** : le modèle bat la meilleure méthode classique réaliste **seulement là où sa covariance est mal adaptée (RMa) et à fort SNR**. Ailleurs, il fait environ 99 % de WMMSE(LMMSE), sans covariance et sans inversion de matrice.
- *Limite* : les écarts sont mesurés sur un seul ensemble d'évaluation (192 tirages par scénario, apparié). La variabilité entre graines est maîtrisée ; la variabilité liée au choix des tirages ne l'est pas encore (un bootstrap sur les tirages le ferait).

### Attention en fréquence (fenêtre 49, mélange) — RETENU (`results/phase3/mixed_p10_fattn_D64.*`, `_fattn_D32.*`)
- **Question** : remplacer la concaténation (filtre fixe) par une attention en fréquence (moyenne pondérée adaptative) récupère-t-il le 0,85 bps/Hz qui reste sur RMa jusqu'à l'oracle ? Le modèle élargit-il sa fenêtre sur RMa ?
- **Modèle** (`freq="attention"`) : pour chaque utilisateur, la SC centrale (requête) interroge ses 49 voisines (clés et valeurs, plus une position relative apprise), softmax sur la fenêtre, 2 têtes. Le résultat s'ajoute à l'embedding de la SC centrale, puis le reste est inchangé (1 bloc linéaire × 4). Softmax volontaire : en fréquence, on lisse. Coût compté avec les projections de chaque SC calculées une seule fois (fenêtre glissante).
- **Réglages** : identiques à W49 concat (mélange, pilote 10 dB, graine 0).

Débit somme à 20 dB (25 dB), bps/Hz :

| | UMi | UMa | RMa | params | MAC/SC |
|---|---|---|---|---|---|
| WMMSE (LMMSE oracle) | 24,02 (27,23) | 23,45 (26,47) | 25,24 (29,61) | | |
| WMMSE (LMMSE mélange) | 24,07 (27,33) | 23,62 (26,79) | 24,20 (27,54) | | |
| concat W49 D64 (3 graines) | 23,99 (27,33) | 23,45 (26,67) | 24,39 (27,98) | 85 k | 738 k |
| **attention en fréquence W49 D64** | **24,08 (27,51)** | **23,52 (26,79)** | **24,84 (28,86)** | **46 k** | **595 k** |
| attention en fréquence W49 D32 | 22,09 (24,26) | 21,47 (23,49) | 22,72 (25,21) | 14 k | 162 k |

Largeur effective de l'attention (1/Σa², nombre de voisines réellement utilisées, SNR 20 dB) : UMi 8,7 SC, UMa 8,2 SC, **RMa 10,9 SC** (D64).

**Conclusions :**
- **L'attention en fréquence fait mieux que la concaténation partout à moyen et fort SNR, et beaucoup mieux sur RMa** : +0,44 bps/Hz à 20 dB et +0,88 à 25 dB. Elle récupère **environ la moitié** de l'écart à l'oracle (il restait 0,85 bps/Hz à 20 dB, il en reste 0,40).
- **Face à WMMSE(LMMSE mélange), la meilleure référence réaliste** : **+0,64 bps/Hz sur RMa à 20 dB (+2,6 %) et +1,32 à 25 dB (+4,8 %)** ; égalité sur UMi à 20 dB et +0,17 à 25 dB ; −0,10 sur UMa à 20 dB, égalité à 25 dB. Au faible SNR (0 à 10 dB), toujours −0,15 à −0,2 partout : c'est la limite du précodeur, pas de l'estimation.
- **Le modèle adapte bien sa fenêtre au canal** : il utilise plus de voisines sur RMa (10,9) que sur UMi/UMa (8,2 à 8,7), sans connaître le scénario. L'écart est modéré mais systématique ; on voit aussi sur la figure que RMa met plus de poids loin de la SC centrale.
- **Il est en plus moins cher que la concaténation** : 46 k paramètres au lieu de 85 k, 595 k MAC au lieu de 738 k.
- **Négatif : à D=32, l'attention en fréquence s'effondre** (−1,9 bps/Hz à 20 dB par rapport à la concaténation D64). La largeur D=64 reste nécessaire.
- *Observation à creuser* : le profil d'attention a des pics à ±1, ±5, ±9 SC… (période 4 = le peigne des pilotes des 4 utilisateurs) et un creux en 0 (la SC centrale est déjà dans l'embedding). Le modèle semble avoir appris la structure des pilotes.
- *Limite* : une seule graine (la dispersion mesurée pour la concaténation était d'environ 0,01 bps/Hz, mais elle n'est pas encore mesurée pour l'attention).

---

## Phase 4 — améliorer le modèle candidat (2026-10-24)

Candidat de départ, figé : attention en fréquence (softmax, fenêtre 49), D=64, un bloc d'attention linéaire entre utilisateurs × 4, perte relative à RZF, entraîné sur le mélange UMi + UMa + RMa, pilote 10 dB (`results/phase3/mixed_p10_fattn_D64`). Chaque essai = un seul changement. Même évaluation (192 tirages par scénario).

### Estimateur classique dans le domaine des retards — RETENU comme baseline, NÉGATIF comme piste (`results/phase4/delay_baseline_p10.json`)
- **Question** : un estimateur « DFT-based » (LS sur le peigne → IDFT → troncature à L retards ou seuil sur le bruit → DFT) montre-t-il une grande marge dans l'estimation, au-delà de LMMSE ?
- **Plancher de fuite** : même **sans bruit**, la reconstruction plafonne à −20 à −21 dB d'erreur. Les retards réels ne tombent pas sur la grille de 54 ns et la bande est coupée net, donc l'énergie fuit sur tous les retards. C'est du même ordre que le bruit à pilote 10 dB : erreur d'estimation à peine meilleure que le LS (−17 à −18 dB, contre −19,5 pour LMMSE).
- **En débit, à 20 dB** (meilleure variante par scénario, ce qui est déjà favorable au classique) :

| | UMi | UMa | RMa |
|---|---|---|---|
| RZF (LS) | 21,81 | 21,37 | 21,68 |
| RZF, domaine des retards (meilleure variante) | 23,15 (seuil k=8) | 22,74 (seuil k=8) | 23,94 (L=16) |
| WMMSE, domaine des retards | 23,56 | 23,19 | 24,54 |
| WMMSE, LMMSE réaliste | 24,07 | 23,62 | 24,20 |
| **notre modèle (candidat)** | **24,08** | **23,52** | **24,84** |

**Conclusions :**
- **Notre modèle bat l'estimateur dans le domaine des retards partout.** Celui-ci ne bat LMMSE réaliste que sur RMa, et encore avec une troncature choisie pour RMa (L=16 ; la meilleure est L=48 en ville), donc avec une connaissance du scénario.
- **La version classique simple ne montre pas de grande marge cachée**, à cause de la fuite. Ce n'est pas la preuve qu'il n'y en a pas : un estimateur parfait par lien (covariance exacte de chaque lien) serait la vraie borne, mais il n'a pas été calculé. **Décision : ne pas pousser plus loin le domaine des retards pour l'instant** (il faudrait gérer la fuite, ce qui est de l'ingénierie).
- Utile pour le paper comme baseline supplémentaire, que notre modèle bat.

### Borne « génie » de l'estimation : LMMSE par lien — RETENU (`results/phase4/genie_bound_p10.json`)
- **Question** : reste-t-il une vraie marge dans l'estimation, au-delà de notre modèle ? On donne à un LMMSE la covariance fréquentielle **exacte de chaque lien**, construite à partir des retards et des puissances de ses trajets (réponse impulsionnelle Sionna). C'est irréalisable tel quel, mais c'est le plafond d'une estimation linéaire adaptée à chaque lien.
- **Réglages** : mêmes tirages et même bruit pilote que le mélange (vérifié : RZF(LS) identique au centième), pilote 10 dB, sans entraînement.

| 20 dB | UMi | UMa | RMa |
|---|---|---|---|
| Erreur d'estimation : LS / **génie** | −17,1 / **−27,6 dB** | −16,8 / **−27,5 dB** | −17,6 / **−30,4 dB** |
| WMMSE (CSI parfait) | 26,74 | 26,46 | 26,56 |
| **WMMSE (génie par lien)** | **26,09** | **25,77** | **26,19** |
| RZF (génie par lien) | 25,66 | 25,30 | 25,59 |
| WMMSE (LMMSE oracle par scénario) | 24,02 | 23,45 | 25,24 |
| notre modèle (candidat) | 24,08 | 23,52 | 24,84 |

**Conclusions :**
- **La marge dans l'estimation est grande : environ 2 bps/Hz (8 %) entre notre modèle et la borne génie, dans les trois scénarios.** La borne génie est à 0,4-0,7 bps/Hz du CSI parfait.
- **Pourquoi le génie est si bon** : le canal d'un lien n'est qu'une somme d'une vingtaine de trajets. Connaissant leurs retards, il ne reste à estimer que leurs gains, environ 20 inconnues pour 153 mesures par antenne, ce qui retire énormément de bruit. C'est la **parcimonie dans le domaine des retards**, exploitée parfaitement.
- **Ce n'est pas qu'une borne théorique** : les retards et puissances des trajets varient lentement (grande échelle), contrairement aux gains rapides. Une BS peut donc les apprendre au fil des slots. Et un modèle qui voit **toute la bande** (et pas une fenêtre de 49 SC) pourrait en exploiter une partie directement.
- **Notre fenêtre de 49 SC n'utilise qu'environ 8 % de la bande** : c'est probablement ce qui limite. **C'est la piste qui peut faire battre WMMSE franchement.**

### Tête de puissance — ÉCARTÉ (gain négligeable) (`results/phase4/mixed_p10_fattn_D64_powerhead.*`)
- **Question** : séparer direction et puissance par utilisateur (softmax sur les puissances, sans inversion) comble-t-il le trou au faible SNR face à WMMSE ?
- **Résultat** : +0,02 à +0,05 bps/Hz partout par rapport au candidat (0 dB : 5,24 / 5,18 / 5,30 contre 5,20 / 5,13 / 5,25). L'écart au faible SNR face à WMMSE(LMMSE réaliste) reste de −0,12 à −0,19 bps/Hz.
- **Décision** : écarté (une option de plus pour presque rien). Le trou au faible SNR (environ 0,15 bps/Hz) pèse peu face à la marge d'estimation (environ 2 bps/Hz, borne génie). L'option `power_head` reste dans le code, désactivée par défaut.

### Fenêtre élargie à 153 SC (attention en fréquence) — NÉGATIF (`results/phase4/mixed_p10_fattn_D64_W153.*`)
- **Question** : élargir simplement la fenêtre (49 → 153 SC, un quart de la bande) rapproche-t-il de la borne génie ?
- **Résultat** (20 dB) : UMi 24,11 / UMa 23,54 / RMa 24,89, soit +0,03 à +0,06 bps/Hz par rapport à 49 SC (+0,12 sur RMa à 25 dB). Largeur effective presque inchangée (9,4 / 8,6 / 12,3 SC). **L'écart à la borne génie reste de 2,0 / 2,2 / 1,3 bps/Hz à 20 dB** (2,8 à 4,1 à 25 dB).
- **Interprétation** (cohérente avec l'hypothèse faite avant le test) : l'attention en fréquence calcule une **moyenne à poids positifs**, c'est-à-dire un lissage passe-bas. Le filtre génie d'un canal à ~20 trajets combine les voisines avec des **coefficients complexes qui tournent en phase** selon les retards ; une moyenne positive ne peut pas le représenter, quelle que soit la fenêtre. Le modèle n'utilise donc qu'une dizaine de SC, même avec 153 disponibles.
- **Décision** : élargir ne suffit pas. Prochain essai : estimer dans le **domaine des retards** (IFFT de la bande → transformer sur les retards → un gain entre 0 et 1 par retard → FFT), où chaque trajet devient un pic et où le bon traitement redevient une sélection.

### Estimation dans le domaine des retards + précodeur par SC — RETENU, résultat principal (`results/phase4/delay_model_D64.*`, `_D32.*`)
- **Question** : remplacer l'attention en fréquence (fenêtre 49) par un module sur **toute la bande** dans le domaine des retards (variante a : IFFT des 153 mesures LS du peigne de chaque utilisateur → 2 couches d'attention softmax entre les 153 retards (D=32) → un gain entre 0 et 1 par retard, commun aux 8 antennes → FFT vers 612 SC), puis le précodeur par SC **sans fenêtre** (1 bloc linéaire × 4). Un seul modèle, entraîné sur le débit (perte relative à RZF), lots de 4 tirages complets.
- **Réglages** : mélange UMi + UMa + RMa, pilote 10 dB, mêmes tirages que les références (vérifié : RZF(LS) identique), graine 0, 40 000 pas (environ 10 min sur L40S, contre 30 à 40 pour le candidat).

Débit à 20 dB (25 dB), bps/Hz :

| | UMi | UMa | RMa |
|---|---|---|---|
| WMMSE (LMMSE réaliste) | 24,07 (27,33) | 23,62 (26,79) | 24,20 (27,54) |
| WMMSE (LMMSE oracle) | 24,02 (27,23) | 23,45 (26,47) | 25,24 (29,61) |
| candidat (attention en fréquence, W=49) | 24,08 (27,51) | 23,52 (26,79) | 24,84 (28,86) |
| **retards + précodeur D64** | **24,80 (28,92)** | **24,40 (28,40)** | **25,05 (29,58)** |
| retards + précodeur D32 | 23,97 (27,38) | 23,56 (26,86) | 23,98 (27,58) |
| WMMSE (génie par lien) | 26,09 | 25,77 | 26,19 |

Écart du modèle D64 à WMMSE(LMMSE réaliste) : **+0,73 / +0,79 / +0,85 bps/Hz à 20 dB (+3,0 / +3,3 / +3,5 %)** et **+1,6 / +1,6 / +2,0 à 25 dB (+5,8 / +6,0 / +7,4 %)** ; +0,11 à +0,21 à 15 dB ; −0,07 à −0,24 de 0 à 10 dB.

**Conclusions :**
- **Pour la première fois, le modèle bat la meilleure méthode classique réaliste dans les TROIS scénarios** à moyen et fort SNR, et pas seulement sur RMa. Il bat même WMMSE avec la covariance oracle sur UMi et UMa (et l'égale presque sur RMa, 25,05 contre 25,24).
- **Il comble environ 40 % de l'écart à la borne génie** (il reste 1,1 à 1,4 bps/Hz à 20 dB, contre 1,3 à 2,2 pour le candidat).
- **L'adaptation est visible et interprétable** (figure de droite) : le modèle garde en moyenne **14 retards sur RMa** (canal compact) contre **23-24 en ville** (canal plus étalé), sans connaître le scénario. Les gains décroissent avec le retard, comme un filtre de Wiener.
- **Le précodeur a besoin de D=64** : avec D=32, on retombe au niveau du candidat. Ce n'est donc pas la compression d'une fenêtre qui exigeait D=64 : c'est la précision du précodage à fort SNR.
- **Coût** : 58 k paramètres, environ 602 k MAC/SC (le précodeur D64 en représente l'essentiel ; l'estimation, faite une fois par bande, est faible une fois répartie).
- **Toujours sous WMMSE au faible SNR** (−0,1 à −0,2 bps/Hz de 0 à 10 dB) : limite du précodeur, inchangée.
- *Limite* : une seule graine.

### Masque retard × angle (variante b) — NÉGATIF (`results/phase4/delay_model_angle_D64.*`)
- **Question** : apprendre le gain par couple (retard, angle), avec une DFT unitaire sur les 4 antennes de chaque polarisation, exploite-t-il la structure spatiale et réduit-il l'écart au génie ?
- **Résultat** : identique au masque par retard (±0,01 bps/Hz partout ; 20 dB : 24,79 / 24,41 / 25,06). Le modèle garde en moyenne 187 / 196 / 114 couples (retard, angle) sur 1 224.
- **Interprétation** : avec 4 antennes par polarisation, la résolution angulaire est très grossière (4 directions) ; il n'y a presque pas de parcimonie angulaire à exploiter. Le précodeur traite déjà les 8 antennes ensemble.
- **Décision** : écarté en M=8. **À retester en massif** (64 antennes : 32 directions par polarisation), où la parcimonie angulaire devrait compter. Option `angle_mask` gardée dans le code.

### Diagnostic du faible SNR — RETENU (`results/phase4/lowsnr.*`)
- **Question** : en CSI parfait, nos précodeurs plafonnent à environ 102,5 % de RZF à 0 dB (WMMSE : 105,4 %), quelle que soit leur taille. Est-ce le compromis d'un seul modèle pour 0 à 25 dB ?
- **Réglages** : précodeur du candidat (D64, 1 bloc linéaire × 4, perte relative), CSI parfait, UMi (données et références de la phase 1), entraîné sur 0-25 dB puis sur 0-10 dB seulement.

| % de WMMSE | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---|---|---|---|---|---|
| entraîné 0-25 dB | 97,1 | 98,1 | 99,1 | 99,7 | **100,0** | 99,7 |
| entraîné 0-10 dB | **99,1** | **99,3** | 99,2 | 96,5 | 82,0 | 37,1 |

**Conclusions :**
- **L'essentiel du trou au faible SNR vient du compromis entre SNR** : spécialisé sur 0-10 dB, le même précodeur passe de 97,1 à 99,1 % de WMMSE à 0 dB. Il reste environ 1 % structurel.
- En CSI parfait, le précodeur généraliste est **au niveau de WMMSE à 15-25 dB** (99,7 à 100 %).
- **Piste de correction** (un seul changement) : mieux conditionner le modèle sur le SNR (le SNR n'entre aujourd'hui qu'une fois, à l'embedding), ou démarrer la boucle depuis RZF et n'apprendre que la correction.

### Référence robuste : WMMSE à erreur moyenne sur LMMSE — RETENU (`results/phase4/robust_ref_p10.json`)
- **Question** : un WMMSE qui tient compte de l'erreur d'estimation (covariance d'erreur diagonale de l'estimateur LMMSE de Sionna, ajoutée aux termes d'interférence) efface-t-il notre gain ?
- **Réglages** : mêmes tirages et même bruit pilote ; deux versions de la variance d'erreur (annoncée par Sionna ; calibrée sur l'erreur réelle moyenne du scénario) ; deux covariances LMMSE (mélange, oracle). Contrôle : WMMSE classique retrouvé au centième.

Débit à 20 dB (bps/Hz) :

| | UMi | UMa | RMa |
|---|---|---|---|
| WMMSE (LMMSE mélange) | 24,07 | 23,62 | 24,20 |
| **WMMSE robuste (LMMSE mélange)**, meilleure des deux versions | **24,18** | **23,74** | **24,35** |
| WMMSE robuste (LMMSE oracle) | 24,12 | 23,57 | 25,33 |
| **notre modèle (retards + D64)** | **24,80** | **24,40** | **25,05** |

Écart de notre modèle à la **meilleure référence classique réaliste** (robuste, covariance mélange, meilleure des deux versions) : **+0,62 / +0,67 / +0,70 à 20 dB** et **+1,36 / +1,35 / +1,74 à 25 dB** ; +0,07 à +0,17 à 15 dB ; −0,09 à −0,24 de 0 à 10 dB.

**Conclusions :**
- **La version robuste n'apporte qu'environ +0,1 bps/Hz** à WMMSE. **Notre gain tient** : il ne vient pas d'une simple meilleure régularisation, mais d'une meilleure estimation du canal (ce qu'un précodage robuste ne peut pas récupérer).
- La variance d'erreur annoncée par Sionna est juste en moyenne quand la covariance est bonne ; avec la covariance du mélange, elle est trop pessimiste sur RMa (0,0099 contre 0,0074) et trop optimiste sur UMa (0,0099 contre 0,0117). La calibrer ne change presque rien.
- **Seule exception** : sur RMa, le WMMSE robuste avec la **covariance oracle** (irréaliste : il faut connaître le scénario) nous bat de 0,28 bps/Hz à 20 dB.

### Sondages précédents (profil des trajets) — NÉGATIF (`results/phase4/history.*`, parties refs / model / history)
- **Question** : donner au masque, pour chaque retard, sa puissance moyenne sur les 4 sondages précédents (espacés de 2 ms, mêmes trajets, phases qui évoluent avec la mobilité) rapproche-t-il du génie ?
- **Réglages** : nouvelles données (5 instants par tirage), donc tout refait sur les mêmes tirages : modèle sans historique, modèle avec historique (seul changement), références. Données à l'instant du dernier sondage.

Débit à 20 dB (bps/Hz) :

| | UMi | UMa | RMa |
|---|---|---|---|
| WMMSE (génie par lien) | 25,79 | 25,86 | 26,22 |
| **modèle sans historique** | **24,86** | **24,49** | **24,85** |
| **modèle avec historique** | **24,86** | **24,49** | **24,83** |
| WMMSE (LMMSE construit sur l'historique) | 23,89 | 23,91 | 24,41 |
| WMMSE robuste (LMMSE mélange) | 23,86 | 23,72 | 24,36 |
| WMMSE (LMMSE mélange) | 23,75 | 23,60 | 24,21 |

**Conclusions :**
- **L'historique n'apporte rien** au modèle (±0,03 bps/Hz). Le LMMSE classique construit sur ce même historique n'est guère meilleur que le LMMSE de covariance moyenne (+0,05 à +0,3).
- **L'écart au génie (0,9 à 1,4 bps/Hz) ne vient donc pas d'un manque de connaissance du profil des trajets sur la grille des retards.** Le génie connaît les retards **exacts**, hors grille : sa covariance n'a pas de fuite. Notre masque et le LMMSE « historique » travaillent tous deux sur la grille des retards, où chaque trajet fuit sur ses voisins. **L'écart restant ressemble donc à de la fuite.**
- Sur ces nouveaux tirages, le modèle (sans historique) bat toujours la meilleure référence classique, historique compris : +0,97 / +0,58 / +0,42 à 20 dB, +1,63 / +1,13 / +1,15 à 25 dB.
- **Règle convenue** : écart > 0,5 bps/Hz et signature de fuite → on fait **le dernier effort d'estimation** : un filtre complexe sur 3 retards voisins au lieu d'un gain par retard. On le garde seulement s'il apporte ≥ +0,2 bps/Hz à 20 dB dans les trois scénarios.

### Filtre complexe sur 3 retards voisins (dernier effort d'estimation) — À LA LIMITE DE LA RÈGLE (`results/phase4/delay_model_bande_D64.*`)
- **Question** : remplacer le gain réel par retard par une combinaison complexe des retards t−1, t, t+1 (commune aux antennes) récupère-t-il une partie de la fuite ?
- **Réglages** : identiques au modèle « retards + D64 » (mêmes tirages, graine 0), seul l'étage de sortie du module change (+165 paramètres, calcul inchangé). *Incident* : l'évaluation finale avait planté (paquets à 3 éléments depuis l'option historique) ; corrigé, et le modèle a été réévalué à partir de ses poids sauvegardés (option `--load`), sans réentraîner.

Débit (bps/Hz) et écart au masque :

| | UMi | UMa | RMa |
|---|---|---|---|
| masque, 20 dB | 24,80 | 24,40 | 25,05 |
| **filtre à bande, 20 dB** | **24,99 (+0,19)** | **24,59 (+0,18)** | **25,29 (+0,24)** |
| filtre à bande, 25 dB | 29,21 (+0,29) | 28,68 (+0,28) | 29,97 (+0,39) |
| filtre à bande, 0 dB | 5,17 (−0,04) | 5,11 (−0,04) | 5,21 (−0,04) |
| écart restant au génie, 20 dB | 1,10 | 1,19 | 0,90 |

**Conclusions :**
- **Le gain est réel et cohérent** (environ 20 fois la variabilité entre graines), il grandit avec le SNR (+0,28 à +0,39 à 25 dB), au prix d'une très légère perte à 0 dB (−0,04).
- **Mais il est juste sous le seuil fixé d'avance** (+0,2 dans les trois scénarios à 20 dB) : +0,19 et +0,18 en ville, +0,24 sur RMa.
- Il ne referme qu'environ 15 % de l'écart restant au génie : la fuite n'explique donc qu'une partie de cet écart.
- **Décision (utilisateur, 2026-10-24) : GARDÉ**, malgré le seuil manqué de peu, parce qu'il ne coûte presque rien. C'est le dernier effort sur l'estimation.

### Attribution : estimation ou précodeur ? — RETENU (`results/phase4/attribution_p10.json`)
- **Question** : l'écart restant au génie (environ 1 bps/Hz à 20 dB) vient-il du module d'estimation ou du précodeur ? On croise estimation (notre module / génie / canal parfait) et précodeur (le nôtre / WMMSE / RZF), avec le modèle retenu rechargé depuis ses poids, sur les mêmes tirages.
- **Découverte** : appris ensemble sur le débit, le module sort le canal **multiplié par environ 4,8** (le précodeur normalise sa puissance, l'échelle est libre), et le précodeur s'est adapté à cette échelle. Donné un canal à l'échelle normale (génie ou parfait), notre précodeur s'effondre (13 à 14 bps/Hz). Même remise à l'échelle (énergie = LS moins bruit, réaliste), la sortie du module a une « erreur » de +2 dB, sans doute une rotation de phase par utilisateur, qui ne change pas le débit. **Le module est une représentation apprise pour le précodeur, pas un estimateur de canal.**

Débit à 20 dB (bps/Hz) :

| | UMi | UMa | RMa |
|---|---|---|---|
| notre modèle complet | 24,99 | 24,59 | 25,29 |
| notre estimation (échelle corrigée) + WMMSE | 25,05 | 24,66 | 25,48 |
| **génie (à l'échelle du module) + notre précodeur** | **26,00** | **25,65** | **25,99** |
| génie + WMMSE | 26,09 | 25,77 | 26,19 |
| canal parfait (échelle du module) + notre précodeur | 26,61 | 26,29 | 26,32 |
| canal parfait + WMMSE | 26,74 | 26,46 | 26,56 |

**Conclusions :**
- **Le goulot est l'estimation** : à précodeur fixé (le nôtre), remplacer notre représentation par le génie fait gagner environ 1,0 bps/Hz (UMi +1,01, UMa +1,06, RMa +0,70).
- **Le précodeur ne pèse que pour une petite part** : WMMSE sur notre estimation fait +0,06 à +0,19 à 20 dB, et environ +0,2 à 0 dB (le trou au faible SNR connu). Avec une bonne estimation, notre précodeur est à 0,1-0,2 de WMMSE.
- **Décision proposée** : rouvrir l'estimation, de façon encadrée (3 essais au plus : plus de données, module plus grand, grille de retards sur-échantillonnée ; seuil +0,2 à 20 dB), puisque c'est là qu'est la marge et que le module est l'entrée du précodeur.

### Estimation rouverte (cadre : 3 essais, seuil +0,2 à 20 dB) — essais (i) et (ii)
Base : modèle « retards + filtre à bande + précodeur D64 » (24,99 / 24,59 / 25,29 à 20 dB). Mêmes tirages d'évaluation.

| Essai (un seul changement) | UMi 20 dB | UMa 20 dB | RMa 20 dB | 25 dB | params | Verdict |
|---|---|---|---|---|---|---|
| **(i) 3× plus de données (1 056 tirages par scénario) et 80 000 pas** (`delay_model_bande_n1056_s80k_D64`) | **25,24 (+0,26)** | **24,83 (+0,25)** | **25,50 (+0,21)** | +0,48 à +0,52 | 58 k | **RETENU** |
| (ii) module plus grand, D=64 au lieu de 32 (`delay_model_bande_est64_D64`) | 25,04 (+0,05) | 24,63 (+0,05) | 25,30 (+0,01) | +0,00 à +0,09 | 114 k | ÉCARTÉ |

**Conclusions :**
- **Le modèle manquait de données, pas de capacité.** Trois fois plus de tirages (et deux fois plus de pas) apportent +0,21 à +0,26 bps/Hz à 20 dB et environ +0,5 à 25 dB, un peu aussi au faible SNR (+0,05 à 0 dB). Doubler la largeur du module n'apporte presque rien.
- *Limite* : (i) change à la fois la quantité de données et le nombre de pas ; on ne sait pas lequel des deux compte le plus.
- **Nouveau candidat** : le même modèle, entraîné avec 1 056 tirages par scénario et 80 000 pas. Face à la meilleure référence classique réaliste (WMMSE robuste, LMMSE mélange) : **+1,07 / +1,10 / +1,16 bps/Hz à 20 dB (+4,4 / +4,6 / +4,8 %)** et **+2,17 / +2,14 / +2,61 à 25 dB** ; égalité à 10 dB ; −0,15 à −0,22 à 0-5 dB.
- **Écart restant au génie à 20 dB : 0,85 / 0,94 / 0,69 bps/Hz.**
- Reste l'essai (iii), la grille de retards sur-échantillonnée, à faire avec ce nouveau réglage de données.

### Estimation rouverte — essai (iii) grille de retards fine, et 2× plus de données
Base : candidat « retards + filtre à bande + précodeur D64, 1 056 tirages/scénario, 80 000 pas » (25,24 / 24,83 / 25,50 à 20 dB).

| Essai (un seul changement) | UMi 20 dB | UMa 20 dB | RMa 20 dB | 25 dB | params | MAC/SC | Verdict |
|---|---|---|---|---|---|---|---|
| **(iii) grille 2× plus fine** (306 retards espacés de 27 ns) (`delay_model_bande_os2_n1056_s80k_D64`) | **25,76 (+0,51)** | **25,36 (+0,53)** | **26,03 (+0,52)** | +0,83 à +0,92 | 63 k | 702 k | **RETENU** |
| 2× plus de données (2 112 tirages/scénario) (`delay_model_bande_n2112_s80k_D64`) | 25,27 (+0,03) | 24,87 (+0,04) | 25,54 (+0,04) | +0,05 à +0,08 | 58 k | 602 k | ÉCARTÉ (saturé) |

**Conclusions :**
- **La fuite était bien le problème principal restant** : avec une grille deux fois plus fine, +0,51 à +0,53 bps/Hz à 20 dB et environ +0,9 à 25 dB, dans les trois scénarios. Le gain apparaît dès 10 dB (+0,12).
- **Les données sont saturées** : doubler encore les tirages n'apporte que +0,03 à +0,04. Pas besoin de plus de données à cette taille de problème.
- **Nouveau candidat** (grille fine, filtre à bande, précodeur D64, 1 056 tirages/scénario, 80 000 pas) face à la meilleure référence classique réaliste : **+1,58 / +1,63 / +1,68 bps/Hz à 20 dB (+6,5 / +6,9 / +6,9 %)** et **+3,00 / +2,99 / +3,53 à 25 dB** ; **désormais au-dessus dès 10 dB** (+0,12 / +0,12 / 0,00) ; encore −0,12 à −0,16 à 0-5 dB.
- **Écart restant au génie à 20 dB : 0,33 / 0,41 / 0,17 bps/Hz** (il était de 1,1 à 1,4 au début de la phase 4). À 25 dB : 0,75 / 0,89 / 0,41.
- **Fin de l'estimation**, comme convenu. Une grille encore plus fine (4×) pourrait apporter un peu plus, mais on s'arrête ici.

### Grille 4× plus fine, et ne garder que les premiers retards — ÉCARTÉS
Base : candidat grille 2× (306 retards ; 25,76 / 25,36 / 26,03 à 20 dB ; 63 k params ; 702 k MAC/SC).

| Essai (un seul changement) | UMi 20 dB | UMa 20 dB | RMa 20 dB | params | MAC/SC | Verdict |
|---|---|---|---|---|---|---|
| grille 4× plus fine (612 retards, 13,5 ns) (`delay_model_bande_os4_n1056_s80k_D64`) | 25,75 (−0,01) | 25,34 (−0,02) | 26,01 (−0,02) | 73 k | 1 020 k | ÉCARTÉ (saturé, plus cher) |
| grille 2×, 64 premiers retards seulement (3,5 µs ; 136 retards) (`delay_model_bande_os2_L64_n1056_s80k_D64`) | 25,59 (−0,17) | 25,07 (−0,29) | 26,01 (−0,02) | 58 k | 593 k | ÉCARTÉ |

**Conclusions :**
- **La finesse de la grille est saturée à 2×** : passer à 4× ne change rien et coûte 45 % de calcul en plus.
- **Couper les retards au-delà de 3,5 µs fait perdre en ville** (−0,17 UMi, −0,29 UMa) mais pas sur RMa (−0,02) : les canaux urbains ont des trajets tardifs au-delà de 3,5 µs, le canal rural est compact. Le gain de calcul est faible (−16 %). On garde toute la plage de retards.
- **Fin de l'estimation** : le candidat reste la grille 2× avec tous les retards.

### Faible SNR : SNR à chaque itération (A) et départ depuis RZF (B)
Base : candidat grille 2× (écart à la meilleure référence classique réaliste à 0 / 5 dB : −0,12 / −0,08 (UMi), −0,13 / −0,10 (UMa), −0,16 / −0,16 (RMa)).

| Essai (un seul changement, dans le précodeur) | 0 dB (vs classique) | 5 dB (vs classique) | 20 dB (vs candidat) | 25 dB (vs candidat) | Verdict |
|---|---|---|---|---|---|
| (A) SNR ajouté avant chaque itération (`..._snrall_...`) | −0,11 / −0,13 / −0,16 | −0,08 / −0,09 / −0,16 | −0,03 / −0,04 / −0,03 | −0,06 à −0,08 | ÉCARTÉ (aucun effet) |
| **(B) précodeur = RZF + correction apprise** (`..._rzf0_...`) | **−0,03 / −0,04 / −0,05** | **−0,01 / −0,01 / −0,04** | +0,04 / +0,03 / +0,04 | **+0,16 / +0,16 / +0,25** | **à décider** |

(UMi / UMa / RMa.)

**Conclusions :**
- **Rappeler le SNR à chaque itération ne change rien.** Le compromis entre SNR ne venait pas d'un manque d'information sur le SNR dans le bloc.
- **Démarrer depuis RZF comble l'essentiel du trou au faible SNR** : de −0,12 à −0,16 vers −0,03 à −0,05 bps/Hz à 0 dB (+0,09 à +0,12), sans rien perdre ailleurs, et avec un petit gain à 25 dB (+0,16 à +0,25). Le modèle est désormais **au niveau de la meilleure référence classique au faible SNR** et au-dessus dès 10 dB.
- **Le prix** : une inversion 4×4 par SC (celle de RZF ; environ 1 400 MAC, négligeable), qui fait perdre l'argument « aucune inversion de matrice ».
- **Décision (utilisateur)** : ne pas mettre RZF au cœur du modèle (cela affaiblit le propos « le transformer apprend à précoder »). Tester d'abord un départ sans inversion (MRT) ; sinon accepter le petit écart.

### Faible SNR : départ depuis MRT (sans inversion) — ÉCARTÉ, écart accepté (`..._mrt0_...`)
- Précodeur = MRT (canal conjugué, aucune inversion) + correction apprise, tête initialisée à zéro.
- **Aucun effet** : 0 dB −0,11 / −0,13 / −0,16 face au classique (identique au candidat), 20 dB −0,03 par rapport au candidat.
- **Interprétation** : ce n'est pas un « raccourci » quelconque qui aidait, c'est la structure de RZF elle-même (l'inversion régularisée), qui donne le bon compromis au faible SNR.
- **Décision finale (règle convenue)** : on garde le modèle **sans inversion** et on **accepte l'écart au faible SNR** (−0,1 à −0,16 bps/Hz à 0-5 dB, soit 2 à 3 %). Le départ depuis RZF est documenté comme **variante** (au niveau du classique partout, au prix d'une inversion 4×4 par SC).

---

## Phase 5 — MIMO massif (2026-10-24)

### Modèle M8K4 transposé tel quel en M=64, K=8 — NÉGATIF (`results/phase5/massive_refs.json`, `massive_model.json`)
- **Réglages** : M=64 (1 rangée de 32 colonnes, double polarisation), K=8, 50 RB = 600 SC (peigne 1 SC sur 8), pilote 10 dB, mélange UMi/UMa/RMa. Données d'évaluation générées une fois (96 tirages/scénario) et partagées par les références et le modèle. Modèle : celui retenu en M8K4 sans modification (retards grille 2× + filtre à bande + précodeur D64, 1 bloc linéaire × 4), 352 tirages/scénario, 40 000 pas (environ 15 min). 76 k paramètres, 1,64 M MAC/SC (contre environ 19 M pour LMMSE + WMMSE).

Débit somme (bps/Hz), UMi (UMa et RMa identiques à ±1) :

| SNR | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| **notre modèle** | **14,1** | **22,1** | **28,7** | **32,5** | **34,3** | **34,9** |
| RZF (LS) | 23,5 | 35,2 | 47,1 | 57,9 | 66,3 | 71,6 |
| WMMSE robuste (LMMSE mélange) | 23,9 | 35,8 | 48,0 | 59,5 | 69,0 | 75,2 |
| WMMSE (génie par lien) | 24,0 | 36,1 | 48,8 | 61,6 | 73,7 | 84,3 |
| WMMSE (CSI parfait) | 24,0 | 36,1 | 49,0 | 62,1 | 75,3 | 88,5 |

**Conclusions :**
- **Le modèle transposé tel quel échoue** : environ 50 à 60 % de RZF(LS), et il plafonne vers 35 bps/Hz. L'objectif d'entraînement (débit relatif à RZF) n'atteint que 0,62.
- **Le contexte change** : avec 64 antennes pour 8 utilisateurs, RZF sur le simple LS est déjà très bon (66 bps/Hz à 20 dB, à 9 bps/Hz du CSI parfait), et la marge d'estimation jusqu'au génie est d'environ 5 bps/Hz.
- **Hypothèse principale (à tester)** : le précodeur D=64 doit produire, pour chaque utilisateur, un vecteur de 64 antennes complexes (128 nombres réels) à partir d'un état de 64 nombres, tout en gérant l'interférence entre 8 utilisateurs. C'est un goulot de largeur, comme celui vu en M8K4 avec la fenêtre.
- **Prochain** : diagnostic en CSI parfait, précodeur seul : (a) D plus large, (b) sortie structurée W = Ĥᴴ C (le modèle ne produit que les K×K coefficients ; le précodeur optimal est toujours de cette forme, et ça supprime la dépendance en M de la sortie, sans inversion).

### Diagnostic : le précodeur seul en massif, CSI parfait — RETENU (`results/phase5/precoder_parfait_*.json`)
- **Question** : le plafond du modèle en massif vient-il du précodeur ? Précodeur seul, CSI parfait, M=64, K=8, mélange, 352 tirages/scénario, 40 000 pas, évalué sur le vrai canal des données d'évaluation du massif. RZF ≈ WMMSE ≈ 75,3 bps/Hz à 20 dB en CSI parfait.

| Précodeur (1 bloc linéaire × 4) | 0 dB | 10 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|---|
| base : D64, sortie directe | 66 % | 65 % | **50 %** | 43 % | 50 k | 1,2 M |
| D128, sortie directe | 87 % | 88 % | **77 %** | 68 % | 166 k | 4,5 M |
| D64, sortie W = Ĥᴴ C | 83 % | 60 % | **42 %** | 36 % | 44 k | 1,2 M |

(% de RZF en CSI parfait, UMi ; UMa et RMa identiques à ±1.)

**Conclusions :**
- **Le goulot est bien le précodeur** : même en CSI parfait, le précodeur actuel n'atteint que 50 % de RZF à 20 dB, et 66 % à 0 dB, alors qu'à faible SNR le simple MRT est déjà proche de l'optimum.
- **Doubler la largeur aide (77 %) mais reste loin, et coûte 4,5 M MAC/SC.**
- **La sortie structurée aide au faible SNR (83 %, elle rend MRT trivial) mais pas au fort SNR (42 %)** : il reste à calculer C ≈ (G + αI)⁻¹, une inversion 8×8, à partir des valeurs brutes du canal.
- **En massif, il y a deux difficultés distinctes** : (a) conserver l'information spatiale de 64 antennes (résolue par la sortie structurée) ; (b) inverser une matrice d'interaction 8×8 entre utilisateurs (non résolue).
- **Fait important** : en massif, RZF en CSI parfait vaut déjà WMMSE (75,26 contre 75,29). Il n'y a rien à gagner en précodage pur ; toute la marge est dans l'estimation (RZF(LS) 66 → génie 74).

### Options 2 et 3 en massif : attention de Gram contre force brute — RETENU (`results/phase5/precoder_parfait_gram.json`, `_d256.json`)
Même diagnostic (précodeur seul, CSI parfait, M=64, K=8). % de RZF (UMi ; UMa et RMa identiques à ±0,3) :

| Précodeur | 0 dB | 10 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|---|
| base (tokens d'utilisateur, D64) | 66,0 | 64,5 | 50,0 | 43,2 | 50 k | 1,21 M |
| D128 | 86,7 | 87,9 | 76,5 | 68,1 | 166 k | 4,52 M |
| **D256 (option 3, force brute)** | 99,9 | 100,1 | 99,8 | 99,4 | 593 k | **17,4 M** |
| **attention de Gram (option 2)** | **100,4** | **100,3** | **100,1** | **99,9** | **6,6 k** | **1,75 M** |
| *WMMSE (CSI parfait)* | *100,1* | *100,1* | *100,0* | *100,0* | | *≈ 19 M (+ LMMSE)* |

**Conclusions :**
- **Le précodeur à attention de Gram atteint WMMSE à tous les SNR**, y compris au faible SNR, avec **6 600 paramètres** (90 fois moins que D256) et 1,75 M MAC/SC (10 fois moins que D256 et que WMMSE), **sans aucune inversion de matrice**. Il converge dès 10 000 pas.
- **La force brute marche aussi (D256 : environ 100 %)**, mais elle coûte autant que WMMSE : l'argument du coût disparaît.
- **Pourquoi ça marche** : (a) la sortie W = Ĥᴴ C garde l'information des 64 antennes sans la faire passer par un état étroit ; (b) l'état par paire d'utilisateurs, mis à jour 4 fois avec la matrice de Gram (une « attention » dont les poids sont les produits scalaires entre canaux), représente une inversion itérative de la matrice 8×8 ; (c) le coût ne dépend de M que par G et W.
- **C'est la confirmation directe de l'intuition de la phase 1** : l'attention linéaire en boucle réalise une inversion itérative. Il suffisait de lui donner les bons objets (paires d'utilisateurs, poids = Gram).
- **Prochain** : (1) en massif, de bout en bout : module des retards + précodeur de Gram, sur le CSI estimé ; (2) en M8K4 : remplacer le précodeur actuel par celui de Gram (devrait combler le trou au faible SNR, sans inversion, et réduire le coût).

### La forme déroulée (4 étapes) est-elle toujours au niveau de RZF ? — NON hors du cas facile (`results/phase5/gram_stress.json`, `experiments/phase5_gram_stress.py`)
Évaluation seule des poids de Gram (entraînés sur le massif standard), CSI parfait, 57 600 SC (UMi+UMa+RMa). Argument testé : l'inverse exact (G+αI)⁻¹ est un polynôme de degré K−1 = 7 en G (Cayley–Hamilton) ; 4 produits par G n'atteignent que le degré 4. Ça doit suffire si G est bien conditionné, et casser sinon.

**1. Par échantillon (distribution d'entraînement)** — rapport modèle/RZF :

| SNR | moyenne | 1er centile | < 95 % de RZF | quartiles de conditionnement (0–50, 50–90, 90–99, 99–100 %) |
|---|---|---|---|---|
| 0 dB | 1,005 | 1,000 | 0 % | 1,003 / 1,005 / 1,009 / 1,014 |
| 20 dB | 1,001 | 0,974 | 0,5 % | 1,002 / 1,002 / 0,998 / 0,953 |
| 25 dB | 0,998 | 0,935 | 1,4 % | 1,001 / 0,999 / 0,985 / 0,905 |
| 30 dB (hors plage) | 0,991 | 0,878 | 3,4 % | 0,998 / 0,992 / 0,961 / 0,845 |

**2. Stress** (débit à 20 dB, modèle / RZF ; WMMSE ≈ RZF sauf M ≤ 16) :

| Cas | cond. médian | modèle | RZF | WMMSE |
|---|---|---|---|---|
| standard | 3,6 | 75,2 | 75,1 | 75,2 |
| utilisateurs corrélés par paires ρ=0,5 | 7,2 | 69,7 | 71,8 | 71,9 |
| ρ=0,8 | 20,5 | 50,4 | 63,5 | 63,5 |
| ρ=0,95 | 84,6 | 26,7 | 48,7 | 48,8 |
| M=32 antennes (K=8) | 7,9 | 62,8 | 63,4 | 64,2 |
| M=16 | 25,4 | 43,3 | 48,5 | 50,8 |
| M=8 | 239 | 18,3 | 25,5 | 34,7 |

**Conclusions :**
- **Sur la distribution d'entraînement, c'est vrai par échantillon, pas seulement en moyenne** (0,5 % des SC sous 95 % de RZF à 20 dB). Mais **le rapport baisse avec le conditionnement de G** : le 1 % des SC les plus mal conditionnées tombe à 95 % (20 dB), 90 % (25 dB), 85 % (30 dB).
- **Dès que G est mal conditionné, le modèle décroche nettement** (ρ=0,8 : 79 % de RZF ; ρ=0,95 : 55 % ; M=8 : 72 %), d'autant plus que le SNR est haut. C'est le comportement attendu d'une inversion polynomiale de bas degré.
- **Réserve importante** : ces cas sont hors de la distribution d'entraînement (conditionnement médian 3,6). On ne sait pas encore si c'est la **forme** (4 produits par G) ou la **généralisation** qui limite. Test qui sépare les deux : réentraîner sur un mélange incluant des cas mal conditionnés, avec 4 puis 8 étapes (et, si besoin, un bloc E·E de type Newton–Schulz qui double le degré à chaque étape).
- **Conséquence pour M8K4** (M/K=2, conditionnement typique plus élevé) : le précodeur de Gram à 4 étapes n'y est pas garanti ; c'est à mesurer directement.
- Note : à M ≤ 16, WMMSE dépasse nettement RZF (M=8 : 34,7 contre 25,5) ; en massif standard, RZF ≈ WMMSE.

### Forme ou généralisation ? Réentraînement avec cas difficiles, 4 contre 8 étapes, et attention triangulaire (`results/phase5/gram_dur_{L4,L8,tri_L4,tri_L8}.json`, `experiments/phase5_gram_hard.py`)
Même banc que le stress. Entraînement : moitié standard, moitié utilisateurs corrélés par paires (ρ ~ U(0, 0,95)), SNR 0–30 dB ; 16 et 8 antennes jamais vues. Variante « triangulaire » : le bloc G·E est remplacé par une attention triangulaire linéaire apprise entre paires, Σ_l a(e_jl)·b(e_lk) (Edge Transformer, AlphaFold), qui permet E·E (Newton–Schulz). Débit à 20 dB (% de RZF) :

| Cas | Gram L4 (std) | Gram L4 | Gram L8 | Tri L4 | Tri L8 | RZF |
|---|---|---|---|---|---|---|
| standard | 75,2 | 75,1 | **75,3** | 75,3 | 75,3 | 75,1 |
| ρ=0,8 | 50,4 (79 %) | 56,7 (89 %) | 62,8 (99 %) | 62,7 (99 %) | **63,2 (99,7 %)** | 63,5 |
| ρ=0,95 | 26,7 (55 %) | 38,3 (79 %) | 45,3 (93 %) | 43,7 (90 %) | **46,7 (96 %)** | 48,7 |
| ρ=0,95, 30 dB | — | 43,6 (58 %) | 59,9 (80 %) | 53,1 (71 %) | **62,3 (83 %)** | 74,9 |
| M=16 (jamais vu) | 43,3 (89 %) | 44,9 (93 %) | **49,1 (101 %)** | 47,7 (98 %) | 48,0 (99 %) | 48,5 |
| M=8 (jamais vu) | 18,3 (72 %) | 22,9 (90 %) | **26,1 (102 %)** | 22,5 (88 %) | 20,0 (78 %) | 25,5 |
| M=8, 30 dB | 19,2 (49 %) | 25,5 (65 %) | **31,8 (81 %)** | 26,9 (69 %) | 23,2 (59 %) | 39,2 |
| params / MAC par SC | 6,6 k / 1,75 M | 6,6 k / 1,75 M | 6,6 k / 3,45 M | 8,6 k / 2,28 M | 8,6 k / 4,50 M | |

**Conclusions :**
- **Les deux causes jouent.** Voir des cas difficiles à l'entraînement aide (ρ=0,95 : 55 → 79 %), mais à 4 étapes l'écart reste et grossit avec le SNR : c'est aussi la forme.
- **8 étapes de Gram ferment l'essentiel de l'écart jusqu'à 20 dB**, y compris à M=16 et M=8 jamais vus (≥ RZF). Bien mieux que la prédiction « Richardson » (≈ 5κ étapes) : la règle apprise se comporte comme une méthode à pas adaptés. Reste un écart au haut SNR avec G très mal conditionné (ρ=0,95 à 30 dB : 80 %).
- **L'attention triangulaire est meilleure à profondeur égale sur les cas vus** (à 4 couches : ρ=0,8 99 % contre 89 %, comme attendu du doublement de degré E·E), et un peu meilleure à 8 couches (ρ=0,95 : 96 % contre 93 %).
- **Mais elle généralise moins bien à un nombre d'antennes jamais vu** (M=8 : 78 % à 8 couches contre 102 % pour Gram ; elle se dégrade même de 4 à 8 couches). Interprétation : Gram remultiplie à chaque étape par la vraie G de l'échantillon et reste ancré dessus ; la forme triangulaire ne voit G qu'à l'entrée et apprend des règles propres à la distribution d'entraînement.
- **Retenu pour l'instant : Gram L8 entraîné avec cas difficiles** (robuste partout, 6,6 k params, 3,45 M MAC/SC). Piste non testée : un bloc qui combine G·E et E·E.
- Note technique : l'évaluation par conditionnement saturait la mémoire d'une P100 (16 Go) ; calcul désormais par paquets, et option `--load` pour réévaluer des poids sauvés (l'historique d'entraînement des deux runs triangulaires n'est pas conservé).

### Attention triangulaire avec G réinjecté à chaque couche (`results/phase5/gram_dur_trig_L{4,8}.json`)
Hypothèse testée : la forme triangulaire généralise mal parce qu'elle ne voit G qu'à l'entrée. Seul changement : G[j,k] (Re, Im, |·|) concaténé à l'entrée de chaque couche, comme caractéristique d'arête (aucun calcul imposé). Même banc. % de RZF :

| Cas | Gram L8 | Tri L8 | **Tri+G L8** | Tri+G L4 |
|---|---|---|---|---|
| standard, 20 dB | 100,3 | 100,2 | 100,2 | 100,2 |
| ρ=0,8, 20 dB | 99,0 | 99,7 | **99,8** | 98,9 |
| ρ=0,95, 20 dB | 93,1 | 96,0 | **97,0** | 90,7 |
| ρ=0,95, 30 dB | 80,0 | 83,1 | **86,9** | 72,6 |
| M=16 (jamais vu), 20 dB | **101,1** | 98,8 | 100,0 | 99,1 |
| M=8 (jamais vu), 20 dB | **102,2** | 78,2 | 84,0 | 89,5 |
| M=8 (jamais vu), 30 dB | **81,1** | 59,1 | 64,1 | 71,0 |
| params / MAC par SC | 6,6 k / 3,45 M | 8,6 k / 4,50 M | 9,0 k / 4,60 M | 9,0 k / 2,32 M |

**Conclusions :**
- **Réinjecter G aide, mais ne règle pas tout.** Sur les cas vus, Tri+G L8 est le meilleur de tous (ρ=0,95 à 30 dB : 87 % contre 80 % pour Gram L8). Sur M=8 jamais vu, il remonte un peu (78 → 84 %) mais reste loin de Gram L8 (102 %).
- **L'hypothèse de l'ancrage n'explique donc qu'une partie de l'écart de généralisation.** La forme imposée G·E généralise intrinsèquement mieux à une configuration jamais vue (M=8, K=8 : conditionnement médian 239, loin de l'entraînement).
- Comme pour la forme sans G, le passage de 4 à 8 couches dégrade M=8 (89 → 84 %) : plus de liberté = plus de spécialisation à la distribution d'entraînement.
- **Conséquence pratique** : pour la forme libre, il faut entraîner sur les configurations visées (mélange de nombres d'antennes / d'utilisateurs ; les paramètres n'en dépendent pas) plutôt que compter sur la généralisation.

## Phase 6 — architecture de précodeur la plus générale (2026-10-26, autonome)

**Question** : quelle architecture de base marche dans les deux configurations visées (M8K4 et massif M64K8), y compris les cas difficiles (utilisateurs corrélés, haut SNR) et des tailles jamais vues, de préférence sans forme « déroulée » imposée ?

**Banc commun** (`experiments/phase6_bench.py`, données `/export/tmp/sala/freqctx/data/phase6`) : CSI parfait, précodeur seul. Entraînement d'UN modèle sur le mélange M64K8 (600 SC) + M8K4 (612 SC), UMi+UMa+RMa, la moitié des échantillons avec utilisateurs corrélés par paires (ρ ~ U(0, 0,95)), SNR 0–30 dB, 40 000 pas, perte relative à RZF, coupure du gradient à 1. Évaluation sur 11 configurations : vues (M64K8, M8K4, et versions corrélées) et non vues (M32K8, M16K8, M8K8, M64K4, M32K4, M16K4). Script d'entraînement `experiments/phase6_train.py`, synthèse `experiments/phase6_summary.py` → `results/phase6/summary.md`.

**Candidats** (tous : sortie W = Ĥᴴ C sauf « axial », aucune inversion, équivariants aux permutations d'utilisateurs) :
- `gram` : état par paire, bloc G·E (forme « déroulée »), partagé ou non ;
- `tri` / `trig` : état par paire, **attention triangulaire apprise** (Edge Transformer), sans / avec G réinjecté à chaque couche, partagée (`tied`) ou **couches distinctes** (`untied` = transformer standard) ;
- `axial` : un token par (antenne, utilisateur), Gram appris à chaque couche, sortie directe (proche des « 2D-Gformer » de Guo et Yang 2025).

### Résultats (% de RZF ; « vues » / « non vues » = moyennes ; `results/phase6/*.json`)

| run | params | MAC/SC M64K8 / M8K4 | vues 20 dB | vues 30 dB | non vues 20 dB | non vues 30 dB | pire config 20 dB | M8K4 % WMMSE 0 / 20 dB |
|---|---|---|---|---|---|---|---|---|
| **trig L8 untied** | 70 k | 4,60 M / 1,11 M | **101,1** | **98,9** | **104,4** | **99,7** | ρ0,95 99,1 | 99,8 / 100,5 |
| trig L8 untied, graine 1 | 70 k | 4,60 M / 1,11 M | 101,1 | 98,9 | 104,1 | 99,3 | ρ0,95 99,3 | 99,8 / 100,4 |
| tri L8 untied (sans G) | 67 k | 4,50 M / 1,09 M | 100,7 | 97,6 | 103,0 | 97,5 | ρ0,95 98,1 | 99,6 / 100,2 |
| gram L8 untied | 51 k | 3,45 M / 0,82 M | 99,1 | 94,2 | 103,0 | 97,9 | ρ0,95 92,5 | 98,9 / 100,2 |
| trig L4 untied | 35 k | 2,32 M / 0,56 M | 99,5 | 94,1 | 102,0 | 96,1 | ρ0,95 95,1 | 99,3 / 100,1 |
| trig L8 (partagé) | 9,0 k | 4,60 M / 1,11 M | 99,4 | 94,5 | 102,2 | 96,6 | ρ0,95 94,3 | 98,6 / 100,1 |
| trig L8 + G normalisé | 9,0 k | 4,60 M / 1,11 M | 99,6 | 94,9 | 101,8 | 96,5 | ρ0,95 94,7 | 99,0 / 100,1 |
| gram L8 (partagé) | 6,6 k | 3,45 M / 0,82 M | 97,9 | 91,5 | 103,2 | 97,7 | ρ0,95 89,0 | 98,1 / 100,0 |
| axial L4 | 8,7 k | **19,0 M** / 1,12 M | 95,6 | 85,9 | 101,1 | 93,5 | ρ0,95 78,9 | 97,2 / 99,2 |

**Conclusions :**
- **Meilleure architecture : transformer standard sur les paires d'utilisateurs** (attention triangulaire apprise, G réinjecté comme caractéristique d'arête, 8 couches distinctes, sortie W = Ĥᴴ C). Au moins 99 % de RZF dans toutes les configurations à 20 dB (y compris ρ = 0,95), 94 % au pire à 30 dB ; ≥ WMMSE en M8K4 dès 10 dB ; reproductible (graine 1 identique à ±0,4).
- **Entraîner sur les deux configurations règle la généralisation** de la forme libre (la phase 5 la voyait échouer à M=8 jamais vu ; ici M8K8 jamais vu : 117,5 % de RZF à 20 dB, 98 % à 30 dB).
- **Ce qui compte, par ordre** : couches distinctes (trig partagé → untied : 94,5 → 98,9 vues à 30 dB) ; profondeur 8 (L4 : 94,1) ; forme triangulaire plutôt que G·E à couches distinctes (gram untied : 94,2) ; G réinjecté (sans : 97,6). La normalisation de G ne change rien.
- **Le modèle axial** (tokens antenne × utilisateur, type Gformer) marche sur les cas faciles mais décroche sur les cas mal conditionnés (ρ = 0,95 à 30 dB : 46 %) et coûte 4 à 5× plus en massif : la représentation par paires d'utilisateurs est la bonne.
- **Coût** : la meilleure version a 70 k paramètres (couches distinctes) mais le même calcul que la version partagée (4,6 M MAC/SC en massif, 1,1 M en M8K4).
- Réserve : le pourcentage de WMMSE sur ρ = 0,95 dépasse parfois 100 % (jusqu'à 113 % à 10 dB) : WMMSE démarré de RZF reste dans un optimum local quand deux utilisateurs sont presque colinéaires ; ce n'est pas une borne.

### Gram partagé avec plus de passages à l'inférence (`results/phase6/loops_gram_L8_d16_h64.json`)
Sans réentraîner : 12 passages (au lieu de 8) améliorent les cas difficiles (ρ = 0,95 : 89 → 93 % de RZF à 20 dB, 74 → 79 % à 30 dB ; M8K8 à 30 dB : 89 → 99 %), 16 passages dégradent. La règle apprise converge en partie seulement ; 4 passages s'effondrent (le modèle n'est pas « anytime »).

### De bout en bout, des pilotes au précodeur (estimateur des retards inchangé + nouveau précodeur)
**M8K4, pilote 10 dB, mélange** (`results/phase4/delay_model_bande_os2_{gramL8,trigL8_untied}_n1056_s80k_D64.json`), débit en bps/Hz :

| | 0 dB (UMi/UMa/RMa) | 20 dB (UMi/UMa/RMa) | params | MAC/SC |
|---|---|---|---|---|
| précodeur phase 4 (tokens d'utilisateur) | 5,25 / 5,19 / 5,28 | 25,76 / 25,36 / 26,03 | 63 k | 0,70 M |
| **+ Gram L8** | 5,36 / 5,32 / 5,43 | 25,81 / 25,40 / 26,11 | 34 k | 0,99 M |
| **+ transformer par paires (trig L8 untied)** | **5,39 / 5,36 / 5,47** | **25,87 / 25,47 / 26,28** | 98 k | 1,27 M |
| WMMSE robuste (LMMSE mélange) | 5,37 / 5,32 / 5,45 | 24,18 / 23,73 / 24,35 | | |
| génie par lien | 5,40 / 5,37 / 5,47 | 26,09 / 25,77 / 26,19 | | |

- **Le trou au faible SNR est comblé sans inversion** : à 0 dB, le modèle égale le WMMSE robuste et le génie (à 0,01 près), alors que le précodeur de la phase 4 y perdait 0,1–0,2.
- À 20 dB : +1,69 / +1,74 / +1,93 bps/Hz sur le meilleur classique réaliste ; en RMa le modèle dépasse la « borne » génie (26,28 contre 26,19 : le génie est WMMSE sur une très bonne estimation, pas un optimum sous CSI imparfait).

**Massif M64K8, pilote 10 dB, mélange** (`results/phase5/massive_model_{gramL8,trigL8_untied}.json`), 352 tirages/scénario, 40 000 pas :

| | 0 dB | 20 dB (UMi/UMa/RMa) | 25 dB (UMi/UMa/RMa) | params | MAC/SC |
|---|---|---|---|---|---|
| modèle M8K4 transposé (phase 5a) | — | 34,3 / 34,0 / 34,3 | — | 76 k | 1,6 M |
| **+ Gram L8** | 23,96 / 23,79 / 23,88 | **72,70 / 71,73 / 73,98** | 82,2 / 80,8 / 85,2 | 33 k | 3,9 M |
| **+ trig L8 untied** | 23,96 / 23,80 / 23,88 | 72,67 / 71,72 / 73,96 | 82,1 / 80,7 / 85,1 | 96 k | 5,0 M |
| RZF (LS) | 23,54 / 23,26 / 23,56 | 66,33 / 64,86 / 68,43 | 71,6 / 69,7 / 74,5 | | |
| WMMSE robuste (LMMSE mélange) | 23,86 / 23,67 / 23,78 | 68,97 / 68,00 / 69,51 | 75,2 / 74,0 / 76,1 | | |
| génie par lien | 24,01 / 23,94 / 23,90 | 73,74 / 73,49 / 74,10 | 84,4 / 83,9 / 85,7 | | |

- **En massif, +3,7 / +3,7 / +4,5 bps/Hz sur le meilleur classique réaliste à 20 dB** (+5,4 à +6,5 %), +7 à +9 à 25 dB ; ≥ classique à tous les SNR ; à 0,1–1,8 du génie.
- **Les deux précodeurs donnent le même débit en massif standard** : le précodeur n'y est plus le facteur limitant, c'est l'estimation (écart au génie 1,0 / 1,8 / 0,1). Le transformer par paires ne se distingue que sur les cas difficiles (M8K4, corrélés, haut SNR).

### Phase 6, compléments : coût, entraînement plus long, graine, données en massif
**Frontière coût / performance** (transformer par paires, couches distinctes ; % de RZF ; `results/phase6/trig_*_untied*.json`) :

| variante | params | MAC/SC M64K8 / M8K4 | vues 20 dB | vues 30 dB | non vues 30 dB | pire config 30 dB |
|---|---|---|---|---|---|---|
| L8, FFN 64, **80 000 pas** | 70 k | 4,60 M / 1,11 M | 101,4 | **100,0** | **100,4** | ρ0,95 97,7 |
| L8, FFN 64 | 70 k | 4,60 M / 1,11 M | 101,1 | 98,9 | 99,7 | ρ0,95 94,4 |
| L8, FFN 32 | 45 k | 3,03 M / 0,72 M | 100,8 | 97,9 | 98,2 | — |
| L6, FFN 64 | 53 k | 3,46 M / 0,83 M | 100,7 | 98,0 | 97,8 | — |
| **L8, FFN 16** | 32 k | **2,24 M / 0,52 M** | 100,1 | 96,2 | 97,3 | — |
| L6, FFN 32 | 33 k | 2,28 M / 0,54 M | 100,2 | 95,8 | 97,8 | — |
| L4, FFN 64 | 35 k | 2,32 M / 0,56 M | 99,5 | 94,1 | 96,1 | — |
| *Gram L8 partagé (réf.)* | 6,6 k | 3,45 M / 0,82 M | 97,9 | 91,5 | 97,7 | ρ0,95 73,9 |

- **Entraîner 2× plus longtemps ferme presque tout l'écart** : 100,0 % de RZF en moyenne à 30 dB sur les configurations vues, 97,7 % au pire (ρ = 0,95).
- **À coût égal, la profondeur bat la largeur** (≈ 2,3 M MAC : L8/FFN16 96,2 > L6/FFN32 95,8 > L4/FFN64 94,1). Avec 8 couches étroites (2,2 M MAC en massif, 0,52 M en M8K4), le transformer par paires bat Gram L8 qui coûte 1,5× plus.
- Seule configuration encore loin de WMMSE : M8K8 jamais vu à 30 dB (74 % de WMMSE ; WMMSE y dépasse RZF de 33 %).

**De bout en bout M8K4, graine 1** (`results/phase4/..._trigL8_untied_s1_...`) : 25,87 / 25,48 / 26,28 à 20 dB, identique à ±0,01 à la graine 0.

**Massif de bout en bout, 3× plus de données (1 056 tirages/scénario, 80 000 pas, Gram L8)** (`results/phase5/massive_model_gramL8_n1056_s80k.json`) : 72,91 / 72,00 / 74,08 à 20 dB (+0,21 / +0,27 / +0,10 sur 352 tirages) ; écart au génie 0,8 / 1,5 / 0. L'écart restant en massif est l'estimation (UMa surtout), pas le précodeur.

**Ménage du 2026-10-28** : pour la phase 6, seuls sont gardés dans le dépôt les résultats retenus (`trig_L8_d16_h64_untied*.json`, version étroite `trig_L8_d16_h16_untied.json`, référence structurée `gram_L8_d16_h64.json`) et `summary.md` (tableau de tous les runs, y compris écartés). Les fichiers des variantes écartées (modèle axial, normalisation de G, tri sans G, Gram à couches distinctes, versions partagées, L4/L6, FFN 32, passages supplémentaires de Gram) et leur code (`models/axial_precoder.py`, option de normalisation, `experiments/phase6_loops.py`) sont retirés ; leurs chiffres restent ci-dessus, et tout est récupérable dans l'historique git (commit 9efa024 et avant).

## Phase 7 — ce qu'apporte le contexte fréquentiel, à précodeur fixe (2026-10-27)

**Question** : d'où vient le gain à partir des pilotes ? Que représente la sortie de l'encodeur fréquentiel ? Est-ce un seul modèle ou deux blocs ? Combien de bande faut-il voir ? Peut-on calculer le précodeur moins souvent en fréquence ?

**Cadre commun** (`experiments/phase7_context.py`) : précodeur = transformer par paires retenu en phase 6 (8 couches distinctes, FFN 64). M8K4, pilote 10 dB, mélange UMi+UMa+RMa, 1 056 tirages/scénario, 80 000 pas, mêmes tirages d'évaluation que les phases 3–6. Une seule chose change par ligne.

Débit (bps/Hz), UMi / UMa / RMa :

| Variante | 0 dB | 20 dB | 25 dB | MAC/SC |
|---|---|---|---|---|
| **modèle complet** (encodeur sur toute la bande, appris avec le précodeur ; phase 6) | 5,39 / 5,36 / 5,47 | **25,87 / 25,47 / 26,28** | 30,80 / 30,27 / 31,84 | 1,27 M |
| sans contexte fréquentiel (LS, SC seule) | 5,32 / 5,28 / 5,40 | 22,43 / 22,04 / 22,61 | 24,75 / 24,31 / 24,98 | 1,11 M |
| contexte classique : LMMSE (mélange) + même précodeur | 5,37 / 5,32 / 5,44 | 24,27 / 23,82 / 24,51 | 27,69 / 27,16 / 28,05 | 1,11 M |
| encodeur sur 1 RB | 5,34 / 5,29 / 5,43 | 23,18 / 22,71 / 24,01 | 25,93 / 25,35 / 27,19 | 1,15 M |
| encodeur sur 3 RB | 5,36 / 5,32 / 5,45 | 24,47 / 23,99 / 25,23 | 28,07 / 27,46 / 29,48 | 1,15 M |
| encodeur sur 17 RB | 5,38 / 5,35 / 5,46 | 25,47 / 25,03 / 26,05 | 29,96 / 29,36 / 31,27 | 1,19 M |
| entraînement séparé (encodeur sur l'erreur quadratique, puis figé) | 5,39 / 5,36 / 5,47 | 25,69 / 25,32 / 26,07 | 30,41 / 29,92 / 31,32 | 1,27 M |
| encodeur sans attention entre retards | 5,39 / 5,35 / 5,47 | 25,68 / 25,29 / 26,12 | 30,42 / 29,90 / 31,44 | 1,16 M |
| *WMMSE robuste (LMMSE mélange)* | *5,37 / 5,32 / 5,45* | *24,18 / 23,73 / 24,35* | *27,57 / 27,04 / 27,84* | |

**Conclusions :**
- **Tout le gain à partir des pilotes vient du contexte fréquentiel.** Sans lui, le même transformer fait 22,4 à 20 dB (+0,7 sur RZF(LS), mais −3,4 sur le modèle complet).
- **L'encodeur appris bat le contexte classique** : à précodeur égal, +1,6 à +1,8 sur LMMSE. Avec LMMSE en entrée, notre précodeur ne fait que +0,1 sur WMMSE robuste : le gain du modèle vient de l'encodeur.
- **Plus de bande, c'est mieux, de façon régulière** : 1 RB 23,2 → 3 RB 24,5 → 17 RB 25,5 → 51 RB 25,9 (UMi, 20 dB). Il faut au moins 3 RB pour égaler le LMMSE classique sur toute la bande. Lien avec le mémoire : le contexte intra-RB (IB) ne suffit pas ; toute la bande (idée de TA-RB), vue dans le domaine des retards, est la bonne façon de l'exploiter.
- **Un seul modèle, pas deux blocs indépendants, mais l'écart est modeste** : l'entraînement séparé perd 0,15 à 0,2 à 20 dB (0,4 à 0,5 à 25 dB), rien à 0 dB.
- **L'attention entre retards apporte environ 0,2** (version sans attention : −0,16 à −0,19) pour un coût faible : gardée.

### Ce que représente la sortie de l'encodeur (`experiments/phase7_analysis.py`, `results/phase7/analysis.json`)
Modèle complet, graine 0. Erreur quadratique normalisée par rapport au vrai canal (dB), UMi / UMa / RMa :

| | LS | LMMSE | encodeur |
|---|---|---|---|
| canal entier (encodeur réaligné par un facteur complexe par utilisateur) | −17,1 / −16,8 / −17,6 | −20,2 / −19,5 / −21,4 | −16,7 / −16,8 / −16,4 |
| **direction seule** (facteur complexe par utilisateur ET par SC) | −17,9 / −17,5 / −18,3 | −20,9 / −20,2 / −22,0 | **−25,8 / −25,3 / −29,5** |

- **L'encodeur n'est pas un estimateur de canal au sens de l'erreur quadratique** (pas meilleur que le LS sur le canal entier, échelle moyenne ×2,1), **mais il estime la direction du canal de chaque utilisateur sur les antennes 5 à 7,5 dB mieux que LMMSE.** C'est ce dont le précodage a besoin ; l'amplitude par SC, à laquelle le débit est peu sensible, est laissée libre.
- WMMSE appliqué à la sortie réalignée : 25,70 / 25,30 / 26,06 à 20 dB (contre 24,07 / 23,62 / 24,20 sur LMMSE) : l'information utile est bien dans la sortie de l'encodeur.
- **Nom retenu : « encodeur fréquentiel »** (pas « estimateur ») ; il apprend une représentation par SC dont l'essentiel est la direction spatiale du canal.

### Granularité en fréquence du précodeur
**Vérification classique** (RZF, canal parfait, `experiments/phase7_gauge_check.py`) : tenir le précodeur constant sur 1 RB (type PRG 5G) coûte 26,1 → 22,5 en UMi à 20 dB (la bande de cohérence, ≈ 9 SC, est plus petite qu'un RB) ; seul RMa tolère le RB. Interpoler les coefficients C (même en fixant l'échelle et la phase par SC) est pire. **Interpoler W linéairement entre des SC « ancres »** marche : 1 SC sur 3 → −0,13 / −0,20 / −0,01.
**Appris** (`--variant interp`) : le précodeur n'est calculé qu'aux ancres (1 SC sur s), W interpolé linéairement, débit appris sur toutes les SC :

| pas s | 20 dB | 25 dB | MAC/SC (modèle total) |
|---|---|---|---|
| 1 (chaque SC) | 25,87 / 25,47 / 26,28 | 30,80 / 30,27 / 31,84 | 1,27 M |
| **3** | **25,80 / 25,37 / 26,27** | 30,70 / 30,12 / 31,84 | **0,53 M** |
| 4 | 25,70 / 25,22 / 26,26 | 30,52 / 29,87 / 31,81 | 0,44 M |
| 6 | 25,36 / 24,79 / 26,23 | 29,98 / 29,19 / 31,76 | 0,35 M |

- **1 SC sur 3 : perte ≤ 0,1, coût total divisé par 2,4.** L'entraînement récupère une partie de la perte de l'interpolation fixe. Au-delà, UMi et UMa perdent, RMa tient (bande de cohérence plus large) : un pas adaptatif à l'étalement des retards est une perspective.
- Lien avec le mémoire : c'est la bonne version de l'idée de TA-RB (précodeur grossier puis interpolé), calée sur la bande de cohérence et non sur le RB.

## Phase 8 — robustesse et références apprises (2026-10-28)

### Robustesse sans réentraîner (`experiments/phase8_robust.py`, `results/phase8/robustness.json`)
Modèle principal M8K4 (entraîné pilote 10 dB, données au slot du sondage) et version 1 SC sur 3. Nouveaux tirages (graines 2026+10i / 3026+10i). Débit à 20 dB, UMi / UMa / RMa :

| SNR pilote | modèle (par SC) | modèle (1 SC sur 3) | WMMSE robuste (LMMSE) |
|---|---|---|---|
| 0 dB | **12,90 / 12,92 / 13,23** | 17,29 / 17,24 / 18,98 | 19,03 / 18,91 / 20,30 |
| 5 dB | 22,64 / 22,58 / 23,64 | 23,76 / 23,68 / 24,76 | 21,83 / 21,79 / 22,55 |
| 10 dB (entraînement) | 25,64 / 25,60 / 26,23 | 25,57 / 25,51 / 26,23 | 23,90 / 23,90 / 24,28 |
| 15 dB | 26,12 / 26,11 / 26,49 | 26,02 / 25,99 / 26,48 | 25,22 / 25,22 / 25,38 |
| 20 dB | 26,27 / 26,27 / 26,58 | 26,15 / 26,12 / 26,56 | 25,93 / 25,93 / 25,98 |

| Retard données (pilote 10 dB) | modèle (par SC) | WMMSE robuste (LMMSE) | écart |
|---|---|---|---|
| 0 ms | 25,80 / 25,63 / 26,19 | 24,13 / 23,85 / 24,25 | +1,67 / +1,78 / +1,94 |
| 2,5 ms | 21,76 / 21,65 / 24,18 | 20,70 / 20,53 / 22,64 | +1,06 / +1,12 / +1,54 |
| 5 ms | 18,15 / 17,98 / 22,00 | 17,38 / 17,18 / 20,71 | +0,77 / +0,80 / +1,29 |

- **Vieillissement : robuste.** Le modèle (entraîné sans vieillissement) garde son avance sur le classique à 2,5 et 5 ms ; les deux perdent autant en absolu (le vieillissement touche le canal, pas l'estimation).
- **SNR pilote plus élevé : robuste** (toujours au-dessus du classique, avance qui se réduit naturellement quand l'estimation devient facile : +0,34 à pilote 20 dB).
- **SNR pilote plus bas : fragile.** À 5 dB, encore +0,8 sur le classique ; **à 0 dB, effondrement** (12,9 contre 19,0), l'encodeur ayant appris un débruitage calé sur 10 dB. La version 1 SC sur 3 tient mieux (17,3). Correction en cours : entraînement avec SNR pilote aléatoire (phase 8, `--variant pilot_rand`).

### Références apprises (même cadre, mêmes données, même perte ; `results/phase8/lmmse_axial.json`, `tarb.json`)
Pilote 10 dB, débit UMi / UMa / RMa :

| | 0 dB | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|---|
| **notre modèle** | **5,39 / 5,36 / 5,47** | **25,87 / 25,47 / 26,28** | **30,80 / 30,27 / 31,84** | 98 k | 1,27 M |
| TA-RB du mémoire (réécrit, `models/tarb.py`, sur LS) | 5,21 / 5,16 / 5,32 | 24,23 / 23,63 / 25,39 | 27,81 / 27,01 / 30,02 | 1 125 k | 1,79 M |
| LMMSE + précodeur type 2D-Gformer (Guo & Yang 2025 ; `models/axial_precoder.py`) | 5,36 / 5,31 / 5,43 | 24,24 / 23,79 / 24,47 | 27,66 / 27,12 / 28,00 | 34 k | 1,12 M |
| *WMMSE robuste (LMMSE)* | *5,37 / 5,32 / 5,45* | *24,18 / 23,73 / 24,35* | *27,57 / 27,04 / 27,84* | | |

- **Notre modèle bat les deux références apprises** : +1,6 / +1,8 / +0,9 bps/Hz sur TA-RB à 20 dB (+3,0 / +3,3 / +1,8 à 25 dB), avec 11× moins de paramètres et 30 % de calcul en moins.
- **TA-RB (mémoire)** : au niveau du classique en ville, mieux en RMa (+1,0 : son moyennage par groupe de 3 SC et son attention sur toute la bande aident quand la bande de cohérence est large) ; en dessous au faible SNR. Dans ce cadre réaliste (vrais pilotes, référence LMMSE), son avance du mémoire ne se retrouve pas en ville.
- **Précodeur de la littérature sur LMMSE** : au niveau de WMMSE robuste (+0,06 à +0,12) — même conclusion que notre précodeur sur LMMSE (phase 7) : sans encodeur appris, un bon précodeur appris n'apporte presque rien.
- Note : TA-RB ne tourne pas sur les P100 de srvgrm08 (convolutions cuDNN), entraîné sur srvgrm12.

### Entraînement avec SNR pilote aléatoire — corrige la fragilité (`results/phase7/pilot_rand.json`, `results/phase8/robustness_pilot_rand.json`)
Même modèle, mêmes données de canal ; seul changement : SNR pilote tiré uniformément entre 0 et 20 dB par lot de 8 tirages à l'entraînement (pas d'entrée supplémentaire). Débit à 20 dB, UMi / UMa / RMa :

| SNR pilote | modèle d'origine (entraîné à 10 dB) | **modèle pilote aléatoire** | WMMSE robuste (LMMSE) |
|---|---|---|---|
| 0 dB | 12,90 / 12,92 / 13,23 | **22,41 / 22,22 / 24,22** | 19,03 / 18,91 / 20,30 |
| 5 dB | 22,64 / 22,58 / 23,64 | **24,45 / 24,38 / 25,56** | 21,83 / 21,79 / 22,55 |
| 10 dB | 25,64 / 25,60 / 26,23 | 25,61 / 25,57 / 26,21 | 23,90 / 23,90 / 24,28 |
| 15 dB | 26,12 / 26,11 / 26,49 | 26,17 / 26,16 / 26,49 | 25,22 / 25,22 / 25,38 |
| 20 dB | 26,27 / 26,27 / 26,58 | 26,40 / 26,41 / 26,60 | 25,93 / 25,93 / 25,98 |

- **La fragilité disparaît** : à pilote 0 dB, +3,4 / +3,3 / +3,9 bps/Hz sur le classique (au lieu de −6) ; au-dessus du classique à tous les SNR pilote et tous les SNR de données testés (à 0 dB de données aussi).
- **Aucune perte au pilote nominal** (−0,03 à 10 dB, mêmes tirages standard : 25,84 / 25,45 / 26,26 contre 25,87 / 25,47 / 26,28), léger gain à 15–20 dB. Vieillissement inchangé (21,76 / 21,66 / 24,17 à 2,5 ms).
- L'encodeur estime donc lui-même le niveau de bruit (sur les retards sans trajet) : pas besoin de lui donner le SNR pilote.
- **Proposition : ce modèle devient le modèle principal.**

## Phase 9 — réduire le coût, et le juger en énergie (2026-10-31)

### Symétrie hermitienne du précodeur par paires (`--variant sym`, `results/phase7/sym.json`)
L'état est hermitien (E[k,j] = conj E[j,k]) ; l'entrée de la paire {j,k} est construite avec ses deux orientations (s_jk = c_jk + conj c_kj) ; le FFN, à poids réels appliqués de la même façon aux parties réelle et imaginaire (donc conjugaison-équivariant), ne tourne que sur les K(K+1)/2 paires j ≤ k ; sortie C = S diag(p) (S hermitienne, une puissance par utilisateur lue sur la diagonale : la forme de WMMSE). Une version naïve (FFN ordinaire sur les paires j ≤ k) cassait l'équivariance aux permutations d'utilisateurs (écart 0,54) : écartée.
- Débit (pilote aléatoire, comme `pilot_rand`) : 25,79 / 25,39 / 26,21 à 20 dB (−0,05 / −0,06 / −0,05) ; 30,62 / 30,09 / 31,66 à 25 dB (−0,12 / −0,13 / −0,13) ; 0 dB inchangé.
- Coût : 0,99 M MAC/SC au lieu de 1,27 M (−22 %), 72,6 k paramètres au lieu de 97,8 k. Précodeur seul : −26 % (M8K4), −29 % (massif).

**Vérification sur le banc de la phase 6 (précodeur seul, cas difficiles et tailles jamais vues ; `results/phase6/trig_L8_d16_h{64,128}_untied_sym.json`) — ÉCARTÉE :**

| (% de RZF) | vues 20 dB | vues 30 dB | ρ = 0,95 à 30 dB | non vues 30 dB | M8K8 jamais vu, 30 dB | MAC/SC M8K4 |
|---|---|---|---|---|---|---|
| base (FFN 64) | 101,1 | **98,9** | **94,4** | **99,7** | **98,0** | 1,11 M |
| symétrique, FFN 64 | 99,9 | 95,6 | 82,4 | 98,1 | 93,3 | 0,83 M |
| symétrique, FFN 128 (mêmes paramètres que la base) | 100,0 | 96,3 | 84,5 | 97,2 | 87,5 | 1,33 M |
| base étroite, FFN 16 (référence de coût) | 100,1 | 96,2 | 80,2 | 97,3 | — | 0,52 M |

- **La symétrie limite le modèle sur les cas difficiles** (ρ = 0,95 à 30 dB : 82 % contre 94 %), et **élargir le réseau ne rattrape pas l'écart** : c'est la contrainte (FFN à poids réels sur Re/Im, aucune rotation de phase interne), pas la capacité. La forme de sortie est juste en théorie, mais la façon de l'imposer réduit l'expressivité.
- **La version étroite sans symétrie fait aussi bien pour 37 % de calcul en moins** que la version symétrique : la symétrie n'apporte rien qu'un réglage de largeur n'apporte déjà.
- **Décision : écartée.** Base sans symétrie (architecture plus simple).

### Vraie FFT dans l'encodeur (recomptage, `DelayDomainPrecoder.macs_per_sc(fft=True)`)
Encodeur : M8K4 161 k → 114 k MAC/SC (−29 % ; l'essentiel est l'attention entre les 306 retards) ; massif 425 k → 102 k (÷ 4,2).

### Énergie (modèle du laboratoire, celui du mémoire ; `experiments/phase9_energy.py`, `results/phase9/energy.json`)
nJ par SC et par mise à jour ; notre modèle en INT8, classique en FP32 :

| | M8K4 | Massif M64K8 |
|---|---|---|
| LMMSE + RZF (FP32) | 0,09 | 0,79 |
| LMMSE + WMMSE, 30 it. (FP32) | 0,57 | 73,1 |
| notre modèle, par SC (INT8) | 0,43 | 1,66 |
| symétrique (INT8) | 0,34 | 1,22 |
| 1 SC sur 3 (INT8) | 0,17 | 0,58 |
| symétrique + 1 SC sur 3 (INT8, *non entraîné ensemble*) | 0,14 | 0,44 |
| notre modèle, par SC, en FP32 (pour comparaison) | 5,36 | 20,5 |

- **En énergie, la précision compte plus que le nombre d'opérations** : notre modèle en INT8 coûte 12 fois moins qu'en FP32.
- **M8K4** : dès la version par SC, moins que LMMSE + WMMSE ; version économique ≈ 1,5 fois LMMSE + RZF, pour +1,7 bps/Hz.
- **Massif** : 44 fois moins que LMMSE + WMMSE en version par SC ; la version économique passe **sous LMMSE + RZF** (0,44 contre 0,79 nJ), pour +4 bps/Hz.
- **Réserves** : (1) le fonctionnement en INT8 n'est pas encore validé (quantification à faire) ; (2) modèle d'énergie générique (pas une synthèse FPGA) ; (3) activations estimées en ordre de grandeur ; (4) classique compté en FP32 alors que notre WMMSE utilise des inversions en double précision (favorable au classique) ; (5) symétrie + interpolation pas encore entraînées ensemble, et ces variantes pas encore entraînées en massif.

### Réglage de taille et BASE FIGÉE (2026-10-31 ; `results/phase7/pilot_rand_{ph32,pd8,ed16,el1}.json`, `results/phase8/robustness_base.json`)
Un seul changement par essai à partir du modèle à pilote aléatoire ; règle fixée à l'avance : garder si la perte est ≤ 0,05 bps/Hz à 20 et 25 dB et si le modèle reste au-dessus du classique à pilote 0 dB. Écart au modèle d'origine, UMi / UMa / RMa :

| Réduction | 20 dB | 25 dB | MAC/SC | Décision |
|---|---|---|---|---|
| FFN du précodeur 64 → 32 | 0 / −0,01 / −0,01 | −0,02 / −0,02 / −0,03 | 1,27 M → 0,88 M | **retenue** |
| état d'une paire 16 → 8 | −0,03 / −0,03 / −0,03 | −0,06 / −0,08 / −0,08 | 0,65 M | écartée |
| largeur de l'encodeur 32 → 16 | −0,03 / −0,03 / −0,02 | −0,06 / −0,07 / −0,06 | 1,21 M | écartée |
| encodeur 2 → 1 couche | −0,07 / −0,08 / −0,06 | −0,15 / −0,16 / −0,14 | 1,22 M | écartée |

**Base figée** : encodeur fréquentiel (domaine des retards, grille 2×, 2 couches d'attention, largeur 32, filtre sur 3 retards) + transformer par paires (attention triangulaire, G réinjecté, 8 couches distinctes, état 16 complexes, FFN 32), sortie W = Ĥᴴ C, entraînement avec SNR pilote aléatoire U(0, 20) dB. **72,2 k paramètres, 0,88 M MAC/SC** (M8K4, FFT comptées en produits de matrices). Robustesse identique au modèle à pilote aléatoire (±0,01) : pilote 0 dB 22,40 / 22,22 / 24,21 contre 19,03 / 18,91 / 20,30 pour WMMSE robuste ; 5 ms de vieillissement 18,15 / 17,99 / 21,99 contre 17,38 / 17,18 / 20,71. Variante de coût : précodeur calculé sur 1 SC sur 3.

## Phase 10 — St-Catherine et quantification ART (chaîne BitAdapt du labo, 2026-11-01)

**Cadre** (`torch_port/`) : port PyTorch de la base figée (équivalence Keras 4e-5, `torch_port/check_equivalence.py`). Jeu St-Catherine du labo (lancer de rayons, mono-porteuse 2 GHz, M=64, K=4, 9 032 positions, SNR par utilisateur de −16 à +35 dB : puissances très inégales). **Précodeur seul, CSI parfait**, protocole et débit du labo (`rate_calculator_3d`, bruit 1e-13). Notre précodeur par paires (même architecture que la base, 44 k paramètres) entraîné de zéro sur St-Catherine avec leur perte (−débit). Test : 20 000 échantillons fixes (`torch_port/stecath.py`, `results/stecath/fp32_comparison.json`).

| 32 bits | params | débit moyen | 5e centile |
|---|---|---|---|
| RZF | — | 17,03 | 8,41 |
| WMMSE 30 it. / 300 it. | — | 24,41 / 26,88 | 13,28 / 13,37 |
| BitAdapt tiny / small / medium | 348 k / 825 k / 1 827 k | 23,04 / 24,12 / 24,45 | 11,61 / 12,10 / 12,22 |
| **notre précodeur par paires** | **44 k** | **26,99** | 12,98 |

- Avec des puissances très inégales, **RZF est mauvais (pas de répartition de puissance) et WMMSE converge lentement** (30 itérations : 24,4 ; 300 : 26,9). Notre précodeur atteint WMMSE convergé avec 40× moins de paramètres que leur « medium » (qui plafonne au niveau de WMMSE 30 it.).

**Modèle d'énergie** (Moons et al. 2017, constantes de Horowitz ISSCC 2014 — vérifiées : MAC entier 8 bits 0,23 pJ, 32 bits 3,2 pJ ⇒ 0,858 (Q/16)^1,9). Quatre erreurs corrigées dans `Pipeline/Energy.py` du labo (branche `fix/energy-model` poussée sur GitHub) : mémoire linéaire en Q (et non (Q/16)^1,9), parallélisme √p = √(64·16/Q) (dépendance inversée), MAC des couches denses × nombre de tokens, longueur de séquence de l'attention (c'était le lot = 1), appels avec (1, K, 2N) au lieu de (1, 64, 128). Effet sur « small » : 8 bits 0,637 → 1,734 µJ ; rapport 32/4 bits 47× → 24×. Notre version : `torch_port/energy.py`.

**Quantification ART** (quantificateurs LSQ et perte du labo, énergie corrigée ; `torch_port/art_stecath.py`, `results/stecath/energy_comparison.json`). Énergie par échantillon :

| | débit quantifié | énergie (µJ) |
|---|---|---|
| nous, λ=1 / 5 / 20 (couches à poids) | 24,71 / 24,48 / 23,39 | 0,236 / 0,173 / 0,129 |
| **nous, λ=5, produits sans poids aussi quantifiés** (G, triangulaire, W) | **23,62** | **0,113** |
| BitAdapt medium λ=1 / small λ=1 / tiny λ=1 | 24,10 / 23,24 / 21,80 | 0,811 / 0,378 / 0,154 |

- **Notre précodeur domine la courbe débit-énergie** : tout quantifié, 23,6 bps/Hz à 0,11 µJ, contre 23,2 à 0,38 µJ pour leur « small » (3,4× moins d'énergie) ; couches seules, λ=5 : mieux que leur « medium » pour 4,7× moins d'énergie.
- **Mais notre modèle perd plus à la quantification** (27,0 → 23,6–24,7, contre −0,4 à −0,9 pour les leurs) : probablement la dynamique des entrées (G) avec des utilisateurs à 50 dB d'écart. Piste : entrée en échelle logarithmique ou normalisation par utilisateur.
- Réserves : débits du labo = les leurs (leur jeu de test, même distribution) ; positions communes entre entraînement et test (leur protocole) ; G et W gardent 8 bits.

### Dynamique des entrées : G en log, ou canaux normalisés par utilisateur (`--inp log|norm`, `results/stecath/art_pair_h32_lam5_qprod_{log,norm}.json`)
Un seul changement, l'entrée du précodeur. « log » : G en module (dB) et phase. « norm » : chaque canal d'utilisateur normalisé (G = corrélations, |G| ≤ 1), puissance relative de chaque utilisateur (dB) donnée à part sur les paires, sortie W = h̃ᴴ C (même espace de canaux). ART λ = 5, tout quantifié :

| entrée | débit 32 bits | débit quantifié | énergie (µJ) |
|---|---|---|---|
| raw (base) | 26,99 | 23,62 | 0,113 |
| log | 27,09 | 23,62 | 0,101 |
| **norm** | **27,60** | **26,61** | 0,109 |
| *BitAdapt medium λ=1 / small λ=1* | *24,49 / 24,15* | *24,10 / 23,24* | *0,811 / 0,378* |
| *WMMSE 300 it. (32 bits)* | *26,88* | | |

- **La normalisation par utilisateur règle le problème de quantification** : perte de quantification 1,0 bps/Hz au lieu de 3,4, et le modèle 32 bits progresse aussi (+0,6). Le module en log seul ne suffit pas : c'est l'écart de puissance entre utilisateurs, pas la dynamique d'une entrée, qui gênait.
- **Tout quantifié (≈ 3,9 bits de poids, 5,8 d'activations ; G et W à 7–8 bits), 26,6 bps/Hz à 0,11 µJ** : +2,5 bps/Hz sur le meilleur modèle BitAdapt quantifié, pour 7,4× moins d'énergie, et au niveau de WMMSE convergé en 32 bits.
- En 32 bits, la variante normalisée dépasse WMMSE à 300 itérations (27,6 contre 26,9) : WMMSE reste dans un optimum local quand les puissances sont très inégales.
- **À reporter dans la base Sionna/OFDM** (entrée normalisée par utilisateur) : changement sans perte d'expressivité, qui rend le modèle robuste aux puissances inégales, le point de réalisme qui restait ouvert.

## Phase 11 — puissances inégales entre utilisateurs dans la base OFDM (2026-11-01)

**Cadre** (`experiments/phase11_power.py`, `results/phase11/power_spread.json`) : affaiblissement de parcours et masque 3GPP de Sionna gardés (`SystemConfig.relative_power`), canal normalisé par tirage : écart médian 28–35 dB entre le plus fort et le plus faible utilisateur d'un tirage (jusqu'à 84 dB). Sondage avec **contrôle de puissance montant complet** (pilotes au même SNR, 10 dB ; la BS remet chaque estimation à l'échelle) ; débit sur le vrai canal à puissances réelles. Base figée (FFN 32, pilote aléatoire), entrée d'origine ou normalisée par utilisateur (`GramPrecoder(inp="norm")`), entraînée à puissances égales ou réelles. M8K4, 192 tirages / scénario.

Débit à 20 dB (25 dB), UMi / UMa / RMa :

| | puissances **réelles** | puissances **égales** (test standard) |
|---|---|---|
| origine, entraînée à puissances égales | 16,51 / 16,68 / 16,87 | 25,84 / 25,44 / 26,25 |
| normalisée, entraînée à puissances égales | 15,85 / 16,10 / 15,87 | 25,82 / 25,42 / 26,26 |
| origine, entraînée à puissances réelles | 17,07 / 17,20 / 17,39 | 25,66 / 25,26 / 26,01 |
| **normalisée, entraînée à puissances réelles** | **17,08 / 17,19 / 17,41** (20,61 / 20,66 / 21,10) | 25,66 / 25,24 / 26,05 |
| WMMSE robuste 100 it. (LMMSE) | 16,22 / 16,26 / 16,38 (18,95 / 19,02 / 19,06) | 24,18 / 23,73 / 24,35 (30 it.) |
| WMMSE robuste 30 it. (LMMSE) | 15,96 / 16,04 / 16,15 | |
| RZF (LMMSE) | 9,82 / 9,88 / 9,93 | |

- **Avec des puissances réelles, le modèle garde l'avantage** : +0,86 / +0,93 / +1,03 bps/Hz sur WMMSE robuste convergé à 20 dB, +1,7 à +2,0 à 25 dB. L'avance est plus petite qu'à puissances égales (+1,7) : les utilisateurs faibles (−25 dB médian) pèsent peu dans le débit somme, et l'estimation y compte moins.
- **Ce qui compte, ce sont les données d'entraînement** : entraîner à puissances réelles apporte +0,5 à +0,6 sur ce test, pour −0,2 sur le test à puissances égales (−0,3 à −0,4 à 25 dB). L'entrée normalisée seule n'apporte rien en virgule flottante (et nuit si le modèle n'a vu que des puissances égales).
- **RZF s'effondre** (9,8) : sans répartition de puissance, il sert mal des utilisateurs aussi inégaux ; WMMSE a besoin de plus d'itérations (100 : +0,25 sur 30).
- **Décision proposée** : nouvelle base = entrée normalisée (même débit en virgule flottante, et elle réduit fortement la perte de quantification, phase 10) **entraînée à puissances réelles**. Le léger recul à puissances égales (−0,2) pourrait se récupérer avec un entraînement sur un mélange des deux ; à tester si besoin.

### BASE FIGÉE (version 2, 2026-11-01)
Validée par l'utilisateur. Encodeur fréquentiel (domaine des retards, grille 2×, 2 couches d'attention, largeur 32, filtre sur 3 retards) + transformer par paires (attention triangulaire, G réinjecté, 8 couches distinctes, état 16 complexes, FFN 32) avec **entrée normalisée par utilisateur** (G = corrélations, puissance relative de chaque utilisateur en dB sur les paires, W = h̃ᴴ C). Entraînement : SNR pilote aléatoire U(0, 20) dB, **puissances réelles entre utilisateurs** (affaiblissement de parcours et masque 3GPP, contrôle de puissance montant), mélange UMi + UMa + RMa. Poids : `/export/tmp/sala/freqctx/weights/phase7/pilot_rand_ph32_norm_relpow.weights.h5` (srvgrm12). 72 k paramètres, 0,88 M MAC/SC (M8K4). Débit à 20 dB : 17,08 / 17,19 / 17,41 à puissances réelles (WMMSE robuste convergé 16,22 / 16,26 / 16,38) ; 25,66 / 25,24 / 26,05 à puissances égales.

## Phase 12 — que fait l'encodeur ? Trois figures du masque (2026-11-01, sans entraînement)

**Cadre** (`experiments/phase12_mask.py`, `results/phase12/mask.json`, `mask_examples`, `mask_vs_pilot`, `mask_kernel`) : base v2, puissances réelles + contrôle de puissance montant, 64 tirages / scénario (graines de la phase 11), pilote 0 à 20 dB. L'encodeur est **linéaire en les pilotes une fois ses coefficients choisis** (g_hat[t] = c0 g[t] + c−1 g[t−1] + c+1 g[t+1], vérifié : noyau × pilotes = sortie de l'encodeur à 8·10⁻⁷ près) ; seuls les coefficients dépendent des données. Comparaison à un **Wiener oracle diagonal** sur la même grille, w(τ) = P(τ)/(P(τ)+σ²) (profil sans bruit du vrai canal ; *ce n'est pas le MMSE exact* : la grille 2× a des retards corrélés et de la fuite), et au noyau LMMSE (covariance du mélange). L'échelle de |c0| est libre (×50 à ×130 : le précodeur normalise chaque canal) ; elle est ramenée à 1 en moyenne sur les retards à plus de 20 dB de SNR.

| pilote 10 dB | UMi | UMa | RMa |
|---|---|---|---|
| retards gardés (Σ gain²), appris / Wiener oracle | 49 / 65 | 61 / 69 | 43 / 57 |
| corrélation masque appris – Wiener | 0,61 | 0,57 | 0,41 |
| largeur du noyau de la SC centrale (pilotes utilisés) : appris / Wiener / LMMSE | **16 / 5 / 3** | **14 / 5 / 3** | **25 / 6 / 3** |

- **Le masque garde les retards des trajets et coupe le plancher de bruit** (≈ 0 contre ≈ 0,15 pour le Wiener diagonal) ; il retrouve un trajet isolé à 5 µs en UMi. Il coupe plus tôt que le Wiener les queues de fuite sous le bruit (UMa, RMa). Pic inexpliqué sur le premier retard (≈ 1,6), à regarder avec les coefficients voisins c±1.
- **Il s'adapte au bruit sans recevoir le SNR pilote** : le seuil (masque = 0,5) passe d'environ 7 dB de SNR par retard (pilote 0 dB) à environ 17 dB (pilote 20 dB). En niveau absolu, il baisse donc de 10 dB quand le bruit baisse de 20 dB : l'adaptation est réelle mais **moins que proportionnelle**, et le masque est **toujours plus sévère que le Wiener diagonal** (seuil 0 dB). Retards gardés : 28 → 70 (UMi) de 0 à 20 dB, contre 25 → 176 pour le Wiener.
- **Le résultat le plus lisible est en fréquence** : l'encodeur moyenne sur bien plus de pilotes que LMMSE (14 à 25 contre 3, soit 56 à 100 SC contre 12), **plus large sur RMa** (canal plat) qu'en ville, et **plus étroit quand le pilote est bon** (UMi 29 → 10 pilotes de 0 à 20 dB). Le LMMSE à covariance fixe moyenne sur ~3 pilotes dans tous les scénarios : c'est ce qui lui coûte sur RMa.
- **Lecture** : un filtre de Wiener dans le domaine des retards dont le profil est estimé tirage par tirage, mais réglé pour le débit (la direction), pas pour l'erreur quadratique : plus sélectif en retard, donc plus lissant en fréquence.
- *Limites* : |c0| seul (sans c±1) ; Wiener diagonal et non le génie exact par lien de la phase 4 ; normalisation de l'échelle choisie par nous.

### Wiener dans le domaine des retards en formule fermée — RETENU comme référence classique, ÉCARTÉ comme encodeur (`results/phase12/wiener.json`, `results/phase7/wiener_k4_ph32_norm_relpow.json`)
- **Question** : combien l'attention entre retards ajoute-t-elle par-dessus un Wiener classique sur la même grille ?
- **Estimateur** (`models/delay_domain.WienerDelayEstimator`, aucun paramètre appris) : même IDFT sur la grille 2× (306 retards) ; P(τ) = puissance moyenne sur les 8 antennes, lissée sur 3 retards ; bruit σ̂² = médiane de P sur les retards (÷ 0,959) ; gain = max(P − κσ̂², 0)/P ; DFT. Seuil κ choisi sur des tirages d'**entraînement** (WMMSE 30 it., 20 dB) : κ = 0,5 / 1 / 2 / **4** / 8 / 16 → 15,91 / 16,64 / 16,87 / **16,91** / 16,79 / 16,46. Le seuil optimal est **sévère** (4σ²), comme le masque appris (phase 12, figures).
- **Réglages** : cadre de la phase 11 (puissances réelles, contrôle de puissance montant, pilote 10 dB, mêmes 192 tirages / scénario ; la base v2 y redonne exactement 17,08 / 17,19 / 17,41). « Wiener + notre précodeur » : variante `--variant wiener --kappa 4` de `phase7_context` (même entraînement que la base v2, seul l'encodeur change ; seul le précodeur apprend).

Débit à 20 dB (25 dB), UMi / UMa / RMa :

| | 20 dB | 25 dB | params | MAC/SC |
|---|---|---|---|---|
| LMMSE (mélange) + WMMSE robuste 100 it. | 16,22 / 16,26 / 16,38 | 18,95 / 19,02 / 19,06 | | |
| **Wiener formule fermée + WMMSE 100 it.** | **16,73 / 16,77 / 17,11** | **19,59 / 19,62 / 20,22** | 0 | |
| Wiener formule fermée + RZF | 10,64 / 10,65 / 11,11 | 12,11 / 12,20 / 12,57 | 0 | |
| Wiener formule fermée + notre précodeur | 16,84 / 16,90 / 17,22 | 20,16 / 20,16 / 20,69 | 44,6 k | 0,77 M |
| **base v2 (encodeur appris)** | **17,08 / 17,19 / 17,41** | **20,61 / 20,66 / 21,10** | 72,3 k | 0,88 M |

Égalité à 0 et 10 dB à ±0,05 près entre les quatre lignes non-RZF (5,15–5,21 ; 10,25–10,51). À puissances égales (test standard de `phase7_context`), Wiener + notre précodeur : 25,16 / 24,69 / 25,74 contre 25,66 / 25,24 / 26,05 pour la base v2.

**Décomposition de l'avance de la base v2 sur LMMSE + WMMSE robuste** (20 dB : 0,86 / 0,93 / 1,03 ; 25 dB : 1,66 / 1,64 / 2,04) :
- **Wiener sur la grille des retards (classique)** : +0,51 / +0,51 / +0,73 à 20 dB (55 à 71 % de l'avance) ; +0,64 / +0,60 / +1,16 à 25 dB.
- **notre précodeur au lieu de WMMSE** (même Wiener) : +0,11 / +0,13 / +0,11 à 20 dB ; +0,57 / +0,54 / +0,47 à 25 dB.
- **attention entre retards au lieu du Wiener** : +0,24 / +0,29 / +0,19 à 20 dB (18 à 31 %) ; +0,45 / +0,50 / +0,41 à 25 dB.

**Conclusions :**
- **L'essentiel de ce que fait l'encodeur s'explique par un Wiener dans le domaine des retards** (grille 2×, profil et bruit estimés sur le tirage, seuil sévère). C'est le résultat d'interprétabilité attendu.
- **L'attention apporte en plus +0,2 à +0,3 à 20 dB et +0,4 à +0,5 à 25 dB**, pour 28 k paramètres et 0,11 M MAC/SC (13 % du calcul). Elle reste dans la base (bien au-delà de la règle des 0,05 bps/Hz).
- **Conséquence importante pour l'article A** : ce Wiener + WMMSE est une **référence classique nettement plus forte que LMMSE + WMMSE robuste**. Face à elle, l'avance de la base v2 tombe à **+0,35 / +0,42 / +0,30 à 20 dB (+2 %)** et **+1,0 / +1,0 / +0,9 à 25 dB (+5 %)**, au lieu de +0,9 à +1,0 (20 dB). Il faut la mettre dans le tableau principal (un relecteur la construirait).
- *Limites* : une seule graine ; robustesse (pilote 0 dB, vieillissement) et puissances égales pas encore évaluées pour la référence Wiener + WMMSE ; κ choisi sur la seule configuration pilote 10 dB.

### Robustesse de la référence Wiener face à la base v2 (`experiments/phase12_robust.py`, `results/phase12/robust.json`)
Sans réentraîner. Cadre de la phase 11 (puissances réelles, contrôle de puissance montant), 192 tirages / scénario ; SNR pilote sur les tirages de la phase 11, vieillissement sur de nouveaux tirages (graines 5026+10i, pilote 10 dB), puissances égales sur les tirages standard (graines 2026+10i, sondage refait : la base v2 y fait 25,39 / 25,37 / 26,02, au lieu de 25,66 / 25,24 / 26,05 dans `phase7_context`, autre tirage du bruit). Méthodes : base v2 / Wiener + notre précodeur / **Wiener + WMMSE 100 it.** / LMMSE + WMMSE robuste 100 it.

Débit à 20 dB de données, UMi / UMa / RMa :

| | base v2 | Wiener + notre précodeur | Wiener + WMMSE | LMMSE + WMMSE robuste | **v2 − (Wiener + WMMSE)** |
|---|---|---|---|---|---|
| pilote 0 dB | 15,48 / 15,33 / 16,30 | 15,17 / 14,90 / 16,09 | 14,96 / 14,68 / 15,88 | 13,96 / 13,74 / 14,35 | **+0,52 / +0,65 / +0,42** |
| pilote 5 dB | 16,50 / 16,51 / 17,03 | 16,20 / 16,14 / 16,81 | 16,03 / 15,96 / 16,65 | 15,23 / 15,18 / 15,49 | +0,47 / +0,55 / +0,38 |
| pilote 10 dB | 17,08 / 17,19 / 17,41 | 16,83 / 16,89 / 17,22 | 16,72 / 16,77 / 17,11 | 16,20 / 16,27 / 16,39 | +0,36 / +0,42 / +0,30 |
| pilote 15 dB | 17,37 / 17,56 / 17,58 | 17,15 / 17,30 / 17,41 | 17,08 / 17,22 / 17,34 | 16,85 / 16,99 / 16,98 | +0,29 / +0,34 / +0,24 |
| pilote 20 dB | 17,50 / 17,72 / 17,66 | 17,29 / 17,47 / 17,51 | 17,24 / 17,41 / 17,45 | 17,24 / 17,43 / 17,33 | +0,26 / +0,31 / +0,21 |
| vieilli 0 ms | 17,22 / 17,58 / 17,01 | 16,97 / 17,28 / 16,82 | 16,88 / 17,14 / 16,74 | 16,36 / 16,62 / 16,03 | +0,34 / +0,44 / +0,27 |
| vieilli 2,5 ms | 15,29 / 15,63 / 15,84 | 15,19 / 15,51 / 15,77 | 14,97 / 15,26 / 15,55 | 14,70 / 14,94 / 15,09 | +0,32 / +0,37 / +0,29 |
| vieilli 5 ms | 13,47 / 13,67 / 14,72 | 13,45 / 13,65 / 14,71 | 13,17 / 13,30 / 14,43 | 12,97 / 13,08 / 14,07 | +0,30 / +0,37 / +0,29 |
| puissances égales | 25,39 / 25,37 / 26,02 | 24,89 / 24,82 / 25,71 | 24,99 / 24,89 / 25,80 | 23,95 / 23,95 / 24,36 | +0,40 / +0,48 / +0,22 |

À 25 dB de données, l'avance de la base v2 sur Wiener + WMMSE est d'environ le double : +1,16 / +1,24 / +0,97 (pilote 0 dB), +1,02 / +1,07 / +0,89 (10 dB), +0,89 / +0,92 / +0,67 (20 dB), +0,78 / +0,88 / +0,51 (puissances égales). À 0 dB de données, toutes les méthodes sont à égalité (±0,03).

**Conclusions :**
- **La base v2 reste au-dessus de Wiener + WMMSE partout** (tous les SNR pilote, tous les retards, puissances égales ou réelles), de **+0,2 à +0,65 bps/Hz à 20 dB** et **+0,5 à +1,2 à 25 dB**. L'avance grandit quand le pilote est mauvais et quand le SNR de données monte.
- **Le Wiener en formule fermée est robuste** : il bat LMMSE + WMMSE robuste partout sauf avec un très bon pilote (20 dB : égalité), et de +1,0 à +1,5 avec un pilote à 0 dB. C'est la référence classique à mettre dans l'article.
- **L'apport de l'attention disparaît avec le vieillissement** : à 5 ms, Wiener + notre précodeur égale la base v2 (±0,02) ; l'avance restante sur Wiener + WMMSE (+0,3) y vient alors du précodeur appris. À pilote 0 dB, l'attention apporte au contraire le plus (+0,21 à +0,43).
- **À puissances égales, Wiener + WMMSE dépasse légèrement Wiener + notre précodeur** (−0,07 à −0,10) : ce précodeur a été entraîné à puissances réelles seulement.

### Encodeur sans softmax : convolution sur les retards, ou attention linéaire (`experiments/phase12_mix.py`, `results/phase12/encoder_mix.json`, `results/phase7/pilot_rand_ph32_em{conv,linear}_norm_relpow.json`)
- **Question** : l'apport de l'attention softmax entre retards (+0,2 à +0,4 sur le Wiener) vient-il de l'interaction entre retards ? Peut-on le garder avec un bloc plus simple en HLS ?
- **Un seul changement** par rapport à la base v2 (même entraînement : pilote aléatoire, puissances réelles, 80 000 pas, graine 0), dans les 2 couches de l'encodeur : « conv » = convolution sur 7 retards voisins + moyenne globale sur les retards (`Block(mixer="conv")`) ; « linéaire » = attention sans softmax (scores / T ; calculable en q(kᵀv), coût linéaire en T).
- Évaluation : cadre de la phase 11 (puissances réelles), mêmes tirages que `phase12_robust` ; puissances égales : test standard de `phase7_context`.

Débit à 20 dB (25 dB), UMi / UMa / RMa :

| | params | MAC/SC | pilote 10 dB | pilote 0 dB | puissances égales (20 dB) |
|---|---|---|---|---|---|
| base v2 (softmax) | 72,3 k | 0,88 M | 17,08 / 17,19 / 17,40 (20,60 / 20,66 / 21,10) | 15,48 / 15,33 / 16,30 | 25,66 / 25,24 / 26,05 |
| **convolution + moyenne** | 82,4 k | 0,82 M | **17,14 / 17,30 / 17,44** (20,73 / 20,87 / 21,17) | **15,62 / 15,50 / 16,36** | **25,83 / 25,44 / 26,14** |
| attention linéaire | 72,3 k | 0,81 M | 17,09 / 17,22 / 17,42 (20,63 / 20,72 / 21,13) | 15,58 / 15,49 / 16,36 | 25,71 / 25,29 / 26,12 |
| Wiener formule fermée + WMMSE | 0 | | 16,73 / 16,78 / 17,10 | 14,96 / 14,68 / 15,88 | — |

**Conclusions :**
- **Le softmax n'est pas nécessaire.** Les deux variantes égalent ou dépassent légèrement la base : convolution +0,04 à +0,11 (pilote 10 dB), +0,06 à +0,17 (pilote 0 dB), +0,09 à +0,20 (puissances égales) ; linéaire +0,01 à +0,16. L'avance sur le Wiener en formule fermée est conservée (+0,3 à +0,8 à 20 dB).
- **Ce qui compte, c'est l'interaction entre retards** (un MLP par retard, phase 7, perdait 0,17), pas la forme softmax : une interaction locale (7 retards, la fuite d'un trajet) plus une statistique globale (le niveau de bruit) suffit.
- **Pour le HLS, la convolution est le bloc le plus simple** (MAC seulement, aucune exponentielle ni division, noyau court), avec 7 % de calcul en moins ; +10 k paramètres (noyaux de 7 × 32 × 32).
- *Limites* : une seule graine (écarts de +0,05 à +0,2, au-dessus de la dispersion entre graines mesurée ailleurs, ±0,01 à 0,03, mais à confirmer) ; pas encore évalué en vieillissement ni en massif.
- **Proposition** : encodeur à convolution comme candidat de base v3, après une 2e graine.

## Phase 13 — canaux jamais vus : lancer de rayons (2026-11-25, exploration)

### DeepMIMO Boston5G (28 GHz), sans réentraîner (`experiments/phase13_deepmimo.py`, `results/phase13/deepmimo_boston.json`)
- Trajets DeepMIMO v2 (angle de départ, phase, retard, puissance ; 105 842 positions avec au moins un trajet, 25 trajets au plus) → H(f) sur nos 612 SC, même réseau BS (4 colonnes, 2 polarisations ; la 2e polarisation reçoit une phase aléatoire par trajet, le lancer de rayons ne donnant pas la polarisation), retards relatifs au premier trajet. 192 tirages de 4 positions au hasard. Mêmes modèles qu'à la phase 12 ; LMMSE avec la covariance 3GPP du mélange (mal adaptée ici, comme en pratique).

Débit à 20 dB :

| | base v2 | encodeur conv | Wiener + notre préc. | Wiener + WMMSE | LMMSE + WMMSE robuste | WMMSE CSI parfait |
|---|---|---|---|---|---|---|
| puissances réelles, pilote 0 dB | 16,97 | **17,11** | 16,79 | 16,60 | 14,67 | 18,48 |
| puissances réelles, pilote 10 dB | 18,09 | **18,17** | 17,87 | 17,80 | 17,01 | 18,48 |
| puissances égales, pilote 0 dB | 23,28 | **23,43** | 22,82 | 22,70 | 19,33 | 25,21 |
| puissances égales, pilote 10 dB | 24,58 | **24,67** | 24,37 | 24,49 | 23,10 | 25,21 |

- **Le modèle entraîné sur TR 38.901 tient sur un canal de lancer de rayons jamais vu** : toujours au-dessus de Wiener + WMMSE (+0,1 à +0,6), à 0,3–0,4 du WMMSE en CSI parfait au pilote 10 dB. L'encodeur à convolution reste le meilleur.
- **LMMSE à covariance fixe s'effondre** (−2 à −4 bps/Hz au pilote 0 dB) : la covariance 3GPP ne correspond pas à ce canal. Les méthodes dans le domaine des retards (Wiener, encodeur appris) n'ont pas ce problème.
- *Limites* : 28 GHz (et non 3,5), polarisation émulée, canaux à peu de trajets (favorables au domaine des retards), une graine.

### Données du labo avec trajets (MIMO GenEE, https://mimogenee.grm.polymtl.ca)
- Le scénario « Raytracing with mobility » (St-Catherine, Décarie, Ericsson) est téléchargeable (`/download/<site>_64x8x8_raytracing_time_series.csv`, copie dans `/export/tmp/sala/freqctx/data/lab/`). Le fichier enchaîne trois tables : 1 000 trajectoires (départ x, y), leurs vitesses (2 500 pas chacune), et une **grille de 91 400 cellules (i, j) avec 1 à 11 trajets chacune** : `los, tau_s, ux, uy, alpha_re, alpha_im, psi_1, psi_2`. Ces trajets suffisent pour construire des canaux OFDM sur les sites du labo, avec mobilité.
- **Les canaux sont presque plats sur 20 MHz** : 2,3 à 3,5 trajets par cellule, étalement des retards médian 51 ns (St-Catherine), 2 ns (Décarie), 1 ns (Ericsson), contre ~400 ns en 3GPP.

### Sites du labo en OFDM, sans réentraîner (`experiments/phase13_lab.py`, `results/phase13/lab_<site>_<conv>.json`, figure `results/phase12/fig_raytracing.png`)
Même chaîne que Boston (puissances réelles, 192 tirages de 4 cellules, une graine). Convention des angles inconnue : testée sur St-Catherine avec trois hypothèses. Débit à 20 dB, pilote 0 dB / 10 dB :

| | encodeur conv | base v2 | Wiener + notre préc. | Wiener + WMMSE | LMMSE + WMMSE robuste | WMMSE CSI parfait |
|---|---|---|---|---|---|---|
| St-Catherine, psi2az | **20,67 / 21,81** | 20,44 / 21,70 | 19,82 / 21,39 | 19,65 / 21,37 | 17,12 / 20,25 | 22,10 / 22,10 |
| St-Catherine, psi1az | **17,16 / 18,19** | 16,84 / 17,96 | 16,38 / 17,74 | 16,34 / 18,11 | 14,23 / 17,12 | 18,85 / 18,85 |
| St-Catherine, angles aléatoires | **21,18 / 22,47** | 21,05 / 22,40 | 20,36 / 22,02 | 20,26 / 22,04 | 17,65 / 20,91 | 22,79 / 22,79 |
| Décarie, psi2az | **21,69 / 22,59** | 21,54 / 22,52 | 20,92 / 22,22 | 20,77 / 22,21 | 17,80 / 20,95 | 22,82 / 22,82 |
| Ericsson, psi2az | **23,40 / 24,14** | 23,25 / 24,06 | 22,44 / 23,70 | 22,38 / 23,78 | 18,88 / 22,36 | 24,38 / 24,38 |

- **Même classement sur les trois sites et les trois conventions d'angles** : encodeur conv > base v2 > Wiener + WMMSE ≫ LMMSE. La convention change le niveau absolu (géométrie entre utilisateurs), pas la conclusion.
- **En % du WMMSE en CSI parfait** : notre modèle 93 à 96 % (pilote 0 dB) et 98,5 à 99 % (10 dB) ; Wiener + WMMSE 89 à 92 % et 96 à 97,5 % ; LMMSE 77 à 79 % et 91 à 92 %.
- **L'avance de l'apprentissage se concentre au pilote faible** (+0,8 à +1,0 sur Wiener + WMMSE à 0 dB ; +0,1 à +0,4 à 10 dB). Exception : St-Catherine psi1az à 10 dB, où Wiener + WMMSE (18,11) passe la base v2 (17,96), mais pas l'encodeur conv (18,19).
- **LMMSE à covariance 3GPP s'effondre** sur ces canaux plats (−4 à −5 bps/Hz au pilote 0 dB) : sa covariance est fausse ici.
- *Limites* : canaux presque plats (cas facile pour le domaine des retards ; ne teste pas vraiment la sélectivité), angles supposés, une graine.

### Compléments sur les sites du labo : CSI parfait, RZF, covariance estimée sur le site (`--extra`, `results/phase13/lab_<site>_psi2az_extra.json`)
Débit à 20 dB, pilote 0 dB / 10 dB (St-Catherine / Décarie / Ericsson) :

| | St-Catherine | Décarie | Ericsson |
|---|---|---|---|
| encodeur conv (depuis les pilotes) | **20,67 / 21,81** | **21,69 / 22,60** | **23,40 / 24,13** |
| Wiener + WMMSE | 19,65 / 21,36 | 20,77 / 22,22 | 22,38 / 23,77 |
| **LMMSE (covariance estimée sur le site) + WMMSE robuste** | 18,55 / 20,79 | 20,42 / 22,11 | 22,17 / 23,65 |
| LMMSE (covariance 3GPP) + WMMSE robuste | 17,12 / 20,26 | 17,80 / 20,95 | 18,88 / 22,36 |
| Wiener + RZF | 13,86 / 15,50 | 16,15 / 17,50 | 19,19 / 20,57 |
| **notre précodeur, CSI parfait** | 22,09 | 22,80 | 24,32 |
| WMMSE 100 it., CSI parfait | 22,10 | 22,82 | 24,38 |
| RZF, CSI parfait | 16,28 | 18,11 | 21,20 |

- **Notre précodeur, entraîné sur 3GPP, égale WMMSE en CSI parfait sur les sites du labo** (à 0,01–0,06 près), sans réentraîner. RZF y perd 3 à 6 bps/Hz (pas de répartition de puissance entre utilisateurs inégaux).
- **La covariance estimée sur le site rend LMMSE bien meilleur** (+1,4 à +3,3 au pilote 0 dB) : la référence 3GPP était injuste ici. Il reste sous Wiener + WMMSE et sous notre modèle (−1,0 à −2,1 au pilote 0 dB ; −0,5 à −1,0 au pilote 10 dB).
- **Notre modèle depuis les pilotes est à 0,2–0,3 du WMMSE en CSI parfait au pilote 10 dB.**

### WMMSE selon le nombre d'itérations, et sondage sans Sionna (`experiments/phase13_checks.py`, `results/phase13/checks.json`)
**CSI parfait, puissances réelles, débit à 20 dB** (aucune fonction Sionna dans ce calcul : canal construit en numpy, précodeurs appliqués directement) :

| | notre précodeur | WMMSE 5 it. | 10 it. | 30 it. | 100 it. | 300 it. | RZF |
|---|---|---|---|---|---|---|---|
| St-Catherine | 22,09 | 20,53 | 21,17 | 21,76 | 22,10 | 22,19 | 16,28 |
| Décarie | 22,80 | 21,07 | 21,66 | 22,39 | 22,82 | 22,91 | 18,11 |
| Ericsson | 24,32 | 23,02 | 23,48 | 24,08 | 24,38 | 24,44 | 21,20 |
| Sionna UMi (3GPP) | 17,65 | 15,55 | 16,24 | 17,12 | 17,60 | 17,69 | — |

- **Notre précodeur (une passe, sans inversion) vaut WMMSE à ~100 itérations** partout ; WMMSE convergé (300 it.) fait +0,04 à +0,1 de plus. À 30 itérations (réglage courant), WMMSE est 0,2 à 0,5 en dessous. On ne dépasse pas WMMSE convergé en CSI parfait ici ; l'intérêt est le coût (une passe contre ~100 itérations avec inversions K×K).

**Sondage sans Sionna** : LS sur le peigne refait à la main (vrai canal aux SC pilotes + bruit gaussien). **Découverte de convention** : le sondeur Sionna (pilotes « kronecker ») amplifie l'énergie des pilotes ×K = ×4 (+6 dB), car chaque utilisateur n'occupe qu'une SC sur 4 à puissance totale égale : variance du bruit LS mesurée 0,25 au « pilote 0 dB » (0,025 à 10 dB). Avec la même amplification, le sondage à la main redonne les débits Sionna au centième (St-Catherine 20,69 / 21,81 contre 20,70 / 21,81 ; Ericsson 23,39 / 24,13 contre 23,41 / 24,14). **Sionna n'ajoute rien de caché** ; mais **tous les « SNR pilote » du journal sont des SNR par RE avant l'amplification de 6 dB du peigne** — à écrire dans l'article.

### CSI parfait : précodeur calculé sur 1 SC sur s, W interpolé (`experiments/phase13_interp.py`, `results/phase13/interp.json`, figure `results/phase12/fig_interp.png`)
Débit à 20 dB, puissances réelles, notre précodeur (non réentraîné pour l'interpolation) / WMMSE 100 it. avec la même interpolation :

| s | 1 | 3 | 6 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|
| Sionna UMi (~400 ns) | 17,65 / 17,60 | 17,61 / 17,56 | 17,43 / 17,38 | 16,87 / 16,81 | 15,61 / 15,52 | 13,68 / 13,57 |
| St-Catherine (51 ns) | 22,09 / 22,10 | 22,08 / 22,09 | 22,00 / 22,00 | 21,66 / 21,62 | 21,09 / 21,02 | 20,48 / 20,36 |
| Décarie (2 ns) | 22,80 / 22,82 | 22,80 / 22,82 | 22,79 / 22,82 | 22,76 / 22,80 | 22,64 / 22,67 | 22,31 / 22,32 |
| Ericsson (1 ns) | 24,32 / 24,38 | 24,32 / 24,38 | 24,31 / 24,37 | 24,28 / 24,35 | 24,16 / 24,23 | 23,87 / 23,94 |

- **La sélectivité fixe le coût du multiporteuse** : en 3GPP, 1 SC sur 3 (−0,2 %) mais déjà −4,5 % à 1 sur 12 ; sur les sites du labo presque plats, 1 SC sur 12 à 24 ne coûte presque rien (Décarie, Ericsson : −0,2 à −0,7 % à 1 sur 24). Un pas adapté à l'étalement des retards diviserait le calcul par 3 (3GPP) à 24 (labo).
- Notre précodeur suit WMMSE 100 it. à toutes les granularités (et le dépasse légèrement quand l'interpolation est grossière en 3GPP).

### Énergie de RZF et WMMSE dans le même modèle que les réseaux (`torch_port/energy_classical.py`, `results/stecath/classical_energy.json`, figure `results/presentation/figB_pareto_energy.png`)
St-Catherine (M=64, K=4, CSI parfait, protocole du labo, 20 000 échantillons). Modèle d'énergie corrigé (= `Pipeline/Energy.py` v5 de BitAdaptHLS, déjà corrigé : mémoire linéaire en Q, parallélisme 64·16/Q, tokens comptés). Opérations de RZF et WMMSE comptées dans leur forme efficace en dimension K (même précodeur que notre implémentation en dimension M), division ou racine = 10 MAC.

| | débit | MAC réelles | énergie 32 bits (µJ) | énergie 16 bits (µJ) |
|---|---|---|---|---|
| RZF | 17,03 | 9,0 k | 0,052 | 0,016 |
| WMMSE 1 / 10 / 30 it. | 19,01 / 22,77 / 24,41 | 18 k / 102 k / 287 k | 0,10 / 0,51 / 1,42 | 0,03 / 0,15 / 0,43 |
| WMMSE 100 / 300 it. | 25,92 / 26,88 | 0,93 M / 2,78 M | 4,62 / 13,7 | 1,39 / 4,13 |
| notre précodeur, 32 bits | 27,60 | — | 3,32 | |
| **notre précodeur, quantifié ART (entrée normalisée)** | **26,61** | | **0,109** | |
| BitAdapt medium, quantifié (λ=1) | 24,10 | | 0,81 | |

- **À débit égal (~26,6 bps/Hz), WMMSE demande ~200 itérations : ~9 µJ en 32 bits, ~2,8 µJ en 16 bits, soit 25 à 85 fois notre précodeur quantifié.** RZF est moins cher (0,016 à 0,05 µJ) mais perd 9,6 bps/Hz.
- **Honnêteté sur l'origine du gain** : à précision égale (32 bits), notre précodeur (3,3 µJ) coûte à peu près autant que WMMSE à ~70 itérations. L'avantage vient de ce que le réseau **tolère la quantification à 4–8 bits** (sans division ni inversion), alors que WMMSE (inversions, écarts de puissance de 50 dB) a besoin d'au moins 16 bits ; la courbe WMMSE 16 bits est déjà optimiste.
- *Limites du modèle* : générique (ASIC 45 nm, pas FPGA), ne compte pas finement les non-linéarités ; à valider par le HLS.

### ⚠ Vérification de la comparaison d'énergie avec BitAdapt (2026-11-25) — COMPARAISON NON VALIDE EN L'ÉTAT
Trois problèmes trouvés en recoupant avec l'export v5 envoyé à F. Leduc-Primeau le 2026-10-24 (`/export/tmp/sala/home_archive/ELE6310/ELE6310E/export_francois_v5_20261024/`) :
1. **Points BitAdapt périmés.** `torch_port/lab_energy_compare.py` lisait les checkpoints de l'article (dépôt mMIMO-ART-FPGA, ancien modèle d'énergie, FP32 80 epochs, architectures `FPGA_*`), pas la v5 (réentraînée, modèle corrigé, 6 tailles × 8 λ). En v5 sur St-Catherine, BitAdapt atteint 26,69 bps/Hz (xlarge, 14–28 µJ) ; à ~0,1 µJ, tiny λ=1 fait 20,61. Les tailles small/medium n'ont pas les mêmes dimensions dans les deux dépôts (v5 : small D128 FFN1024, medium D128 FFN2048).
2. **Deux implémentations du modèle d'énergie qui ne concordent pas** : à architecture identique (tiny, 8 bits), `torch_port/energy.py` donne 0,755 µJ et `Pipeline/Energy.py` v5 donne 1,049 µJ (×1,39). Les formules par couche sont les mêmes ; l'écart vient du comptage MAC / poids / activations par couche. Il faut calculer TOUS les modèles avec le même code.
3. **Énergie de WMMSE : conventions très différentes.** La v5 (`eval_baselines.py`) utilise E = K·I·N_c, N_c = éq. 12 du papier (terme dominant (8/3)·N_T³·N_U, c'est-à-dire une inversion M×M par utilisateur et par itération) calibré sur 296 µJ à I = 92,8 : 3,2 µJ à I = 1, 79,7 µJ à I = 25. La forme efficace (dimension K, identité push-through, même précodeur) compte ~9 k MAC par itération : 0,10 µJ à I = 1 en 32 bits (×33 de moins), 1,2 µJ à I = 25. **Les gains « 30 à 73× face à WMMSE » de la v5 reposent sur la forme coûteuse** ; avec la forme efficace, ils diminuent fortement (tiny λ=1 : 20,61 @ 0,103 µJ contre WMMSE I=1 efficace 19,0 @ 0,10 µJ en 32 bits).
- Débits : nos 20 000 échantillons (graine 2026) contre 200 000 en v5, même distribution ; WMMSE 30 it. 24,41 (nous) contre I = 25 23,86 (v5), cohérent.
- **Conséquence** : la figure `results/presentation/figB_pareto_energy` et les phrases « +10 % de débit, 7× moins d'énergie que BitAdapt » ne sont pas valides tant que (1)–(3) ne sont pas réglés. À faire : calculer l'énergie de notre précodeur avec `Pipeline/Energy.py` v5 (un seul code), utiliser le front v5, et montrer WMMSE avec les deux conventions.

**Corrections faites (2026-11-25)** :
- **Un seul code d'énergie** (`torch_port/energy_v5.py`, `results/stecath/energy_v5_ours.json`) : `Pipeline/Energy.py` v5 redonne exactement la colonne E8 de la v5 pour tiny (1,049 µJ, contrôle passé). Notre précodeur avec ce code : **0,113 µJ quantifié**, 0,315 µJ à 8 bits, 3,32 µJ en 32 bits (738 k MAC par échantillon). L'écart de l'ancien calcul venait des points BitAdapt (architectures `FPGA_*` différentes de la v5), pas de notre précodeur.
- **Même jeu de test que la v5** (`torch_port/eval_v5_testset.py`, `results/stecath/v5_testset_ours.json`, 200 000 canaux, chargeur `Data_Load` du dépôt) : notre précodeur **27,62 (32 bits) / 26,64 (quantifié)** ; contrôles RZF 17,04 (v5 : 17,14), notre WMMSE I = 1 / 5 / 10 / 25 = 19,03 / 21,64 / 22,76 / 24,13 (v5 : 19,29 / 21,53 / 22,56 / 23,86 ; v5 cherche μ par bissection, nous en forme fermée).
- **Figure B refaite** : front de Pareto BitAdapt v5 (non dominés sur 48 modèles), WMMSE sous les deux conventions.

- **Énergies BitAdapt v5 recalculées** (`torch_port/check_v5_energy.py`, `results/stecath/check_v5_energy.json`) avec `Energy.py` v5 à partir des bits par couche de l'export (`bitwidths_all_v5.csv`) : les 48 modèles de St-Catherine sont retrouvés à 5 % près au maximum (arrondis des bits de l'attention).
- **Conditions de notre quantification** (`torch_port/art_stecath.py`) : quantificateurs LSQ du code du labo (`quan.quantizer.lsq.LsqQuan`), bits initialisés à 8, bornes 1–8 (poids) et 2–8 (activations), comme le plancher v5 ; perte ART = −débit/débit₃₂ + λ·E/E₃₂ (λ = 5) pendant 20 000 pas (lots de 2 048), puis 5 000 pas avec bits arrondis et figés ; produits sans poids (G, produit triangulaire, W) quantifiés aussi et comptés dans l'énergie. Entraînement plus court que la v5 (QAT 120 epochs + phase 3). Bits obtenus : poids 1–6 bits (moyenne ~3,9), activations 2–8 (~5,8), G et W 7–8 bits. Données : St-Catherine, CSI parfait, M=64, K=4, mono-porteuse ; mêmes 9 032 positions pour l'entraînement et le test (protocole du labo).
- **Non comptés, pour les deux réseaux** : LayerNorm, GELU/ReLU, softmax (BitAdapt), normalisation par utilisateur (rsqrt, nous).

**Lecture de la figure corrigée (St-Catherine, CSI parfait)** :
- Avec la convention de l'article, tout réseau bat WMMSE de loin. **Avec la forme efficace, le front BitAdapt v5 suit à peu près la courbe WMMSE 32 bits, et WMMSE 16 bits le domine** : l'avantage de BitAdapt sur WMMSE dépend de la convention.
- **Notre précodeur quantifié (26,64 bps/Hz, 0,113 µJ) domine les deux** : BitAdapt v5 atteint ce débit à ~14–28 µJ (xlarge), WMMSE efficace vers ~200 itérations (~9 µJ en 32 bits, ~2,8 µJ en 16 bits, déjà optimiste). C'est le seul point qui se détache nettement, quelle que soit la convention.

### Massif M64K8 en CSI parfait, puissances réelles, sans réentraîner — NÉGATIF (`experiments/phase13_massive.py`, `results/phase13/massive_csi_parfait.json`)
Sionna UMi, 64 tirages, 600 SC. Débit à 20 dB (30 dB) : WMMSE 300 it. 39,84 (56,78) ; 100 it. 38,54 ; 30 it. 36,19 ; 10 it. 33,71 (47,61) ; RZF 24,50 (36,30) ; **notre précodeur (entraîné en M8K4) 29,00 (36,06)** ; précodeur de la phase 6 (entraîné sur M64K8 + M8K4 mais à puissances égales) 30,18 (35,02).
- **En massif à puissances réelles, nos précodeurs actuels décrochent** (73–76 % du WMMSE convergé à 20 dB, sous WMMSE 10 it.) et plafonnent au fort SNR. Ni l'un ni l'autre n'a vu ce cas : le premier n'a vu que M8K4, le second que des puissances égales. En phase 6 (puissances égales), le même précodeur égalait WMMSE en massif. **À faire avant toute figure massive : entraîner le précodeur sur M64K8 (+ M8K4) à puissances réelles, entrée normalisée.**

### Massif à puissances réelles : précodeur entraîné sur M64K8 + M8K4 — RETENU (`experiments/phase13_train_massive.py`, `results/phase13/train_massive.json`, figure `results/presentation/figA2_massif.png`)
Seul changement par rapport au banc de la phase 6 : les données (Sionna UMi + UMa + RMa à puissances réelles, 1 SC sur 3 ; 76 800 échantillons M64K8, 313 344 M8K4), entrée normalisée, FFN 32, 60 000 pas (~33 min). CSI parfait, mêmes 64 tirages que `phase13_massive` :

| M64K8, SNR | 0 | 10 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| **notre précodeur (puissances réelles)** | 13,50 | 24,67 | **40,62** | 50,19 | **60,67** |
| WMMSE 300 it. | 13,50 | 24,66 | 39,84 | 48,19 | 56,78 |
| WMMSE 30 it. | 13,47 | 23,62 | 36,19 | 43,20 | 50,81 |
| RZF | 10,22 | 16,04 | 24,50 | 29,74 | 36,30 |
| (avant : entraîné M8K4 seulement) | 12,22 | 20,31 | 29,00 | 32,89 | 36,06 |

- **Avec les bonnes données, un seul précodeur égale WMMSE convergé en massif et le dépasse au fort SNR** (+2 % à 20 dB, +7 % à 30 dB : WMMSE reste dans un optimum local ou converge lentement avec 8 utilisateurs de puissances très inégales). Le même réseau en M8K4 : 17,68 à 20 dB (WMMSE 300 it. : 17,69).
- L'échec précédent venait bien des données (configuration ou puissances jamais vues), pas de l'architecture.

### Massif depuis les pilotes, puissances réelles (`experiments/phase13_massive_pilots.py`, `results/phase13/massive_pilots.json`, panneau (d) de `figA_sumrate_snr`)
Encodeur à convolution + précodeur par paires (initialisé par le précodeur CSI parfait massif), appris de bout en bout ; M64K8, 600 SC, UMi + UMa + RMa, puissances réelles, contrôle de puissance montant, SNR pilote U(0, 20) dB, 256 tirages/scénario, 30 000 pas (~28 min). Évaluation : 64 tirages UMi (ceux du panneau CSI parfait), pilote 10 dB :

| SNR | 0 | 10 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| **notre modèle** | 13,48 | 24,49 | **39,34** | 47,36 | **55,24** |
| LMMSE (covariance massif, mélange) + WMMSE robuste 100 it. | 13,43 | 23,97 | 36,76 | 43,31 | 49,83 |
| Wiener formule fermée + WMMSE 100 it. | 13,45 | 24,13 | 36,44 | 41,83 | 46,61 |
| LMMSE + RZF | 10,13 | 15,37 | 21,38 | 24,43 | 28,04 |

- **En massif depuis les pilotes : +7 % à 20 dB, +11 % à 30 dB sur la meilleure chaîne classique**, à 1,3 bps/Hz du CSI parfait (40,62) à 20 dB.
- En massif, LMMSE + WMMSE dépasse Wiener + WMMSE au fort SNR (l'inverse du cas M8K4) : avec 64 antennes, la covariance moyenne suffit mieux, et le Wiener (κ réglé en M8K4) n'a pas été réajusté.

### Interpolation et bords de la bande (`experiments/phase13_interp2.py`, `results/phase13/interp2.json`)
- **Interpoler mieux n'apporte presque rien** (CSI parfait, 20 dB) : aligner la phase des colonnes de W avant d'interpoler, ou utiliser une spline, gagne +0,05 à 1 SC sur 12 en UMi (16,87 → 16,92–16,98) et +0,4 seulement à 1 sur 48 ; rien sur St-Catherine. La limite est la variation du canal entre ancres, pas la méthode d'interpolation. On garde l'interpolation linéaire.
- **Les bords de la bande perdent un peu** (depuis les pilotes, pilote 10 dB, UMi, 20 dB) : notre modèle 16,46 sur les 24 SC des bords contre 17,17 au milieu (−0,7) ; Wiener + WMMSE 15,13 contre 16,79 (−1,7). C'est l'effet de bord de l'IFFT sur une bande coupée net, moindre chez nous. Sur toute la bande, l'enjeu est ~0,03 bps/Hz : donner l'indice de SC (ou la distance au bord) ne vaut pas la peine pour l'instant.

### HLS, étape 1 : référence « golden » du précodeur quantifié (`hls/model.py`, `hls/golden.py`)
- `hls/model.py` (`QuantPairPrecoder`) extrait le modèle ART (St-Catherine, entrée normalisée, produits quantifiés) : poids entiers, échelles LSQ (par canal pour les poids), bits. Convention LSQ du labo : q = round(clamp(x/s, ±(2^b − 1))), donc **b + 1 bits signés en matériel**.
- `Golden` (flottant, opération par opération) redonne le modèle PyTorch quantifié : 26,643 contre 26,641 bps/Hz (2 000 canaux).
- `GoldenFixed` (arithmétique du matériel : MAC entiers exacts, échelles = entier 16 bits × 2^e, valeurs intermédiaires à F bits fractionnaires) — 5 000 canaux de St-Catherine, golden flottant 26,627 :

| F (bits fractionnaires) | 16 | 12 | 10 | 8 |
|---|---|---|---|---|
| débit | 26,627 | **26,612** | 26,591 | 26,211 |

- **Deux points de conception trouvés en chemin** : (1) la normalisation par utilisateur doit être en **virgule flottante par bloc** (un exposant par utilisateur : les puissances s'étalent sur ~50 dB ; en virgule fixe uniforme, le log de la puissance relative tombe à log 0) ; (2) une table GELU brute (pas 1/32) coûtait 0,18 bps/Hz ; **table de 257 valeurs avec interpolation linéaire** : plus de perte.
- **Choix : F = 12 bits fractionnaires** (−0,015 bps/Hz), ~11 bits entiers (max |x| intermédiaire ≈ 890), soit des valeurs intermédiaires sur ~24 bits ; les produits quantifiés restent sur b + 1 bits.

### Notre précodeur en plus petit : front de Pareto (St-Catherine, ART λ = 5) (`scripts/queue13_pareto.sh`, `results/stecath/art_pair_h*_L*_lam5_qprod_norm.json`, `energy_v5_ours.json`, `v5_testset_ours.json`)
Une seule valeur de λ par taille (λ = 5), produits quantifiés, entrée normalisée ; énergie avec `Energy.py` v5, débit sur le jeu de test v5 (200 000 canaux) :

| taille | param. | MAC | débit 32 bits | débit quantifié | énergie (µJ) |
|---|---|---|---|---|---|
| 8 couches, FFN 32 | 44 k | 738 k | 27,60 | 26,65 | 0,113 |
| 4 couches, FFN 32 | ~23 k | 375 k | 27,36 | 25,94 | 0,064 |
| 4 couches, FFN 16 | ~16 k | 274 k | 27,22 | 25,58 | 0,049 |
| 2 couches, FFN 16 | 8 k | 143 k | 26,39 | 24,13 | **0,026** |

- **La version à 2 couches passe sous l'énergie de RZF en 32 bits** (0,026 contre 0,052 µJ) avec 24,1 bps/Hz (RZF : 17,0), et fait mieux que WMMSE à 10 itérations (22,8 bps/Hz, 0,51 µJ) pour 20 fois moins d'énergie.
- La perte de quantification grandit quand le modèle rétrécit (1,0 → 2,3 bps/Hz) : peu de couches laisse peu de marge.

### HLS, étape 2 : générateur C++ et simulation C (`hls/intref.py`, `hls/codegen.py`, `hls/README.md`)
- `IntPairPrecoder` (référence entière, exactement les calculs du C++) : 26,622 bps/Hz contre 26,627 pour PyTorch quantifié (5 000 canaux).
- `codegen.py` (classes `ConstantsWriter`, `SourceWriter`, `TestbenchWriter`) écrit `weights.h`, `precoder.cpp`, `vectors.h`, `tb.cpp`, `run_csim.tcl` (projet dans `/export/tmp/sala/hls_work/pair_precoder`).
- **Simulation C Vitis HLS 2022.2 : PASS, identique bit à bit à la référence entière** (écart 0, cosinus 1,000000, 32 vecteurs St-Catherine).
- Version 1 : normalisation par utilisateur avant le bloc, racines en double, entiers C++. Avant la synthèse : racines en entiers, largeurs `ap_int`, pragmas, normalisation intégrée.
- **Première synthèse de la version 1** (srvgrm07, xcu280, 100 MHz, aucune directive : boucles séquentielles, entiers 32/64 bits ; rapport `/users/sala/Documents/test_projet/hls_work/pair_precoder_v1/csynth_v1.rpt`) : **latence 628 951 cycles = 6,3 ms par précodeur** (≈ 1 MAC par cycle pour 738 k MAC), Fmax estimée 137 MHz ; 240 DSP (2,7 %), 331 BRAM 18K (8 %, poids stockés sur 32 bits), 13,5 k FF, 41 k LUT (3 %). C'est la référence avant optimisation.
- **Ordre de grandeur de la cible** : 612 précodeurs par slot de 0,5 ms (205 avec 1 SC sur 3) → ~0,8 µs (2,4 µs) par précodeur, soit ~3 000 MAC par cycle pour le modèle à 8 couches et ~600 pour celui à 2 couches (143 k MAC) : faisable sur l'U280 (9 024 DSP), à condition de dérouler les boucles et de réduire les largeurs (b + 1 bits par produit, deux à quatre produits par DSP).
- **Synthèse de la version 2 (tout en entiers, sans directive)** : simulation C identique bit à bit à la référence entière ; latence 633 397 cycles (6,3 ms), 1 405 DSP (15 %), 332 BRAM, 52 k FF, 103 k LUT. Plus aucun flottant, mais les divisions 64 bits (réciproques de la normalisation et de la LayerNorm) coûtent cher en DSP : à remplacer par une inverse de racine par Newton (multiplications seulement). Les boucles à nombre de tours variable (racine, bit de tête) ont dû être réécrites à nombre de tours fixe pour que la latence soit bornée.
- **Version 2 de la référence entière (tout en entiers)** : entrée en virgule flottante par bloc (mantisses 16 bits, un exposant par utilisateur), racine entière exacte et réciproque entière pour la normalisation par utilisateur, |G|, LayerNorm et W ; log2 par table interpolée. Débit 26,620 (PyTorch quantifié 26,627) : plus aucune opération flottante, sans perte.

- **À confirmer avec Jérôme avant de s'en servir** : la convention des angles (psi_1, psi_2 : azimut et zénith de départ ? ux, uy : direction à l'arrivée ou vitesse ?), le pas de la grille, et le réseau d'antennes utilisé pour les 64 SC × 64 antennes du site.

## Phase 14 — suites de la réunion du 2026-11-25 (zero-shot, CSI imparfait)

### Zero-shot du précodeur entraîné sur 3GPP (`experiments/phase14_zeroshot.py --train existant`, `results/phase14/zeroshot_existant.json`)
Précodeur par paires entraîné sur M8K4 + M64K8 (UMi + UMa + RMa, puissances réelles), CSI parfait, testé sans réentraîner (mêmes tirages que `phase13_bench`). Débit (bps/Hz), 0 / 10 / 20 / 25 / 30 dB :

| site | notre précodeur | WMMSE 300 it. | WMMSE 30 it. |
|---|---|---|---|
| St-Catherine | 5,49 / 12,68 / 22,05 / 27,27 / 32,62 | 5,51 / 12,70 / 22,19 / 27,53 / 33,11 | 5,51 / 12,69 / 21,76 / 26,59 / 31,37 |
| Décarie | 5,41 / 12,73 / 22,83 / 28,41 / 34,00 | 5,45 / 12,79 / 22,91 / 28,62 / 34,39 | 5,45 / 12,77 / 22,39 / 27,45 / 32,58 |
| Ericsson | 5,44 / 13,58 / 24,31 / 30,05 / 35,77 | 5,47 / 13,64 / 24,44 / 30,38 / 36,37 | 5,47 / 13,62 / 24,08 / 29,55 / 35,05 |
| UMi (vu) | 5,16 / 10,35 / 17,68 / 22,09 / 26,93 | 5,17 / 10,36 / 17,69 / 22,05 / 26,60 | |

- Toujours au-dessus de WMMSE 30 it. ; écart à WMMSE convergé de 0,1 à 0,15 jusqu'à 20 dB et de 0,3 à 0,6 bps/Hz à 25–30 dB sur les sites jamais vus : c'est l'écart que le fine-tuning devra combler.

### CSI imparfait : erreur synthétique à l'entraînement, vrai estimateur au déploiement (`experiments/phase14_csi.py`, `results/phase14/csi_imparfait.json`)
Précodeur M8K4 initialisé par le précodeur CSI parfait, réentraîné 20 000 pas avec h_est = h + e (erreur gaussienne relative à la puissance de chaque utilisateur, de −25 à −5 dB). Déploiement : Sionna UMi, puissances réelles, 192 tirages, estimation LS + interpolation LMMSE (covariance du mélange) ou LS + interpolation linéaire ; tous les précodeurs reçoivent la même estimation. Débit à 20 dB de données (BER 16-QAM non codé entre parenthèses, approximation gaussienne à partir du SINR de chaque flux) :

| estimation, pilote | entraîné CSI parfait | entraîné erreur synthétique | WMMSE 100 it. (robuste pour LMMSE) | RZF |
|---|---|---|---|---|
| LMMSE, 0 dB | 13,96 (0,149) | **14,30** (0,168) | 13,98 (0,149) | 7,80 |
| LMMSE, 10 dB | 16,25 (0,143) | 16,22 (0,162) | 16,23 (0,142) | 9,83 |
| LMMSE, 20 dB | **17,34** (0,142) | 17,06 (0,162) | 17,28 (0,140) | 11,10 |
| linéaire, 0 dB | 10,88 (0,168) | **11,63** (0,184) | 10,66 (0,169) | 5,39 |
| linéaire, 10 dB | 15,29 (0,144) | **15,44** (0,164) | 15,13 (0,143) | 8,83 |
| linéaire, 20 dB | **17,17** (0,142) | 16,94 (0,162) | 17,06 (0,141) | 10,91 |

- **L'erreur synthétique aide quand l'estimation est mauvaise** (pilote faible, interpolation simple : +0,75 bps/Hz, +7 % avec LS linéaire à pilote 0 dB) **et coûte un peu quand elle est bonne** (−0,2 à −0,3 à pilote 20 dB) : le modèle est réglé pour une erreur moyenne. Piste : lui donner le niveau d'erreur (SNR pilote) en entrée, ou tirer l'erreur sur une plage plus réaliste.
- **Le précodeur entraîné en CSI parfait égale ou dépasse WMMSE (robuste) avec un vrai estimateur**, sans réentraîner.
- **BER** : les précodeurs qui maximisent le débit somme ont un BER moyen plus élevé (puissance concentrée sur les bons utilisateurs ; les flux faibles ont un SINR très bas). Le BER moyen sur les flux n'est pas l'objectif optimisé ; pour une figure BER, il faudra fixer la modulation par flux ou viser un autre objectif (débit minimal, MSE).

### Diagnostic du fine-tuning et raffinement WMMSE (`experiments/phase14_ft_diag.py`, `results/phase14/ft_diag_<site>.json`)
Débit à 30 dB (protocole du fine-tuning, M8K4) :
| site | zero-shot | oracle (toutes les données, 20 k pas) | oracle, SNR U(20,30) | de zéro, même budget | **réseau + 3 it. WMMSE** | WMMSE 300 it. |
|---|---|---|---|---|---|---|
| Ericsson | 35,32 | 35,77 | 35,85 | 29,85 | **36,33** | 36,19 |
| Décarie | 34,86 | — | — | — | **35,48** | 35,04 |
| St-Catherine | 33,28 | — | — | — | **33,87** | 33,56 |
- Pas un manque de données : l'oracle ne comble que la moitié de l'écart. Le départ 3GPP est essentiel (de zéro : 29,85). Pondérer le fort SNR aide peu (+0,05 à +0,2). Écart peu corrélé au conditionnement de G : limite diffuse du nombre d'« itérations » à fort SNR.
- **1 à 3 itérations de WMMSE initialisées par le réseau dépassent WMMSE 300 it.** (initialisé par RZF) sur les 3 sites : le réseau mène à un meilleur point stationnaire. Coût : ~3 itérations en forme Gram (O(K³) chacune). Recommandation : réseau + 1-2 it. WMMSE plutôt que fine-tuner ou agrandir.

### CSI imparfait : variance d'erreur par utilisateur en entrée (`experiments/phase14_csi_errin.py`, `GramPrecoder(err_in=True)`, `results/phase14/csi_errin.json`)
Erreur synthétique tirée par utilisateur, niveau fourni au réseau ; déploiement : variance donnée par l'estimateur Sionna. Débit à 20 dB, LMMSE :
| taille, pilote | CSI parfait | synth. (M8K4 seul) | **synth. + variance** | WMMSE robuste | RZF |
|---|---|---|---|---|---|
| M8K4, 0 dB | 13,96 | 14,30 | **14,32** | 13,98 | 7,80 |
| M8K4, 10 dB | 16,25 | 16,22 | **16,32** | 16,23 | 9,83 |
| M8K4, 20 dB | 17,34 | 17,06 | 17,30 | 17,28 | 11,10 |
| M64K8, 0 dB | 32,98 | 30,31 | **33,09** | 32,92 | 18,06 |
| M64K8, 10 dB | 37,42 | 33,01 | **37,42** | 36,77 | 21,41 |
| M64K8, 20 dB | 39,77 | 33,83 | 39,63 | 38,48 | 23,60 |
- La variance en entrée supprime le compromis (gain à pilote faible sans perte à pilote fort) et égale ou dépasse WMMSE robuste partout, y compris en massif (+0,65 à +1,15 à 10–20 dB) : Ĝ + variance par utilisateur suffit dans ces tests ; la restriction W = Ĥ^H C ne coûte rien de mesurable.
- Réserve : la colonne « synth. » n'a été entraînée qu'en M8K4 (zero-shot en massif) ; le témoin « synth. sans variance entraîné aussi en M64K8 » manque pour attribuer le gain massif à la seule variance.

### Énergie à Ericsson et WMMSE en forme Gram (`torch_port/*.py --site ericsson`, `results/stecath/classical_energy_ericsson.json`, `art_pair_*_ericsson.json`)
WMMSE en forme Gram : itération sur C (K x K) seulement, G une fois, W = H^H C à la fin ; identique à wmmse() (écart ~1e-7), coût par itération indépendant de M (`torch_port.stecath.wmmse_gram`). Énergie (µJ, Moons v5) :
| méthode | Ericsson débit / énergie | St-Catherine débit / énergie |
|---|---|---|
| RZF 32 / 16 bits | 33,44 / 0,052 / 0,016 | 17,03 / 0,052 / 0,016 |
| WMMSE 5 it. Gram, 32 / 16 bits | 35,45 / 0,088 / 0,028 | 21,62 / 0,088 / 0,028 |
| WMMSE 30 it. Gram, 32 / 16 bits | 37,31 / 0,267 / 0,084 | 24,41 / 0,267 / 0,084 |
| ART 2 couches λ = 1 | 34,91 / 0,046 | — |
| ART 2 couches λ = 5 | 34,35 / 0,031 | 24,07 / 0,025 |
| ART 4 couches λ = 5 | 35,96 / 0,057 | — |
- Débit ≥ RZF et énergie < RZF 32 bits : oui sur les deux sites ; < RZF 16 bits : non.
- **À Ericsson, WMMSE Gram 16 bits domine nos points** (5 it. : 35,45 à 0,028 µJ). Sur St-Catherine, nos points restent 3 à 4× sous WMMSE Gram 16 bits à débit égal. Notre avance en énergie vient uniquement de la précision ; elle disparaît sur les canaux bien conditionnés où WMMSE converge en peu d'itérations.

### BER réel par simulation de symboles (`experiments/phase14_ber.py`, `results/phase14/ber.{json,png}`)
QAM Gray (QPSK à 256-QAM), démodulation dure, gain effectif connu du récepteur ; UMi M8K4, puissances réelles. Modulation adaptative par flux (BER cible 1e-3, sans codage) → débit transmis. Pilote LMMSE 10 dB, débit transmis (BER 16-QAM sur flux actifs) :
| précodeur | 10 dB | 20 dB | 30 dB |
|---|---|---|---|
| entraîné erreur synthétique | **5,53** (0,062) | **9,98** (0,028) | **13,40** (0,014) |
| entraîné CSI parfait | 5,29 (0,086) | 9,62 (0,052) | 13,28 (0,034) |
| WMMSE 100 it. robuste | 5,28 (0,084) | 9,56 (0,051) | 13,15 (0,034) |
| MRT | 3,92 | 5,02 | 5,16 |
| RZF | 1,71 | 2,73 | 4,94 |
- L'ancienne figure (BER moyen gaussien) était trompeuse : à modulation réelle, la concentration de puissance des précodeurs somme-débit permet les ordres élevés ; RZF garde des SINR juste au-dessus du seuil QPSK (≈ 1,3 bit par flux actif contre ≈ 3,8).
- L'erreur synthétique gagne partout en débit transmis (+10 % à pilote 0 dB, +1 % à 20 dB), y compris là où elle perdait en débit de Shannon.

### Zero-shot : entraînement sur UMa seul ou sur les 3 modèles 3GPP (`phase14_zeroshot.py --train uma|3gpp`, `results/phase14/zeroshot_{uma,3gpp}.json`)
Débit à 10 / 20 / 25 / 30 dB, CSI parfait (mêmes tirages que `phase13_bench`) :

| site | 3GPP, M8K4 + M64K8 (existant) | 3GPP, M8K4 | UMa seul | WMMSE 300 it. |
|---|---|---|---|---|
| St-Catherine | 12,68 / 22,05 / 27,27 / 32,62 | 12,65 / 21,98 / 27,10 / 32,32 | 12,64 / 21,97 / 27,14 / 32,33 | 12,70 / 22,19 / 27,53 / 33,11 |
| Décarie | 12,73 / 22,83 / 28,41 / 34,00 | 12,76 / 22,77 / 28,26 / 33,74 | 12,71 / 22,72 / 28,24 / 33,66 | 12,79 / 22,91 / 28,62 / 34,39 |
| Ericsson | 13,58 / 24,31 / 30,05 / 35,77 | 13,58 / 24,23 / 29,91 / 35,44 | 13,52 / 24,21 / 29,93 / 35,37 | 13,64 / 24,44 / 30,38 / 36,37 |

- UMa seul ≈ 3 modèles 3GPP (±0,1) : la variété entre modèles 3GPP n'aide presque pas pour les sites du labo. Entraîner sur deux tailles (M8K4 + M64K8) aide un peu plus (+0,3 à 30 dB).

### Fine-tuning avec peu de données sur un site (`phase14_finetune.py`, `results/phase14/finetune_<site>.json`)
Départ : précodeur 3GPP (M8K4 + M64K8). Cellules du site séparées en deux moitiés disjointes (adaptation / test, 64 tirages de test). Adaptation sur N échantillons (une sous-porteuse de 4 utilisateurs ; utilisateurs permutés au hasard), trois méthodes : complet (44,6 k paramètres appris, pas 1e-4), LoRA rang 4 (9,8 k, pas 1e-3), valeurs singulières seules (« vecteurs propres », 1,0 k, pas 1e-3). Débit à 30 dB (à 20 dB, toutes les variantes sont à ±0,1 du zero-shot, lui-même au niveau de WMMSE convergé) :

| site (30 dB) | zero-shot | complet 40 / 400 / 4000 | LoRA 40 / 400 / 4000 | valeurs sing. 40 / 400 / 4000 | WMMSE 300 it. |
|---|---|---|---|---|---|
| St-Catherine | 33,28 | 33,15 / 33,13 / **33,35** | 33,17 / 33,16 / 33,33 | 33,21 / 33,12 / 33,34 | 33,56 |
| Décarie | 34,86 | 34,48 / 34,85 / **34,92** | 34,65 / 34,81 / 34,89 | 34,68 / 34,82 / 34,86 | 35,04 |
| Ericsson | 35,32 | 34,82 / 35,21 / 35,51 | 34,72 / 35,08 / **35,53** | 34,89 / 35,06 / 35,26 | 36,19 |

- **Le fine-tuning comble peu l'écart au fort SNR** : au mieux +0,07 à +0,21 bps/Hz avec 4 000 échantillons (un quart à un tiers de l'écart à WMMSE convergé). Avec 40 échantillons, le fine-tuning complet DÉGRADE (−0,1 à −0,5 : sur-apprentissage) ; LoRA et les valeurs singulières limitent la casse sans gagner.
- LoRA (9,8 k paramètres) fait autant que le fine-tuning complet à N = 4 000 ; les valeurs singulières seules (1 k paramètres) suffisent sur St-Catherine, moins sur Ericsson.
- **Lecture** : l'écart restant au fort SNR n'est pas surtout dû au site (sinon l'adaptation le comblerait) mais à une limite du précodeur à fort SNR sur des canaux très bien conditionnés et peu dispersifs (Ericsson). Pistes : plus de couches ou plus d'entraînement à fort SNR ; ou une ou deux itérations de WMMSE partant de la sortie du réseau.

### Écart au fort SNR : capacité ou entraînement ? (`phase14_capacity.py`, `results/phase14/capacity_*.json`)
Un seul changement par rapport au précodeur de référence (8 couches, M8K4 + M64K8, SNR U(0, 30) dB, 60 000 pas) ; zero-shot, CSI parfait, mêmes tirages que `phase13_bench`. Débit à 0 / 20 / 30 dB :

| site | 8 couches (référence) | **12 couches** | 8 couches, SNR U(10, 35) | WMMSE 300 it. |
|---|---|---|---|---|
| St-Catherine | 5,49 / 22,05 / 32,62 | 5,49 / 22,12 / **32,84** | 5,34 / 21,91 / 32,31 | 5,51 / 22,19 / 33,11 |
| Décarie | 5,41 / 22,83 / 34,00 | 5,43 / 22,88 / **34,14** | 5,25 / 22,64 / 33,71 | 5,45 / 22,91 / 34,39 |
| Ericsson | 5,44 / 24,31 / 35,77 | 5,45 / **24,46** / **36,08** | 5,14 / 24,05 / 35,36 | 5,47 / 24,44 / 36,37 |
| UMi (vu) | 5,16 / 17,68 / 26,93 | 5,16 / 17,68 / 26,96 | 5,12 / 17,65 / 26,91 | 5,17 / 17,69 / 26,60 |

- **La profondeur est le levier** : 12 couches comblent un tiers à la moitié de l'écart à WMMSE convergé à 30 dB sur les sites jamais vus (Ericsson : 35,77 → 36,08 ; à 20 dB, au-dessus de WMMSE), sans aucune donnée du site, mieux que le fine-tuning avec 4 000 échantillons. Coût : ×1,5 en calcul (12 couches au lieu de 8).
- **Déplacer l'entraînement vers le fort SNR dégrade partout** (−0,3 à −0,4 à 30 dB, −0,2 à −0,3 à 0 dB) : ÉCARTÉ.
- **16 couches ≈ 12 couches** (30 dB : St-Catherine 32,88, Décarie 34,27, Ericsson 36,10 ; 3GPP au-dessus de WMMSE300) : la profondeur sature, l'écart résiduel aux sites (0,1 à 0,3) ne se comble pas par la capacité seule.
- Cohérent avec la lecture « itérations » du précodeur (chaque couche ≈ une étape d'inversion) : les canaux très bien conditionnés à fort SNR demandent plus d'étapes.

### Modèle de coût FPGA pour ART (agent, `/users/sala/Documents/test_projet/hls_work/cost_model/` : `RAPPORT.md`, `cost_table.csv`, `fpga_cost.py`)
68 synthèses Vitis HLS (couche dense 16 × 16 déroulée, II = 1, xcu280, 100 MHz), b_w ∈ {1, 2, 3, 4, 6, 8}, b_a ∈ {2, 3, 4, 6, 8}, poids constants (cas A, design déroulé) ou variables (cas B), plus quelques multiplications forcées en LUT ou DSP.
- **Cas A, b_w ≤ 4 : aucun DSP**, LUT par MAC ≈ 0,80·b_w·b_a − 1,76·b_a + 11,6 (b_w = 2 à 4, R² = 0,83) ; b_w = 1 : 0,36·b_a + 3,9. À b_w = 6 ou 8, Vitis place une partie des produits en DSP de façon irrégulière (0,17 à 0,43 DSP/MAC) : tabulé seulement.
- Cas B : 0,5 DSP par MAC (deux produits par DSP48E2) dès b_w ≥ 2.
- **Comparé à Moons** : même choix optimal (b_w = 1, b_a = 2) mais classement global différent (Spearman 0,47) : Moons sous-estime le saut b_w = 1 → 2 et surestime le poids de b_a, qui coûte peu sur FPGA.
- Fonction différentiable prête (`layer_cost_torch`) pour une perte ART orientée FPGA ; recommandation : pénaliser b_w bien plus que b_a, limiter b_w ≤ 4.

### HLS déroulé et pipeliné (agent Fable, `hls/codegen.py --arch seq|flow|stream`, rapports `hls_work/<variante>/`)
Modèle 2 couches h16 (143 k MAC), xcu280, horloge cible 4 ns. Toutes les variantes identiques bit à bit à `IntPairPrecoder` (banc natif et simulation C).
| variante | 64 SC | Fmax | DSP | LUT | instances U280 |
|---|---|---|---|---|---|
| départ (int64, sans pragma) | 7,05 M cycles (51 ms) | 137 MHz | 1 802 | 83 k | 5 |
| `seq` tableaux (ressources minimales) | 1,06 M (4,2 ms) | 255 MHz | 364 | 90 k | 14 |
| `flow` streams (1 SC par appel) | 4 209 (12,3 µs) | 342 MHz | 1 945 | 294 k | 4 |
| **`stream` (64 SC par appel, DATAFLOW de 13 processus)** | **1 341 (3,9 µs)** | 342 MHz | 2 147 | 299 k | **4** |
- `stream` : couches denses déroulées à poids constants, 4 antennes/cycle, le flot ne se vide pas entre sous-porteuses (~17 cycles/SC en régime établi) ; 4 instances → 256 SC en ~1 080 cycles. Déroulage complet (1 SC/cycle) : ~16× les LUT, impossible sur l'U280.
- `flow` limité par la profondeur de pipeline (~45 cycles) face à 16 tokens ; dérouler les tokens double les ressources sans gain.
- Aucune division : inverse de racine par table + 1 tour de Newton, perte 0,004 bps/Hz (2 couches), 0,002 (8 couches).
- 8 couches (738 k MAC), `stream` : 64 SC en 1 878 cycles (7,5 µs à 250 MHz), II du lot 1 080 ; 7 252 DSP (80 % de l'U280), 1,28 M LUT (98 %), 1 325 BRAM : une seule instance, à la limite (estimation HLS, le placement-routage peut échouer). À faire : cosimulation RTL (profondeurs de FIFO), multiplications par constantes 16 bits en LUT pour une 5e instance.

### HLS, 2e passe : co-conception ART-FPGA et architecture repliée (`hls/foldgen.py`, `--arch fold --fold S`, `hls/README.md`)
Modèle 2 couches, toutes variantes bit à bit = `IntPairPrecoder`. Budget 5G : 1 092 SC par créneau de 0,5 ms (1/3 de 100 MHz).
| variante | poids | 64 SC | DSP | LUT | instances U280 | M SC/s par U280 | 1/3 bande, 1 instance |
|---|---|---|---|---|---|---|---|
| stream (réf.) | ART énergie (24,07) | 3,9 µs | 2 147 | 299 k | 4 | 81 | oui (9,3×) |
| stream | ART-FPGA λ = 0,3 (24,56) | 3,8 µs | 1 788 | 279 k | 4 | 82 | oui (9,4×) |
| fold S = 4 | λ = 0,3 | 12,8 µs | 502 | 270 k | 4 | 21 | oui (2,4×, ≥ 141 MHz) |
| **fold S = 8** | λ = 0,3 | 24,7 µs | **335** | **171 k** | 7 | 19 | oui (1,2×, ≥ 280 MHz) |
- Co-conception seule : gain modeste (−17 % DSP) ; les poids étaient déjà en LUT, les DSP venaient des constantes 16 bits, du produit triangulaire, de W, LN et irsqrt (irsqrt passé de 64 bits à largeurs exactes : 39 → 14 DSP).
- Repli S = 8 : DSP ÷ 6, LUT ÷ 1,7 ; une instance (13 % LUT, 3,7 % DSP) tient le 1/3 de bande si Fmax ≥ 280 MHz (estimée 344). Les LUT restantes sont structurelles (tokens stockés, FIFO), pas les poids.
- En SC/s par carte entière, stream reste 4× meilleur (coûts fixes non repliables). Écartés : `--cmul fabric` (plantage Vitis 2022.2), empaquetage DSP (inutile après repli). À faire : placement-routage (Fmax réelle), cosim RTL.

### ART avec le coût FPGA (`torch_port/art_stecath.py --cost fpga`)
Premier essai (poids jusqu'à 8 bits, coût calculé avec b_w borné à 4) : b_w > 4 gratuit → tous les poids à 8 bits, activations à 2 bits, débit 22,3 (λ = 5) contre 24,1 avec l'objectif énergie. Défaut de la perte, pas un résultat : relancé avec b_w ≤ 4 (régime sans DSP), λ = 1 et 5.
Avec b_w ≤ 4 (St-Catherine, 2 couches h16, mêmes options que la référence énergie λ = 5 : 24,07 bps/Hz, poids 2 à 8 bits) :
| objectif | débit | poids des couches denses | activations | coût FPGA (LUT-éq.) |
|---|---|---|---|---|
| **FPGA λ = 0,3** | **24,56** | 2-3 bits (entrée/sortie 4) | 4-8 | 181 k |
| FPGA λ = 1 | 23,21 | 1-2 bits (entrée/sortie 4) | 3-8 | 125 k |
| FPGA λ = 5 | 21,74 | 1 bit (entrée/sortie 3) | 3-6 | 37 k |
- L'objectif FPGA pousse les poids vers 1-2 bits (multiplication par constante sans DSP) et garde des activations plus larges : l'inverse de Moons, qui pénalise max(b_w, b_a). λ = 0,3 : 24,56 bps/Hz, AU-DESSUS de la référence énergie (24,07) avec des poids ≤ 3 bits dans les blocs (aucun DSP pour les poids sur FPGA) ; énergie Moons 23,7 nJ contre 24,5.
- Limite : G, W et le produit triangulaire (variable × variable, 8 bits) ne sont pas dans le coût FPGA.

## Phase 15 — comprendre l'écart au fort SNR (2026-11-26, évaluation seule)

### Plancher d'interférence et structure du précodeur (`experiments/phase15_diag.py`, `results/phase15/diag.json`)
Précodeur 3GPP M8K4 + M64K8 (phase 13), CSI parfait, M8K4 ; UMi (tirages `phase13_bench`) et les 3 sites (tirages seed 77). Fuite = Σ_{j≠k}|h_k w_j|² / |h_k w_k|² par flux, médiane en dB.

| SNR | UMi : réseau / WMMSE 300 (fuite) | Ericsson : réseau / WMMSE 300 (fuite) |
|---|---|---|
| 20 dB | 17,65 / 17,66 (−31 / −31) | 24,08 / 24,03 (−41 / −42) |
| 30 dB | 26,89 / 26,56 (−43 / −49) | 35,26 / 36,15 (−46 / −61) |
| 40 dB | 36,28 / 35,61 (−42 / −70) | 43,71 / 47,84 (−43 / −79) |

- **Plancher confirmé** : la fuite du réseau sature vers −42 à −47 dB partout (UMi, 3 sites, et SC UMi les mieux conditionnées), celle de WMMSE continue de baisser. Réseau + 1 it. WMMSE dépasse toujours WMMSE 300 (Ericsson 40 dB : 47,92).
- **Effet de régime, pas de site** : le plancher ne pénalise que lorsque le SINR visé le dépasse (Ericsson à fort SNR) ; c'est pourquoi le fine-tuning sur le site ne le comble pas et la profondeur seulement un peu.
- **Structure de Björnson** C = (ΛG + σ²I)⁻¹P^½ ajustée au C du réseau (2K scalaires, Adam 400 pas, 256 échantillons) : erreur relative médiane 1 à 5 % ; à 40 dB, la forme exacte avec ces (λ, p) dépasse le réseau (UMi 37,85 contre 36,04 ; Ericsson 46,26 contre 43,05). Le réseau semble avoir la bonne forme et une inversion imprécise. Réserve : moyennes tirées par des ajustements ratés (Ericsson 20 dB : forme 21,66 < réseau 23,73), donc indicatif seulement.
- **Newton–Schulz depuis une initialisation générique** (λ, p de WMMSE, X0 = Aᴴ/(‖A‖₁‖A‖∞), 512 MAC/pas) : 15 pas égalent le réseau à 20 dB (~8,7 k MAC/SC contre 717 k), insuffisant à 30 dB (Ericsson 35,13 contre 35,52). Référence seulement : décision du 2026-11-26, le modèle principal reste libre.

### Entraîner jusqu'à 40 dB — ÉCARTÉ (`results/phase15/snr40.json`, code non gardé)
Seul changement par rapport au précodeur 3GPP de la phase 13 : SNR d'entraînement U(0, 40) au lieu de U(0, 30) (60 000 pas, plateau dès 20 000). Mêmes tirages que le diagnostic. Débit ancien / nouveau / WMMSE 300 :

| SNR | UMi | St-Catherine | Décarie | Ericsson |
|---|---|---|---|---|
| 20 dB | 17,65 / 17,62 / 17,66 | 22,35 / 22,21 / 22,36 | 23,27 / 23,09 / 23,22 | 24,08 / 23,59 / 24,03 |
| 30 dB | 26,89 / 26,87 / 26,56 | 33,32 / 33,17 / 33,58 | 34,80 / 34,65 / 35,00 | 35,26 / 34,74 / 36,15 |
| 40 dB | 36,28 / 37,28 / 35,61 | 43,00 / 43,95 / 44,68 | 44,51 / 45,72 / 45,88 | 43,71 / 44,40 / 47,84 |

- Critère fixé à l'avance (≥ +0,2 à 30 dB sur les sites, ≥ −0,05 de 0 à 20 dB) : échoue des deux côtés (−0,28 en moyenne à 30 dB ; Ericsson −0,49 à 20 dB).
- À 40 dB, le plancher descend (fuite médiane −48 à −51 dB contre −43 à −44) et le débit gagne +0,7 à +1,2, mais tout ce qui est en dessous se dégrade : **le réseau (8 couches, état 16) fait un compromis précision / plage de SNR à capacité fixe**. Le plancher n'est pas qu'une question de plage d'entraînement.

### Répartition des MAC du précodeur (`experiments/phase15_macs.py`, `results/phase15/macs.json`)
Base (8 couches, état 16, FFN 32) : 717 k MAC/SC en M8K4, 724 k en M64K4. **95 % dans les couches denses appliquées à chaque paire** (16 paires × 8 couches), 4,6 % dans le produit triangulaire, 0,1 % pour G et pour W : le coût ne dépend presque pas de M. Leviers sans forme imposée : pas fréquentiel (1 SC sur 12 sur les sites : −0,04 à −0,43 bps/Hz, ÷ 12 ; UMi −0,78), largeur des couches par paire, profondeur (4 couches : 375 k), quantification.

### CSI estimé sur les sites du labo (`experiments/phase15_lab_csi.py`, `results/phase15/lab_csi.json`)
Sites convertis en OFDM (`phase13_lab`, convention psi2az), pilotes Sionna, LS + interpolation LMMSE (covariance du mélange), M8K4, puissances réelles, 192 tirages par site (seed 13) ; tous les précodeurs reçoivent la même estimation, débit sur le vrai canal. Débit à 30 dB (20 dB entre parenthèses pour le pilote 10 dB) :

| site, pilote | réseau CSI parfait | réseau erreur synth. | WMMSE robuste 300 it. | RZF |
|---|---|---|---|---|
| St-Catherine, 0 dB | 19,83 | **20,15** | 19,84 | 13,19 |
| St-Catherine, 10 dB | 25,78 (20,31) | 25,87 (20,32) | 25,87 (20,36) | 18,11 |
| St-Catherine, 20 dB | 30,11 | 29,93 | **30,49** | 22,39 |
| Décarie, 0 / 10 / 20 dB | — / 26,92 / 31,33 | 21,17 / 26,99 / 31,14 | 20,87 / 27,01 / **31,67** | — / 21,21 / 25,62 |
| Ericsson, 0 / 10 / 20 dB | — / 27,99 / 32,95 | 21,63 / 28,03 / 32,67 | 21,41 / 28,05 / **33,35** | — / 24,70 / 29,83 |

- **Généralisation de l'entraînement avec erreur synthétique à des sites jamais vus : oui** jusqu'à pilote 10 dB (égal ou devant WMMSE robuste ; +0,2 à +0,3 à pilote 0 dB).
- **Le plancher du réseau réapparaît à pilote 20 dB** (−0,4 à −0,7 à 30 dB) : à bon SNR pilote, l'erreur d'estimation ne le masque plus. À pilote 10 dB, égalité (≤ 0,06). Le plancher n'est donc pas qu'une curiosité du CSI parfait.
- À pilote 20 dB, le modèle entraîné avec erreur synthétique perd en plus 0,2-0,3 sur le modèle CSI parfait (réglé pour une erreur moyenne, cf. phase 14) ; la variante « + variance » (poids non sauvés en phase 14, réentraînement en cours) devrait supprimer cette part.
- Réserves : une graine, une convention d'angles.
- **Variante « erreur synthétique + variance » (réentraînée à l'identique, UMi reproduit à 0,03 près ; `--errin-only`)** : à 30 dB, St-Catherine 20,14 / 25,73 / 29,44, Décarie 21,19 / 26,80 / 30,56, Ericsson 21,53 / 27,80 / 32,21 (pilote 0 / 10 / 20 dB). Elle bat WMMSE robuste à pilote 0 dB mais fait MOINS bien que le modèle sans variance à pilote 10-20 dB (jusqu'à −1,1 contre WMMSE à pilote 20 dB). Le gain de la variance vu sur UMi ne se transfère pas aux sites : probablement parce que la variance d'erreur fournie par l'estimateur est calculée avec la covariance du mélange 3GPP, fausse sur des canaux presque plats (variance surestimée → précodeur trop prudent). À vérifier avant de montrer la variante.


### Deuxième passe qui voit sa propre erreur — RETENU (`models/refine.py`, `experiments/phase15_refine.py`, `phase15_refine_eval.py`, `results/phase15/refine.json`)
Passe 1 = précodeur 3GPP figé (C1) ; passe 2 = 4 blocs triangulaires (d 16, FFN 32) qui reçoivent par paire les caractéristiques de G plus celles du canal effectif E = H̃W1 (≈ G·C1), et sortent ΔC (sortie initialisée à zéro : départ = base exactement) ; C2 = C1 + ΔC. Seule la passe 2 est entraînée (mêmes données, perte, SNR U(0, 30), 60 000 pas, 23 min). Aucune division, aucune forme imposée. Débit base / base + passe 2 / WMMSE 300 (fuite médiane du réseau, dB) :

| CSI parfait | 20 dB | 30 dB | 40 dB |
|---|---|---|---|
| UMi | 17,65 / 17,65 / 17,66 | 26,89 / **26,99** / 26,56 (−43 → −47) | 36,28 / **37,81** / 35,61 |
| St-Catherine | 22,35 / 22,40 / 22,36 | 33,32 / **33,69** / 33,58 (−47 → −56) | 43,00 / **45,29** / 44,68 |
| Décarie | 23,27 / 23,32 / 23,22 | 34,80 / **35,28** / 35,00 (−47 → −56) | 44,51 / **47,02** / 45,88 |
| Ericsson | 24,08 / 24,18 / 24,03 | 35,26 / 35,88 / 36,15 (−46 → −55) | 43,71 / 46,43 / 47,84 |

CSI estimé, pilote 20 dB, 30 dB : St-Catherine 29,96 → 30,27 (WMMSE robuste 30,49), Décarie 31,31 → 31,58 (31,67), Ericsson 32,89 → 33,24 (33,35).
- Critère fixé à l'avance : rempli (+0,49 à 30 dB en moyenne sur les sites, aucune perte de 0 à 20 dB, perte en CSI estimé réduite de moitié environ).
- **Le plancher descend d'environ 9 dB** (−46 → −55 dB) ; le réseau dépasse WMMSE 300 à 30 dB sur St-Catherine et Décarie et à 40 dB sur UMi, St-Catherine, Décarie. Seul Ericsson (le plus haut SINR) reste sous WMMSE au fort SNR.
- **Coût : +363 k MAC/SC (+51 %, total 1,08 M en M8K4)**, à peu près celui de 12 couches, qui ne gagnaient que la moitié (Ericsson 30 dB +0,31 contre +0,62, protocoles de tirage différents) : à coût égal, voir l'erreur vaut mieux que plus de profondeur.
- Réserves : une graine ; en CSI estimé, WMMSE est repris de `lab_csi.json` (mêmes réglages, sous-ensemble de tirages possiblement différent : la base y vaut 29,96 contre 30,11) ; passe 2 entraînée seulement en CSI parfait.

### Deuxième passe : appliquée deux fois, et seulement au fort SNR (`experiments/phase15_refine_more.py`, `results/phase15/refine_more.json`, évaluation seule)
- **Deux applications des mêmes poids** (C3 = C2 + ΔC(G, E2), sans réentraîner) : le plancher descend encore d'~1-2 dB (Ericsson 30 dB : −55,4 → −57,4 dB) ; débit à 30 / 40 dB : St-Catherine 33,77 / 45,53, Décarie 35,34 / 47,39, Ericsson 36,01 / 46,92 (WMMSE 300 : 36,15 / 47,84), UMi 26,99 / 37,89. Aucune perte nulle part. Gain modeste (+0,06 à +0,13 à 30 dB) pour +363 k MAC : non retenu par défaut, option documentée.
- **Passe 2 activée seulement au-dessus d'un seuil de SNR** (le SNR est connu) : seuil 20 dB → perte max 0,018 bps/Hz sur 0-40 dB (UMi + sites), coût moyen 838 k MAC/SC au lieu de 1,08 M pour un SNR uniforme sur 0-30 dB (−22 %) ; seuil 25 dB perd 0,10. **Retenu : passe 2 au-delà de 20 dB.**

### Deuxième passe entraînée jusqu'à 40 dB — retenue de justesse, effet marginal (`experiments/phase15_refine40.py`, `results/phase15/refine40.json`)
Seul changement : SNR d'entraînement de la passe 2 U(0, 40) (passe 1 figée). Débit passe 2 U(0,30) → U(0,40), CSI parfait : 40 dB UMi 37,81 → 37,91, St-Catherine 45,29 → 45,52, Décarie 47,02 → 47,28, Ericsson 46,43 → 46,92 (WMMSE 300 : 47,84) ; à 35 dB +0,0 à +0,2 ; de 0 à 30 dB ±0,04 ; CSI estimé (pilote 20 dB) identique.
- Critère rempli de justesse (+0,21 en moyenne à 35-40 dB, pire −0,035 de 0 à 30 dB). Contrairement au réseau entier, **pas de compromis** quand seule la passe 2 voit le fort SNR (passe 1 figée).
- Gain du même ordre que d'appliquer deux fois la passe 2 U(0,30), et seulement à 40 dB : sans effet en pratique. Même coût ; une graine.

### Pas fréquentiel adaptatif par tirage (`experiments/phase15_adaptive_step.py`, `results/phase15/adaptive_step.json`, évaluation seule)
Précodeur 3GPP, CSI parfait, W interpolé entre ancres (1 SC sur s, s ∈ {1…24}). Oracle = plus grand s par tirage perdant ≤ 1 % ; règle = s tel que s·Δf ≤ B_c = 1/(α τ_rms), τ_rms estimé par ajustement de l'autocorrélation en fréquence (l'IDFT pleine bande surestime τ_rms ×10-50 par fuite), α = 26,3 réglé sur UMi à 20 dB. s moyen harmonique, débit 20 / 30 dB :

| jeu (τ_rms médian) | pas fixe 3 | oracle 1 % (s à 20 → 30 dB) | règle (s ; perte 20 / 30 dB) |
|---|---|---|---|
| UMi (108 ns) | 17,64 / 26,73 | 6,9 → 3,8 ; 17,58 / 26,79 | 7,2 ; 0,95 % / 3,9 % |
| UMa (182 ns) | 17,84 / 26,97 | 3,9 → 2,5 ; 17,92 / 27,43 | 4,4 ; 1,2 % / 4,2 % |
| RMa (27 ns) | 17,74 / 26,62 | 16,4 → 12,1 ; 17,70 / 26,53 | 16,3 ; 0,4 % / 1,4 % |
| St-Catherine (65 ns) | 22,04 / 32,56 | 8,8 → 5,5 ; 21,94 / 32,44 | 13,7 ; 1,7 % / 5,3 % |
| Décarie (17 ns) | 22,83 / 34,00 | 20,4 → 16,4 ; 22,79 / 33,91 | 21,9 ; 0,35 % / 1,1 % |
| Ericsson (12 ns) | 24,31 / 35,77 | 20,4 → 17,1 ; 24,26 / 35,68 | 22,3 ; 0,37 % / 1,05 % |

- **Le pas adaptatif vaut le coup** : l'oracle calcule 1,3× (UMa) à 6,8× (Décarie, Ericsson) moins que le pas fixe 3 pour < 0,6 % de perte.
- **Le bon pas dépend aussi du SNR** (il diminue quand le SNR monte : l'erreur d'interpolation agit comme une fuite d'interférence, cf. plancher). La règle à un seul α (étalement seul) perd 1 à 5 % à 30 dB : il faut y ajouter le SNR (α(SNR)), avant d'envisager d'apprendre le pas.

### RZF et WMMSE en virgule fixe : combien de bits vraiment ? (`torch_port/fixedpoint_classical.py`, `results/stecath/fixedpoint_classical.json`, CPU)
Virgule fixe à exposant de bloc (mantisse b bits, arrondi + saturation après chaque opération, accumulateurs larges), H normalisé par utilisateur, un exposant par colonne pour les matrices K×K et W, inversion par équilibrage de Jacobi + LDLᴴ, réciproques et rsqrt par table + Newton, aucune division. St-Catherine et Ericsson (M64K4, CSI parfait, 20 000 échantillons ; chemin flottant = classical_energy*.json à 1e-6).
- **Bits minimaux (perte ≤ 1 % / ≤ 0,1 bps/Hz)** : St-Catherine 12 / 13 pour RZF et WMMSE ; Ericsson 13 / 14-15. Un exposant unique par échantillon (format naïf) demande 17-19 bits : G casse (50 dB d'écarts). Le goulot est l'inversion, puis G et W ; l'entrée H tient en 8 bits.
- **Énergie au nombre de bits qui tient (µJ, Moons v5)** : Ericsson RZF 33,2-33,3 à 0,012-0,013 ; WMMSE 5 it. 35,1-35,4 à 0,020-0,025 ; WMMSE 30 it. 37,0-37,2 à 0,061-0,068. St-Catherine RZF 16,9-17,0 à 0,010-0,012 ; WMMSE 30 it. 24,3-24,4 à 0,053-0,061. Nos points : ART 2 couches 34,35 à 0,031 (Ericsson), 24,07 à 0,025 (St-Catherine) ; 8 couches quantifié 26,64 à 0,113 (St-Catherine).
- **Conclusions** : (1) à Ericsson, WMMSE 5 it. à 13-15 bits nous domine (plus de débit, moins d'énergie) : la phrase « WMMSE Gram nous domine à Ericsson » tient, plus fort qu'annoncé ; (2) à St-Catherine, à débit égal (~24,1-24,3), nous restons ~2× sous WMMSE 30 it., et le point 26,64 n'a pas d'équivalent WMMSE bon marché (~200 it.) ; (3) **nous ne battons pas RZF en énergie** : RZF coûte 2 à 3× moins, mais perd 7 bps/Hz à St-Catherine et ~0,9 à Ericsson. Notre avantage sur RZF est en débit, pas en énergie.
- Réserves : format généreux pour les classiques (exposants dynamiques, détection de zéros de tête et décalages non comptés) ; Moons est un modèle ASIC générique ; CSI parfait, mono-porteuse.

## Phase 16 — refontes d'architecture (2026-11-27)

### Forme imposée ENTRAÎNÉE : petit transformer → (λ, p) + Newton–Schulz (`models/structured_precoder.py`, `experiments/phase16_structured.py`, `phase16_eval.py`, `results/phase16/structured.json`)
Expérience de comparaison (la v0 est libre). Transformer par paires de 2 couches (d 16, FFN 32, entrées normalisées) → deux scalaires positifs par utilisateur (λ_k, p_k) lus sur les tokens diagonaux ; A = ΛG + σ²I (G physique), inversion Newton–Schulz déroulée à T pas (X0 mis à l'échelle par une norme : une réciproque scalaire), C = X P^½, W = H̃ᴴC normalisé. De zéro, mêmes données, perte, SNR U(0, 30), 60 000 pas. Débit CSI parfait, M8K4 (T20 / libre / libre + 2e passe / WMMSE 300) :

| | 10 dB | 20 dB | 30 dB | 40 dB |
|---|---|---|---|---|
| UMi | 10,34 / 10,33 / 10,33 / 10,34 | 17,65 / 17,65 / 17,65 / 17,66 | **26,98** / 26,89 / 26,99 / 26,56 | **37,84** / 36,28 / 37,81 / 35,61 |
| St-Catherine | 12,67 / 12,64 / 12,65 / 12,69 | 22,39 / 22,35 / 22,40 / 22,36 | **33,96** / 33,32 / 33,69 / 33,58 | **46,41** / 43,00 / 45,29 / 44,68 |
| Décarie | 12,91 / 12,92 / 12,93 / 12,95 | 23,20 / 23,27 / 23,32 / 23,22 | **35,37** / 34,80 / 35,28 / 35,00 | **48,28** / 44,51 / 47,02 / 45,88 |
| Ericsson | 13,40 / 13,45 / 13,47 / 13,49 | 23,92 / 24,08 / 24,18 / 24,03 | **36,27** / 35,26 / 35,88 / 36,15 | **49,32** / 43,71 / 46,43 / 47,84 |

CSI estimé, pilote 20 dB, 30 dB (T20 / libre / libre + 2e passe / WMMSE robuste) : St-Catherine 30,28 / 29,96 / 30,27 / 30,49 ; Décarie 31,59 / 31,31 / 31,58 / 31,67 ; Ericsson 33,11 / 32,89 / 33,24 / 33,35.
- **MAC/SC (M8K4) : 193 k (T20), 188 k (T10)** contre 717 k (libre) et 1,08 M (libre + 2e passe) : 3,7× moins que le réseau libre ; Newton–Schulz ne pèse que ~10 k.
- **T20 dépasse le réseau libre et sa 2e passe au fort SNR partout** (30-40 dB), et WMMSE 300 à 30-40 dB sur les 4 jeux (Ericsson 40 dB : 49,32 contre 47,84). Pas de plancher : l'inversion est exacte au nombre de pas près.
- Petites pertes à 10-20 dB sur les canaux les plus faciles : Ericsson −0,16 à 20 dB, Décarie −0,07 ; ailleurs ±0,05.
- T10 : Newton–Schulz trop court, −1 à −5 au fort SNR (le diagnostic phase 15 le prévoyait).
- Réserves : une graine par T ; mélange de conventions (A sur G physique, transformer sur entrées normalisées) non testé autrement ; Newton–Schulz en virgule fixe non vérifié (l'inversion est le goulot des classiques : 13-15 bits) ; 20 pas séquentiels → latence. Contredit l'hypothèse « la liberté coûte peu » : à MAC égal, la forme imposée gagne nettement.

### Encodeur angle × retard en massif — ÉCARTÉ (code non gardé)
BS 1×32 au pas λ/2, deux polarisations : DFT unitaire de 32 points par polarisation avant le traitement par retard, filtre complexe 3×3 (retard × angle) par faisceau, log de la puissance par angle en entrée (départ = référence exacte) ; reste identique à `phase13_massive_pilots` (30 000 pas). 64 tirages UMi, pilote 10 dB : 13,47 / 24,47 / **39,27** / 47,26 / 55,15 à 0 / 10 / 20 / 25 / 30 dB contre 13,48 / 24,49 / 39,34 / 47,36 / 55,24 pour la référence. À 20 dB : pilote 0 dB 36,49 contre 36,19, pilote 20 dB 40,05 contre 40,23. Encodeur +56 % de MAC (FFT comptées). Critère (+0,3) non atteint : l'angle n'apporte rien de mesurable, en massif comme en M8K4. Non testé : jetons par couple (retard, faisceau), ~64× plus cher.

## Prochaine étape
(1) 3 graines pour l'attention en fréquence D64. (2) Réduire le coût. (3) Bootstrap sur les tirages. (4) Faire le point avant d'aller plus loin (pilote 20 dB, vieillissement, M64).
Cible principale ensuite (phases 2–3) : le régime pilote 10 dB, où il reste 2,7 à 5,9 bps/Hz après LMMSE.
