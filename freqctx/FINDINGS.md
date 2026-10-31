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

## Prochaine étape
(1) 3 graines pour l'attention en fréquence D64. (2) Réduire le coût. (3) Bootstrap sur les tirages. (4) Faire le point avant d'aller plus loin (pilote 20 dB, vieillissement, M64).
Cible principale ensuite (phases 2–3) : le régime pilote 10 dB, où il reste 2,7 à 5,9 bps/Hz après LMMSE.
