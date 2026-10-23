# Journal des résultats

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

## Prochaine étape
Phase 3 : donner au même modèle les W SC voisines du CSI estimé (W ∈ {1, 3, 5, 9, 13}), d'abord par simple concaténation dans le token, pilote 10 dB. Cible : battre RZF(LMMSE) (23,5 à 20 dB).
Cible principale ensuite (phases 2–3) : le régime pilote 10 dB, où il reste 2,7 à 5,9 bps/Hz après LMMSE.
