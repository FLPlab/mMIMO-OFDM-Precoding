# Journal des résultats

## SYNTHÈSE (mise à jour 2026-10-23, à relire avant tout)

**Le modèle actuel** (`models/per_sc.py`) : par sous-porteuse (SC), un token par utilisateur. Entrée = CSI estimé de la SC (et de ses W voisines, concaténées) → Dense(D) → **un seul bloc répété 4 fois** [attention **linéaire** entre utilisateurs + FFN] → Dense(2M) = w_k → normalisation de puissance. Perte = débit relatif à RZF. Mêmes poids pour toutes les SC.

**Ce qui est établi** (M8K4, 20 MHz, UMi, débit à 20 dB sauf mention) :

| # | Constat | Preuve |
|---|---|---|
| 1 | En CSI parfait, il n'y a presque rien à gagner par SC (WMMSE = RZF + 1,7 %) | phase 0 |
| 2 | La marge est dans l'estimation : à pilote 10 dB, RZF passe de 26,2 (parfait) à 21,7 (LS) et 23,5 (LMMSE) | phase 0 |
| 3 | Le softmax empêche de soustraire : +19 points avec des poids signés | phase 1, 2 × 2 |
| 4 | Ce sont les itérations qui comptent, pas les paramètres : 1 bloc × 4 = 4 blocs, avec 3,7 fois moins de paramètres | phase 1, highsnr |
| 5 | L'attention linéaire (sans exp, division ni tanh) vaut l'attention signée | phases 1, 2, 3 |
| 6 | La perte relative à RZF corrige le faible SNR | phase 1, candidat |
| 7 | **Sans voisines, entraîner sur du CSI estimé n'apporte rien** (= RZF(LS)) | phase 2 |
| 8 | **Avec les voisines, le modèle bat le lissage classique** (23,15 contre 22,67) | phase 3 |
| 9 | Le goulot avec la fenêtre est la largeur : D=64 donne 23,65 | phase 3, diagnostic |
| 10 | **Mais il reste sous la meilleure méthode classique WMMSE(LMMSE)** : 23,65 contre 23,93 (98,8 %), 97 à 99 % à tous les SNR | phase 3, `wmmse_ref_p10` |

**Négatifs** : sortie structurée W = Hᴴ C, biais Gram, normalisation de phase, attention signée avec fenêtre.

**Ce que le modèle a pour lui face à WMMSE(LMMSE)** : il n'a pas besoin de la covariance du canal, il n'itère pas sur une inversion de matrice, il s'exécute en un temps fixe, et son attention n'a que des multiplications et des additions. **Ce qui lui manque** : 1 à 3 % de débit, et un coût de calcul de 590 k MAC/SC à D=64 (environ 420 fois RZF).

| 11 | **Scénarios mélangés : LMMSE avec covariance moyenne perd 1 bps/Hz sur RMa (canal plat)** ; notre modèle à filtre fixe ne le récupère pas (24,09 contre 24,20 pour WMMSE(LMMSE mélange), 25,24 pour l'oracle) | phase 3, `mixed_p10` |

**Questions ouvertes** : (a) est-ce la fenêtre trop étroite (W=13) ou le filtre fixe qui empêche de profiter de RMa ? (b) une attention en fréquence (adaptative) récupère-t-elle le 1 bps/Hz ? (c) peut-on traiter la fenêtre plus économiquement que D=64 partout ?

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

## Prochaine étape
Sur le mélange UMi + UMa + RMa : (1) fenêtre plus large (W = 25, 49) avec la même concaténation, pour savoir si c'est la largeur qui manque ; (2) si ça ne suffit pas, une attention en fréquence (filtre adaptatif).
Cible principale ensuite (phases 2–3) : le régime pilote 10 dB, où il reste 2,7 à 5,9 bps/Hz après LMMSE.
