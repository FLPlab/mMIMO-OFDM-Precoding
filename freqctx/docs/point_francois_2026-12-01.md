# Point d'avancement — rencontre du 1er décembre 2026 (brouillon)

Suite de la réunion du 25 novembre. Débits en bps/Hz, M = 8 antennes et K = 4 utilisateurs sauf mention contraire.
Détails et chiffres complets : `FINDINGS.md`, phases 14 à 19.

## 1. Zero-shot sur les sites du labo (entraîné sur 3GPP seulement)
- Jusqu'à 20 dB, le précodeur est au niveau de WMMSE convergé (300 it.) sur St-Catherine, Décarie et Ericsson, sans aucune donnée de site.
- L'écart au fort SNR est expliqué : ce n'est pas un problème de site mais une limite de précision du réseau (l'interférence résiduelle plafonne vers −45 dB, quel que soit le canal).
- Il est corrigé par une **deuxième passe** : un petit transformer qui reçoit le canal effectif de la première solution (ce que chaque utilisateur recevrait) et la corrige. Activée seulement au-delà de 20 dB.
- Résultat, CSI parfait, 30 dB : St-Catherine 33,7 (WMMSE 33,6), Décarie 35,3 (35,0), Ericsson 35,9, et 36,1 en répétant la passe (36,2).

## 2. Fine-tuning (complet, LoRA)
- Peu utile : +0,03 à +0,3 bps/Hz avec 400 à 4 000 canaux du site, même en n'adaptant que la deuxième passe.
- Cohérent avec le diagnostic : l'écart ne vient pas du site. **Le zero-shot suffit.**

## 3. CSI imparfait (erreur synthétique à l'entraînement, vrai estimateur au déploiement)
- Précodeur seul + estimateur LMMSE : au niveau de WMMSE robuste ou au-dessus sur UMi et sur les 3 sites jusqu'à un pilote de 10 dB ; avec un bon pilote (20 dB), 0,1 à 0,4 en dessous.
- **Chaîne complète depuis les pilotes** (encodeur fréquentiel + précodeur + deuxième passe), pilote 20 dB, 30 dB : St-Catherine 31,9 contre 30,7 (meilleure chaîne classique), Décarie 33,5 contre 32,3, Ericsson 35,0 contre 34,7. À pilote 0 dB, Ericsson : 30,6 contre 27,5.

## 4. BER avec modulation
- Modulation adaptative par flux (BER cible 10⁻³, non codé) : meilleur débit transmis que WMMSE et RZF (9,98 contre 9,56 et 2,7 à 20 dB).

## 5. Complexité et énergie face à RZF (et WMMSE)
- **Correction par rapport au 25 novembre** : la forme de WMMSE comptée alors n'était pas la meilleure. La meilleure implémentation publiée, R-WMMSE (Zhao et al., IEEE TSP 2023 : itération sur une matrice K×K), donne le même précodeur pour bien moins de calcul. RZF et WMMSE tiennent en 12-15 bits en virgule fixe (vérifié).
- Un audit des calculs a corrigé une erreur de notre côté (bits comptés b au lieu de b + 1).
- **Face à RZF** : il coûte 3 à 5 fois moins en calcul, mais perd 7 bps/Hz à St-Catherine et 5 à Ericsson. **Battre RZF en complexité n'est pas atteignable** ; notre gain est en débit.
- **Face à R-WMMSE** (énergie de calcul, débit égal) : 1,4 à 2 fois moins cher sur le canal difficile (St-Catherine), à égalité sur le canal facile (Ericsson).
- Face à WMMSE compté comme dans la littérature classique (une inversion M×M par itération) : 40 à 1 600 fois moins cher.
- **À l'échelle du système**, si le précodeur est recalculé une fois par créneau, son énergie devient négligeable devant la puissance émise dès quelques watts : l'efficacité énergétique est dominée par le débit.
- Notre avantage vient de la tolérance à la quantification (3 à 9 bits contre 13 à 15), pas d'un nombre d'opérations plus faible.
- Pour réduire le calcul : calculer moins de sous-porteuses selon l'étalement des retards et le SNR (÷ 1,3 à ÷ 1,9 en moyenne, sans perte notable). Réduire la largeur du réseau fait perdre le zero-shot.

## 6. FPGA (à compléter avec les résultats de placement-routage et de cosimulation)
- Précodeur 2 couches quantifié (St-Catherine), Vitis HLS, Alveo U280 : 64 sous-porteuses en 3,8 µs (4 instances par carte) ; architecture repliée : une instance tient un tiers de bande 5G.
- Coût FPGA intégré à ART (poids 2-3 bits sans DSP).

## Résultat nouveau
En OFDM, la « décision » du précodeur (qui reçoit quelle puissance) varie très lentement en fréquence, alors que « l'exécution » (annuler l'interférence avec le canal local) doit être refaite à chaque sous-porteuse.

## Suite proposée
- FPGA sur l'architecture retenue ; quantification de l'encodeur.
- Contrainte de puissance par antenne.
- Contrôle de débit par utilisateur (suite d'Emery et al.).
