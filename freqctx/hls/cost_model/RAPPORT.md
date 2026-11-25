# Modele de cout FPGA pour ART -- calibre par synthese HLS

## 1. Objectif et methode

ART (Adaptive Rounding/biT-width, ou equivalent) apprend b_w (bits de poids)
et b_a (bits d'activation) par couche en minimisant `-debit + lambda * cout`.
Le `cout` utilise jusqu'ici le modele d'energie de Moons et al. 2017,
`E(b) ~= (b/16)^1.9` avec `b = max(b_w, b_a)` : un modele pense pour un
circuit ASIC dedie, pas pour un FPGA. Ce travail remplace ce `cout` par un
modele calibre sur de la synthese HLS reelle (Vitis HLS 2022.2, cible
`xcu280-fsvh2892-2L-e`, horloge visee 10 ns), pour une couche dense deroulee
IN=16, OUT=16 (256 MAC), avec le noyau accumule en largeur `acc_w` adaptee a
(b_w, b_a).

Deux scenarios de circuit ont ete synthetises :
- **Cas A (poids constants)** : les poids sont figes a la compilation
  (`ap_int<b_w> const W[...]`), representatif du deploiement vise par ART
  (quantifier puis figer). Vitis HLS remplace `x * constante` par un reseau
  d'additionneurs/decaleurs (sauf exception, voir plus bas).
- **Cas B (poids variables)** : poids passes en argument (`ap_int<b_w>`),
  representatif d'un accelerateur reconfigurable generique ; sert de point
  de comparaison, pas le scenario cible.

Grille : `b_w in {1,2,3,4,6,8}`, `b_a in {2,3,4,6,8}`, 30 points cas A + 30
points cas B. En plus, 8 points `BIND_OP` forces (impl=fabric -> LUT pur,
impl=dsp -> DSP pur) sur 4 configs representatives du cas B
(`bw=ba in {1-2,4,6,8}`), pour isoler le cout d'un multiplieur pur LUT vs un
multiplieur DSP48. **68 points au total, tous synthetises avec succes**
(0 echec). Chaque synthese a ete lancee sur `srvgrm07` (licence Xilinx
distante) via `vitis_hls -f run_synth.tcl`, au plus 4 en parallele.

Extraction : `extract_results.py` parse chaque `report/csynth.xml` ->
`cost_table.csv` / `cost_table.json` (LUT, FF, DSP, BRAM, latence, II,
Fmax estime). Ajustement : `fit_model.py` (regression lineaire moindres
carres, numpy) -> `fit_coeffs.json`. Modele final exploitable : `fpga_cost.py`.

## 2. Tableau resume (LUT, DSP par MAC)

### Cas A -- poids constants (deploiement cible)

| b_w | DSP/MAC | LUT/MAC (min-max sur b_a) |
|---|---|---|
| 1 | 0 | 4,5 -- 6,5 |
| 2 | 0 | 7,2 -- 11,6 |
| 3 | 0 | 12,0 -- 16,4 |
| 4 | 0 | 16,9 -- 21,4 |
| 6 | **0,17 -- 0,23** (regime DSP, non lineaire) | 22,0 -- 25,0 |
| 8 | **0,39 -- 0,43** (regime DSP, non lineaire) | 23,3 -- 25,7 |

Constat central : **le cas A n'est "sans DSP" que pour b_w <= 4** (DSP=0
observe sur les 20 points). A partir de b_w=6, Vitis HLS commence a inferer
des DSP48 pour une fraction des multiplicateurs constants (ex. b_w=8 ->
100-110 DSP / 256 MAC), avec un ratio qui varie avec b_a de facon non
lineaire -- ce n'est pas le reseau shift-add "pur LUT" suppose au depart.
Ce changement de regime n'est pas modelise par une formule continue (voir
Limites).

### Cas B -- poids variables (reference, hors deploiement cible)

| b_w | DSP/MAC | LUT/MAC (min-max sur b_a) |
|---|---|---|
| 1 | 0 (palier exact) | 15,1 -- 26,6 |
| 2,3,4,6,8 | **0,5 (palier exact, tous b_a)** | 9,4 -- 31,2 |

Vitis empaquette systematiquement 2 petites multiplications variable x
variable dans 1 DSP48E2 (SIMD packing), independamment de b_a, pour tout
b_w >= 2 -- sauf b_w=1 (poids ternaire, pas de vrai multiplieur).

### Points `BIND_OP` forces (cas B, 4 configs : bw=ba in {1/2, 4, 6, 8})

| config | impl=dsp : DSP/MAC | impl=dsp : LUT | impl=fabric (LUT pur) : LUT |
|---|---|---|---|
| bw=1,ba=2 | 1,0 | 3456 | 6016 |
| bw=4,ba=4 | 1,0 | 1970 | 8624 |
| bw=6,ba=6 | 1,0 | 2418 | 15216 |
| bw=8,ba=8 | 1,0 | 2866 | 21808 |

Sans packing (1 DSP/MAC exact, confirme sur les 4 points), le DSP reste
avantageux en LUT par rapport a un multiplieur 100% LUT, et l'ecart
croit fortement avec la largeur : 1,7x a bw=1-2, 4,4x a bw=4, 6,3x a bw=6,
jusqu'a 7,6x a bw=8.

## 3. Modele ajuste

### 3.1 Cas A (modele retenu pour ART), en 2 morceaux

Le fit trilineaire global sur `b_w in {1,2,3,4}` donnait un residu de signe
incorrect sur `b_w=1` (le cout mesure CROIT avec b_a, le fit global le
faisait decroitre) : `b_w=1` est un cas degenere (poids ternaire {-1,0},
pas de vrai multiplieur, juste mux/negation), sa pente en b_a n'a pas de
raison de suivre la meme loi que `b_w>=2` (vrai reseau shift-add). Le
modele est donc separe en 2 morceaux, raccordes a `b_w=1,5` :

```
b_w == 1        : LUT/MAC(1, b_a)   = 0,359 * b_a + 3,862      (R2=0,853, RMSE=0,32, n=5)
b_w in {2,3,4}  : LUT/MAC(bw, ba)   = 0,798*bw*ba - 1,763*ba + 11,615   (R2=0,834, RMSE=1,59, n=15)
```

R2 global pondere sur les 20 points du regime propre : ~0,85 (equivalent a
l'ancien fit a 1 morceau, mais avec un comportement physiquement correct
sur b_w=1).

**Decomposition "cout par poids + cout par activation"** (pour la perte
ART), choisie pour que `cout_poids + cout_activation == LUT/MAC` exactement :

```
b_w == 1       : cout_poids(1, ba)  = 3,862         (constante, pas de terme croise)
                 cout_activation(ba) = 0,359 * ba
b_w in {2,3,4} : cout_poids(bw, ba) = 0,798*bw*ba + 11,615
                 cout_activation(ba) = -1,763 * ba
```

Pour une couche a `N_w` poids et `N_a` activations :
`cout_layer = N_w * cout_poids(bw, ba) + N_a * cout_activation(ba)`.

Implementation dans `fpga_cost.py` : `lut_per_mac_caseA`, `cost_per_weight`,
`cost_per_activation`, `layer_cost`, et une version **torch differentiable**
`layer_cost_torch(n_weights, n_activations, bw, ba)` (bw, ba peuvent etre
des tenseurs flottants pendant la relaxation continue d'ART, le branchement
entre les 2 morceaux se fait avec `torch.where`, differentiable partout
sauf exactement au point de jonction b_w=1,5 qui n'est jamais une valeur
entiere recherchee par ART).

### 3.2 Resultat surprenant : b_a pese peu (voire negativement) sur le cout cas A

Pour `b_w in {2,3,4}`, le coefficient de `b_a` seul est **negatif**
(-1,763) : a `b_w` fixe, augmenter `b_a` de 2 a 8 bits fait LEGEREMENT
BAISSER le LUT/MAC (ex. bw=2 : 11,3 -> 10,3). Le terme croise `0,798*bw*ba`
domine seulement quand `bw` augmente aussi. Pour `b_w=1`, le cout croit
doucement avec `b_a` (+0,36/bit). Dans tous les cas, **le modele FPGA cas A
est tres majoritairement sensible a `b_w`, beaucoup moins (voire pas du
tout) a `b_a`** -- contrairement au modele de Moons qui traite `b_w` et
`b_a` de facon symetrique via `max(b_w, b_a)`.

### 3.3 Comparaison avec le modele de Moons

Grille comparee : `b_w in {1,2,3,4}` x `b_a in {2,3,4,6,8}` (20 configs,
seul domaine calibre proprement). Correlation de rang de Spearman entre
`LUT/MAC` (cas A) et `(max(bw,ba)/16)^1.9` (Moons) : **rho = 0,47**
(correlation moderee, pas forte). Le choix optimal (minimum absolu) est le
meme sous les deux modeles (`b_w=1, b_a=2`), mais le classement complet
diverge vite ensuite : Moons classe `b_w=2,b_a=2` comme 2e meilleur choix
(car `max(2,2)=2` reste petit), alors que le modele FPGA le classe tres
mal (11,3 LUT/MAC, pire que b_w=1 a n'importe quel b_a) -- parce que passer
de b_w=1 (mux/negation) a b_w=2 (vrai multiplieur constant) a un cout
materiel disproportionne que Moons ne capture pas. **Conclusion : le
classement des choix de bits N'EST PAS le meme sous les deux modeles au-
dela du strict optimum** ; Moons sous-estime fortement le cout de passer de
b_w=1 a b_w>=2, et surestime le cout de b_a.

## 4. Qualite de l'ajustement

- Cas A, bw=1 : R2=0,85, RMSE=0,32 LUT/MAC (n=5, extrapolation limitee).
- Cas A, bw in {2,3,4} : R2=0,83, RMSE=1,59 LUT/MAC (n=15). Residus jusqu'a
  ~3,6 LUT/MAC sur des valeurs de 7-24 (erreur relative ~15-25% par point,
  acceptable pour un terme de regularisation dans une perte, pas pour un
  dimensionnement precis).
- Cas B (defaut) : R2=0,76, RMSE=3,5 LUT/MAC (n=30) -- plus bruite, logique
  de routage autour des DSP moins reguliere.
- Cas B force LUT (multiplieur pur) : R2=0,99 (n=4 seulement -- trop peu de
  points pour un intervalle de confiance serieux, mais la tendance est tres
  nette).
- Cas B force DSP : DSP/MAC = 1,0 exact sur les 4 points (pas une
  regression, une verification).

## 5. Limites

1. **Regime b_w>=6 non modelise** : Vitis HLS alloue des DSP de facon
   fractionnaire et non lineaire au-dela de b_w=6 pour le cas A (poids
   constants). Le modele de `fpga_cost.py` emet un avertissement
   (`warnings.warn`) et extrapole quand meme la formule bw in{2,3,4} au-
   dela de 4, ce qui est FAUX pour la partie DSP -- a n'utiliser que si
   ART est contraint a b_w<=4 (regime d'interet pour la quantification
   agressive).
2. **Un seul gabarit de noyau** : IN=16, OUT=16, acc_w ajuste par point.
   Le passage a un gabarit plus grand (fan-in/fan-out different, registres
   de pipeline plus larges) changera les constantes absolues ; la forme
   fonctionnelle (terme croise bw*ba) devrait rester pertinente mais n'a
   pas ete verifiee a une autre taille.
3. **Cout = aire combinatoire (LUT/FF/DSP), pas energie ni frequence** :
   Fmax estime varie fortement entre points (143,7 -- 362,1 MHz sur les 68
   points), or le modele actuel ignore cette variation. Un design moins
   couteux en LUT peut aussi etre plus lent, ce qui n'est pas capture par
   `fpga_cost.py` en l'etat (pas de terme de debit materiel/Fmax).
4. **DSP et LUT ne sont pas convertis en une seule unite de cout de facon
   rigoureuse.** Le modele ART cas A reste purement LUT (puisque DSP=0 sur
   son domaine valide), donc ce n'est pas un probleme pour bw<=4, mais cela
   signifie qu'il n'y a pas de poids LUT<->DSP calibre pour un usage plus
   general (cas B, ou bw>=6).
5. **Peu de points pour les ajustements fins** : le morceau bw=1 (n=5) et
   les points `BIND_OP` forces (n=4 chacun) laissent peu de marge pour
   estimer un intervalle de confiance.
6. **Un seul seed/une seule instanciation de poids par point** (cas A) :
   la variabilite point-a-point due au motif particulier des constantes
   (ex. plus ou moins de puissances de 2, donc plus ou moins de decalages
   purs) n'a pas ete quantifiee (pas de repetition avec plusieurs seeds).
7. **Accumulateur** : la largeur `acc_w` est choisie automatiquement par
   point (dependant de bw, ba) ; le cout de l'accumulateur n'est pas
   isole separement du cout du multiplieur dans le `LUT/MAC` rapporte.

## 6. Recommandation pour un ART oriente FPGA

1. **Utiliser le modele cas A (poids constants), pas Moons**, car c'est le
   scenario de deploiement reel vise par ART (poids figes a l'issue de
   l'entrainement). Integrer `fpga_cost.py::layer_cost_torch` dans la perte
   (`-debit + lambda * layer_cost_torch(...)`), qui est differentiable en
   (b_w, b_a) pour la relaxation continue d'ART.
2. **Ponderer b_w beaucoup plus fort que b_a** dans la recherche de bits :
   le cout materiel reel est tres majoritairement pilote par b_w (terme
   croise + saut b_w=1->2) ; b_a a un effet faible, parfois meme negatif
   sur ce gabarit. Un ART qui utilise Moons (symetrique en max(bw,ba))
   va sur-penaliser b_a et sous-penaliser le passage a b_w>=2, menant a
   des choix de bits sous-optimaux une fois deployes sur FPGA.
3. **Contraindre la recherche a b_w <= 4** (ou ajouter une penalite forte
   / une resynthese dediee pour explorer au-dela), le modele actuel n'etant
   pas fiable pour b_w in {6,8} (inference DSP non lineaire, non calibree).
4. **Ne pas traiter ce modele comme un cout d'energie** : c'est un cout
   d'aire combinatoire (LUT-equivalent). Si le critere final d'ART est
   l'energie ou le debit materiel, il faudra completer avec une estimation
   de Fmax/puissance (non couverte ici, cf. limite 3).
5. Si une precision plus fine est necessaire a terme (ex. pour un papier),
   envisager : (a) plusieurs seeds par point cas A pour quantifier le bruit,
   (b) un second gabarit de taille differente pour verifier la portabilite
   de la forme fonctionnelle, (c) une resynthese cible du regime b_w in
   {6,8} avec un modele dedie (DSP fractionnaire).

## 7. Fichiers produits

- `cost_table.csv` / `cost_table.json` : 68 points, LUT/FF/DSP/BRAM/latence/
  II/Fmax.
- `fit_coeffs.json` : coefficients ajustes (cas A 2 morceaux, cas B defaut,
  cas B force LUT/DSP), avec R2/RMSE/n.
- `fpga_cost.py` : modele exploitable (fonctions scalaires + version torch
  differentiable), comparaison avec Moons (`compare_rankings`).
- `fit_model.py` : script d'ajustement (reproductible depuis `cost_table.csv`).
- `extract_results.py` : extraction des rapports de synthese -> `cost_table.*`.
- `manifest.json`, `_status/` : liste des points prevus et marqueurs de
  completion (68/68 `.done`, 0 echec, cf. `run_all.log`).
