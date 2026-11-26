# HLS du précodeur par paires quantifié

Chaîne : modèle ART (PyTorch, `torch_port/art_stecath.py`) → extraction (`model.py`) → références → C++ généré.

| Fichier | Rôle |
|---|---|
| `model.py` | `QuantPairPrecoder` : poids entiers, échelles LSQ, bits (convention LSQ du labo : b + 1 bits signés) ; `hidden`/`layers` déduits du nom (`art_pair_h16_L2_...`) |
| `golden.py` | `Golden` (flottant, = PyTorch quantifié) et `GoldenFixed` (choix des formats : F = 12 bits fractionnaires) |
| `intref.py` | `IntPairPrecoder` : référence ENTIÈRE, exactement les calculs du C++ (constantes entier 16 bits × 2^-e, racine entière, inverse de racine par table + Newton, log2 et GELU par tables) ; relève les maxima (`peak`) pour dimensionner les `ap_int` |
| `codegen.py` | une petite classe par bloc du réseau (`MathGen`, `DenseGen`, `QuantGen`, `LayerNormGen`, `NormalizeGen`, `GramGen`, `InitGen`, `ProjGen`, `FfnGen`, `ReadGen`, `CombineGen`), chacune écrit ses constantes et son C++ ; `Project` les assemble selon `--arch` ; options `--cmul fabric` (constantes 16 bits en LUT), `--fold S` |
| `foldgen.py` | architecture repliée `fold` (`FoldDense`, `FoldStages`, `FoldProject`) : S cycles par token, voir ci-dessous |

```bash
python -m hls.golden      # débit : PyTorch / golden flottant / golden virgule fixe
python -m hls.intref      # débit de la référence entière + maxima des grandeurs intermédiaires
python -m hls.codegen --arch flow --out /users/sala/Documents/test_projet/hls_work/l2_flow [--stats-n 5000] [--n 32]
python -m hls.codegen --arch fold --fold 8 --model art_pair_h16_L2_lam0.3_costfpga_qprod_norm --out .../l2_fold8_l03
cd .../l2_flow && g++ -O2 -I$VITIS_HLS/include -o tb_native tb.cpp precoder.cpp && ./tb_native   # banc natif (bit à bit)
vitis_hls -f run_csim.tcl          # simulation C (srvgrm12) ; run_synth_grm07.tcl : synthèse (srvgrm07, horloge 4 ns)
```

Fichiers générés : `types.h` (dimensions, `typedef ap_int<...>`), `weights.h` (un `struct L_<couche>` par couche dense :
poids, multiplicateurs, décalages, biais, bits ; tables GELU/log2/inverse de racine), `precoder.h/.cpp`, `vectors.h`
(entrées en virgule flottante par bloc et sorties attendues de `IntPairPrecoder`), `tb.cpp`, `widths.txt`.

## Architectures (`--arch`, options `--nsc`, `--ant`)

Les étages (normalisation ×2, G ×2, init, [proj, ffn] × couches, lecture, W ×2) sont des processus reliés par des
`hls::stream` : un élément par token de paire (`vec_t`), par groupe d'antennes (`mgrp_t`, `hgrp_t`, `wgrp_t`) ou par
sous-porteuse (`ri_sc_t`, `gsc_t`, `qa_sc_<i>`, `qb_sc_<i>`, `c_sc_t`, émis au dernier token et lus au premier — ce qui remplace le
double tampon des étages qui ont besoin de toute la sous-porteuse : puissance, G, produit triangulaire, W).

| arch | principe |
|---|---|
| `seq` | appels séquentiels (pas de DATAFLOW), une seule fonction `dense_rt<NIN, NOUT>` partagée par forme (poids en ROM, une sortie par cycle), 1 antenne par cycle — le minimum de ressources |
| `flow` | processus en `DATAFLOW`, une sous-porteuse par appel ; dans chaque processus une boucle sur les 16 tokens `PIPELINE II=1`, couches denses entièrement déroulées (poids = constantes), 4 antennes par cycle ; 64 SC = latence + 63 × II |
| `stream` | idem mais 64 sous-porteuses par appel : chaque processus est une boucle aplatie (sc, token) de 1 024 itérations à II = 1, le flot ne se vide jamais entre deux sous-porteuses (latence 64 SC ≈ 1 024 + somme des profondeurs) |
| `fold` (`--fold S`) | `stream` replié : chaque processus traite un token en S cycles (boucle (token, s) à II = 1, 1/S du travail par cycle). Couches denses : NOUT/S sorties par cycle, poids en petites ROM de S entrées par position de MAC (produits en LUT, `BIND_OP fabric`), quantification de l'entrée pipelinée sur le token précédent dans des registres à décalage (NIN/S multiplieurs par la même constante), NOUT/S rescalers ; LayerNorm en deux processus (`ln1` moyenne + variance par tranches + une irsqrt, `ln2` affine) ; `normalize_fin` en boucle sur k (une irsqrt et un log2 au lieu de K déroulés) ; ef relayé dans le token ; étages par antenne à ANT = max(1, 4/S) ; FIFO de tokens à profondeur 1, `h_w` = NG·NSC |

Historique : `flow` version tableaux (DATAFLOW par tableaux PIPO, résultats l2_flow/l2_flow2) remplacée par les streams.

## Numérique
- Entrée : mantisses 16 bits + un exposant par utilisateur ; F = 12 bits fractionnaires ; échelles = entier 16 bits × 2^-e.
- Aucune division : inverse de racine `irsqrt` = table de 193 valeurs interpolée + un tour de Newton (Q30).
  Perte de débit (5 000 canaux St-Catherine) contre les divisions exactes : 0,004 bps/Hz (2 couches), 0,002 (8 couches).
- Largeurs : opérandes quantifiés sur b + 1 bits, accumulateurs par bornes exactes, valeurs intermédiaires (`act_t`)
  sur max relevé + 2 bits de marge (`--stats-n` canaux). `irsqrt` en largeurs exactes (33 × 34 bits : y ≤ 2^31, Newton
  < 2^32) au lieu d'`ap_int<64>` : mêmes valeurs, ~14 DSP par instance au lieu de ~39.
- Simulation C et banc natif identiques bit à bit à `IntPairPrecoder` pour toutes les architectures.

## Résultats de synthèse (Vitis HLS 2022.2, xcu280, horloge cible 4 ns, modèle 2 couches h16, 143 k MAC)

Projets et rapports : `/users/sala/Documents/test_projet/hls_work/<variante>/pair_precoder_csynth.rpt` (résumé : `scripts/report.py`).
Toutes les variantes : banc natif et simulation C Vitis identiques bit à bit à `IntPairPrecoder` (64 sous-porteuses).

| variante | générateur | latence | II | 64 SC (cycles) | 64 SC (µs) | Fmax est. | DSP | LUT | FF | BRAM | instances U280 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| l2_v3 (agent précédent, int64, sans pragma) | v3 | 110 222 | 110 223 | 7,05 M | 51 500 | 137 MHz | 1 802 | 82,7 k | 33,4 k | 98 | 5 |
| l2_seq2 : `seq`, tableaux, `dense_rt` partagée | tableaux | 16 599 | 16 600 | 1,06 M | 4 171 | 255 MHz | 364 | 89,6 k | 33,1 k | 14 | 14 |
| l2_seq_s : `seq`, streams | final | 33 962 | 33 963 | 2,17 M | 8 534 | 255 MHz | 348 | 175 k | 109 k | 368 | 7 |
| l2_flow : DATAFLOW tableaux, 1 token/cycle | tableaux | 533 | 101 | 6 896 | 21,7 | 317 MHz | 1 962 | 376 k | 192 k | 75 | 3 |
| l2_flow2 : idem, 2 tokens/cycle | tableaux | 478 | 101 | 6 841 | 21,6 | 317 MHz | 4 079 | 636 k | 310 k | 133 | 2 |
| l2_flow_s : `flow`, streams, 1 SC/appel | final | 303 | 62 | 4 209 | 12,3 | 342 MHz | 1 945 | 294 k | 181 k | 345 | 4 |
| **l2_stream : `stream`, 64 SC/appel** | final | **1 341** (lot de 64 SC) | 1 079 (lot) | **1 341** | **3,9** | 342 MHz | 2 147 | 299 k | 189 k | 343 | **4** |

- `flow`/`flow2` (tableaux) : II = 101 imposé par l'étage `normalize` (deux boucles de 16 + irsqrt + log2 en série) ;
  dérouler les tokens (flow2) ne change rien et double les ressources — la profondeur de pipeline (~45 cycles) domine
  les 16 itérations, taux d'occupation ~27 %.
- `flow_s` (streams, étages scindés en acc/scale) : II 62 (étage proj, 16 + 45), latence 303.
- `stream` : les mêmes processus bouclent sur 64 × 16 tokens sans se vider : 64 SC en 1 341 cycles (1 024 + ~300 de
  profondeur), un nouveau lot toutes les 1 079 cycles soit ~17 cycles par sous-porteuse en régime établi ; 4 instances
  tiennent sur l'U280 (limite LUT, puis DSP) → 256 SC par 1 079 cycles. C'est l'architecture recommandée : 5× plus
  rapide que `flow` pour les mêmes ressources, et le déroulage complet des 16 tokens (II = 1 par SC) coûterait ~16× les
  LUT de proj/ffn (~3,5 M), impossible sur l'U280.
- Première synthèse de `stream` : II = 4 dans `normalize_acc` (lectures multiples de `ex[sc][k]` sur un port BRAM) →
  corrigé en lisant chaque exposant une fois (tableau `exk`) et en partitionnant `ex` : 4 380 → 1 341 cycles.
- DSP : surtout les multiplications par les constantes 16 bits (rescale, quantification) et les LayerNorm
  (carrés, × inv) ; les MAC des couches denses (poids 4-8 bits) sont en LUT.
- Limites : `seq` en version streams (final) est moins bon que la version tableaux (seq2) ; les profondeurs de FIFO
  (h_w = 1024, ef = 128·(2i+1)) sont dimensionnées pour 64 SC sans cosimulation RTL (à vérifier par cosim).

Passage à l'échelle (`art_pair_h32_lam5_qprod_norm`, 8 couches, 738 k MAC, `stream`, synthèse 1 h 20) : l8_stream,
64 SC en 1 878 cycles (6,4 µs @ 295 MHz), lot toutes les 1 080 cycles, 7 252 DSP, 1 275 k LUT, 728 k FF, 1 325 BRAM —
une seule instance, à la limite des LUT de l'U280 (98 %) : il faudrait replier une partie des couches denses
(DSP ou `dense_rt`) pour garder une marge de placement. Banc natif et csim bit à bit.

## Co-conception ART-FPGA et repli (25 nov., soir)

Budget 5G NR (100 MHz à 30 kHz) : 3 276 sous-porteuses par créneau de 0,5 ms (6,55 M SC/s), 1 092 avec l'interpolation
1 SC sur 3 validée côté ML (2,18 M SC/s) : à 300 MHz, ~137 cycles par SC sont disponibles là où `stream` en prend 17.
`scripts/report.py` donne SC/s par instance et par U280, et si une instance tient le créneau.

Poids : ART à coût FPGA calibré (`torch_port/art_stecath.py --cost fpga`) : `lam1_costfpga` (23,24 bps/Hz en référence
entière, b_w 1-2 bits, b_a 3-8), **`lam0.3_costfpga` (24,56 bps/Hz, b_w 2-4, b_a 4-8 ; meilleur que la référence
énergie `lam5_qprod_norm` à 24,07 avec b_w ≤ 7)**. Toutes les variantes : banc natif et csim bit à bit (64 SC).

| variante | arch | poids | 64 SC cycles / µs | Fmax | DSP | LUT | FF | BRAM | inst. U280 | M SC/s (inst. / U280) | 1/3 bande ; bande complète (1 inst.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| l2_stream (réf.) | stream | énergie λ5 | 1 341 / 3,9 | 342 | 2 147 | 299 k | 189 k | 343 | 4 | 20,3 / 81 | oui 9,3× ; oui 3,1× |
| l2_stream_fpga1 | stream | lam1_costfpga | 1 291 / 4,4 | 295 | 1 795 | 253 k | 187 k | 357 | 5 | 17,7 / 88 | oui ; oui |
| l2_stream_l03 | stream | lam0.3_costfpga | 1 294 / 3,8 | 342 | 1 788 | 279 k | 191 k | 357 | 4 | 20,5 / 82 | oui 9,4× ; oui 3,1× |
| l2_fold2 | fold S=2 | lam1_costfpga | 2 324 / 7,6 | 306 | 920 | 316 k | 179 k | 195 | 4 | 9,4 / 38 | oui 4,3× ; oui 1,4× |
| l2_fold4 | fold S=4 | lam1_costfpga | 4 381 / 12,8 | 343 | 488 | 228 k | 161 k | 152 | 5 | 5,3 / 27 | oui 2,4× ; non 0,8× |
| l2_fold4_l03 | fold S=4 | lam0.3_costfpga | 4 380 / 12,8 | 343 | 502 | 270 k | 169 k | 152 | 4 | 5,3 / 21 | oui 2,4× ; non 0,8× |
| l2_fold8 | fold S=8 | lam1_costfpga | 8 477 / 24,7 | 344 | 331 | 151 k | 154 k | 144 | 8 | 2,7 / 21 | oui 1,2× (≥ 280 MHz) ; non |
| **l2_fold8_l03** | fold S=8 | lam0.3_costfpga | 8 478 / 24,7 | 344 | **335** | 171 k | 158 k | 144 | 7 | 2,7 / 19 | oui 1,2× (≥ 280 MHz) ; non |
| l2_fold8_nrj | fold S=8 (v3) | énergie λ5 (b_w ≤ 7) | 8 597 / 25,0 | 344 | 329 | 182 k | 126 k | 142 | 7 | 2,6 / 18 | oui 1,2× ; non |
| l2_fold8_v1 | fold S=8, v1 | lam1_costfpga | 8 487 / 24,7 | 344 | 567 | 171 k | 182 k | 144 | 7 | 2,7 / 19 | (quantificateurs dupliqués, voir ci-dessous) |
| l2_fold8_l03_v3 / v4 | fold S=8, essais | lam0.3_costfpga | 8 598 / 25,0 | 344 | 335 | 182 k / 233 k | 138 k / 107 k | 144 | 7 / 5 | | (essais écartés, voir ci-dessous) |

Lecture :
- **Co-conception seule (`stream`, piste 1)** : les MAC des couches denses étaient déjà en LUT ; passer aux poids 1-4 bits
  ne fait qu'enlever les ~60 DSP des poids ≥ 5 bits et resserrer les accumulateurs : 2 147 → 1 788 DSP (dont ~180 par
  l'`irsqrt` resserrée), 299 → 253-279 k LUT, 4-5 instances. Les DSP de `stream` sont les multiplications par des
  constantes 16 bits (quantification ~430, rescale ~220), le produit triangulaire (256, 1 DSP par produit 8 × 8), W (100)
  et les LayerNorm (~190) — pas les poids. `--cmul fabric` (BIND_OP dans `quant`) fait planter la pré-synthèse de Vitis
  2022.2 sans message (l2_fold8_l03_fab) ; non poursuivi : en repli les DSP ne limitent plus (335 → 26 instances
  possibles contre 7 par les LUT), les passer en LUT ne ferait que perdre.
- **Repli (piste 4)** : S=8 divise les DSP par 5-6 (335) et les LUT par 1,7 (171 k) par rapport à `stream`, pour 17×
  moins de débit : une instance (13 % des LUT, 3,7 % des DSP de l'U280) tient les 1 092 SC du créneau si Fmax ≥ 280 MHz
  (estimée 344). S=4 (141 MHz suffisent) est le choix prudent ; S=2 tient la bande complète (1,4×). En SC/s par U280
  entier, `stream` reste 3-4× meilleur : les coûts fixes (irsqrt, FIFO, contrôle, registres de tokens) ne se replient pas.
- Où vont les LUT du repli (fold8_l03, 171 k) : couches denses 6 × 9-14 k = 97 k (pour ~1 000 positions de MAC : les
  multiplieurs LUT 5 × 3 bits ~29 LUT + arbres d'addition ~10 LUT/MAC = 40 k ; le reste = registres à décalage et
  multiplexeurs de chargement des tokens (~1 LUT par bit stocké : 665-1 273 bits par token), décaleurs des rescale,
  blocs a/b par sous-porteuse 2 × 1 280 bits), tri 2 × 8 k (copie des 4 blocs), FIFO 24 k (dont ~8 k pour les 8 FIFO de
  blocs de 2 304 bits ; profondeur 1 ou 2 : même coût),
  LN 13 k, normalize_fin 7 k, combine 6 k, gram 5 k. Le coût n'est donc plus dans les poids (lam5 énergie à b_w ≤ 7 en
  LUT : 182 k, même ordre) mais dans la structure : chaque bit de token stocké avec condition coûte ~1 LUT dans
  l'estimation Vitis.
- Essais écartés (mesurés) : v1 — tranche de quantification à indice variable `xq[s*QN+u]` : Vitis duplique le
  multiplieur constant sur les NIN éléments (567 DSP) → registres à décalage à indices constants (v2, 335 DSP) ;
  v3 — tokens a/b émis par token et collectés dans tri par registres à décalage : +9 k LUT par couche ; v4 — MAC par
  multiplexeur des multiples {0, ±x, ±2x, ±3x} (+3 k LUT par couche dense) et blocs a/b en LUTRAM double banque avec
  extraction de canal par décalage variable (+30 k LUT par couche). Retenu : v2 (= v5, FIFO de tokens à profondeur 1 :
  résultats identiques, Vitis implante une FIFO de profondeur 1 comme une de profondeur 2).
- Piste 3 (empaquetage DSP) non faite : après repli les produits variable × variable (tri 40, W 32, G 16, carrés 50)
  ne limitent rien. Piste 5 (8 couches h32) non synthétisée : extrapolation fold S=8 ≈ 35 k + 8 × ~70-130 k LUT
  (hidden 32 → couches denses 2-4× plus larges) ≈ 0,6-1,1 M LUT, soit une instance sur l'U280 avec marge incertaine ;
  `stream` 8 couches (1,28 M LUT) ne tient pas.
- Vérifications : II = 1 sur toutes les boucles de tous les processus (fold et stream) ; produits poids × activation
  tous en LUT (`mul_4s_2s`, `mul_5s_3s`… : 0 DSP) ; profondeurs de FIFO dimensionnées sans cosimulation RTL (comme
  avant) ; `h_w` = NG·NSC groupes (4 096 à ANT = 1, 16 BRAM) ne peut pas bloquer ; Vitis conseille d'augmenter les
  FIFO de profondeur 1 (indication de performance, débit mesuré inchangé : II du lot 8 219 → 8 331 cycles).

Pistes restantes : (a) tokens plus étroits — `act_t` 19 bits partout alors que les activations quantifiées font 4-8
bits : quantifier dans le producteur (ln2, tri, f1) et ne transporter que des entiers quantifiés diviserait par 3-4 les
registres/FIFO de tokens (~30 k LUT) et supprimerait la double file xq ; (b) rescale à décalage constant par couche
(mantisses alignées sur l'exposant minimal ; change la numérique, à mesurer) ; (c) ANT = 1/2 pour S = 8 (étages par
antenne à mi-régime) ; (d) cosimulation RTL d'un lot pour valider les profondeurs de FIFO.

