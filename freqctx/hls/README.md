# HLS du précodeur par paires quantifié

Chaîne : modèle ART (PyTorch, `torch_port/art_stecath.py`) → extraction (`model.py`) → références → C++ généré.

| Fichier | Rôle |
|---|---|
| `model.py` | `QuantPairPrecoder` : poids entiers, échelles LSQ, bits (convention LSQ du labo : b + 1 bits signés) ; `hidden`/`layers` déduits du nom (`art_pair_h16_L2_...`) |
| `golden.py` | `Golden` (flottant, = PyTorch quantifié) et `GoldenFixed` (choix des formats : F = 12 bits fractionnaires) |
| `intref.py` | `IntPairPrecoder` : référence ENTIÈRE, exactement les calculs du C++ (constantes entier 16 bits × 2^-e, racine entière, inverse de racine par table + Newton, log2 et GELU par tables) ; relève les maxima (`peak`) pour dimensionner les `ap_int` |
| `codegen.py` | une petite classe par bloc du réseau (`MathGen`, `DenseGen`, `QuantGen`, `LayerNormGen`, `NormalizeGen`, `GramGen`, `InitGen`, `ProjGen`, `FfnGen`, `ReadGen`, `CombineGen`), chacune écrit ses constantes et son C++ ; `Project` les assemble selon `--arch` |

```bash
python -m hls.golden      # débit : PyTorch / golden flottant / golden virgule fixe
python -m hls.intref      # débit de la référence entière + maxima des grandeurs intermédiaires
python -m hls.codegen --arch flow --out /users/sala/Documents/test_projet/hls_work/l2_flow [--stats-n 5000] [--n 32]
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

Historique : `flow` version tableaux (DATAFLOW par tableaux PIPO, résultats l2_flow/l2_flow2) remplacée par les streams.

## Numérique
- Entrée : mantisses 16 bits + un exposant par utilisateur ; F = 12 bits fractionnaires ; échelles = entier 16 bits × 2^-e.
- Aucune division : inverse de racine `irsqrt` = table de 193 valeurs interpolée + un tour de Newton (Q30).
  Perte de débit (5 000 canaux St-Catherine) contre les divisions exactes : 0,004 bps/Hz (2 couches), 0,002 (8 couches).
- Largeurs : opérandes quantifiés sur b + 1 bits, accumulateurs par bornes exactes, valeurs intermédiaires (`act_t`)
  sur max relevé + 2 bits de marge (`--stats-n` canaux).
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
