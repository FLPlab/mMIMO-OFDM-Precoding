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
Dans le mémoire (LOS forcé, utilisateurs dans 15°, 2,88 MHz), un pilote à 20 dB faisait tomber RZF de 40 à 27,6. En drops standard, la perte n'est que de 26,1 → 25,1. **Le gros gain de TA-RB tenait en grande partie à ce régime difficile.**

---

## Prochaine étape
Phase 1 : le transformer minimal par SC, en CSI parfait (ligne de base ≈ RZF, plus petit modèle possible).
Cible principale ensuite (phases 2–3) : le régime pilote 10 dB, où il reste 2,7 à 5,9 bps/Hz après LMMSE.
