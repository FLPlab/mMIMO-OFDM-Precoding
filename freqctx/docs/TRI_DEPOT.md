# Tri du dépôt freqctx — avant transfert vers équipe

> **Correction (session principale, 2026-11-25)** : cet inventaire a été fait par un agent léger, qui s'est appuyé
> sur la section « Plan » de `FINDINGS.md` datée du 2026-10-24, aujourd'hui dépassée. **Les phases 5 à 14 ne sont
> PAS des explorations secondaires** : elles contiennent le cœur actuel du travail (précodeur par paires, massif,
> base v2, puissances réelles, St-Catherine et ART, Wiener, encodeur à convolution, sites du labo, figures de
> présentation, HLS). Elles sont à **garder**. Restent valables : retirer `__pycache__/`, ne pas publier
> `docs/notes/` (notes personnelles) ni, au choix de l'auteur, `results/presentation/NOTES_FRANCOIS.txt` ;
> la proposition de `.gitignore`.

**Date:** 2026-11-25  
**Branche:** `freqctx`  
**Dépôt parent:** `/users/sala/Documents/test_projet/Trans/freq/Sionna`

---

## 1. État git (fichiers modifiés et non suivis)

### Modifiés (6 fichiers)
| Fichier | Raison |
|---------|--------|
| `FINDINGS.md` | Journal des résultats, à publier |
| `experiments/phase7_context.py` | Travail en cours |
| `models/delay_domain.py` | Modèle retenu (phase 4) |
| `models/per_sc.py` | Modèle retenu (phases 1–3) |
| `torch_port/art_stecath.py` | Port quantification ART (phase 10) |
| `torch_port/stecath.py` | Port quantification ART (phase 10) |

### Non suivis (40 items)
- **Scripts phases 12–13** (16 fichiers `phase12_*.py`, `phase13_*.py`, `presentation_figs.py`) : explorations en cours
- **Dossiers résultats** (`results/phase12/`, `results/phase13/`, `results/phase7/`, + autres) : résultats expérimentaux
- **Dossier HLS** : code synthèse pour FPGA (phase 10)
- **Fichier script queue** : `scripts/queue13_pareto.sh`
- **Fichiers torch_port** : scripts d'énergie et d'évaluation (phase 10)
- **Fichier PDF synthèse** : `docs/make_architecture_pdf.py` et résultats

---

## 2. Arborescence et tailles

| Dossier | Taille | Contenu |
|---------|--------|---------|
| `results/` | **11 M** | Résultats JSON/PNG (phases 0–13 + stecath) |
| `docs/` | 1,2 M | Figures, PDF, notes (`notes/note.pdf` = 1,2 M personnel) |
| `experiments/` | 741 K | Scripts d'expérience (85 scripts Python) |
| `torch_port/` | 182 K | Port PyTorch quantification ART (8 fichiers) |
| `models/` | 127 K | Modèles Keras/TensorFlow (4 fichiers clés) |
| `__pycache__/` | 60 K | **Cache Python, à supprimer** |
| `scripts/` | 40 K | Scripts queue/bash (3 fichiers) |
| `hls/` | 91 K | Génération code HLS (5 fichiers) |
| `precoders/` | 15 K | Précodeurs classiques (2 fichiers) |
| `tests/` | 66 K | Tests unitaires (3 fichiers `.py` + cache) |

**Plus gros fichiers (> 1 MB):**
- `docs/notes/note.pdf` — 1,2 MB (notes personnelles, à retirer)
- `FINDINGS.md` — 97 KB (journal article, à garder)
- Figures PNG dans `results/` — 50–400 KB chacune (20+ fichiers)

**Caches et compilés:**
- `__pycache__/` dans 5 dossiers (~60 KB total)
- `.pyc` fichiers (générés)

---

## 3. Classification détaillée des fichiers

### A. À GARDER (cœur du code et résultats article)

#### Code source actif
| Item | Taille | Raison |
|------|--------|--------|
| `models/per_sc.py` | 11 K | Précodeur attention linéaire par SC (phase 1–3, retenu) |
| `models/delay_domain.py` | 15 K | Estimation domaine retards (phase 4, **résultat principal**) |
| `models/gram_precoder.py` | 11 K | Variante Gram (exploré, référence) |
| `models/tarb.py` | 7,4 K | Modèle baseline (exploration) |
| `models/axial_precoder.py` | 4,0 K | Variante axiale (exploration) |
| `precoders/classical.py` | 3,1 K | Références classiques (RZF, WMMSE) |
| `channel.py` | 4,5 K | Utilitaire génération canal Sionna |
| `config.py` | 4,5 K | Configuration commune |
| `csi.py` | 8,5 K | Estimation CSI (LS, LMMSE) |
| `utils.py` | 4,5 K | Utilitaires (chemins, merge, etc.) |
| `metrics.py` | 4,5 K | Débit, SINR, plots |
| `train.py` | 8,5 K | Boucle entraînement commune |

#### Résultats phases 0–4 (article, mentionnés dans FINDINGS.md)
| Fichier | Taille | Raison |
|---------|--------|--------|
| `results/phase0/{baselines,channel_stats}.*` | ~300 KB | Baselines classiques (phase 0, synthèse A1–A3) |
| `results/phase1/{size_sweep,signed_gram,highsnr,candidate}.*` | ~1,2 M | Architectures per-SC (phase 1, synthèse A3–A4) |
| `results/phase2/{estimated_csi_p10,p20}.*` | ~400 KB | CSI estimé par SC (phase 2, négatif → justification phase 3) |
| `results/phase3/{window_p10,diagnostic_p10,mixed_p10,wmmse_ref,fattn}.*` | ~2 M | Fenêtre fréquentielle + mélange scénarios (phase 3, synthèse A5–A8) |
| `results/phase4/delay_model_bande_os2_n1056_s80k_D64*` | ~800 KB | **Résultat principal** (phase 4, synthèse A10–A12) |
| `results/phase4/{genie_bound,attribution,robust_ref}.*` | ~600 KB | Bornes et robustesse (phase 4, A9, A11, A12) |

#### Documentation article
| Item | Taille | Raison |
|------|--------|--------|
| `FINDINGS.md` | 97 K | Journal article (synthèse A–G + plan + positionnement) |
| `docs/RESUME.md` | — | Plan d'exécution (où sont les poids, résultats) |
| `README.md` | 4,5 K | Présentation générale |
| `requirements.txt` | 4,5 K | Dépendances reproductibles |
| `hls/README.md` | — | Documentation HLS (phase 10) |

#### Figures présentations (si destinées à l'article final)
| Item | Taille | Raison |
|------|--------|--------|
| `results/presentation/fig{A..F}*.{pdf,png}` | ~1 M | Figures finales de synthèse (Nov 25) |
| `results/presentation/architecture.pdf` | 50 K | Diagramme architecture |

#### Tests
| Item | Taille | Raison |
|------|--------|--------|
| `tests/test_*.py` | 66 K | Validation reproductibilité |

### B. À VÉRIFIER (explorations, non dans le plan article)

#### Phases 5–9 (explorations mises de côté selon plan)
| Item | Taille | Raison |
|------|--------|--------|
| `experiments/phase5_*.py` | ~100 K | Variantes Gram massif (exploré, non retenu) |
| `results/phase5/` | ~3 M | Résultats variantes (non dans synthèse A–G) |
| `experiments/phase6_*.py` | — | Loops Gram (exploré) |
| `results/phase6/` | ~400 KB | Résultats (non dans synthèse) |
| `experiments/phase7_context.py` | 20 K | Contexte en temps (exploration) |
| `results/phase7/` | ~3,2 M | Résultats (non publiés) |
| `experiments/phase8_*.py` | — | Robustesse (exploré, A11 retenu) |
| `results/phase8/` | ~600 KB | Résultats |
| `experiments/phase9_*.py` | — | Énergie baseline (exploré) |
| `results/phase9/` | ~300 KB | Résultats |

**Action recommandée:** Ces phases ont produit des résultats intéressants (disponibles dans git), mais ne figurent **pas dans la synthèse A–G** du papier. Elles peuvent rester dans `results/` pour archive, ou être archivées séparément (elles prennent ~11 M).

**Vérifier:** Sont-elles mentionnées dans les figures de présentation ou dans le texte du papier ?

#### Phases 12–13 (exploration récente, hors plan article)
| Item | Taille | Raison |
|------|--------|--------|
| `experiments/phase12_{figs,mask,mix,robust,wiener}.py` | ~40 KB | Synthèse sans entraînement (nov 1, "RETENU" = figure cache) |
| `results/phase12/` | 1,2 M | Figures synthèse encoder + masque + robustesse |
| `experiments/phase13_{bench,checks,deepmimo,figA,interp,interp2,lab,massive,massive_pilots,train_massive}.py` | ~100 KB | Exploration ray-tracing (nov 25, "exploration") |
| `results/phase13/` | 110 KB | Résultats préliminaires |
| `experiments/presentation_figs.py` | ~25 KB | Figures présentations (hors article ?) |
| `torch_port/{check_v5_energy,energy_*.py,eval_v5_testset,mac_table}.py` | ~40 KB | Énergie + évaluation phase 10 |
| `results/stecath/` | ~400 KB | Résultats quantification ART |
| `hls/` | 91 K | Code HLS synthèse (phase 10) |

**Action recommandée:** 
- Phase 12 : Figures synthèse, utiles si elles figurent dans la présentation. Les scripts **ne réentraînent pas** (c'est du post-traitement). À garder si les figures sont dans le papier.
- Phase 13 : "Exploration" = travail préliminaire, test sur ray-tracing jamais vu. **Pas dans le plan article** (section E dit "mis de côté : ray-tracing"). Peut rester en archive.
- Phase 10 (ART/quantification) : Sur FPGA (BitAdapt), **pas prioritaire** pour l'article d'algorithme. À archiver.

**Vérifier auprès de l'auteur:** Incluent-elles des résultats mentionnés dans le papier final ?

### C. CANDIDATS À RETIRER

#### Caches et compilés (supprimer sans hésiter)
| Item | Taille | Action |
|------|--------|--------|
| `__pycache__/` (tous niveaux) | 60 K | Supprimer (auto-généré) |
| `*.pyc` fichiers | — | Supprimer (auto-généré) |
| `.pytest_cache/` | — | Supprimer si présent |

#### Notes personnelles (ne pas publier)
| Item | Taille | Raison |
|------|--------|--------|
| `docs/notes/note.pdf` | 1,2 M | Fichier personnel (non code) |
| `docs/notes/` | — | Dossier notes privées |
| `results/presentation/NOTES_FRANCOIS.txt` | 6,6 K | Notes texte (non code) |

#### Fichiers temporaires (non critiques)
| Item | Taille | Raison |
|------|--------|--------|
| `scripts/queue13_pareto.sh` | — | Queue job (serveur spécifique, non portable) |
| `docs/make_architecture_pdf.py` | — | Utilitaire build local |

#### Phases anciennes (explorations mises à l'écart)
Optionnel : archiver séparément si déjà dans git
- `results/phase5/`, `phase6/`, `phase8/` : ~4 M (explorations) → à decider si garder ou archiver
- Garder au minimum 1 commit git pour traçabilité historique

---

## 4. Propositions `.gitignore`

### Ajouter au `.gitignore` du dépôt (ou `.gitignore` local si le dépôt d'équipe en a un)

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data / Results (optionnel : si trop gros pour git)
# Dé-commenter seulement si les results/ doivent rester locaux :
# results/phase*/
# results/stecath/

# Notes personnelles
docs/notes/
*.pdf  # Si seulement notes ; sinon commenter (architecture.pdf à garder)

# Logs
*.log
nohup.out

# Weights / Checkpoints (gros fichiers, à déployer séparément si nécessaire)
# *.ckpt
# *.h5
# weights/

# OS
.DS_Store
Thumbs.db

# Fichiers d'exécution serveur
scripts/queue*.sh
```

### Remarques
- **Ne pas inclure** `requirements.txt` dans `.gitignore` (c'est une dépendance).
- **Vérifier** si `results/` doit être incluse : si la taille dépasse ~50–100 MB, préférer un stockage séparé (S3, Zenodo, dépôt data distinct) et commiter `.gitkeep` ou README.
- **Poids de modèles** : s'il y a des fichiers `.ckpt`, `.h5`, `.pt` volumineux, les commiter avec Git LFS ou les déployer séparément.

---

## 5. Fichiers à ne pas publier

| Item | Type | Raison | Action |
|------|------|--------|--------|
| `docs/notes/note.pdf` | Note personnelle | Contenu privé | Retirer avant commit |
| `results/presentation/NOTES_FRANCOIS.txt` | Notes texte | Remarques privées | Retirer ou nettoyer |
| Chemins personnels | (si présents dans les scripts) | `/users/sala/...` hardcodé | Vérifier et remplacer par chemins relatifs |
| Identifiants/tokens | (si présents) | Secrets | Chercher avant commit |

### Scan recommandé avant commit
```bash
grep -r "/users/sala" freqctx/  # Chemins personnels
grep -r "PASSWORD\|TOKEN\|KEY\|SECRET" freqctx/ --include="*.py"
```

Aucun secret détecté jusqu'à présent, mais vérifier si des fichiers `.env` ou de config ont été oubliés.

---

## 6. Résumé des recommandations

| Action | Fichiers | Raison |
|--------|----------|--------|
| **Inclure** | `models/{per_sc,delay_domain}.py` + code support | Code article retenu (phases 1, 4) |
| **Inclure** | `results/phase{0,1,2,3,4}/*.{json,pdf,png}` | Résultats mentionnés dans FINDINGS.md (synthèse A–G) |
| **Inclure** | `FINDINGS.md`, `README.md`, `requirements.txt`, `docs/RESUME.md` | Documentation article |
| **Inclure** | `tests/`, `hls/README.md` | Validation et portabilité |
| **Vérifier** | `results/phase{5..11,12,13}/`, phase scripts | Explorations : inclure ou archiver ? |
| **Vérifier** | `torch_port/`, `hls/codegen.py` | Quantification FPGA : inclure ou repo séparé ? |
| **Exclure** | `__pycache__/`, `*.pyc` | Auto-généré |
| **Exclure** | `docs/notes/note.pdf` | Contenu personnel |
| **Exclure** | `results/presentation/NOTES_FRANCOIS.txt` | Remarques privées |

---

## 7. Taille et impact

| Scénario | Taille repo |
|----------|-------------|
| Minimal (code + phases 0–4) | ~2–3 MB |
| + phases 5–13 explorations | ~15–17 MB |
| + cache Python inclus | ~17–18 MB |
| Avec LFS pour résultats PNG | ~1–2 MB (sans images) |

**Recommandation :** Commiter phases 0–4 (essentielles), archiver phases 5–13 séparément sur GitHub (ou Zenodo) avec un `README-RESULTS.md` pointant vers eux.

---

## 8. Checklist avant transfert

- [ ] Lire `FINDINGS.md` section E (plan) : phases 12–13 y sont listées "Mis de côté"
- [ ] Lire `docs/RESUME.md` : vérifier qu'il pointe vers le bon modèle/résultats
- [ ] Vérifier `results/presentation/` : ces figures sont-elles dans le papier ?
- [ ] `grep -r "/users/sala"` : remplacer chemins personnels par `.` ou `$PROJECT_ROOT`
- [ ] Nettoyer `docs/notes/note.pdf` et `NOTES_FRANCOIS.txt`
- [ ] `git status --short freqctx` : valider que les 6 fichiers modifiés sont préparés
- [ ] Tester `python -m experiments.phase0_baselines` sur une graine pour validation

---

**Généré le 2026-11-25 — audit avant transfert vers dépôt équipe**
