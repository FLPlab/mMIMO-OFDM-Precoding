# mMIMO-OFDM-Precoding

Précodage MU-MIMO OFDM appris, des pilotes de sondage au précodeur, avec une cible FPGA.
Polytechnique Montréal, FLPlab — Glodi Sala-Mangituka, dir. François Leduc-Primeau.

## L'idée

- **Un petit transformer par sous-porteuse**, appliqué à toutes les sous-porteuses avec les mêmes poids,
  avec **un token par paire d'utilisateurs** et une sortie W = h̃ᴴC : **sans aucune inversion de matrice**.
  En CSI parfait, il égale WMMSE convergé, en M=8 / K=4 comme en massif M=64 / K=8 (44 k paramètres).
- **Depuis les pilotes**, un encodeur sur toute la bande, dans le domaine des retards, débruite le canal avant
  le précodeur ; il fait mieux que les chaînes classiques (LMMSE ou Wiener + WMMSE).
- **Lien avec le labo** : le même modèle fonctionne sans réentraînement sur les sites de lancer de rayons du labo
  (MIMO GenEE : St-Catherine, Décarie, Ericsson), et il est quantifié avec ART (méthode BitAdapt) ; un générateur
  HLS produit le C++ du précodeur quantifié pour l'Alveo U280.

## Contenu

| Dossier | Contenu |
|---|---|
| [`freqctx/`](freqctx/) | **Travail actuel** : simulation Sionna (OFDM, pilotes, 3GPP), modèles, expériences, résultats, port PyTorch et quantification ART (`torch_port/`), générateur HLS (`hls/`). Voir [`freqctx/README.md`](freqctx/README.md). |
| [`mimo_precoding/`](mimo_precoding/) | Code du mémoire de maîtrise (TA-RB, attention signée). Description détaillée : [`archive/README_memoire.md`](archive/README_memoire.md). |
| [`archive/`](archive/) | Anciennes explorations. |

## Où lire les résultats

- [`freqctx/FINDINGS.md`](freqctx/FINDINGS.md) : journal de toutes les expériences (question, chiffres, verdict), négatifs compris.
- [`freqctx/results/presentation/`](freqctx/results/presentation/) : figures de synthèse et [`architecture.pdf`](freqctx/results/presentation/architecture.pdf) (l'architecture expliquée en 3 pages).
- [`freqctx/hls/README.md`](freqctx/hls/README.md) : chaîne modèle quantifié → référence entière → C++ HLS.

## Démarrer

Environnements (versions exactes dans [`freqctx/requirements.txt`](freqctx/requirements.txt)) :
- **TensorFlow 2.20 + Sionna 1.2.1** pour la simulation et l'entraînement (`freqctx/`) ;
- **PyTorch** pour `freqctx/torch_port/` (quantification ART avec les quantificateurs du labo, dépôt `mMIMO-ART-FPGA`).

Chaque expérience est un script ; lancer depuis `freqctx/`, toujours avec `python -u` :

```bash
cd freqctx
python -u -m experiments.phase13_bench --set umi        # précodeur contre WMMSE / RZF, CSI parfait
python -u -m experiments.presentation_figs             # figures de synthèse (relit results/)
python -u -m hls.codegen --out <dossier>               # projet HLS du précodeur quantifié
```

Les poids entraînés et les données générées ne sont pas dans le dépôt (trop volumineux) : ils sont sur
`/export/tmp/sala/freqctx/` (srvgrm12) et se régénèrent avec les scripts.
