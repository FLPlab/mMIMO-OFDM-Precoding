"""PDF d'explication de l'architecture (pour F. Leduc-Primeau) : 3 pages, texte simple + schemas dessines
avec matplotlib (pas de dependance en plus). Sortie : results/presentation/architecture.pdf

    python docs/make_architecture_pdf.py
"""
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "presentation", "architecture.pdf")
INK, MUTED, BLUE, ORANGE, AQUA, LIGHT = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834", "#1baf7a", "#eef4fc"


def page():
    fig = plt.figure(figsize=(8.27, 11.69))          # A4
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 141)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fill=LIGHT, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fill, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))


def para(ax, x, y, text, width=95, size=9.5, color=INK, step=2.3):
    """Ecrit un paragraphe (retours a la ligne automatiques) ; renvoie la hauteur utilisee."""
    n = 0
    for block in text.split("\n"):
        lines = textwrap.wrap(block, width) or [""]
        for line in lines:
            ax.text(x, y - n * step, line, fontsize=size, color=color, va="top")
            n += 1
    return n * step


def title(ax, t, sub=None):
    ax.text(5, 136, t, fontsize=16, color=INK, weight="bold", va="top")
    if sub:
        ax.text(5, 130.5, sub, fontsize=10, color=MUTED, va="top")


def page1(pdf):
    fig, ax = page()
    title(ax, "1. Vue d'ensemble", "Des pilotes de sondage au précodeur, pour les 612 sous-porteuses")
    # schema
    box(ax, 5, 104, 18, 14, "Pilotes reçus\n(LS en peigne)\n612 sous-porteuses", AQUA, "#e8f7f1")
    box(ax, 30, 100, 30, 22, "Encodeur fréquentiel\n(toute la bande d'un coup)\n\nIFFT → retards\n→ garder les retards utiles\n(convolution sur les retards)\n→ FFT", BLUE)
    box(ax, 67, 104, 28, 14, "Canal débruité ĥ(f)\npour chaque\nsous-porteuse f", AQUA, "#e8f7f1")
    arrow(ax, 23.5, 111, 29.5, 111)
    arrow(ax, 60.5, 111, 66.5, 111)
    for i, yy in enumerate([86, 79, 72]):
        box(ax, 30, yy, 30, 5, f"précodeur par paires  (SC {['1', '2', '...'][i]})", ORANGE, "#fdf0ea", 8.5)
    ax.text(45, 69.5, "même réseau, mêmes poids", ha="center", fontsize=8, color=MUTED)
    arrow(ax, 81, 103.5, 61, 88.5)
    box(ax, 67, 74, 28, 12, "Précodeurs W(f)\n(M × K par sous-porteuse)", ORANGE, "#fdf0ea")
    arrow(ax, 60.5, 80, 66.5, 80)
    h = para(ax, 5, 63, (
        "Deux blocs, appris ensemble sur le débit :\n"
        "• L'encodeur fréquentiel regarde TOUTE la bande. Il passe les pilotes dans le domaine des retards (IFFT) : le canal "
        "n'a qu'une vingtaine de trajets, donc son énergie tient sur quelques retards, alors que le bruit est étalé partout. "
        "On garde les retards utiles, on revient en fréquence (FFT) : le canal est débruité sur chaque sous-porteuse.\n"
        "• Le précodeur par paires calcule W(f) à partir de ĥ(f), sous-porteuse par sous-porteuse, sans inversion de matrice.\n"
        "\n"
        "Est-il appliqué à chaque sous-porteuse ? Oui : c'est le même petit réseau (mêmes poids) pour les 612 sous-porteuses ; "
        "il ne reçoit pas l'indice de la sous-porteuse, seulement son canal. Le contexte fréquentiel vient de l'encodeur.\n"
        "\n"
        "Est-ce une boucle ? Sur GPU, non : les 612 sous-porteuses (et plusieurs tirages) sont empilées en un seul lot et "
        "calculées en parallèle. Sur FPGA, on fera passer les sous-porteuses l'une après l'autre dans le même circuit "
        "(pipeline), avec les poids chargés une seule fois sur la puce. Contrainte : 612 précodeurs par slot de 0,5 ms, "
        "soit environ 1,2 million par seconde.\n"
        "\n"
        "En CSI parfait (figures B, C, D, F), l'encodeur n'est pas utilisé : le précodeur reçoit directement le vrai canal."))
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig, ax = page()
    title(ax, "2. Le précodeur par paires", "Un petit transformer (44 k paramètres), appliqué à une sous-porteuse")
    box(ax, 5, 112, 20, 10, "canal ĥ\nK utilisateurs\n× M antennes", AQUA, "#e8f7f1", 8.5)
    box(ax, 31, 112, 22, 10, "normaliser chaque\nutilisateur, puis\nG = ĥ ĥᴴ (K × K)", BLUE, LIGHT, 8.5)
    box(ax, 59, 112, 36, 10, "un token par PAIRE (j, k) : K² = 16 tokens\ncaractéristiques : G(j,k), |G(j,k)|,\nSNR, puissances relatives", BLUE, LIGHT, 8.5)
    arrow(ax, 25.5, 117, 30.5, 117)
    arrow(ax, 53.5, 117, 58.5, 117)
    ax.add_patch(FancyBboxPatch((20, 80), 60, 25, boxstyle="round,pad=0.4,rounding_size=1.5", fc="white", ec=ORANGE, lw=1.4, ls="--"))
    ax.text(50, 103.5, "× 8 couches (poids différents à chaque couche)", ha="center", fontsize=8.5, color=ORANGE)
    box(ax, 23, 92, 16, 7, "normalisation\n+ projections a, b", ORANGE, "#fdf0ea", 8)
    box(ax, 42, 92, 18, 7, "produit triangulaire\nΣₗ a(j,l) · b(l,k)", ORANGE, "#fdf0ea", 8)
    box(ax, 63, 92, 14, 7, "petit réseau\n(FFN)", ORANGE, "#fdf0ea", 8)
    arrow(ax, 39.5, 95.5, 41.5, 95.5)
    arrow(ax, 60.5, 95.5, 62.5, 95.5)
    ax.text(50, 85, "+ résiduel ; G est redonné à chaque couche", ha="center", fontsize=8, color=MUTED)
    arrow(ax, 77, 111.5, 60, 105.5)
    box(ax, 22, 64, 26, 9, "lecture : coefficients\nC (K × K)", BLUE, LIGHT, 8.5)
    box(ax, 54, 64, 28, 9, "W = ĥᴴ C  (M × K)\npuis normalisation de puissance", ORANGE, "#fdf0ea", 8.5)
    arrow(ax, 40, 79.5, 35, 73.5)
    arrow(ax, 48.5, 68.5, 53.5, 68.5)
    para(ax, 5, 57, (
        "Pourquoi des paires d'utilisateurs ? Le précodage, c'est gérer l'interférence entre utilisateurs : ce qui compte, "
        "c'est la relation entre chaque paire (j, k), c'est-à-dire la matrice G = ĥĥᴴ. Chaque token porte une case de G.\n"
        "\n"
        "Le produit triangulaire : la paire (j, k) reçoit Σₗ a(j,l)·b(l,k), c'est-à-dire un produit de matrices K × K "
        "appris. Huit couches font comme huit étapes d'une inversion itérative de (G + αI), sans jamais inverser de matrice. "
        "C'est pour cela qu'il égale WMMSE.\n"
        "\n"
        "La sortie W = ĥᴴC a la même forme que RZF et WMMSE (combinaison des canaux des utilisateurs) : le réseau ne "
        "fabrique que les K × K coefficients C, quel que soit le nombre d'antennes M.\n"
        "\n"
        "Chiffres : 44 k paramètres, environ 0,7 M MAC par sous-porteuse (M = 64, K = 4). Quantifié avec ART : poids de 1 à "
        "6 bits, activations de 2 à 8 bits."))
    pdf.savefig(fig)
    plt.close(fig)


def page3(pdf):
    fig, ax = page()
    title(ax, "3. L'interpolation : précodeur sur 1 sous-porteuse sur 3", "Calculer moins souvent, interpoler entre les deux")
    y = 110
    ax.plot([5, 95], [y, y], color=MUTED, lw=1)
    for i in range(16):
        x = 8 + i * 5.6
        anchor = i % 3 == 0
        ax.plot(x, y, "o", ms=9 if anchor else 7, color=BLUE if anchor else "white", mec=BLUE, mew=1.5)
        ax.text(x, y - 3.5, str(i), ha="center", fontsize=7, color=MUTED)
    ax.text(5, y + 6, "● ancre : le précodeur est calculé     ○ entre deux ancres : W interpolé", fontsize=9, color=INK)
    ax.text(50, y - 8, "indice de sous-porteuse", ha="center", fontsize=8, color=MUTED)
    box(ax, 18, 82, 64, 11, "Pour une sous-porteuse f entre les ancres a et a + 3 :\n"
        "W(f) = (1 − t) · W(a) + t · W(a + 3),   t = (f − a) / 3\npuis on renormalise la puissance de W(f)", ORANGE, "#fdf0ea", 9)
    para(ax, 5, 74, (
        "Comment ça se passe :\n"
        "1. Le précodeur est calculé seulement sur les « ancres » : 1 sous-porteuse sur 3, soit 205 sur 612.\n"
        "2. Pour les sous-porteuses entre deux ancres, W est la moyenne pondérée des deux précodeurs voisins (interpolation "
        "linéaire), puis on renormalise sa puissance.\n"
        "3. Le modèle est entraîné avec cette interpolation : il apprend à la compenser.\n"
        "\n"
        "Pourquoi ça marche : le canal varie lentement d'une sous-porteuse à la voisine. En 3GPP, il reste presque "
        "identique sur environ 9 sous-porteuses (bande de cohérence). Avec un pas de 3, on reste bien en dessous.\n"
        "\n"
        "Ce que ça coûte et rapporte :\n"
        "• Canaux 3GPP (sélectifs) : 1 sur 3 perd au plus 0,1 bps/Hz, pour un calcul divisé par 2,4. Un pas plus grand "
        "coûte vite (1 sur 12 : −4,5 %).\n"
        "• Sites du labo (presque plats) : on peut aller jusqu'à 1 sur 12 ou 24 presque sans perte.\n"
        "• L'interpolation coûte seulement 2 multiplications par coefficient de W.\n"
        "\n"
        "Pour le FPGA : le pas pourrait s'adapter à l'étalement des retards du canal, et diviser le calcul par 3 à 24 "
        "selon l'environnement."))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        page1(pdf)
        page2(pdf)
        page3(pdf)
    print("->", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
