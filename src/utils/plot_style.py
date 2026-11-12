"""Estilo único dos gráficos do artigo (paleta categórica fixa, eixos discretos, 300 DPI)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Ordem categórica fixa — a cor acompanha a entidade, nunca a posição
CORES = {
    "Baseline Gulosa": "#eb6834",
    "Aleatória": "#eda100",
    "Menor sala que cabe": "#e87ba4",
    "Random Forest": "#1baf7a",
    "XGBoost": "#2a78d6",
    "Oráculo": "#52514e",
    "Média (Dummy)": "#eda100",
    "Regras v1 (manual)": "#e87ba4",
}
AZUL = "#2a78d6"
LARANJA = "#eb6834"
TEXTO = "#0b0b0b"
TEXTO2 = "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_azul", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d95926", "#f0efec", "#2a78d6"])


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TEXTO, "axes.titlecolor": TEXTO,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "xtick.color": TEXTO2, "ytick.color": TEXTO2, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2,
    })


def salvar(fig, pasta: str, nome: str):
    caminho = os.path.join(pasta, nome)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {caminho}")
