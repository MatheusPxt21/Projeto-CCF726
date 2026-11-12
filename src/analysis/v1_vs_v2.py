"""
Figura "antes × depois" do alvo: dataset v1 (degenerado) vs. v2 (incidentes sobre grades reais).

Recalcula o score v1 a partir do bruto arquivado (data/raw/eventos_academicos_raw.json,
também em s3://.../archive/v1/) com o código v1 original (src/legacy/feature_engineering_v1.py).

Uso: python -m src.analysis.v1_vs_v2
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.legacy.feature_engineering_v1 import calcular_sobreposicoes
from src.utils.plot_style import AZUL, LARANJA, TEXTO2, aplicar_estilo, salvar

RAW_V1 = os.path.join(config.RAW_DIR, "eventos_academicos_raw.json")


def score_v1() -> pd.DataFrame:
    with open(RAW_V1, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["trilha"] = df["trilha"].fillna("Geral")
    ocup = df["participantes_estimados"] / df["capacidade_sala"]
    hora = df["horario_inicio"].str[:2].astype(int)
    pico = (((hora >= 10) & (hora < 12)) | ((hora >= 14) & (hora < 16))).astype(int)
    sobrep = np.array(calcular_sobreposicoes(df))
    tipo = df["tipo_cenario"]
    pen = (np.where(ocup > 1, (ocup - 1) * 40, 0) + sobrep * 25 + pico * 10
           + np.where(tipo == "saturacao", 8, 0) + np.where(tipo == "atraso", 5, 0))
    return pd.DataFrame({"score": np.clip(100 - pen, 0, 100), "sobreposicao": sobrep})


def main():
    aplicar_estilo()
    v1 = score_v1()
    v2 = pd.read_csv(config.PROCESSED_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    bins = np.linspace(0, 100, 41)
    for ax, s, cor, tit in [
        (axes[0], v1["score"], LARANJA, "v1: cenários isolados + fórmula determinística"),
        (axes[1], v2[config.TARGET], AZUL, "v2: incidentes sobre grades reais + simulador latente"),
    ]:
        ax.hist(s, bins=bins, color=cor, edgecolor="white", linewidth=0.5)
        ax.axvline(config.LIMIAR_VIAVEL, color=TEXTO2, ls="--", lw=1)
        ax.set(title=tit, xlabel="score_viabilidade", ylabel="instâncias")
        ax.text(0.12, 0.93, f"média {s.mean():.1f} | DP {s.std():.1f}\nscore = 0: {(s == 0).mean():.1%}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=TEXTO2)
    salvar(fig, config.FIG_DIR, "hist_00_alvo_v1_vs_v2.png")

    print(f"v1: score=0 em {(v1.score == 0).mean():.1%} | sobreposição média {v1.sobreposicao.mean():.0f} "
          f"(máx {v1.sobreposicao.max()})")
    print(f"v2: score=0 em {(v2[config.TARGET] == 0).mean():.1%} | sobreposição média "
          f"{v2.indice_sobreposicao_trilha.mean():.2f} (máx {v2.indice_sobreposicao_trilha.max()})")


if __name__ == "__main__":
    main()
