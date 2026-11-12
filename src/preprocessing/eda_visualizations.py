"""
Etapa 3.2 — Análise exploratória (EDA). Gera figuras 300 DPI em docs/figures/.

Uso: python -m src.preprocessing.eda_visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils.plot_style import AZUL, DIV, LARANJA, TEXTO2, aplicar_estilo, salvar


def main():
    aplicar_estilo()
    import matplotlib as mpl; mpl.rcParams["axes.grid"] = False  # heatmaps sem grade
    df = pd.read_csv(config.PROCESSED_FILE)
    print(f"EDA sobre {len(df)} candidatos / {df[config.GROUP].nunique()} incidentes")

    # Resumo descritivo para o artigo
    desc = df[config.FEATURES + [config.TARGET]].describe().T.round(3)
    desc.to_csv(f"{config.DOCS_DIR}/estatisticas_descritivas.csv")

    # a) Matriz de correlação (Spearman: robusta a relações monotônicas não lineares)
    corr = df[config.FEATURES + [config.TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap=DIV, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=1, linecolor="white",
                cbar_kws={"label": "ρ de Spearman", "shrink": 0.8}, ax=ax)
    ax.set_title("Correlação entre atributos e score de viabilidade")
    salvar(fig, config.FIG_DIR, "eda_01_correlacao.png")

    # b) Distribuições do alvo e do fator de ocupação
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[config.TARGET], bins=50, color=AZUL, edgecolor="white", linewidth=0.5)
    axes[0].axvline(config.LIMIAR_VIAVEL, color=LARANJA, ls="--", lw=1.5)
    axes[0].text(config.LIMIAR_VIAVEL - 2, axes[0].get_ylim()[1] * 0.92,
                 f"limiar de viabilidade = {config.LIMIAR_VIAVEL:.0f}", ha="right", color=TEXTO2, fontsize=9)
    axes[0].set(title="Distribuição do score_viabilidade", xlabel="score (0–100)", ylabel="candidatos")
    axes[1].hist(df["fator_ocupacao_sala"].clip(upper=5), bins=60, color=AZUL, edgecolor="white", linewidth=0.5)
    axes[1].axvline(1.0, color=LARANJA, ls="--", lw=1.5)
    axes[1].text(1.08, axes[1].get_ylim()[1] * 0.92, "capacidade = público", color=TEXTO2, fontsize=9)
    axes[1].set(title="Distribuição do fator de ocupação (truncado em 5)",
                xlabel="público estimado / capacidade", ylabel="candidatos")
    salvar(fig, config.FIG_DIR, "eda_02_distribuicoes.png")

    # c) Ocupação x sobreposição de trilha x score
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    amostra = df.sample(6000, random_state=config.RANDOM_STATE)
    sc = axes[0].scatter(amostra["fator_ocupacao_sala"].clip(upper=4), amostra[config.TARGET],
                         c=amostra["indice_sobreposicao_trilha"].clip(upper=3), cmap="viridis_r",
                         s=8, alpha=0.6, linewidths=0)
    cb = fig.colorbar(sc, ax=axes[0], ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["0", "1", "2", "3+"])
    cb.set_label("sessões da mesma trilha em paralelo")
    axes[0].axvline(1.0, color=TEXTO2, ls=":", lw=1)
    axes[0].set(title="Score vs. ocupação (cor = sobreposição de trilha)",
                xlabel="fator de ocupação (truncado em 4)", ylabel="score_viabilidade")

    df["faixa_ocupacao"] = pd.cut(df["fator_ocupacao_sala"], [0, 0.5, 0.8, 1.0, 1.25, 1.5, np.inf],
                                  labels=["<0,5", "0,5–0,8", "0,8–1,0", "1,0–1,25", "1,25–1,5", ">1,5"])
    df["sobrep"] = df["indice_sobreposicao_trilha"].clip(upper=2).map({0: "0", 1: "1", 2: "2+"})
    piv = df.pivot_table(index="sobrep", columns="faixa_ocupacao", values=config.TARGET,
                         aggfunc="mean", observed=False)
    sns.heatmap(piv, cmap="Blues", annot=True, fmt=".0f", linewidths=1, linecolor="white",
                cbar_kws={"label": "score médio"}, ax=axes[1])
    axes[1].set(title="Score médio por ocupação × sobreposição",
                xlabel="fator de ocupação", ylabel="sobreposição de trilha")
    salvar(fig, config.FIG_DIR, "eda_03_ocupacao_sobreposicao_score.png")

    # d) Score por tipo de imprevisto
    fig, ax = plt.subplots(figsize=(6, 4))
    ordem = ["atraso", "choque_sala", "saturacao"]
    ax.grid(axis="y")
    sns.boxplot(data=df, x="tipo_cenario", y=config.TARGET, order=ordem, color="#86b6ef",
                fliersize=1, linewidth=1, ax=ax)
    ax.axhline(config.LIMIAR_VIAVEL, color=LARANJA, ls="--", lw=1.2)
    ax.set(title="Score por tipo de imprevisto", xlabel="", ylabel="score_viabilidade")
    salvar(fig, config.FIG_DIR, "eda_04_score_por_cenario.png")

    print("Taxa de viáveis por cenário:")
    print(df.groupby("tipo_cenario")["viavel"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
