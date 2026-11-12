"""
Etapa 3.4 — Baseline Gulosa e estratégias de decisão de referência.

Todas as estratégias escolhem UM candidato por incidente, a partir do MESMO
conjunto de opções livres. A qualidade da decisão é medida pelo score REALIZADO
(simulado) do candidato escolhido — nunca pelo score predito.

  * Baseline Gulosa: primeiro horário livre em ordem cronológica a partir do
    horário original (empate -> primeira sala na ordem alfabética), ignorando
    público, tráfego e sobreposição temática.
  * Aleatória: escolhe qualquer opção livre (piso de desempenho).
  * Menor sala que cabe: heurística forte de "bom senso" (capacidade >= público,
    menor sala; empate -> mais cedo). Evita vencer apenas um espantalho.
  * Oráculo: melhor opção pelo score realizado (teto teórico).
  * Modelo de ML: argmax do score predito.

Uso isolado: python -m src.baseline.greedy_baseline
"""
import numpy as np
import pandas as pd

from src import config


def _hora(df):
    return df["hora_inicio_min"]


def escolher_gulosa(df: pd.DataFrame) -> pd.Series:
    """Retorna o índice do candidato escolhido pela heurística gulosa em cada incidente."""
    orig = df["horario_original"].map(lambda s: int(s[:2]) * 60 + int(s[3:5]))
    d = df.assign(_apos=(_hora(df) < orig).astype(int), _h=_hora(df))
    d = d.sort_values([config.GROUP, "_apos", "_h", "sala"])
    return d.groupby(config.GROUP).head(1).set_index(config.GROUP)["_idx"]


def escolher_aleatoria(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.Series:
    return df.groupby(config.GROUP).sample(1, random_state=seed).set_index(config.GROUP)["_idx"]


def escolher_menor_sala(df: pd.DataFrame) -> pd.Series:
    d = df.assign(_nao_cabe=(df["capacidade_sala"] < df["participantes_estimados"]).astype(int),
                  _folga=(df["capacidade_sala"] - df["participantes_estimados"]).abs(), _h=_hora(df))
    d = d.sort_values([config.GROUP, "_nao_cabe", "_folga", "_h"])
    return d.groupby(config.GROUP).head(1).set_index(config.GROUP)["_idx"]


def escolher_por_score(df: pd.DataFrame, col: str) -> pd.Series:
    d = df.sort_values([config.GROUP, col, "hora_inicio_min"], ascending=[True, False, True])
    return d.groupby(config.GROUP).head(1).set_index(config.GROUP)["_idx"]


def avaliar_decisoes(df: pd.DataFrame, estrategias: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df precisa conter as colunas originais + colunas de predição.
    estrategias: {nome: Series(incidente_id -> _idx escolhido)}.
    Retorna (resumo por estratégia, score realizado por incidente x estratégia).
    """
    df = df.copy()
    oraculo = df.groupby(config.GROUP)[config.TARGET].max()
    rank = df.groupby(config.GROUP)[config.TARGET].rank(ascending=False, method="min")
    por_inc = {}
    linhas = []
    for nome, esc in estrategias.items():
        escolhido = df.loc[esc.values].set_index(config.GROUP)
        s = escolhido[config.TARGET]
        regret = oraculo.loc[s.index] - s
        r = rank.loc[esc.values].values
        por_inc[nome] = s
        linhas.append({
            "estrategia": nome,
            "score_realizado_medio": s.mean(),
            "score_realizado_dp": s.std(),
            "regret_medio": regret.mean(),
            "taxa_conflito_critico": (s < config.LIMIAR_VIAVEL).mean(),
            "top1_acc": (r == 1).mean(),
            "top3_acc": (r <= 3).mean(),
        })
    return pd.DataFrame(linhas), pd.DataFrame(por_inc)


def estrategias_referencia(df: pd.DataFrame) -> dict:
    return {
        "Baseline Gulosa": escolher_gulosa(df),
        "Aleatória": escolher_aleatoria(df),
        "Menor sala que cabe": escolher_menor_sala(df),
        "Oráculo": escolher_por_score(df, config.TARGET),
    }


if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_FILE)
    df["_idx"] = df.index
    resumo, _ = avaliar_decisoes(df, estrategias_referencia(df))
    print(resumo.round(3).to_string(index=False))
