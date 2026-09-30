import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from src.utils.s3_client import S3Storage

RAW_FILE = os.path.join("data", "raw", "eventos_academicos_raw.json")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "eventos_processados.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def calcular_duracao(inicio_str: str, fim_str: str) -> int:
    fmt = "%H:%M"
    try:
        t_inicio = datetime.strptime(inicio_str, fmt)
        t_fim = datetime.strptime(fim_str, fmt)
        return int((t_fim - t_inicio).total_seconds() / 60)
    except Exception:
        return 60  # Duração default de 1 hora em caso de falha

def calcular_horario_pico(inicio_str: str) -> int:
    try:
        hora = int(inicio_str.split(":")[0])
        # Picos tradicionais em conferências: meio da manhã e início/meio da tarde
        if (10 <= hora < 12) or (14 <= hora < 16):
            return 1
        return 0
    except Exception:
        return 0

def calcular_sobreposicoes(df: pd.DataFrame) -> list:
    """Calcula sessões da mesma trilha simultâneas no mesmo evento e data."""
    fmt = "%H:%M"

    # Converte horários para minutos a partir de meia-noite (int) — mais rápido
    def to_min(s: str) -> int:
        try:
            h, m = map(int, s.split(":"))
            return h * 60 + m
        except Exception:
            return 0

    df_work = df[["evento", "data", "trilha", "horario_inicio", "horario_fim"]].copy()
    df_work["_ini"] = df_work["horario_inicio"].map(to_min)
    df_work["_fim"] = df_work["horario_fim"].map(to_min)
    df_work["_idx"] = df_work.index

    conflitos = pd.Series(0, index=df.index)

    # Processa por grupo para evitar explosão de memória no merge (OOM em 50k instâncias)
    for _, group in df_work.groupby(["evento", "data", "trilha"]):
        if len(group) < 2:
            continue

        merged = group.merge(group, on=["evento", "data", "trilha"], suffixes=("_a", "_b"))
        merged = merged[merged["_idx_a"] != merged["_idx_b"]]

        # max(ini_a, ini_b) < min(fim_a, fim_b)
        overlaps = merged[
            np.maximum(merged["_ini_a"], merged["_ini_b"]) < np.minimum(merged["_fim_a"], merged["_fim_b"])
        ]

        if not overlaps.empty:
            counts = overlaps.groupby("_idx_a").size()
            conflitos = conflitos.add(counts, fill_value=0)

    return conflitos.astype(int).tolist()

def process_features():
    print(f"Lendo dados brutos de {RAW_FILE}...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)

    # Limpeza de valores ausentes
    df["capacidade_sala"]          = df["capacidade_sala"].fillna(100).astype(int)
    df["participantes_estimados"]  = df["participantes_estimados"].fillna(50).astype(int)
    df["trilha"]                   = df["trilha"].fillna("Geral")
    df["fonte"]                    = df["fonte"].fillna("curado")

    # tipo_cenario: mapeamento ordinal para feature numérica
    _TIPO_MAP = {"atraso": 0, "choque_sala": 1, "saturacao": 2}
    if "tipo_cenario" in df.columns:
        df["tipo_cenario"]         = df["tipo_cenario"].fillna("atraso")
        df["tipo_cenario_cod"]     = df["tipo_cenario"].map(_TIPO_MAP).fillna(0).astype(int)
    else:
        df["tipo_cenario"]         = "atraso"
        df["tipo_cenario_cod"]     = 0

    # Atributos Numéricos Derivados
    df["duracao_minutos"] = [
        calcular_duracao(ini, fim) for ini, fim in zip(df["horario_inicio"], df["horario_fim"])
    ]

    df["fator_ocupacao_sala"] = np.round(
        df["participantes_estimados"] / df["capacidade_sala"], 3
    )

    df["horario_pico"] = [calcular_horario_pico(ini) for ini in df["horario_inicio"]]
    df["indice_sobreposicao_trilha"] = calcular_sobreposicoes(df)

    # Definição do Score de Viabilidade
    # Penaliza: ocupação excessiva, sobreposição, horário de pico e saturação.
    penalidade_ocupacao    = np.where(
        df["fator_ocupacao_sala"] > 1.0,
        (df["fator_ocupacao_sala"] - 1.0) * 40, 0
    )
    penalidade_sobreposicao = df["indice_sobreposicao_trilha"] * 25
    penalidade_pico         = df["horario_pico"] * 10
    penalidade_saturacao    = np.where(df["tipo_cenario_cod"] == 2, 8, 0)
    penalidade_atraso       = np.where(df["tipo_cenario_cod"] == 0, 5, 0)

    # Score entre 0 e 100 (quanto maior, mais viável e menos danosa é a alocação)
    score = 100 - (
        penalidade_ocupacao + penalidade_sobreposicao
        + penalidade_pico + penalidade_saturacao + penalidade_atraso
    )
    df["score_viabilidade"] = np.clip(np.round(score, 2), 0, 100)

    # Salvar dataset tabular processado
    df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8")
    print(f"Dataset tabular salvo em {PROCESSED_FILE} com {len(df)} registros.")

    # Distribuição dos tipos de cenário
    if "tipo_cenario" in df.columns:
        dist = df["tipo_cenario"].value_counts()
        print("  Distribuicao tipo_cenario:")
        for t, c in dist.items():
            print(f"    {t}: {c} ({100*c/len(df):.1f}%)")

    print(f"  Score medio de viabilidade : {df['score_viabilidade'].mean():.2f}")
    print(f"  Score min / max            : {df['score_viabilidade'].min():.2f} / {df['score_viabilidade'].max():.2f}")

    # Backup do dataset processado no S3
    s3 = S3Storage()
    s3.upload_file(PROCESSED_FILE, "processed/eventos_processados.csv")


if __name__ == "__main__":
    process_features()