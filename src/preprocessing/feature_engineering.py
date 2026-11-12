"""
Etapa 3.2 — Preparação e engenharia de atributos.

Lê as grades reais (data/raw/eventos_base_reais.json, gerado pelo crawler),
gera incidentes de realocação com K candidatos cada, calcula os atributos
tabulares observáveis e o alvo `score_viabilidade` (simulado com variáveis
latentes + ruído) e salva data/processed/eventos_processados.csv.

Uso: python -m src.preprocessing.feature_engineering
"""
import json

import pandas as pd

from src import config
from src.simulation.incident_generator import carregar_grades, gerar_incidentes


def process_features() -> pd.DataFrame:
    print(f"Lendo grades reais de {config.BASE_EVENTS_FILE}...")
    grades = carregar_grades(config.BASE_EVENTS_FILE)
    print(f"  {len(grades)} grades (evento, dia) | {sum(map(len, grades.values()))} sessões reais")

    linhas = gerar_incidentes(grades, config.N_INCIDENTES, config.K_CANDIDATOS, config.RANDOM_STATE)
    with open(config.RAW_CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(linhas, f, ensure_ascii=False)

    df = pd.DataFrame(linhas)
    df["viavel"] = (df[config.TARGET] >= config.LIMIAR_VIAVEL).astype(int)
    df.to_csv(config.PROCESSED_FILE, index=False, encoding="utf-8")

    print(f"Dataset salvo em {config.PROCESSED_FILE}: {len(df)} linhas, "
          f"{df[config.GROUP].nunique()} incidentes")
    print(df[config.TARGET].describe().round(2).to_string())
    print(f"  Taxa de alocações viáveis (score >= {config.LIMIAR_VIAVEL}): {df['viavel'].mean():.1%}")

    try:
        from src.utils.s3_client import S3Storage
        S3Storage().upload_file(config.PROCESSED_FILE, "processed/eventos_processados.csv")
    except Exception as e:  # sessão AWS Academy expirada não deve travar o pipeline
        print(f"[AVISO] Backup S3 ignorado: {e}")
    return df


if __name__ == "__main__":
    process_features()
