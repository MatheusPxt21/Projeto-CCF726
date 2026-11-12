"""Configuração central do projeto (caminhos, sementes e hiperparâmetros do experimento)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
DOCS_DIR = os.path.join(ROOT, "docs")
FIG_DIR = os.path.join(DOCS_DIR, "figures")
MODELS_DIR = os.path.join(ROOT, "models")

BASE_EVENTS_FILE = os.path.join(RAW_DIR, "eventos_base_reais.json")
RAW_CANDIDATES_FILE = os.path.join(RAW_DIR, "incidentes_candidatos_raw.json")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "eventos_processados.csv")
RESULTS_FILE = os.path.join(DOCS_DIR, "resultados_experimentos.csv")
DECISION_RESULTS_FILE = os.path.join(DOCS_DIR, "resultados_decisao.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "best_relocation_model.joblib")

RANDOM_STATE = 42
N_INCIDENTES = 2500          # 2500 incidentes x 20 candidatos = 50.000 instâncias
K_CANDIDATOS = 20
LIMIAR_VIAVEL = 60.0         # score >= 60 -> alocação viável; < 60 -> conflito crítico
TEST_SIZE = 0.20

# Atributos observáveis pelo modelo (nenhuma variável latente do simulador entra aqui)
FEATURES = [
    "duracao_minutos",
    "capacidade_sala",
    "participantes_estimados",
    "fator_ocupacao_sala",
    "indice_sobreposicao_trilha",
    "sessoes_concorrentes",
    "densidade_bloco",
    "horario_pico",
    "hora_inicio_min",
    "deslocamento_minutos",
    "mesma_sala",
    "tipo_cenario_cod",
]
TARGET = "score_viabilidade"
GROUP = "incidente_id"

for _d in (PROCESSED_DIR, FIG_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)
