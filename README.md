# Sistema Preditivo de Realocação de Atividades para Organização Autônoma de Eventos Científicos

> **Disciplina:** CCF726 - Engenharia de Aprendizado de Máquina  
> **Instituição:** Universidade Federal de Viçosa (UFV) – Campus Florestal  
> **Professor:** Fabrício A. Silva  
---

## 📌 Visão Geral do Problema

A gestão em tempo real de grandes eventos científicos e conferências acadêmicas (ex: CSBC, simpósios e semanas acadêmicas) é sujeita a imprevistos dinâmicos constantes, como atrasos de palestrantes, choque de capacidade física em salas e cancelamentos de sessões.

Embora ecossistemas baseados em sistemas multi-agentes e interfaces conversacionais identifiquem a necessidade de remanejar a agenda, a escolha de um novo horário e espaço físico demanda inteligência analítica. Uma alocação inadequada gera saturação em salas, esvaziamento de sessões e sobreposição temática.

Este projeto desenvolve um **motor de recomendação e predição orientado a dados tabulares**. O modelo recebe um conflito de cronograma e prevê o score de viabilidade de diferentes hipóteses de realocação (sala e faixa de horário). Ele serve como núcleo analítico de decisão para agentes autônomos: em vez de usar o primeiro horário livre, o agente escolhe a opção com maior viabilidade prevista.

---

## 🚀 Metodologia e Pipeline

1. **Coleta de Dados (*Data Collection*):**
   - Web scraping das páginas públicas de programação do SBRC 2025 e do IHC 2024 (`requests` + `BeautifulSoup`).
   - Base curada com **42 sessões reais** de 6 eventos acadêmicos:
     CSBC/WIT 2025, XII SECOM UFV Florestal, Manna Quantum Festival,
     Semana Acadêmica EEL-USP, XI Mostra Bioeconomia e Fórum CPAs UFPB.
   - Fusão em uma base de **172 sessões reais** (22 grades evento × dia).
   - Persistência em nuvem via **Amazon S3** (AWS Academy Learner Lab).

2. **Geração de Incidentes e Engenharia de Atributos:**
   - Unidade de análise: **incidente de realocação**. Sobre uma grade real, uma sessão sofre um
     imprevisto (atraso, choque de sala ou saturação) e recebe **20 opções candidatas**
     (sala, horário) realmente livres naquela grade.
   - **2.500 incidentes × 20 candidatos = 50.000 instâncias.**
   - O alvo `score_viabilidade` [0–100] vem de um **simulador de impacto com variáveis latentes**
     que o modelo não observa (comparecimento real, afinidade de público entre trilhas,
     sensibilidade ao deslocamento, atratividade do horário) + ruído. Assim, o alvo não é uma
     função determinística das features.
   - Limiar de classificação: score ≥ 60 → **alocação viável**; < 60 → **conflito crítico**.

3. **Análise Exploratória (EDA):** correlação de Spearman, distribuições do alvo e da ocupação,
   relação ocupação × sobreposição × score e score por tipo de imprevisto.

4. **Modelagem Supervisionada e Persistência:**
   - **Random Forest** e **XGBoost** (regressão do score).
   - Split 80/20 **agrupado por incidente** e `RandomizedSearchCV` com `GroupKFold(5)`.
   - Persistência do melhor modelo via `joblib`.

5. **Validação vs. Baselines:**
   - **Avaliação por candidato:** RMSE, MAE, R² e, pelo limiar, Acurácia, Precisão, Recall,
     F1 e Matriz de Confusão.
   - **Avaliação da decisão por incidente:** cada estratégia escolhe uma opção, e mede-se o score
     realizado dessa escolha. Estratégias comparadas:
     - **Baseline Gulosa** (primeiro horário livre em ordem cronológica);
     - Aleatória;
     - Menor sala que cabe;
     - Modelo de ML;
     - Oráculo (teto teórico).
   - Teste de Wilcoxon pareado e IC 95% por bootstrap.

---

## 📈 Principais Resultados (conjunto de teste)

| Modelo | RMSE | MAE | R² | F1 (conflito crítico) |
|---|---|---|---|---|
| Média (Dummy) | 26,27 | 19,73 | −0,01 | 0,000 |
| Regras manuais (v1) | 21,69 | 15,98 | 0,31 | 0,713 |
| **Random Forest** | **11,40** | **8,18** | **0,81** | **0,813** |
| XGBoost | 11,41 | 8,23 | 0,81 | 0,813 |

| Estratégia (500 incidentes) | Score médio da escolha | % com conflito crítico |
|---|---|---|
| Aleatória | 66,9 | 29,2% |
| Baseline Gulosa | 67,7 | 31,4% |
| Menor sala que cabe | 77,9 | 16,0% |
| **Random Forest** | **91,4** | **0,0%** |
| Oráculo (teto) | 94,2 | 0,0% |

Ganho do Random Forest sobre a Baseline Gulosa: **+23,7 pontos** por incidente
(IC 95% [+21,2; +26,3]; Wilcoxon p < 0,001).

---

## 📂 Estrutura do Repositório

```text
Projeto-CCF726/
├── data/
│   ├── raw/
│   │   └── eventos_base_reais.json        <- 172 sessões reais (versionado)
│   └── processed/
│       └── eventos_processados.csv        <- 50.000 candidatos com features e alvo (não versionado)
├── docs/
│   ├── figures/                           <- Figuras da EDA e dos resultados (300 DPI)
│   ├── resultados_experimentos.csv        <- Métricas por candidato
│   ├── resultados_decisao.csv             <- Métricas de decisão por incidente
│   └── estatisticas_descritivas.csv
├── models/                                <- best_relocation_model.joblib (não versionado)
├── src/
│   ├── config.py                          <- Caminhos, seed (42), features e limiar
│   ├── crawler/event_crawler.py           <- Scraping (requests + BS4) + base curada
│   ├── simulation/incident_generator.py   <- Incidentes + simulador do score (variáveis latentes)
│   ├── preprocessing/
│   │   ├── feature_engineering.py         <- Gera o dataset processado
│   │   └── eda_visualizations.py          <- Figuras exploratórias
│   ├── baseline/greedy_baseline.py        <- Baseline Gulosa e estratégias de referência
│   ├── models/train_evaluate.py           <- Tuning, avaliação, figuras e persistência
│   ├── analysis/v1_vs_v2.py               <- Comparação do alvo v1 (degenerado) × v2
│   ├── legacy/feature_engineering_v1.py   <- Código v1 arquivado (apenas registro)
│   └── utils/
│       ├── s3_client.py                   <- Cliente AWS S3 (credenciais temporárias)
│       └── plot_style.py                  <- Estilo padrão dos gráficos
├── requirements.txt
├── .env                                   <- Credenciais AWS (não versionado)
└── README.md
```

---

## ⚙️ Como Executar

### Pré-requisitos

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows CMD  (PowerShell: .\venv\Scripts\Activate.ps1)
source venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

### Variáveis de Ambiente (`.env`)

```env
S3_BUCKET_NAME=seu-bucket-s3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...            # Obrigatório para AWS Academy Learner Lab
AWS_DEFAULT_REGION=us-east-1
```

As credenciais do Learner Lab expiram a cada ~4 h. Se estiverem expiradas, o upload ao S3 é
ignorado com um aviso e o pipeline local continua normalmente.

### Pipeline (a partir da raiz do repositório)

| Passo | Comando | Saída | Tempo |
|---|---|---|---|
| 1. Coleta (opcional) | `python -m src.crawler.event_crawler` | `data/raw/eventos_base_reais.json` | ~1 min |
| 2. Dataset | `python -m src.preprocessing.feature_engineering` | `data/processed/eventos_processados.csv` (+ upload S3) | ~10 s |
| 3. EDA | `python -m src.preprocessing.eda_visualizations` | `docs/figures/eda_*.png` | ~10 s |
| 4. Modelos e validação | `python -m src.models.train_evaluate` | `docs/resultados_*.csv`, `docs/figures/res_*.png`, `models/best_relocation_model.joblib` | ~9 min |
| 5. Comparação v1 × v2 (opcional) | `python -m src.analysis.v1_vs_v2` | `docs/figures/hist_00_alvo_v1_vs_v2.png` | ~10 s |

Todas as etapas usam `random_state = 42`, portanto os resultados são reprodutíveis.

---

## 📊 Dataset Processado

| Coluna | Tipo | Descrição |
|---|---|---|
| `incidente_id` | id | Identificador do incidente (agrupa os 20 candidatos) |
| `evento`, `data`, `atividade`, `trilha` | contexto | Sessão real afetada |
| `tipo_cenario` / `tipo_cenario_cod` | feature | Imprevisto: atraso (0), choque_sala (1), saturacao (2) |
| `sala_original`, `horario_original` | contexto | Alocação original |
| `sala`, `horario_inicio`, `horario_fim` | contexto | Opção candidata |
| `duracao_minutos` | feature | Duração da atividade |
| `capacidade_sala` | feature | Capacidade da sala candidata |
| `participantes_estimados` | feature | Público estimado |
| `fator_ocupacao_sala` | feature | Público estimado ÷ capacidade |
| `indice_sobreposicao_trilha` | feature | Sessões reais da mesma trilha em paralelo |
| `sessoes_concorrentes` | feature | Total de sessões em paralelo |
| `densidade_bloco` | feature | Sessões em paralelo ÷ nº de salas |
| `horario_pico` | feature | 1 se o início está entre 10–12h ou 14–16h |
| `hora_inicio_min` | feature | Início em minutos desde 0h |
| `deslocamento_minutos` | feature | Distância do horário original |
| `mesma_sala` | feature | 1 se usa a sala original |
| `score_viabilidade` | **alvo** | Score realizado [0–100] (simulado) |
| `viavel` | alvo binário | 1 se score ≥ 60 |

### Histórico: dataset v1

A primeira versão do dataset (cenários isolados + fórmula determinística) produzia um **alvo
degenerado**: 99,4% das instâncias tinham score 0. A causa era que a sobreposição de trilha era
contada entre 50 mil cenários sintéticos que nunca coexistiriam na mesma grade. A versão atual (v2)
corrige isso. O bruto v1 está arquivado em `s3://<bucket>/archive/v1/` e o código original em
`src/legacy/`.
