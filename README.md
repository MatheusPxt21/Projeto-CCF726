# Sistema Preditivo de Realocação de Atividades para Organização Autônoma de Eventos Científicos

> **Disciplina:** CCF726 - Engenharia de Aprendizado de Máquina  
> **Instituição:** Universidade Federal de Viçosa (UFV) – Campus Florestal  
> **Professor:** Fabrício A. Silva  
---

## 📌 Visão Geral do Problema

A gestão em tempo real de grandes eventos científicos e conferências acadêmicas (ex: CSBC, simpósios e semanas acadêmicas) é sujeita a imprevistos dinâmicos constantes, como atrasos de palestrantes, choque de capacidade física em salas e cancelamentos de sessões.

Embora ecossistemas baseados em sistemas multi-agentes e interfaces conversacionais identifiquem a necessidade de remanejar a agenda, a escolha de um novo horário e espaço físico demanda inteligência analítica. Uma alocação inadequada gera saturação em salas, esvaziamento de sessões e sobreposição temática.

Este projeto desenvolve um **motor de recomendação e predição orientado a dados tabulares**. O modelo atua recebendo um conflito de cronograma e prevendo a taxa de sucesso/score de viabilidade para diferentes hipóteses de realocação (sala e faixa de horário), servindo como o núcleo analítico de decisão para agentes autônomos.

---

## 🚀 Metodologia e Pipeline

O projeto cumpre o fluxo completo de Engenharia de Machine Learning:

1. **Coleta de Dados (*Data Collection*):**
   - Web scraping funcional de páginas públicas de programação de eventos da SBC
     (SBRC 2025, IHC 2024) usando `requests` + `BeautifulSoup`, com parsers
     especializados para grades matriciais e tratamento tolerante a falhas.
   - Base curada com **42 sessões reais** de 6 eventos acadêmicos:
     CSBC/WIT 2025 (3 dias), XII SECOM UFV Florestal, Manna Quantum Festival (MON),
     Semana Acadêmica EEL-USP, XI Mostra Bioeconomia e Fórum CPAs UFPB.
   - Fusão da base curada com os dados extraídos, totalizando **172 sessões reais** pré-expansão.
   - Expansão combinatória (*Data Augmentation*) gerando **50.000 instâncias** de
     cenários sintéticos de realocação, divididos em três tipos de imprevisto (atraso, choque de sala, saturação).
   - Persistência estruturada em nuvem via **Amazon S3** (AWS Academy Learner Lab).

2. **Engenharia de Atributos (*Feature Engineering*):**
   - Limpeza e padronização temporal das grades coletadas.
   - Construção de features tabulares numéricas:
     - `duracao_minutos`: tempo total da atividade em minutos.
     - `fator_ocupacao_sala`: público estimado ÷ capacidade da sala candidata.
     - `indice_sobreposicao_trilha`: sessões concorrentes da mesma trilha no mesmo bloco (self-join vetorizado).
     - `horario_pico`: indicador binário para janelas de alta saturação (10h–12h e 14h–16h).
     - `score_viabilidade`: target sintético contínuo [0–100], penalizando superlotação, sobreposição temática e horário de pico.

3. **Modelagem Supervisionada e Persistência:**
   - Treinamento e comparação de modelos clássicos para dados tabulares (**Random Forest** e **XGBoost**).
   - Otimização de hiperparâmetros (*Grid Search* / *Random Search*).
   - Persistência do artefato final serializado via `joblib`.

4. **Validação Quantitativa vs. Baseline:**
   - **Baseline:** Algoritmo guloso determinístico (alocação simples no primeiro slot vago).
   - **Métricas:** RMSE e MAE (para scores contínuos de viabilidade) ou F1-Score e Matriz de Confusão (para classificação binária de conflito/viabilidade).

---

## 📂 Estrutura do Repositório

```text
Projeto-CCF726/
├── data/
│   ├── raw/             <- Dados brutos coletados pelos crawlers (ignorado no git)
│   │   └── eventos_academicos_raw.json   (50.000 instâncias de realocação)
│   └── processed/       <- Dados limpos com features tabulares
│       └── eventos_processados.csv       (50.000 × 17 colunas)
├── docs/                <- Artigo final e apresentações
├── models/              <- Artefatos treinados e serializados (.joblib)
├── notebooks/           <- Análise exploratória de dados e prototipação
├── src/
│   ├── crawler/
│   │   ├── event_crawler.py     <- Scraping web (requests + BS4) + base curada + expansão
│   │   └── dataset_expander.py  <- Módulo auxiliar de expansão de cenários
│   ├── preprocessing/
│   │   └── feature_engineering.py  <- Limpeza, features tabulares e target sintético
│   ├── utils/
│   │   └── s3_client.py         <- Cliente AWS S3 com suporte a credenciais temporárias
│   ├── models/          <- Scripts de treino, otimização e inferência
│   └── baseline/        <- Implementação da heurística gulosa de comparação
├── requirements.txt     <- Dependências do projeto
├── .env                 <- Credenciais AWS (não versionado)
├── .gitignore
└── README.md
```

---

## ⚙️ Como Executar

### Pré-requisitos

```bash
# Criar e ativar o ambiente virtual
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/macOS

# Instalar dependências
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

### Pipeline de Coleta (1ª Etapa)

```bash
# A partir da raiz do repositório:
python -m src.crawler.event_crawler
```

Saída esperada:
- `data/raw/eventos_academicos_raw.json` — 50.000 instâncias de realocação
- Upload automático em `s3://<bucket>/raw/eventos_academicos_raw.json`

### Pipeline de Feature Engineering (2ª Etapa)

```bash
python -m src.preprocessing.feature_engineering
```

Saída esperada:
- `data/processed/eventos_processados.csv` — dataset processado com 17 colunas
- Upload automático em `s3://<bucket>/processed/eventos_processados.csv`

## 📊 Dataset

| Atributo | Descrição |
|---|---|
| `evento` | Nome do evento de origem |
| `data` | Data da sessão (YYYY-MM-DD) |
| `horario_inicio` / `horario_fim` | Faixa horária da sessão |
| `sala` | Sala candidata para realocação |
| `capacidade_sala` | Lotação máxima da sala (pessoas) |
| `trilha` | Área temática da sessão |
| `atividade` | Título da atividade |
| `participantes_estimados` | Público esperado (perturbado com distribuição normal) |
| `duracao_minutos` | *(feature)* Duração calculada |
| `fator_ocupacao_sala` | *(feature)* Taxa de ocupação [0–1.4+] |
| `horario_pico` | *(feature)* Binário: 1 se horário de pico |
| `indice_sobreposicao_trilha` | *(feature)* N° de sessões paralelas da mesma trilha |
| `tipo_cenario` | *(feature)* Categoria do imprevisto (atraso, choque_sala, saturação) |
| `tipo_cenario_cod` | *(feature)* Codificação ordinal (0, 1, 2) do tipo de cenário |
| `score_viabilidade` | *(target)* Score de viabilidade de realocação [0–100] |