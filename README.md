# Sistema Preditivo de Realocação de Atividades para Organização Autônoma de Eventos Científicos

> **Disciplina:** CCF726 - Engenharia de Aprendizado de Máquina  
> **Instituição:** Universidade Federal de Viçosa (UFV) – Campus Florestal  
> **Professor:** Fabrício A. Silva  


---

## 📌 Visão Geral do Problema

A gestão em tempo real de grandes eventos científicos e conferências acadêmicas (ex: CSBC, simpósios e semanas acadêmicas) é sujeita a imprevistos dinâmicos constantes, como atrasos de palestrantes, choque de capacidade física em salas e cancelamentos de sessões.

Embora ecossistemas baseados em sistemas multi-agentes e interfaces conversacionais identifiquem a necessidade de remanejar a agenda, a escolha de um novo horário e espaço físico demanda inteligência analítica. Uma alocação inadequada gera saturação em salas, esvaziamento de sessões correlatas e sobreposição temática.

Este projeto desenvolve um **motor de recomendação e predição orientado a dados tabulares**. O modelo atua recebendo um conflito de cronograma e prevendo a taxa de sucesso/score de viabilidade para diferentes hipóteses de realocação (sala e faixa de horário), servindo como o núcleo analítico de decisão para agentes autônomos.

---

## 🚀 Metodologia e Pipeline

O projeto cumpre o fluxo completo de Engenharia de Machine Learning:

1. **Coleta de Dados (*Data Collection*):**
   - Construção de crawlers e raspadores customizados para extração de dados públicos de grades de eventos científicos (horários, trilhas, salas e estimativa de participantes).
   - Persistência estruturada em nuvem via **Amazon S3** (formato JSON/CSV).

2. **Engenharia de Atributos (*Feature Engineering*):**
   - Limpeza e padronização temporal das grades coletadas.
   - Construção de features tabulares:
     - `fator_ocupacao_sala`: público estimado dividido pela capacidade da sala candidata.
     - `indice_sobreposicao_trilha`: sessões concorrentes da mesma área ocorrendo no mesmo bloco.
     - `horario_pico`: indicador binário para janelas de alta saturação do evento.
     - `duracao_minutos`: tempo de duração total da atividade.

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
│   └── processed/       <- Dados limpos e conjunto com features tabulares
├── docs/                <- Artigo científico final e apresentações
├── models/              <- Artefatos treinados e serializados
├── notebooks/           <- Análise exploratória de dados e prototipação
├── src/
│   ├── crawler/         <- Scripts de raspagem e envio para a AWS S3
│   ├── preprocessing/   <- Pipelines de limpeza e engenharia de atributos
│   ├── models/          <- Scripts de treino, otimização e inferência
│   └── baseline/        <- Implementação da heurística gulosa de comparação
├── requirements.txt     <- Dependências do projeto
├── .gitignore
└── README.md