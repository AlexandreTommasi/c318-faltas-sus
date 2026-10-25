# Previsão de Faltas em Consultas do SUS

Projeto da disciplina **C318 — Fundamentos de Machine Learning** (INATEL).  
Classificação binária para prever se um paciente comparecerá ou não à consulta agendada pelo SUS.

---

## Integrantes

- Alexandre Tommasi Alves
- Fernando Puebla Stein
- Matheus Vieira Honorio de Souza
- Antonio Augusto D'Assumpção

---

## Contexto e Problema de Negócio

Consultas médicas não comparecidas (no-show) representam desperdício de recursos no sistema público de saúde: vagas ocupadas mas não utilizadas poderiam ser redistribuídas para outros pacientes. No Brasil, o SUS enfrenta filas extensas e capacidade limitada, tornando cada falta um custo real para o sistema e para quem aguarda atendimento.

Este projeto analisa dados reais de agendamentos do SUS de Vitória-ES (2016) para identificar os fatores associados às faltas e construir um modelo preditivo que auxilie na priorização de lembretes ou no preenchimento de vagas liberadas.

---

## Objetivos

### Negócio
- Identificar perfis de pacientes com maior risco de falta.
- Subsidiar estratégias de redução de no-show (ex.: reforço de lembretes, lista de espera ativa).

### Ciência de Dados
- Construir e comparar modelos de classificação binária (`No-show`: Sim / Não).
- Avaliar desempenho com métricas adequadas ao desbalanceamento de classes.
- Interpretar os modelos para extrair insights acionáveis.

---

## Perguntas de Negócio

> **Rascunho — a serem refinadas ao longo do projeto**

1. Quais fatores (idade, doença crônica, bairro, dia da semana etc.) mais influenciam a probabilidade de falta?
2. O envio de lembrete por SMS reduz efetivamente a taxa de não comparecimento?
3. Qual o impacto do tempo de espera entre o agendamento e a data da consulta na taxa de falta?
4. Quantas consultas adicionais seria possível realizar se as faltas previstas fossem substituídas por pacientes em lista de espera?

---

## Enquadramento

| Item | Detalhe |
|---|---|
| Tipo de aprendizagem | Supervisionada |
| Tarefa | Classificação binária |
| Variável-alvo | `No-show` — `Yes` (faltou) / `No` (compareceu) |
| Desafio principal | Desbalanceamento de classes (~20% faltas) |

---

## Dataset

**Medical Appointment No Shows** — Kaggle  
~110 mil agendamentos de consultas em Vitória-ES entre novembro de 2015 e junho de 2016.

### Como baixar

1. Acesse: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
2. Faça login no Kaggle e clique em **Download**.
3. Extraia o arquivo e coloque `KaggleV2-May-2016.csv` em `data/raw/`.

Ou via CLI do Kaggle (requer `pip install kaggle` e configuração de credenciais):

```bash
kaggle datasets download -d joniarroba/noshowappointments -p data/raw/ --unzip
```

> O arquivo CSV **não é versionado** (ver `.gitignore`).

---

## Estrutura do Repositório

```
c318-faltas-sus/
├── data/
│   ├── raw/            # Dataset original (ignorado no git)
│   └── processed/      # Dados tratados (ignorado no git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py         # Carregamento e limpeza
│   ├── features.py     # Engenharia de atributos
│   ├── models.py       # Treinamento e tuning
│   └── evaluate.py     # Métricas e relatórios
├── docs/
│   └── decisoes.md     # Log de decisões do projeto
├── reports/
│   └── figures/        # Gráficos gerados
├── requirements.txt
└── README.md
```

---

## Como Rodar

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd c318-faltas-sus

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar o dataset (ver seção Dataset)

# 5. Abrir os notebooks
jupyter notebook
```

---

## Pesquisa da primeira entrega (S1)

A [pesquisa de abordagens e métricas para classificação desbalanceada](docs/pesquisa_classificacao_desbalanceada.md)
registra a proposta de avaliação, os cuidados contra vazamento e as referências
localizadas no Kaggle, incluindo os limites da consulta. Corresponde à tarefa
de pesquisa de Fernando (#4); o baseline será desenvolvido na S2 (#8).

## Limitações

- Os dados são de **uma única cidade** (Vitória-ES), o que pode limitar a generalização para outros contextos do SUS.
- Os registros cobrem apenas **2015–2016**; mudanças estruturais no sistema de saúde desde então não estão refletidas.
- Não há informações sobre o motivo da falta (transporte, esquecimento, melhora espontânea etc.).
