# AGENTS.md

## Ambiente

- Ambiente verificado com Python 3.15.2. Criar e usar a venv local (ignorada no git):
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```
- Dataset: `data/raw/KaggleV2-May-2016.csv` (ignorado no git). Fonte:
  <https://www.kaggle.com/datasets/joniarroba/noshowappointments> —
  `kaggle datasets download -d joniarroba/noshowappointments -p data/raw/ --unzip`.

## Comandos

- Testes: `.venv/bin/python -m unittest discover -s tests -v`
- Limpeza + features (CSV processado): `python -m src.data`
- Baselines e correlações (sprints 2–3): `python -m src.analysis`
- EDA agregada do treino (S3): `python -m src.eda` → `reports/sprints-3-4/eda/`
- Comparação de balanceamento (S4): `python -m src.imbalance` → `reports/sprints-3-4/imbalance/`
- Figuras a partir das tabelas congeladas: `python -m src.sprint_plots` → `reports/sprints-3-4/figures/`
- Rodar comandos a partir da raiz do repositório (pacote `src`).

## Guardrails de vazamento e avaliação

- Alvo `no_show`: 1 = faltou. Partição por paciente (`split_by_patient`,
  `GroupShuffleSplit` seed 42); o teste (20%, 22.169 linhas) fica reservado —
  nunca entra em EDA, CV ou seleção de modelo.
- Features aprovadas: `NUMERIC_FEATURES` + `CATEGORICAL_FEATURES` em
  `src/features.py`. Excluídas: `patient_id`, `appointment_id`, `sms_received`
  (posterior ao agendamento), `same_day`, datas brutas e features de histórico
  fora do escopo aprovado.
- Pré-processamento e reamostradores só são ajustados dentro de cada fold
  (`imblearn.pipeline.Pipeline` em `src/imbalance.py`); folds são
  `StratifiedGroupKFold` sem pacientes compartilhados.
- SMOTENC trata como discretas `bolsa_familia`, `hypertension`, `diabetes`,
  `alcoholism`, `handicap` e as categóricas — valores nunca interpolados.
- Ranking top-20% usa captura esperada sob empates; desvios entre folds são
  desvio-padrão amostral, não intervalo de confiança.
- Scores de modelos com `class_weight` ou reamostragem não são probabilidades
  calibradas; associações de EDA não são efeitos causais.
- Figuras e notebooks de relatório leem apenas artefatos agregados congelados;
  não publicar registros individuais nem deixar outputs de notebook com dados
  brutos commitados.
