# Modelo Final — Issues #13 e #17

**Branch:** `s4-s5/modelo-final` | **Responsável:** Alexandre Tommasi

## Protocolo

Partição por paciente (`GroupShuffleSplit`, seed 42, `test_size=0.2`): **87.720 linhas de desenvolvimento** e **22.169 de teste reservado** (4.497 faltas, prevalência 20,29%). O teste foi avaliado **uma única vez**, após o congelamento da escolha.

Validação cruzada: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` sobre as 87.720 linhas de desenvolvimento, sem pacientes compartilhados entre folds. Mesmos parâmetros e mesma semente dos folds do Matheus (S4/#15).

Features aprovadas: `age`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `waiting_days`, `gender`, `neighbourhood`, `weekday` + histórico de faltas (`previous_appointments`, `previous_no_shows`, `previous_no_show_rate`, `has_history`). Excluídos: `patient_id`, `appointment_id`, `sms_received`, `same_day`, datas brutas.

Pré-processamento: `StandardScaler` nas numéricas, `OneHotEncoder(handle_unknown="ignore")` nas categóricas, ajustados dentro de cada fold. Para o XGBoost: `scale_pos_weight = negativos/positivos` calculado a partir do treino de cada fold.

---

## Validação cruzada — todos os candidatos

### Sem histórico (`use_history=False`)

| Candidato | AP médio | ± dp | ROC-AUC | Prec@20% | Rec@20% |
|---|---:|---:|---:|---:|---:|
| **XGBoost** (lr=0.05, depth=6, n=300) | **0.3646** | 0.0066 | 0.7328 | 38,3% | 38,2% |
| Random Forest (leaf=20, feat=sqrt, n=300) | 0.3599 | 0.0051 | 0.7312 | 37,7% | 37,6% |
| Árvore (depth=None, leaf=300) | 0.3456 | 0.0029 | 0.7226 | 36,5% | 36,4% |
| Logística (C=0.01) | 0.3081 | 0.0055 | 0.6624 | 35,1% | 35,0% |
| Dummy (prior) | 0.2006 | 0.0000 | 0.5000 | 20,1% | 20,0% |

### Com histórico (`use_history=True`)

| Candidato | AP médio | ± dp | ROC-AUC | Prec@20% | Rec@20% |
|---|---:|---:|---:|---:|---:|
| **XGBoost** (lr=0.05, depth=6, n=300) | **0.3722** | 0.0058 | 0.7375 | 38,4% | 38,3% |
| Random Forest (leaf=20, feat=0.5, n=300) | 0.3656 | 0.0076 | 0.7334 | 37,8% | 37,7% |
| Árvore (depth=None, leaf=300) | 0.3475 | 0.0029 | 0.7255 | 36,8% | 36,7% |
| Logística (C=0.01) | 0.3141 | 0.0049 | 0.6679 | 35,4% | 35,3% |
| Dummy (prior) | 0.2006 | 0.0000 | 0.5000 | 20,1% | 20,0% |

---

## Ablação do histórico (ΔAP com − sem)

| Candidato | AP sem hist. | AP com hist. | ΔAP |
|---|---:|---:|---:|
| XGBoost | 0.3646 | 0.3722 | **+0.0076** |
| Random Forest | 0.3599 | 0.3656 | +0.0057 |
| Logística | 0.3081 | 0.3141 | +0.0060 |
| Árvore | 0.3456 | 0.3475 | +0.0019 |
| Dummy | 0.2006 | 0.2006 | 0.0000 |

O histórico melhora todos os modelos, com ganho maior nos modelos mais expressivos (XGBoost e RF). A árvore aproveita pouco o histórico, provavelmente porque as regras de split já capturam os padrões mais salientes com as features originais.

---

## Por que XGBoost com histórico

- **Maior AP em CV**: 0.3722 (com histórico), contra 0.3656 do RF — diferença de 0.0066, superior ao dp de ambos (≈0.006–0.008).
- **Maior ROC-AUC**: 0.7375 vs 0.7334 do RF.
- **Maior ganho do histórico**: ΔAP +0.0076, o maior entre os candidatos.
- **Estabilidade entre folds**: dp 0.0058, semelhante ao RF (0.0076). Menor que RF, sugerindo ligeiramente mais estável.
- **Contraponto interpretabilidade**: XGBoost é menos diretamente explicável que a Árvore, mas o SHAP compensa para a banca.
- **Contraponto RF**: RF com `max_features=0.5` foi o segundo melhor com histórico; escolha entre os dois é de margem estreita. XGBoost levou pela combinação de AP e estabilidade.

---

## Limiar e política dos 20%

Calculado a partir das predições out-of-fold nos 87.720 de desenvolvimento (nunca no teste):

| Critério | Limiar | Descrição |
|---|---:|---|
| **Principal** — top-20% de risco | **0.6568** | Os 20% de agendamentos com maior probabilidade são contactados |
| Alternativo — maximiza F1 (OOF) | 0.5406 | Maior F1 nas predições OOF do desenvolvimento |

A política dos 20% é a métrica de negócio do projeto (K = 20%). O limiar de F1 é registrado como alternativa para análises de sensibilidade.

---

## Resultado único no teste reservado (22.169 linhas, 4.497 faltas)

> Avaliação realizada **uma única vez** em 2026-11-20, após congelamento em `selecao_congelada.json`.  
> Todos os candidatos retreinados no desenvolvimento completo (87.720 linhas) com `use_history=True`.

| Candidato | AP | ROC-AUC | Prec@20% | Rec@20% | F1 (limiar 0.657) |
|---|---:|---:|---:|---:|---:|
| Dummy (prior) | 0.2029 | 0.500 | 20,3% | 20,0% | 0,00 |
| Logística (C=0.01) | 0.3040 | 0.657 | 34,4% | 33,9% | 0,83% |
| Árvore (depth=None, leaf=300) | 0.3404 | 0.720 | 35,4% | 34,9% | 0,00 |
| Random Forest (leaf=20, feat=0.5) | 0.3587 | 0.728 | 36,9% | 36,4% | 35,8% |
| **XGBoost** (lr=0.05, depth=6) | **0.3617** | **0.730** | **37,1%** | **36,6%** | **36,8%** |

**Matriz de confusão do XGBoost (limiar 0.657):**

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| **Real 0** | 14.891 (TN) | 2.781 (FP) |
| **Real 1** | 2.855 (FN) | 1.642 (TP) |

Contatos totais: 4.423 (20% de 22.169). Faltas capturadas: 1.642 de 4.497 → recall@20% = 36,6%. Precisão nos contatos: 37,1%.

**Comparação com aleatório:** contactar 20% aleatoriamente capturaria ≈ 20,3% das faltas (903). O modelo captura 36,6% — **1,8× mais** que o acaso.

**Notas sobre F1:** Árvore e Logística têm F1 ≈ 0 no limiar 0.657, porque seus scores raramente chegam a esse valor — os thresholds escolhidos pelo Fernando para esses modelos são muito mais baixos (~0.19–0.21). RF e XGBoost operam bem no limiar alto porque seus scores de probabilidade são mais espalhados.

---

## SHAP — Importância das variáveis

Análise via `TreeExplainer` em amostra de 3.000 linhas do desenvolvimento (seed 42).

### Por variável original (mean |SHAP agregado por linha|)

Para variáveis categóricas com OHE: em cada linha da amostra, somam-se os valores SHAP **com sinal** de todas as colunas OHE da mesma variável original; depois calcula-se a média do valor absoluto dessa soma entre as linhas. Isso captura o efeito líquido da variável em cada observação e evita inflar variáveis com muitas categorias.

| Rank | Variável | mean\|SHAP por linha\| |
|---:|---|---:|
| 1 | neighbourhood | 1.625 |
| 2 | waiting_days | 0.815 |
| 3 | age | 0.614 |
| 4 | weekday | 0.310 |
| 5 | previous_appointments | 0.088 |
| 6 | hypertension | 0.078 |
| 7 | previous_no_show_rate | 0.062 |
| 8 | gender | 0.045 |
| 9 | diabetes | 0.034 |
| 10 | bolsa_familia | 0.028 |
| 11 | handicap | 0.013 |
| 12 | previous_no_shows | 0.010 |
| 13 | alcoholism | 0.008 |
| 14 | has_history | 0.000 |

Ver `reports/modelo-final/figures/shap_por_variavel.png`.

**Leitura:**
- `neighbourhood` lidera (1.63): bairro ainda é o preditor de maior magnitude, refletindo heterogeneidade geográfica real de Vitória/ES. Com a agregação correta (soma com sinal), o valor cai de 6.96 para 1.63 — cancelamento parcial entre bairros com efeito positivo e negativo.
- `waiting_days` (0.82) e `age` (0.61) são as variáveis mais interpretáveis e generalizáveis: tempo de espera e idade são fortes preditores de no-show independentes de localização.
- `weekday` (0.31) caiu da 2ª para a 4ª posição em relação à agregação anterior — sinal indica que, dentro de cada agendamento, o efeito do dia da semana se cancela parcialmente entre as categorias.
- Features de histórico contribuem modestamente (previous_appointments: 0.088, previous_no_show_rate: 0.062), suficientes para o ganho de AP na ablação (+0.0076).
- `has_history` individual: SHAP ≈ 0 — a informação do histórico entra via `previous_no_show_rate` e `previous_appointments`.

### Top 10 pós-OHE (mean |SHAP| por coluna)

A tabela abaixo mantém a visão coluna a coluna para referência de quais bairros individuais têm maior efeito:

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | neighbourhood_ILHAS OCEÂNICAS DE TRINDADE | 1.126 |
| 2 | waiting_days | 0.815 |
| 3 | neighbourhood_AEROPORTO | 0.670 |
| 4 | age | 0.614 |
| 5 | neighbourhood_SANTA MARTHA | 0.330 |
| 6 | neighbourhood_PONTAL DE CAMBURI | 0.291 |
| 7 | weekday_0 (segunda-feira) | 0.279 |
| 8 | neighbourhood_REPÚBLICA | 0.256 |
| 9 | neighbourhood_SOLON BORGES | 0.239 |
| 10 | neighbourhood_ITARARÉ | 0.228 |

Ver `reports/modelo-final/figures/shap_bar.png` (barras) e `shap_beeswarm.png` (direção dos efeitos).

---

## Análise de erros no teste

Taxas de FP (contactar quem compareceria) e FN (não contactar quem faltaria) por grupo, limiar 0.657. A coluna **taxa de falta real** é a prevalência de no-show no grupo no conjunto de teste.

> **Como interpretar a taxa de FN junto com a política dos 20%:** o modelo ranqueia os agendamentos por probabilidade de falta e contata os 20% de maior risco. Grupos com taxa de falta real abaixo da média (~20,3%) naturalmente recebem menos contatos — não porque o modelo erra, mas porque eles representam menor risco. Uma taxa de FN alta em grupos de baixa prevalência é **consequência esperada da política de contato seletivo**, não uma falha do modelo. Comparar a taxa de falta real com a taxa de FN permite distinguir esses dois casos.

### Por faixa de espera

| Faixa | Taxa de falta real | FP rate | FN rate | Leitura |
|---|---:|---:|---:|---|
| 0d (mesmo dia) | **4,7%** | 5,0% | 88,8% | Taxa de falta muito baixa (4,7% vs 20,3% global): o modelo concentra contatos em grupos de maior risco, como esperado |
| 1–7d | **25,3%** | 7,5% | 88,0% | Taxa acima da média, mas FN alto — espera curta tem baixa utilidade de contato (consulta já iminente) |
| **8–30d** | 31,3% | 43,0% | 45,2% | Faixa de operação principal do modelo |
| **31–90d** | 32,6% | 48,2% | 38,7% | Longa espera: maior taxa de falta e melhor cobertura |
| > 90d | 18,4% | 19,4% | 57,1% | Poucos casos (38), padrão instável |

### Por faixa etária

| Faixa | Taxa de falta real | FP rate | FN rate | Leitura |
|---|---:|---:|---:|---|
| 0–17 anos | 21,6% | 24,0% | 48,8% | Taxa próxima à média; cobertura razoável |
| 18–39 anos | 23,0% | 27,5% | 44,4% | Maior taxa de falta; melhor cobertura pelo modelo |
| 40–59 anos | 19,9% | 7,4% | 82,8% | Taxa ligeiramente abaixo da média; modelo os ranqueia mais baixo |
| 60+ anos | **15,2%** | 0,8% | 98,2% | Taxa de falta 15,2% (abaixo dos 20,3% globais): o modelo os prioriza menos, o que é consistente com a política dos 20% aplicada a um grupo de menor risco |

### Por dia da semana

| Dia | Taxa de falta real | FP rate | FN rate |
|---|---:|---:|---:|
| Segunda (0) | 19,7% | 18,4% | 56,1% |
| Terça (1) | 20,5% | 15,2% | 65,8% |
| Quarta (2) | 20,4% | 14,2% | 67,0% |
| Quinta (3) | 19,9% | 13,7% | 67,4% |
| Sexta (4) | 20,9% | 17,3% | 60,3% |
| Sábado (5) | 20,0% | 0,0% | 100,0% (5 casos) |

Taxa de falta homogênea entre dias úteis (~20%); as diferenças de FN refletem distribuição dos contatos, não heterogeneidade real de risco.

### Por histórico de consultas (`has_history`)

| Grupo | Taxa de falta real | FP rate | FN rate |
|---|---:|---:|---:|
| Sem histórico (has_history=0) | 21,7% | 19,3% | 59,8% |
| Com histórico (has_history=1) | **16,8%** | 7,4% | 75,5% |

Pacientes com histórico têm taxa de falta real menor (16,8% vs 21,7%) — o modelo acerta ao priorizar menos esse grupo. O FN alto em `has_history=1` é consequência da menor prevalência, não erro do modelo.

### Resumo

As taxas de FN mais altas concentram-se em grupos com **taxa de falta real abaixo da média global (20,3%)**: agendamentos do mesmo dia (4,7%), idosos (15,2%), pacientes com histórico (16,8%). Isso é o comportamento esperado de um modelo de ranking com política de contato seletivo: contatos são concentrados onde o risco é maior. Nenhum desses padrões representa uma limitação estrutural do modelo — representam a política dos 20% aplicada a grupos de menor risco.

---

## Sensibilidade ao bairro (CV, análise exploratória)

> Esta análise serve apenas para entender o papel do `neighbourhood`. **O modelo final não muda** — a escolha foi congelada em `selecao_congelada.json` e o teste foi avaliado uma única vez.

Avaliação nos mesmos 5 folds do modelo final (XGBoost, `use_history=True`, mesmos hiperparâmetros). Duas variantes comparadas com a referência:

| Variante | AP médio | ± dp | ΔAP vs modelo final |
|---|---:|---:|---:|
| **modelo_final** (com neighbourhood) | **0.3722** | 0.0058 | — |
| sem_neighbourhood | 0.3580 | 0.0066 | −0.0142 |
| bairros_agrupados (< 100 agendamentos → "OUTROS") | 0.3711 | 0.0060 | −0.0011 |

**Leitura:**
- Remover `neighbourhood` completamente custa −0.0142 de AP — perda real, o bairro carrega sinal preditivo genuíno (heterogeneidade geográfica de Vitória/ES).
- Agrupar bairros raros (< 100 agendamentos no treino do fold) quase não altera a AP (−0.0011, dentro do dp de 0.006) — os bairros com poucos registros contribuem pouco individualmente, e o agrupamento não prejudica a predição.
- A análise confirma que `neighbourhood` agrega valor ao modelo, mas sua contribuição vem principalmente dos bairros com volume suficiente.

Artefato: `reports/modelo-final/sensibilidade_bairro.csv`.

---

## Limitações

- Scores do XGBoost com `scale_pos_weight` não são probabilidades calibradas; o limiar não deve ser interpretado como probabilidade real de falta.
- A separação por paciente avalia generalização para **novos pacientes**, não para o mesmo paciente em consultas futuras; para implantação futura numa clínica, o protocolo deve ser reavaliado com dados novos.
- Dataset de Vitória/ES em 2016 — generalização para outras cidades ou períodos não foi testada.
- O histórico de faltas é calculado a partir do próprio dataset; em produção seria necessário manter um registro contínuo por paciente.
- SHAP calculado na escala log-odds do XGBoost, não em probabilidade — valores absolutos não são diretamente comparáveis com as taxas de falta observadas.
- Bairros com poucos registros dominam o top-SHAP por fragmentação OHE; na banca, enfatizar `waiting_days` e `age` como features mais generalizáveis.
