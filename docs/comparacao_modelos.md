# Comparação dos modelos (issue #14)

Tabela comparativa dos modelos de classificação de faltas. **Versão 2 (18/11):** inclui Dummy, Regressão Logística (baseline e ajustada), Árvore de Decisão e as estratégias de balanceamento. **Random Forest e XGBoost (issue #13) ainda não foram entregues** e entram numa próxima rodada.

## Fontes

| Bloco | Origem | Commit |
|---|---|---|
| A. Dummy e Logística baseline | `main`: `reports/sprints-2-3/baselines.json` (`python -m src.analysis`) | `main` |
| B. Logística ajustada e Árvore | branch `feat/s4-s5-modelagem-tuning` (Fernando): `reports/sprints-4-5/results.json`, `s4_cv.csv` | `0f3cb9a` |
| C. Balanceamento | branch `feat/matheus-sprints-3-4` (Matheus): `reports/sprint-5/evaluation/comparison.csv`, `confusion_matrices.csv` | `3a45142` |

As branches de B e C ainda não foram mescladas na `main`. Os valores abaixo são os salvos nos arquivos, arredondados.

## Como cada bloco foi avaliado (não misturar)

Existem **quatro** protocolos diferentes. Valores de protocolos diferentes não são comparáveis entre si.

| Protocolo | Dados | Uso |
|---|---|---|
| **Teste reservado por paciente** | `GroupShuffleSplit` por `patient_id`, `test_size=0,2`, `random_state=42`: 87.720 de desenvolvimento e **22.169 de teste (4.497 faltas, prevalência 20,29%)**. Mesma função `split_by_patient` nas três branches, o que confere nos arquivos. | Bloco A e Bloco B |
| **Validação cruzada, 5 folds** (`StratifiedGroupKFold`) | Bloco C: sobre os 87.720 de desenvolvimento. Bloco B (`s4_cv.csv`): população a confirmar com o Fernando. Não usa o teste. | Blocos B e C |
| **Busca de hiperparâmetros** | 70.443 linhas de ajuste, 5 folds, objetivo AP. O limiar foi escolhido em outra reserva independente de 17.277 linhas, sem usar o teste. | Seleção de B |
| **Temporal** | Treino com `appointment_day < 25/05/2016`, teste com `scheduled_day ≥ 25/05/2016` (22.908 agendamentos, prevalência 11,80%). | Só Dummy e Logística da `main` |

**Variáveis (iguais em todos os blocos):** `age`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `waiting_days`, `gender`, `neighbourhood`, `weekday`. Ficam de fora `SMS_received` (enviado depois do agendamento, decisão do grupo), identificadores, datas brutas e `same_day`. **Nenhum modelo usa o histórico de faltas do paciente.**

**Métricas:** AP é a *Average Precision*, usada como PR-AUC. Precisão@20% e recall@20% consideram os 20% de agendamentos de maior risco (4.434 no teste), o corte de contato da métrica de negócio. Precisão, recall e F1 dependem do limiar.

## Bloco A e B: teste reservado por paciente (22.169 agendamentos, 4.497 faltas)

### Métricas de ranking (não dependem do limiar)

| Modelo | AP (PR-AUC) | ROC-AUC | Precisão@20% | Recall@20% | Faltas capturadas no top 20% | Ganho sobre o acaso |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 20,3% | 0,500 | 20,3% | 20,0% | ≈ 899 | 1,00x |
| Regressão Logística (baseline, `main`) | 29,4% | 0,644 | 33,7% | 33,2% | ≈ 1.493 | 1,66x |
| Regressão Logística ajustada (C = 0,01) | 29,5% | 0,649 | 33,5% | 33,0% | 1.484 | 1,65x |
| **Árvore de Decisão** (`min_samples_leaf` = 300, sem limite de profundidade) | **33,4%** | **0,713** | **35,7%** | **35,2%** | **1.581** | **1,76x** |
| Random Forest | não entregue | | | | | |
| XGBoost | não entregue | | | | | |

"Ganho sobre o acaso" = faltas capturadas ÷ 899 esperadas ao contatar 20% ao acaso. As capturadas do baseline são derivadas do recall salvo.

### Métricas por limiar

| Modelo / limiar | Precisão | Recall | F1 | Acurácia | Agendamentos sinalizados (% do teste) | Matriz de confusão `[[VN, FP], [FN, VP]]` |
|---|---:|---:|---:|---:|---:|---|
| Dummy (0,5) | 0,0% | 0,0% | 0,0% | 79,7% | 0 | `[[17.672, 0], [4.497, 0]]` (derivada) |
| Logística baseline (0,5) | 29,1% | 1,3% | 2,6% | | 206 (0,9%) | `[[17.526, 146], [4.437, 60]]` |
| Logística ajustada (0,5) | 28,4% | 1,2% | 2,3% | 79,3% | 190 (0,9%) | `[[17.536, 136], [4.443, 54]]` |
| Logística ajustada (**0,194**) | 29,8% | 56,5% | 39,0% | 64,2% | 8.528 (38,5%) | `[[11.687, 5.985], [1.954, 2.543]]` |
| Árvore (0,5) | 0,0% | 0,0% | 0,0% | 79,7% | 0 | `[[17.672, 0], [4.497, 0]]` |
| Árvore (**0,213**) | 30,6% | 73,6% | 43,3% | 60,8% | 10.804 (48,7%) | `[[10.177, 7.495], [1.188, 3.309]]` |

Os limiares 0,194 e 0,213 maximizam o F1 numa reserva de validação independente (F1 de 0,415 e 0,448 nessa reserva), sem usar o teste.

## Bloco C: estratégias de balanceamento (Matheus, Regressão Logística, 5 folds sobre 87.720 linhas)

Média ± desvio entre os folds, limiar fixo 0,5. **Não é teste final** (`test_evaluated: false`): é validação do desenvolvimento, por isso não se compara diretamente com os blocos A e B.

| Estratégia | AP | ROC-AUC | Precisão | Recall | F1 | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sem balanceamento | 0,307 ± 0,005 | 0,658 ± 0,004 | 36,6% | 2,1% | 3,9% | 35,0% | 34,9% |
| `class_weight` | 0,307 ± 0,005 | 0,665 ± 0,004 | 31,3% | 55,7% | 40,1% | 35,0% | 34,9% |
| Subamostragem | 0,306 ± 0,005 | 0,664 ± 0,004 | 31,1% | 56,0% | 40,0% | 34,6% | 34,5% |
| SMOTENC | 0,303 ± 0,005 | 0,654 ± 0,005 | 30,6% | 55,7% | 39,5% | 35,0% | 34,9% |

Matrizes de confusão (soma dos folds, 87.720 linhas e 17.594 faltas): sem balanceamento `[[69.497, 629], [17.232, 362]]`; `class_weight` `[[48.608, 21.518], [7.788, 9.806]]`; subamostragem `[[48.331, 21.795], [7.742, 9.852]]`; SMOTENC `[[47.872, 22.254], [7.796, 9.798]]`.

## Bloco B (validação cruzada): `s4_cv.csv` (Fernando, 5 folds)

| Modelo | AP | ROC-AUC | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,304 ± 0,003 | 0,659 ± 0,003 | 35,9% | 2,1% | 3,9% |
| Árvore de Decisão | 0,229 ± 0,003 | 0,559 ± 0,004 | 30,1% | 27,3% | 28,6% |

Médias e desvios calculados a partir dos 5 folds do arquivo. **Atenção:** a árvore aqui (AP 0,229) é muito pior que a árvore ajustada (AP 0,341 na busca e 0,334 no teste). Provavelmente é a árvore com hiperparâmetros padrão, sem restrição de folhas, que sobreajusta. Isso é uma inferência pelos números e precisa ser confirmado com o Fernando antes de a tabela ser apresentada.

## Validação temporal (teste: 22.908 agendamentos, 2.702 faltas)

Só existe para Dummy e Regressão Logística baseline (`main`).

| Modelo | Precisão | Recall | F1 | AP | ROC-AUC | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 0,0% | 0,0% | 0,0% | 11,8% | 0,500 | 11,8% | 20,0% |
| Regressão Logística | 0,0% | 0,0% | 0,0% | 16,5% | 0,585 | 17,6% | 29,8% |
| Árvore, Random Forest, XGBoost | não avaliados | | | | | | |

## O que os resultados mostram

- **A árvore de decisão regularizada é o melhor modelo até agora, em ranking.** No teste reservado, AP 0,334 contra 0,295 da logística e 0,203 da prevalência; ROC-AUC 0,713 contra 0,649; os 20% de maior risco capturam 35,2% das faltas (1,76x o acaso) contra 33,0% (1,65x). O resultado é de uma única partição, sem intervalo de confiança. O AP da busca (0,341) e o do teste (0,334) são próximos, o que indica que o ganho não é só efeito da seleção.
- **Ganho moderado.** Todos os modelos ficam entre 1,65x e 1,76x o acaso no top 20%. A diferença entre logística e árvore (cerca de 2 pontos percentuais de recall@20%) é pequena em termos práticos.
- **Balancear muda o ponto de operação, não o ranking.** `class_weight`, subamostragem e SMOTENC elevam o recall em 0,5 de cerca de 2% para cerca de 56%, mas AP (0,303 a 0,307) e recall@20% (cerca de 35%) praticamente não mudam. A conclusão é que ajustar o limiar produz o mesmo efeito sem reamostrar, o que é coerente com os limiares ajustados dos blocos A e B.
- **No limiar padrão a árvore e a logística não servem para decisão:** a árvore não prevê nenhuma falta e a logística prevê menos de 1% dos agendamentos. Só o ranking ou o limiar ajustado são utilizáveis.
- **Limiar ajustado (F1) contata muito:** a árvore sinaliza 48,7% da agenda (10.804 contatos) para capturar 73,6% das faltas (3.309 de 4.497) com precisão de 30,6%. A logística sinaliza 38,5% (8.528) para capturar 56,5% (2.543). Cerca de 70% dos contatos vão a quem compareceria. Esse custo entra na análise da #18.
- **Nenhum modelo usa o histórico do paciente.** A análise da P5 indicou que o histórico separa bem quem nunca faltou (13,5% de falta), com cobertura de só 28% das linhas. Testar essa variável (sem vazamento) é uma melhoria pendente.

## Pendências

- Random Forest e XGBoost (#13, Alexandre), na mesma partição por paciente (`random_state=42`), com e sem limiar ajustado.
- Confirmar com o Fernando a configuração da árvore e a população dos folds de `s4_cv.csv`.
- Decisão do grupo sobre histórico de faltas e `same_day`.
- Avaliação das estratégias de balanceamento no teste reservado (hoje só em validação cruzada) e dos modelos de B na validação temporal.
- Modelo final e SHAP (#17).
- Mesclar as branches `feat/s4-s5-modelagem-tuning` e `feat/matheus-sprints-3-4` na `main`, para que os arquivos citados nas fontes existam no repositório.
- Arquivos `decision_tree_search.csv`, `logistic_search.csv` e `*_thresholds.csv` não foram copiados aqui; ficam na branch do Fernando.
