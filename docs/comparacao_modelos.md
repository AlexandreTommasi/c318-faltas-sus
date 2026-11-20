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
| **Teste reservado por paciente** | `GroupShuffleSplit` por `patient_id`, `test_size=0,2`, `random_state=42`: 87.720 de desenvolvimento e **22.169 de teste (4.497 faltas, prevalência 20,29%)**. Mesma função `split_by_patient` nas três branches, o que confere nos arquivos. | Reservado para avaliação final (#17) |
| **Validação cruzada, 5 folds** (`StratifiedGroupKFold`) | Bloco C: sobre os 87.720 de desenvolvimento. Bloco B (`s4_cv.csv`): população a confirmar com o Fernando. Não usa o teste. | Blocos B e C |
| **Busca de hiperparâmetros** | 70.443 linhas de ajuste, 5 folds, objetivo AP. O limiar foi escolhido em outra reserva independente de 17.277 linhas, sem usar o teste. | Seleção de B |
| **Temporal** | Treino com `appointment_day < 25/05/2016`, teste com `scheduled_day ≥ 25/05/2016` (22.908 agendamentos, prevalência 11,80%). | Só Dummy e Logística da `main` |

**Variáveis (iguais em todos os blocos):** `age`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `waiting_days`, `gender`, `neighbourhood`, `weekday`. Ficam de fora `SMS_received` (enviado depois do agendamento, decisão do grupo), identificadores, datas brutas e `same_day`. **Nenhum modelo usa o histórico de faltas do paciente.**

**Métricas:** AP é a *Average Precision*, usada como PR-AUC. Precisão@20% e recall@20% consideram os 20% de agendamentos de maior risco, o corte de contato da métrica de negócio. Precisão, recall e F1 dependem do limiar.

## Seleção em validação (Fernando, S4/S5)

Resultados da busca de hiperparâmetros e escolha de limiar, **sem usar o teste reservado**. O AP abaixo é a média dos 5 folds na partição de ajuste (70.443 linhas); o limiar foi escolhido numa reserva de validação independente (17.277 linhas), também sem o teste.

| Modelo | AP médio (CV busca) | Limiar escolhido | F1 na reserva de validação |
|---|---:|---:|---:|
| Dummy (prior) | — | — | — |
| Regressão Logística (C = 0,01) | 0,3057 | 0,194 | 0,415 |
| **Árvore de Decisão** (`max_depth=None`, `min_samples_leaf=300`) | **0,3410** | **0,213** | **0,448** |
| Random Forest | não entregue | | |
| XGBoost | não entregue | | |

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

Médias e desvios calculados a partir dos 5 folds do arquivo. **Atenção:** a árvore aqui (AP 0,229) é muito pior que a árvore ajustada (AP 0,341 na busca). Provavelmente é a árvore com hiperparâmetros padrão, sem restrição de folhas, que sobreajusta. Isso é uma inferência pelos números e precisa ser confirmado com o Fernando antes de a tabela ser apresentada.

## Validação temporal (teste: 22.908 agendamentos, 2.702 faltas)

Só existe para Dummy e Regressão Logística baseline (`main`).

| Modelo | Precisão | Recall | F1 | AP | ROC-AUC | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 0,0% | 0,0% | 0,0% | 11,8% | 0,500 | 11,8% | 20,0% |
| Regressão Logística | 0,0% | 0,0% | 0,0% | 16,5% | 0,585 | 17,6% | 29,8% |
| Árvore, Random Forest, XGBoost | não avaliados | | | | | | |

## O que os resultados mostram

- **A árvore de decisão regularizada lidera em validação.** Na busca de hiperparâmetros (5 folds sobre 70.443 linhas de ajuste), a árvore (`max_depth=None`, `min_samples_leaf=300`) alcançou AP médio 0,341 contra 0,306 da logística (C = 0,01). O resultado ainda não tem avaliação no teste reservado; todos os candidatos (Dummy, logística e árvore ajustadas, RF e XGBoost) serão avaliados uma única vez na #17.
- **Balancear muda o ponto de operação, não o ranking.** `class_weight`, subamostragem e SMOTENC elevam o recall em 0,5 de cerca de 2% para cerca de 56%, mas AP (0,303 a 0,307) e recall@20% (cerca de 35%) praticamente não mudam. A conclusão é que ajustar o limiar produz o mesmo efeito sem reamostrar.
- **No limiar padrão os modelos não servem para decisão:** a árvore não prevê nenhuma falta e a logística prevê menos de 1% dos agendamentos. Só o ranking ou o limiar ajustado são utilizáveis.
- **Limiar ajustado (F1) contata muito:** na reserva de validação, a logística (0,194) atingiu F1 0,415 e a árvore (0,213) F1 0,448. Cerca de 70% dos contatos vão a quem compareceria. Esse custo entra na análise da #18.
- **Nenhum modelo usa o histórico do paciente.** A análise da P5 indicou que o histórico separa bem quem nunca faltou (13,5% de falta), com cobertura de só 28% das linhas. Testar essa variável (sem vazamento) é uma melhoria pendente.

## Pendências

- Random Forest e XGBoost (#13, Alexandre), na mesma partição por paciente (`random_state=42`), com e sem limiar ajustado.
- Confirmar com o Fernando a configuração da árvore e a população dos folds de `s4_cv.csv`.
- Decisão do grupo sobre histórico de faltas e `same_day`.
- Avaliação dos modelos de B na validação temporal.
- Avaliação final de todos os candidatos (Dummy, logística e árvore ajustadas, RF e XGBoost) no teste reservado (#17).
- Mesclar a branch `feat/matheus-sprints-3-4` na `main`, para que os arquivos do Bloco C existam no repositório.
- Arquivos `decision_tree_search.csv`, `logistic_search.csv` e `*_thresholds.csv` não foram copiados aqui; ficam na branch do Fernando.
