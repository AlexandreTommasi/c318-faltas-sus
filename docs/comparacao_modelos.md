# Comparação dos modelos (issue #14)

Tabela comparativa dos modelos de classificação de faltas. **Versão 3 (22/11):** reúne Dummy, Regressão Logística, Árvore de Decisão, Random Forest, XGBoost e as estratégias de balanceamento. O modelo final escolhido pelo grupo é o **XGBoost com histórico de faltas do paciente**.

## Fontes

| Bloco | Origem |
|---|---|
| A. Dummy e Logística baseline (sem histórico) | `main`: `reports/sprints-2-3/baselines.json` (`python -m src.analysis`) |
| B. Cinco candidatos no teste reservado (com histórico) e ablação | branch `s4-s5/modelo-final` (Alexandre, commit `43e47a3`): `reports/modelo-final/teste_final.csv`, `teste_final.json`, `cv_candidatos.csv`, `cv_ablacao_historico.csv`, `selecao_congelada.json` |
| C. Logística ajustada e Árvore (sem histórico) | Fernando (PR #40): `reports/sprints-4-5/` |
| D. Balanceamento | Matheus (PR #39): `reports/sprint-5/evaluation/` |

## Como cada bloco foi avaliado (não misturar)

| Protocolo | Dados | Onde aparece |
|---|---|---|
| **Teste reservado por paciente** | `GroupShuffleSplit` por `patient_id`, `test_size=0,2`, `random_state=42`: 87.720 de desenvolvimento e **22.169 de teste (4.497 faltas, prevalência 20,29%)**. A mesma função `split_by_patient` é usada em todas as entregas. | Blocos 1, 2 e 3 |
| **Validação cruzada, 5 folds** (`StratifiedGroupKFold`) | Sobre os 87.720 de desenvolvimento, sem usar o teste. | Blocos 4 e 5 |
| **Seleção do modelo final** | Hiperparâmetros e limiar escolhidos por AP em validação cruzada e previsões *out-of-fold* do desenvolvimento, **congelados antes de abrir o teste** (`selecao_congelada.json`). | Bloco 1 |
| **Temporal** | Treino com `appointment_day < 25/05/2016`, teste com `scheduled_day ≥ 25/05/2016` (22.908 agendamentos, prevalência 11,80%). | Bloco 6 |

**Variáveis.** Base de 10: `age`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `waiting_days`, `gender`, `neighbourhood`, `weekday`. **Com histórico**, entram mais 4: `previous_appointments`, `previous_no_shows`, `previous_no_show_rate`, `has_history`, calculadas só com consultas de data estritamente anterior. `SMS_received`, `same_day`, identificadores e datas brutas ficam de fora em todos os casos.

**Métricas.** AP é a *Average Precision* (PR-AUC). Precisão@20% e recall@20% consideram os 20% de maior risco (4.434 agendamentos no teste), o corte da métrica de negócio. Precisão, recall e F1 dependem do limiar.

## Bloco 1: cinco candidatos no teste reservado, **com histórico** (22.169 agendamentos)

Todos avaliados uma única vez no mesmo teste, depois do congelamento da seleção.

| Modelo | AP | ROC-AUC | Precisão@20% | Recall@20% | Faltas capturadas no top 20% | Ganho sobre o acaso |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 20,3% | 0,500 | 20,3% | 20,0% | ≈ 899 | 1,00x |
| Regressão Logística | 30,4% | 0,657 | 34,4% | 33,9% | ≈ 1.526 | 1,70x |
| Árvore de Decisão | 34,0% | 0,720 | 35,4% | 34,9% | ≈ 1.571 | 1,75x |
| Random Forest | 35,9% | 0,728 | 36,9% | 36,4% | ≈ 1.638 | 1,82x |
| **XGBoost (modelo final)** | **36,2%** | **0,730** | **37,1%** | **36,6%** | **≈ 1.645** | **1,83x** |

As faltas capturadas são derivadas do recall salvo (recall × 4.497). "Ganho sobre o acaso" = capturadas ÷ 899.

**Configurações.** Logística: `C = 0,01`, `max_iter = 2000`. Árvore: sem limite de profundidade, `min_samples_leaf = 300`. Random Forest: 300 árvores, `min_samples_leaf = 20`, `max_features = 0,5`, `class_weight = "balanced"`. XGBoost: 300 árvores, `max_depth = 6`, `learning_rate = 0,05`, `scale_pos_weight` = negativos ÷ positivos, calculado por fold.

### Random Forest e XGBoost com o limiar congelado

O limiar principal (0,657) é o que isola os 20% de maior risco nas previsões *out-of-fold* do XGBoost. Pelo código de `run_final`, o mesmo valor é aplicado a todos os candidatos; por isso só faz sentido como política de contato para o XGBoost.

| Modelo | Precisão | Recall | F1 | Acurácia | Agendamentos sinalizados | Matriz `[[VN, FP], [FN, VP]]` |
|---|---:|---:|---:|---:|---:|---|
| XGBoost (0,657) | 37,1% | 36,5% | 36,8% | 74,6% | 4.423 (19,95%) | `[[14.891, 2.781], [2.855, 1.642]]` |
| Random Forest (0,657) | 37,2% | 34,6% | 35,8% | 74,9% | 4.181 (18,9%) | `[[15.045, 2.627], [2.943, 1.554]]` |

Precisão, recall e acurácia foram calculadas a partir das matrizes salvas. Para Dummy, Logística e Árvore, o F1 com esse limiar é próximo de zero (o limiar fica acima dos escores desses modelos) e **não é comparável** com o dos modelos acima.

## Bloco 2: sem histórico, teste reservado (baselines e ajustes da S4/S5)

Mesmo teste de 22.169 agendamentos, 10 variáveis, sem histórico.

| Modelo | Precisão | Recall | F1 | AP | ROC-AUC | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (prior), limiar 0,5 | 0,0% | 0,0% | 0,0% | 20,3% | 0,500 | 20,3% | 20,0% |
| Logística baseline (`main`), limiar 0,5 | 29,1% | 1,3% | 2,6% | 29,4% | 0,644 | 33,7% | 33,2% |
| Logística ajustada (C = 0,01), limiar 0,5 | 28,4% | 1,2% | 2,3% | 29,5% | 0,649 | 33,5% | 33,0% |
| Logística ajustada, limiar 0,194 | 29,8% | 56,5% | 39,0% | 29,5% | 0,649 | 33,5% | 33,0% |
| Árvore (leaf 300), limiar 0,5 | 0,0% | 0,0% | 0,0% | 33,4% | 0,713 | 35,7% | 35,2% |
| Árvore (leaf 300), limiar 0,213 | 30,6% | 73,6% | 43,3% | 33,4% | 0,713 | 35,7% | 35,2% |

Com o limiar ajustado pelo F1, a logística sinaliza 38,5% da agenda e a árvore 48,7%. Os limiares foram escolhidos numa reserva de validação independente, sem usar o teste.

**Atenção.** Estes números vêm da versão original de `reports/sprints-4-5/results.json` do Fernando. A versão atual desse arquivo na `main` não traz mais os blocos de teste, só a seleção. A fonte dos valores acima precisa ser confirmada com o Fernando.

## Bloco 3: efeito do histórico (validação cruzada, AP médio em 5 folds)

| Modelo | AP sem histórico | AP com histórico | Ganho |
|---|---:|---:|---:|
| Dummy | 0,2006 | 0,2006 | 0,0000 |
| Regressão Logística | 0,3081 | 0,3141 | +0,0060 |
| Árvore de Decisão | 0,3456 | 0,3475 | +0,0019 |
| Random Forest | 0,3599 | 0,3656 | +0,0057 |
| XGBoost | 0,3646 | 0,3722 | +0,0076 |

O histórico melhora o AP de forma consistente, mas pouco (de 0,002 a 0,008). Isso é coerente com a cobertura baixa: só 28% dos agendamentos têm alguma consulta anterior na janela de 6 semanas do dataset.

## Bloco 4: estratégias de balanceamento (Matheus, Regressão Logística, 5 folds sobre 87.720 linhas)

Média ± desvio entre os folds, limiar 0,5. Não é teste final.

| Estratégia | AP | ROC-AUC | Precisão | Recall | F1 | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sem balanceamento | 0,307 ± 0,005 | 0,658 ± 0,004 | 36,6% | 2,1% | 3,9% | 35,0% | 34,9% |
| `class_weight` | 0,307 ± 0,005 | 0,665 ± 0,004 | 31,3% | 55,7% | 40,1% | 35,0% | 34,9% |
| Subamostragem | 0,306 ± 0,005 | 0,664 ± 0,004 | 31,1% | 56,0% | 40,0% | 34,6% | 34,5% |
| SMOTENC | 0,303 ± 0,005 | 0,654 ± 0,005 | 30,6% | 55,7% | 39,5% | 35,0% | 34,9% |

Matrizes de confusão (soma dos folds, 87.720 linhas e 17.594 faltas): sem balanceamento `[[69.497, 629], [17.232, 362]]`; `class_weight` `[[48.608, 21.518], [7.788, 9.806]]`; subamostragem `[[48.331, 21.795], [7.742, 9.852]]`; SMOTENC `[[47.872, 22.254], [7.796, 9.798]]`.

## Bloco 5: `s4_cv.csv` (Fernando, 5 folds, sem histórico)

| Modelo | AP | ROC-AUC | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,304 ± 0,003 | 0,659 ± 0,003 | 35,9% | 2,1% | 3,9% |
| Árvore de Decisão | 0,229 ± 0,003 | 0,559 ± 0,004 | 30,1% | 27,3% | 28,6% |

A árvore deste arquivo (AP 0,229) é muito pior que a árvore ajustada (AP 0,34). Provavelmente é a árvore com hiperparâmetros padrão, que sobreajusta. É inferência pelos números, a confirmar com o Fernando.

## Bloco 6: validação temporal (22.908 agendamentos, 2.702 faltas)

Só existe para Dummy e Logística baseline (`main`), sem histórico.

| Modelo | Precisão | Recall | F1 | AP | ROC-AUC | Precisão@20% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 0,0% | 0,0% | 0,0% | 11,8% | 0,500 | 11,8% | 20,0% |
| Regressão Logística | 0,0% | 0,0% | 0,0% | 16,5% | 0,585 | 17,6% | 29,8% |

A temporal tem outra prevalência (11,8% contra 20,3%). **Não compare seus valores com os dos outros blocos.**

## O que os resultados mostram

- **O XGBoost com histórico é o melhor modelo, mas por margem pequena.** No teste reservado, AP 0,362 e ROC-AUC 0,730; os 20% de maior risco capturam 36,6% das faltas (1,83x o acaso). O Random Forest fica praticamente empatado (36,4%). Todos os candidatos ficam entre 1,70x e 1,83x o acaso; a distância entre a logística e o XGBoost é de cerca de 2,7 pontos percentuais de recall@20%, numa única partição e sem intervalo de confiança.
- **A complexidade do modelo rende pouco.** Entre a logística (33,9%) e o XGBoost (36,6%) a diferença é de cerca de 119 faltas capturadas a mais no teste, ou 5 por 1.000 agendamentos. Com e = 20%, isso equivale a cerca de 1 vaga recuperada a mais por 1.000 agendamentos (ver `docs/impacto_negocio.md`).
- **O histórico ajuda pouco, e para poucos.** O ganho de AP é de 0,002 a 0,008, e só 28% das linhas têm histórico.
- **Balancear muda o ponto de operação, não o ranking.** `class_weight`, subamostragem e SMOTENC elevam o recall em 0,5 de cerca de 2% para cerca de 56%, mas AP (0,303 a 0,307) e recall@20% (cerca de 35%) praticamente não mudam. O ajuste de limiar produz o mesmo efeito sem reamostrar.
- **Limiar padrão não serve para decisão.** Logística e árvore quase não preveem a classe "faltou" em 0,5. Só o ranking (top 20%) ou um limiar ajustado são utilizáveis.
- **Quais variáveis pesam (SHAP do modelo final).** Em ordem: `neighbourhood` (1,63), `waiting_days` (0,82), `age` (0,61), `weekday` (0,31) e `previous_appointments` (0,09). O bairro lidera em parte, provavelmente, porque agrega muitas categorias one-hot (é uma leitura minha, não testada aqui). O grupo fez uma análise de sensibilidade ao bairro (`sensibilidade_bairro.csv`), que não foi revisada neste documento.
- **Limitação de calibração.** Os escores do XGBoost e do Random Forest usam `scale_pos_weight` e `class_weight`, e não são probabilidades calibradas.

## Pendências

- Fernando: confirmar a configuração da árvore de `s4_cv.csv` e onde estão hoje os números de teste da logística e da árvore ajustadas, já que o `results.json` da `main` não os traz mais.
- Alexandre: confirmar que o mesmo limiar de 0,657 é aplicado a todos os candidatos em `run_final`; se for, a coluna de F1 só vale para o XGBoost.
- Decisão do grupo sobre `same_day` (não testado em nenhum modelo).
- Validação temporal do modelo final e dos demais candidatos (hoje só Dummy e Logística).
- Avaliação das estratégias de balanceamento no teste reservado (hoje só em validação cruzada).
- Registrar no relatório que o teste reservado já foi usado em relatórios anteriores (baselines e ajustes da S4/S5), além da avaliação única dos candidatos finais.
- Mesclar a branch `s4-s5/modelo-final` na `main`, para que os arquivos do Bloco 1 existam no repositório.
