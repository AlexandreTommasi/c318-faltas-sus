# Agenda sem desperdício: previsão de faltas em consultas do SUS

Projeto da disciplina **C318 – Fundamentos de Machine Learning** (Inatel, Eng. de Software e Eng. de Computação), prof. Ricardo Augusto. Entregável da NP2.

> Status: versão 5 (final), de 24/nov/2026. Integra os resultados das sprints 2 a 5 (limpeza, análise exploratória, balanceamento, seleção e avaliação de modelos, modelo final) e a análise de impacto de negócio. Os números do dataset vêm de `src/data.py` e do dicionário de dados (`docs/dicionario_dados.md`, issue #3).

## Resumo dos resultados

- **Problema:** prever, no momento do agendamento, quais pacientes têm maior risco de faltar, para priorizar confirmação ativa ou reagendamento.
- **Dados:** 109.889 agendamentos após a limpeza, 62.298 pacientes, 20,10% de faltas. Consultas de 29/04 a 08/06/2016, uma única cidade (Vitória/ES).
- **Modelo final:** XGBoost com histórico de faltas do paciente. No teste reservado por paciente (22.169 agendamentos, 4.497 faltas): AP 0,362 e ROC-AUC 0,730. Contatar os 20% de maior risco captura **36,6% das faltas**, 1,8 vez o que um contato ao acaso captura, com precisão de 37,1%.
- **Impacto estimado (hipóteses, não medido):** a cada 1.000 agendamentos, contatar 200 pacientes recuperaria cerca de **14,8 vagas** com efeito do contato de 20% (7,4 a 22,3 com 10% a 30%), contra 8,1 do contato ao acaso. O contato compensa se o efeito for maior que cerca de 2% (câmbio de sensibilidade de R$ 5 por US$).
- **Limites:** ganho moderado, diferença pequena entre os melhores modelos, uma única partição de teste, dados de 2016 e de uma cidade, e impacto que depende de hipóteses sobre o efeito do contato, o valor da vaga e o custo por contato.

## 1. Equipe

| Integrante | Frente principal | Entregas |
| :--- | :--- | :--- |
| Alexandre Tommasi Alves | Coordenação e engenharia | Repositório e pipeline; features de histórico (#9); Random Forest, XGBoost, modelo final e SHAP (#13, #17) |
| Antonio Augusto D'Assumpção | Negócio e documentação | README e contexto (#2); métrica de negócio (#6); perguntas de negócio (#10); tabela comparativa (#14); impacto de negócio (#18); relatório final (#22) |
| Matheus Vieira Honorio de Souza | Dados e análise exploratória | Dicionário de dados (#3); limpeza (#7); EDA (#11); balanceamento (#15); curvas ROC e precisão-recall (#19) |
| Fernando Puebla Stein | Modelagem e avaliação | Baselines e seleção inicial (#8, #12); logística e árvore com validação cruzada (#16); tuning e limiar de decisão (#20) |

## 2. Contexto

Faltas sem aviso em consultas agendadas (absenteísmo) são um problema recorrente no SUS: a vaga fica ociosa, o recurso é desperdiçado e quem aguarda na fila espera mais. As taxas relatadas variam bastante conforme o local, o nível de atenção e a especialidade. Os números abaixo são exemplos de contextos específicos e **não devem ser lidos como uma taxa única para o SUS**.

**Taxas de absenteísmo relatadas**

- **Mundo e América do Sul:** média de 23% no mundo e de 27,8% na América do Sul, segundo a revisão sistemática de Dantas et al. (2018), reportada em Beltrame et al. (2019) (Referências 9 e 10).
- **Brasil, faixa de 22% a 30%:** citada em um trabalho de conclusão de curso (Referência 2), que a atribui a Santos (2008) (Referência 11). A dissertação de Santos não foi consultada diretamente; trate a faixa como estimativa de fonte secundária.
- **Espírito Santo:** 38,6% das consultas e 32,1% dos exames especializados regulados na Região de Saúde Metropolitana do ES, em 2014–2016, a partir do sistema de regulação estadual (Referência 9). O escopo é a atenção especializada regulada, não necessariamente o mesmo das consultas do dataset.
- **Vitória (ES):** Salazar et al. (2022) relatam faltas de cerca de 30% das consultas em centros de saúde em 2014–2015, com custo médio de aproximadamente US$ 37 por consulta e perda estimada em cerca de US$ 17,5 milhões para o município (Referência 4). São números **históricos e estimados**, atribuídos pelo artigo a Furtado et al. (2016), fonte que não foi localizada nas buscas feitas.
- **Mauá (SP), exemplo local recente:** entre 30/jan e 21/fev/2026, 178 faltas em 862 ultrassonografias agendadas (20,65%) e 92 faltas em 923 tomografias (9,97%), segundo reportagem do Diário do Grande ABC (Referência 3). São exames, e não consultas, de um único município, em fonte jornalística.

**Intervenções e uso de modelos de previsão**

- Em uma unidade de São Paulo, a gestão das agendas por contato telefônico prévio, com confirmação de presença e liberação das vagas não confirmadas, foi acompanhada por uma queda da taxa de absenteísmo de 21% para 19% (2 pontos percentuais), comparando abr–set/2015 com abr–set/2016, sem grupo de controle (Referência 5). O resultado sugere que a confirmação ativa pode ajudar, mas não permite atribuir a queda apenas ao contato telefônico.
- A revisão de Salazar et al. (2022) aponta que só um dos 24 estudos analisados validou o modelo em ambiente real. Nele, lembrar os 25% de pacientes de maior risco segundo o modelo reduziu as faltas de 19,3% para 15,9% (3,4 pontos percentuais), em exames de ressonância em Singapura (Referência 7). Isso ilustra o uso pretendido aqui, **priorizar quem contatar**, mas o contexto é diferente do SUS.

## 3. Objetivos

### Objetivo de negócio

Reduzir consultas perdidas e liberar vagas, apoiando ações de **confirmação ativa** ou **reagendamento** direcionadas aos pacientes de maior risco.

### Objetivo de ciência de dados

Prever a falta (sim/não) de um paciente a uma consulta **no momento do agendamento**, estimando a probabilidade de falta para ranquear os agendamentos por risco.

## 4. Enquadramento do problema

- **Tipo:** aprendizagem supervisionada, classificação binária (falta / comparecimento).
- **Variável alvo:** `No-show` (no dataset, "Yes" = o paciente **faltou**, "No" = compareceu). Na limpeza é codificada como `no_show` (1 = faltou).
- **Desbalanceamento:** 22.091 faltas e 87.798 comparecimentos entre 109.889 agendamentos após a limpeza (20,10%). No CSV bruto são 22.319 faltas em 110.527 agendamentos (20,2%), valor coerente com os 20,19% reportados na literatura (Referência 7). Acurácia sozinha não é uma boa métrica: um modelo que sempre prevê "compareceu" acertaria cerca de 80%.
- **Uso esperado do modelo:** produzir um score de risco, e não apenas um rótulo, para que a unidade escolha quantos pacientes contatar conforme sua capacidade operacional.

## 5. Perguntas de negócio

**Perfil de quem falta**

1. Qual é a taxa geral de faltas e como ela varia por faixa etária, sexo e bairro?
2. O perfil socioeconômico (beneficiário do Bolsa Família) e as condições de saúde mudam a chance de falta?

**Comportamento do agendamento**
3. O tempo entre o agendamento e a consulta (dias de espera) influencia a falta?
4. O dia da semana da consulta importa?
5. Pacientes que já faltaram antes têm maior chance de faltar de novo?
6. Receber SMS está associado a menos faltas? A análise deve controlar o tempo de espera: no dicionário de dados, nenhum agendamento com espera de até 2 dias recebeu SMS, então o envio se relaciona com a antecedência do agendamento. Apenas associação, sem concluir causalidade.

**Valor para a unidade**
7. Se o modelo indicar os 20% de agendamentos com maior risco, que fração das faltas reais ele captura?
8. Quantas consultas poderiam ser recuperadas com confirmação ativa, dada uma capacidade de contato limitada?
9. Em que ponto o custo de contatar um paciente deixa de compensar o custo de uma vaga perdida?

### Respostas

As perguntas 1 a 6 são associações descritivas (`docs/respostas_perguntas_negocio.md`); nenhuma prova causa. As perguntas 7 a 9 usam o modelo final e as hipóteses da seção 7 (resultados nas seções 9 e 10).

| Pergunta | Resposta |
| :--- | :--- |
| 1. Perfil | A idade importa bastante e o sexo não. Adolescentes de 13 a 17 anos têm a maior taxa (26,6%) e quem tem 60 anos ou mais a menor (15,2%). Há diferença entre bairros, que deve ser lida com cuidado nos de poucos agendamentos |
| 2. Bolsa Família e saúde | Beneficiários do Bolsa Família faltam 4 pontos percentuais a mais. Hipertensos e diabéticos faltam menos, o que pode refletir a idade (hipótese não testada). Alcoolismo não muda a taxa |
| 3. Tempo de espera | É a relação mais forte: 4,6% de faltas nos agendamentos do mesmo dia (35% do total) contra cerca de 33% com mais de 30 dias de espera |
| 4. Dia da semana | Diferença pequena entre os dias úteis (19,3% a 21,2%), com sexta-feira no topo |
| 5. Histórico | Quem tem histórico e nunca faltou falta bem menos (13,5%). Quem já faltou (21,4%) fica no nível de quem não tem histórico (21,7%). Só 28% dos agendamentos têm alguma consulta anterior na janela de 6 semanas |
| 6. SMS | Dentro de cada faixa de espera a partir de 3 dias, quem recebeu SMS faltou menos (de 2,8 a 7,1 pontos percentuais). É associação, não efeito do SMS, e a variável fica fora do modelo |
| 7. Faltas capturadas no top 20% | O XGBoost final captura 36,6% das faltas reais no teste (cerca de 1.645 de 4.497), contra 20% ao acaso |
| 8. Consultas recuperadas | Cerca de 14,8 vagas por 1.000 agendamentos com e = 20% (7,4 a 22,3 com e de 10% a 30%), condicionado às hipóteses da seção 7 |
| 9. Ponto de equilíbrio | O contato compensa se o efeito do contato (e) passar de cerca de 2,2% (câmbio de R$ 5 por US$). |

## 6. Dataset

- **Fonte:** *Medical Appointment No Shows*, publicado no Kaggle por Joni Hoppen (Aquarela Advanced Analytics) (Referência 8). Arquivo utilizado: `KaggleV2-May-2016.csv`. A página do Kaggle foi consultada em 28/10/2026 e indica a versão 5, atualizada em 20/08/2017. A data em que o arquivo local foi baixado não foi registrada.
- **Como obter:** baixar pelo site do Kaggle ou pela CLI (`kaggle datasets download -d joniarroba/noshowappointments -p data/raw/ --unzip`) e deixar o CSV em `data/raw/` (diretório ignorado no git).

**Visão geral do CSV bruto** (dicionário de dados, issue #3)

- **Linhas (agendamentos):** 110.527
- **Colunas:** 14
- **Pacientes distintos:** 62.299 (24.379 com mais de um agendamento)
- **Valores nulos:** 0
- **Período de agendamento (`ScheduledDay`):** 10/11/2015 a 08/06/2016
- **Período das consultas (`AppointmentDay`):** 29/04/2016 a 08/06/2016
- **Taxa de falta:** 20,2% (22.319 `Yes`, 88.208 `No`)

**Limpeza** (`src/data.py`, issue #7)

| Métrica | CSV bruto | Após a limpeza |
| :--- | ---: | ---: |
| Linhas | 110.527 | 109.889 |
| Pacientes distintos | 62.299 | 62.298 |
| Faltas | 22.319 | 22.091 |
| Taxa geral de faltas | 20,19% | 20,10% |
| Taxa de faltas no mesmo dia | 4,65% | 4,63% |

**Problemas identificados e como foram tratados**

- **Dias de espera negativos:** 5 linhas em que a consulta precede o agendamento, removidas.
- **Idades:** 1 linha com idade -1, removida. Sete linhas acima de 100 anos, entre elas 5 com 115 (quatro do mesmo paciente), foram mantidas, e a idade é suspeita.
- **Duplicatas escondidas:** 618 linhas repetem tudo menos `AppointmentID`; em 14 desses grupos o alvo diverge. A limpeza compara as linhas ignorando `appointment_id` e `no_show`, mantém a primeira ocorrência de cada grupo e remove 632 linhas (os 618 do dicionário mais os 14 grupos com alvo divergente). Os 14 grupos são uma fonte possível de ruído no alvo.
- **Consultas no mesmo dia:** cerca de 35% dos agendamentos são para o próprio dia, com taxa de falta bem menor (4,6% contra 28,5% quando há espera).
- **Identificadores:** `AppointmentID` cresce com a data de agendamento e não é usado como variável.
- **Sem horário da consulta:** `AppointmentDay` tem sempre hora `00:00:00`, então só dá para usar o dia da semana.
- **Documentação do dataset:** a Referência 4 aponta que falta documentação sobre a coleta e o pré-processamento dos dados.

## 7. Métrica de negócio

A métrica separa o que o modelo **mede** no teste do que depende de **hipóteses** sobre o efeito do contato.

- **Corte de contato (K):** foco nos **20% de agendamentos de maior risco** segundo o modelo. A capacidade real de contato de uma unidade não é conhecida; 20% é um corte de referência.
- **Faltas capturadas:** número de faltas reais dentro dos K agendamentos contatados (resultado observado no teste, sem hipóteses).
- **Consultas recuperadas (estimativa):** faltas capturadas × efeito incremental do contato (*e*), a fração das faltas dos contatados que deixa de ocorrer por causa do contato.
- **Custo líquido (estimativa):** (consultas recuperadas × valor da vaga) − (número de contatos K × custo por contato).
- **Linha de base:** contatar o mesmo número K de pacientes escolhidos ao acaso. O ganho do modelo é a diferença para esse resultado.
- **Ponto de equilíbrio (pergunta 9):** contatar um paciente compensa quando p × *e* × valor da vaga > custo por contato, em que p é a probabilidade de falta do paciente.


| Parâmetro | Valor adotado | Base |
| :--- | :--- | :--- |
| K | 20% dos agendamentos de maior risco | Decisão do grupo (28/out) |
| *e* (efeito incremental) | Cenários de hipótese: 10%, 20% e 30% | Sem valor direto na literatura. Como ordem de grandeza, a Ref. 5 (21% para 19%) equivale a uma redução relativa de cerca de 10% das faltas, e a Ref. 7 (19,3% para 15,9%) a cerca de 18%, em contextos diferentes e sem grupo de controle na Ref. 5 |
| Valor da vaga | Cerca de US$ 37 por consulta | Ref. 4, dado histórico de Vitória (2014–2015) de fonte primária não localizada |
| Custo por contato | R$ 1,50 por contato, hipótese de trabalho | Estimativa de referência para mensagem automatizada, não é custo observado |
| Câmbio | R$ 4,00, 5,00 e 6,00 por US$ | Sensibilidade: o valor da vaga está em dólares e o custo por contato em reais. Não é cotação |

Os cenários de *e*, o valor da vaga, o custo por contato e o câmbio são **hipóteses de trabalho**, não resultados do projeto. Qualquer número de consultas recuperadas deve ser apresentado como **estimativa condicionada a essas hipóteses**. Os resultados estão na seção 10.

## 8. Metodologia e cuidados

- **Variável `SMS_received`:** fora do modelo, por ocorrer depois do agendamento. Usada apenas em análises exploratórias, controlando o tempo de espera.
- **Variáveis cadastrais** (`Age`, `Gender`, `Neighbourhood`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`): premissa de trabalho de que estão disponíveis no momento do agendamento.
- **Histórico de faltas:** calculado só com consultas do mesmo paciente de data estritamente anterior à data do agendamento. Consultas marcadas para o mesmo dia não entram, porque o resultado delas ainda não é conhecido (`add_patient_history`, `src/features.py`). Entram `previous_appointments`, `previous_no_shows`, `previous_no_show_rate` e `has_history`.
- **Variáveis do modelo final:** as 10 cadastrais e de agendamento (`age`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `waiting_days`, `gender`, `neighbourhood`, `weekday`) mais as 4 de histórico. Ficam de fora `SMS_received`, `same_day`, identificadores e datas brutas.
- **Partição por paciente:** `GroupShuffleSplit` por `patient_id`, `test_size` 0,2 e `random_state` 42: 87.720 agendamentos de desenvolvimento e 22.169 de teste (4.497 faltas, 20,29%), sem pacientes em comum.
- **Validação cruzada:** `StratifiedGroupKFold` com 5 folds sobre o desenvolvimento, sem usar o teste. Os hiperparâmetros, o modelo e o limiar foram escolhidos aí e congelados em `reports/modelo-final/selecao_congelada.json` antes de abrir o teste.
- **Desbalanceamento:** `class_weight` balanceado no Random Forest e `scale_pos_weight` no XGBoost, sem reamostragem. O limiar de decisão vem das previsões *out-of-fold* do desenvolvimento, pela política dos 20% de maior risco.
- **Métricas de avaliação:** AP (PR-AUC), ROC-AUC, matriz de confusão, e precisão e recall no corte de 20%. Acurácia isolada é insuficiente.
- **Validação temporal:** só foi feita para os baselines (Dummy e Regressão Logística), porque a janela de dados é curta (cerca de 6 semanas de consultas). Tem outra prevalência (11,8%) e não é comparável com a validação por paciente.
- **Comparabilidade:** a literatura reporta resultados heterogêneos com este dataset (AUC entre 0,68 e 0,88 e acurácia entre 78% e 95%, Referência 7), em condições diferentes. Os resultados do projeto não devem ser comparados com esses valores sem verificar se a validação é equivalente.
- **Fatores relevantes na literatura:** idade, histórico de faltas e intervalo entre agendamento e consulta (Referência 7). No projeto, o tempo de espera e a idade aparecem entre as variáveis mais importantes, e o histórico contribui pouco (seção 9).

## 9. Resultados do modelo final

Detalhes em `docs/modelo_final.md`, `docs/comparacao_modelos.md` e `reports/modelo-final/`.

### 9.1 Escolha do modelo (validação cruzada, com histórico)

AP médio e desvio-padrão em 5 folds sobre os 87.720 agendamentos de desenvolvimento:

| Candidato | AP médio | Desvio | ROC-AUC | Precisão@20% | Recall@20% |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | **0,3722** | 0,0058 | 0,7375 | 38,4% | 38,3% |
| Random Forest | 0,3656 | 0,0076 | 0,7334 | 37,8% | 37,7% |
| Árvore de Decisão | 0,3475 | 0,0029 | 0,7255 | 36,8% | 36,7% |
| Regressão Logística | 0,3141 | 0,0049 | 0,6679 | 35,4% | 35,3% |
| Dummy (prior) | 0,2006 | 0,0000 | 0,5000 | 20,1% | 20,0% |

O XGBoost (300 árvores, `max_depth` 6, `learning_rate` 0,05, `scale_pos_weight` = negativos ÷ positivos) teve o maior AP e o segundo menor desvio. A margem sobre o Random Forest é estreita.

**Efeito do histórico.** Incluir as 4 variáveis de histórico melhora o AP de todos os modelos, mas pouco: de 0,002 (árvore) a 0,008 (XGBoost, de 0,3646 para 0,3722). O histórico só existe para 28% dos agendamentos.

### 9.2 Teste reservado (22.169 agendamentos, avaliação única)

Todos os candidatos foram retreinados no desenvolvimento completo, com histórico, e avaliados uma única vez no teste.

| Modelo | AP | ROC-AUC | Precisão@20% | Recall@20% | Ganho sobre o acaso |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Dummy (prior) | 0,203 | 0,500 | 20,3% | 20,0% | 1,00x |
| Regressão Logística | 0,304 | 0,657 | 34,4% | 33,9% | 1,70x |
| Árvore de Decisão | 0,340 | 0,720 | 35,4% | 34,9% | 1,75x |
| Random Forest | 0,359 | 0,728 | 36,9% | 36,4% | 1,82x |
| **XGBoost (final)** | **0,362** | **0,730** | **37,1%** | **36,6%** | **1,83x** |

Com o limiar congelado de 0,657 (os 20% de maior risco no desenvolvimento), o XGBoost sinaliza 4.423 agendamentos no teste (19,95%). A matriz de confusão (linhas = real, colunas = previsto) é `[[14.891, 2.781], [2.855, 1.642]]`: 1.642 das 4.497 faltas capturadas, precisão de 37,1%, recall de 36,5%. Os resultados do teste confirmam a ordem da validação cruzada, mas as diferenças entre os modelos de topo são pequenas e vêm de uma única partição, sem intervalo de confiança.

### 9.3 Quais variáveis pesam

SHAP do XGBoost final (valor absoluto médio por agendamento, em log-odds), com as colunas de uma mesma variável somadas: `neighbourhood` 1,63, `waiting_days` 0,82, `age` 0,61, `weekday` 0,31 e `previous_appointments` 0,09. O bairro lidera, e uma análise de sensibilidade em validação cruzada mostra que removê-lo reduz o AP em 0,014, enquanto agrupar os bairros raros quase não muda (−0,001). Para a explicação ao público, `waiting_days` e `age` são as variáveis mais generalizáveis.

### 9.4 Onde o modelo erra

O modelo ranqueia por risco e contata os 20% do topo, então os falsos negativos se concentram em grupos de menor taxa de falta: agendamentos do mesmo dia (4,7% de faltas), pessoas com 60 anos ou mais (15,2%) e pacientes com histórico (16,8%). É consequência da política de contato seletivo, e não uma falha estrutural. A faixa de espera de 8 a 90 dias é a de operação principal.

### 9.5 Balanceamento e validação temporal

- **Balanceamento** (Regressão Logística, 5 folds): `class_weight`, subamostragem e SMOTENC mudam o ponto de operação, mas não o ranking. O recall com limiar 0,5 sobe de cerca de 2% para cerca de 56%, ao custo de precisão, enquanto o AP fica entre 0,303 e 0,307 e o recall@20% em cerca de 35%.
- **Validação temporal** (só Dummy e Logística): a prevalência é de 11,8% e os valores não são comparáveis com os demais blocos (detalhes em `docs/comparacao_modelos.md`).

## 10. Impacto de negócio

Tradução das métricas do teste em vagas recuperadas e custo de contato, pela métrica da seção 7. Detalhes, outras políticas e todos os cenários em `docs/impacto_negocio.md` e `reports/impacto_negocio.csv`. **Todos os valores abaixo são estimativas condicionadas às hipóteses da seção 7**: o efeito do contato (e) não foi medido neste dataset.

Por 1.000 agendamentos, política dos 20% de maior risco (200 contatos, custo de R$ 300):

| Modelo | Faltas capturadas | Vagas recuperadas (e = 10% / 20% / 30%) | Custo líquido em R$ (e = 10% / 20% / 30%) |
| :--- | ---: | :--- | :--- |
| Contato ao acaso | 40,6 | 4,1 / 8,1 / 12,2 | |
| Regressão Logística | 68,8 | 6,9 / 13,8 / 20,7 | 973 / 2.247 / 3.520 |
| **XGBoost (final)** | **74,2** | **7,4 / 14,8 / 22,3** | **1.073 / 2.445 / 3.818** |

O custo líquido usa o câmbio hipotético de R$ 5,00 por US$ (vaga de cerca de R$ 185). Com o mesmo custo, o XGBoost recupera 6,7 vagas a mais que o contato ao acaso por 1.000 agendamentos (com e = 20%).

- **Ponto de equilíbrio:** o contato compensa quando o efeito do contato passa de 2,7%, 2,2% ou 1,8% (câmbios de R$ 4, 5 e 6). Com e = 10%, o retorno por real gasto fica entre 3,4 e 5,5 vezes.
- **A escolha do modelo pesa pouco no resultado financeiro:** o XGBoost recupera cerca de 1 vaga a mais por 1.000 agendamentos que a logística. Boa parte do valor vem de contatar quem está no topo do ranking, seja qual for o modelo.
- **Custo para o paciente:** cerca de 63% dos contatos vão a quem compareceria (precisão de 37,1%).
- **Contato mais amplo:** o limiar que maximiza o F1 contata 38% a 49% da agenda, mas só foi avaliado para a logística e a árvore, sem o modelo final, e ficou fora desta análise.

## 11. Limitações

**Dados**

- Uma única cidade (Vitória/ES) e o ano de 2016, em janela curta (cerca de 6 semanas de consultas) e sem sazonalidade. Os resultados não representam o SUS como um todo.
- O dataset não traz motivo da falta, distância até a unidade, especialidade nem horário da consulta, e tem pouca documentação sobre a coleta.
- 14 grupos de duplicatas tinham alvo divergente: podem ser agendamentos duplos reais ou erro de sistema, e é uma fonte possível de ruído.
- O histórico de faltas só existe para 28% dos agendamentos, por causa da janela curta. Em produção, seria necessário manter um registro contínuo por paciente.

**Modelo e avaliação**

- Uma única partição de teste e sem intervalo de confiança. A diferença entre o XGBoost e o Random Forest (0,2 ponto percentual de recall@20%) não é conclusiva.
- O teste reservado foi avaliado uma única vez para a escolha final, mas já tinha sido usado em relatórios anteriores (baselines e ajustes de limiar da S4/S5).
- A partição por paciente mede a generalização para **pacientes novos**, não para o mesmo paciente em consultas futuras.
- Os escores do XGBoost e do Random Forest usam `scale_pos_weight` e `class_weight`, e não são probabilidades calibradas. O limiar não deve ser lido como probabilidade real de falta.
- O bairro é a variável de maior peso no SHAP e tem muitas categorias de pouco volume; a interpretação deve destacar `waiting_days` e `age`.
- O modelo final não foi avaliado na validação temporal, e `same_day` não foi testado como variável.
- O ganho sobre o acaso é moderado (1,8 vez) e a literatura reporta resultados heterogêneos com este dataset, em condições diferentes.

**Impacto de negócio**

- O efeito do contato (e), o valor da vaga (US$ 37, fonte secundária e histórica), o custo por contato (R$ 1,50) e o câmbio são hipóteses. Nenhum foi medido.
- O cálculo supõe que a vaga liberada é preenchida (por reagendamento ou fila de espera) e não inclui tempo de equipe, remarcação, sobreagendamento nem o incômodo ao paciente.
- 35% dos agendamentos são do mesmo dia e quase não faltam (4,6%); não foi verificado quantos deles entram entre os 20% contatados.
- A diferença de faltas associada ao SMS é uma associação e não deve ser usada como valor de e.
- A literatura tem pouca validação em ambiente real: só um dos 24 estudos revisados na Referência 7 validou o modelo na prática.

## 12. Próximos passos

- Medir o efeito do contato (e) com um piloto controlado, comparando agendamentos contatados e não contatados.
- Levantar o custo real de contato e o valor da vaga na unidade, e refazer a análise de impacto.
- Validar o modelo final no teste temporal e em dados de outras unidades ou períodos.
- Testar `same_day` e outras variáveis, e avaliar um limiar de contato mais amplo para o modelo final.
- Calibrar os escores, para que o limiar possa ser interpretado como probabilidade.
- Se for para produção, manter o registro contínuo do histórico por paciente e monitorar o desempenho ao longo do tempo.

## 13. Registro de decisões

| Data | Decisão | Responsável |
| :--- | :--- | :--- |
| 23/out | Estrutura do repositório e divisão inicial de pastas | Alexandre |
| 23/out | Primeira leitura do dataset e dicionário de dados | Matheus |
| 23/out | Enquadramento como classificação supervisionada | Grupo |
| 28/out | `SMS_received` isolado do modelo preditivo | Líder e grupo |
| 28/out | Validação principal por `PatientId`, com teste temporal como checagem extra | Fernando e grupo |
| 28/out | Corte de contato fixado em K = 20% | Antonio |
| 28/out | Definição da métrica de negócio, com efeito do contato e custo por contato como hipóteses | Antonio |
| 1/nov | Limpeza: remove 6 linhas inválidas (idade negativa e espera negativa) e 632 duplicatas, ignorando `appointment_id` e `no_show` e mantendo a primeira ocorrência (issue #7) | Matheus e Alexandre |
| 7/nov | Histórico de faltas só com consultas de data estritamente anterior à do agendamento | Alexandre |
| 20/nov | Modelo final: XGBoost com histórico, limiar de 0,657 pela política dos 20%, escolha congelada antes de abrir o teste (#13 e #17) | Alexandre |

O registro completo e os detalhes de cada decisão estão em `docs/decisoes.md`.

## 14. Estrutura do repositório e como reproduzir

```
c318-faltas-sus/
├── data/
│   ├── raw/            # dataset original (ignorado no git)
│   └── processed/      # dados tratados (ignorado no git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   ├── 04_avaliacao.ipynb
│   ├── 05_balanceamento.ipynb
│   ├── 06_perguntas_negocio.ipynb
│   ├── 07_selecao_modelos.ipynb
│   └── 08_modelo_final.ipynb
├── src/
│   ├── data.py             # carregamento e limpeza
│   ├── features.py         # engenharia de atributos e histórico do paciente
│   ├── models.py           # modelos e partições
│   ├── evaluate.py         # métricas
│   ├── analysis.py         # baselines e análise das sprints 2 e 3
│   ├── eda.py              # tabelas da EDA
│   ├── imbalance.py        # estratégias de balanceamento
│   ├── model_selection.py  # validação cruzada, hiperparâmetros e limiar (Fernando)
│   ├── final_model.py      # seleção, congelamento, teste único e SHAP
│   ├── impacto_negocio.py  # vagas recuperadas e custo de contato
│   ├── validation_curves.py, validation_plots.py, sprint_plots.py
├── tests/
├── docs/               # decisões, dicionário, resultados e comparações
├── reports/            # tabelas e figuras geradas
├── requirements.txt
└── README.md
```

**Ambiente.** Python 3.14.5; as versões dos pacotes estão fixadas em `requirements.txt`.

```
git clone https://github.com/AlexandreTommasi/c318-faltas-sus.git
cd c318-faltas-sus
python -m venv .venv
.venv\Scripts\activate          # Linux e macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Dados.** Baixe `KaggleV2-May-2016.csv` (seção 6) e coloque em `data/raw/`. O arquivo não é versionado. A CLI do Kaggle exige conta e token; sem ela, use o download pelo site.

**Passo a passo** (cada comando roda a partir da raiz do repositório, com o CSV em `data/raw/`; os argumentos opcionais `--raw` e `--output` têm valores padrão):

```
python -m unittest discover -s tests -v    # testes do projeto
python -m src.analysis                     # baselines e correlações (reports/sprints-2-3/)
python -m src.eda                          # tabelas da EDA (reports/sprints-3-4/eda/)
python -m src.imbalance                    # balanceamento (reports/sprints-3-4/imbalance/)
python -m src.sprint_plots                 # figuras das sprints 3 e 4 (precisa de eda e imbalance)
python -m src.validation_curves            # curvas e matrizes (reports/sprint-5/evaluation/)
python -m src.validation_plots             # figuras da sprint 5 (precisa de validation_curves)
python -m src.model_selection              # S4/S5: busca de hiperparâmetros e limiar (reports/sprints-4-5/); --jobs N para paralelizar
```

**Modelo final** (`src.final_model` exige `--fase`; as fases rodam nesta ordem e gravam em `reports/modelo-final/`):

```
python -m src.final_model --fase cv                              # validação cruzada dos candidatos, com e sem histórico
python -m src.final_model --fase freeze --model xgboost --history   # congela a escolha antes de abrir o teste
python -m src.final_model --fase final                           # avaliação única no teste reservado
python -m src.final_model --fase shap                            # SHAP agregado por variável original
python -m src.final_model --fase sensibilidade                   # sensibilidade ao bairro (validação cruzada)
python -m src.impacto_negocio                                    # impacto de negócio (reports/impacto_negocio.csv)
```

`src.model_selection` e `src.final_model` são independentes: um não lê os arquivos do outro. O `src.impacto_negocio` precisa de `teste_final.csv` e `teste_final.json`, gerados pela fase `final`. A fase `freeze` deve receber o modelo e o `--history` correspondentes à escolha desejada. Rodar de novo a fase `final` abre o teste outra vez, então só faça isso para reproduzir o resultado já publicado.

Para o detalhe de cada etapa, veja `docs/sprints-4-5.md` e `docs/modelo_final.md`. Os notebooks acompanham a história do projeto na ordem numérica.

## 15. Referências

1. Fritz, A. G. et al. *Vagas ociosas, cuidados interrompidos: o absenteísmo como barreira ao acesso em saúde*. Resumo simples, Congresso Internacional em Saúde, Unijuí (edição e ano não confirmados). https://www.publicacoeseventos.unijui.edu.br/index.php/conintsau/article/download/27042/25594/63517
2. Carvalho, J. I. S. *O absenteísmo dos usuários da UBS Parque Vitória, município de Franco da Rocha-SP*. TCC (Especialização em Saúde da Família), UNIFESP/UNA-SUS, 2021. https://ares.unasus.gov.br/acervo/html/ARES/27009/1/jean_israel_de_souza_carvalho.pdf
3. Diário do Grande ABC. *Carreta com especialistas realizou mais de 1.400 mil exames em Mauá*. 25/02/2026. https://www.dgabc.com.br/Noticia/4287157/carreta-com-especialistas-realizou-mais-de-1-4-mil-exames-em-maua
4. Salazar, L. H. A. et al. *Application of Machine Learning Techniques to Predict a Patient's No-Show in the Healthcare Sector*. Future Internet, 2022, 14(1), 3. https://doi.org/10.3390/fi14010003 (cita Furtado et al., 2016, para os dados de Vitória)
5. *Estratégia para enfrentamento do absenteísmo em uma unidade de saúde do município de São Paulo*. 6º Simpósio Científico Digital CEJAM (autores e ano não identificados no documento). https://simposio.cejam.org.br/documentos/trabalhos/6o-simposio-cientifico-cejam-estrategia-para-enfrentamento-do-absenteismo-em-uma-unidade-de-saude-do-municipio-de-sao-paulo-10.pdf
6. Carreras-García, D. et al. *Patient No-Show Prediction: A Systematic Literature Review*. Entropy, 2020, 22(6), 675. https://doi.org/10.3390/e22060675
7. Salazar, L. H. A. et al. *No-Show in Medical Appointments with Machine Learning Techniques: A Systematic Literature Review*. Information, 2022, 13(11), 507. https://doi.org/10.3390/info13110507
8. Hoppen, J. (Aquarela Advanced Analytics). *Medical Appointment No Shows*. Kaggle, versão 5, atualizada em 20/08/2017. https://www.kaggle.com/datasets/joniarroba/noshowappointments
9. Beltrame, S. M. et al. *Absenteísmo de usuários como fator de desperdício: desafio para sustentabilidade em sistema universal de saúde*. Saúde em Debate, 2019, 43(123), 1015–1030. https://doi.org/10.1590/0103-1104201912303
10. Dantas, L. F. et al. *No-shows in appointment scheduling – a systematic literature review*. Health Policy, 2018, 122(4), 412–421. https://doi.org/10.1016/j.healthpol.2018.02.002 (texto completo não consultado; os valores de 23% e 27,8% foram conferidos na citação feita por Beltrame et al., 2019)
11. Santos, J. S. *Absenteísmo dos usuários em consultas e procedimentos especializados agendados no SUS: um estudo em um município baiano*. Dissertação (Mestrado), Universidade Federal da Bahia, 2008. (não consultada diretamente; citada na Referência 2)

## 16. Entregas por sprint

Os documentos de cada integrante, com os detalhes e as ressalvas de cada etapa.

### 16.1 Sprints 2 e 3 (Fernando)

Tarefas do Fernando (#8 e #12): baseline ingênuo, regressão logística e
seleção inicial de atributos: [resultados, perguntas de negócio e limitações](docs/sprints-2-3.md).

```bash
python -m unittest discover -s tests -v
python -m src.analysis
```

O segundo comando lê o CSV original e gera métricas, tabelas e gráficos agregados
em `reports/sprints-2-3/`. Os notebooks 01 e 03 permitem acompanhar a análise.

### 16.2 Sprints 3 e 4 (Matheus)

Tarefas do Matheus:

- **#11 (S3):** EDA sobre a partição de treino — tabelas agregadas congeladas em
  `reports/sprints-3-4/eda/` (`summary.json` + um CSV por variável) e seção nova no
  fim de `notebooks/01_eda.ipynb`.
- **#15 (S4):** comparação de métodos de balanceamento — `unbalanced`,
  `class_weight`, `undersampling` e `smote` (SMOTENC) em 5 folds
  `StratifiedGroupKFold` no treino, em `reports/sprints-3-4/imbalance/`
  (`comparison.csv`, `fold_metrics.json`, `metadata.json`) e em
  `notebooks/05_balanceamento.ipynb`.

Figuras geradas a partir das tabelas congeladas ficam em
`reports/sprints-3-4/figures/`. Reprodução (`.venv` opcional; basta ter as
dependências de `requirements.txt`):

```bash
python -m src.eda           # tabelas agregadas do treino
python -m src.imbalance     # comparação das 4 estratégias
python -m src.sprint_plots  # figuras em reports/sprints-3-4/figures/
python -m unittest discover -s tests -v
```

Leitura dos resultados (validação cruzada, teste reservado, sem conclusões de teste
final): no treino, a taxa de falta é 20,06%; a espera e a faixa etária mostram
associações descritivas fortes, e o SMS só se associa a menos faltas dentro de
faixas de espera a partir de 3 dias — confundido, não é evidência de eficácia e
segue fora dos preditores. No balanceamento, a AP média fica entre 0,303 e 0,307
nas quatro estratégias — diferença pequena e sem evidência de superioridade —; o
recall a 0,5 sobe para ~56% com `class_weight`/reamostragem à custa de precisão,
enquanto o recall@20% permanece ~35%. Scores balanceados não são probabilidades
calibradas. Detalhes e ressalvas nos notebooks.

### 16.3 Sprint 5 (Matheus)

Tarefa #19 do Matheus: curvas ROC, precisão-recall e matrizes de confusão das
quatro estratégias de balanceamento, na validação cruzada do treino (teste
reservado). Detalhes em [`docs/sprint-5.md`](docs/sprint-5.md) e no notebook
[`notebooks/04_avaliacao.ipynb`](notebooks/04_avaliacao.ipynb).

```bash
python -m src.validation_curves   # reports/sprint-5/evaluation/
python -m src.validation_plots    # reports/sprint-5/figures/
python -m unittest discover -s tests -v
```

### 16.4 Sprints 4 e 5 (Fernando)

Tarefas #16 e #20: validação cruzada estratificada por paciente, busca de hiperparâmetros e limiar de decisão da Regressão Logística e da Árvore de Decisão. Detalhes em [`docs/sprints-4-5.md`](docs/sprints-4-5.md).

### 16.5 Modelo final (Alexandre)

Tarefas #13 e #17: Random Forest e XGBoost, ablação do histórico, escolha do modelo final, avaliação única no teste, SHAP e análise de erros. Detalhes em [`docs/modelo_final.md`](docs/modelo_final.md). O notebook-resumo do projeto ([`notebooks/08_modelo_final.ipynb`](notebooks/08_modelo_final.ipynb)) reúne todas as etapas, do contexto ao impacto de negócio, lendo apenas artefatos congelados de `reports/`.

### 16.6 Comparação de modelos e impacto de negócio (Antonio)

Tarefas #14 e #18: [`docs/comparacao_modelos.md`](docs/comparacao_modelos.md) compara todos os modelos por protocolo de validação, e [`docs/impacto_negocio.md`](docs/impacto_negocio.md) traduz as métricas em vagas recuperadas e custo de contato.
