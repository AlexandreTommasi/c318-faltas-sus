# Impacto de negócio dos modelos (issue #18)

Tradução das métricas técnicas em consultas recuperadas e custo de contato, conforme a métrica da seção 7 do README. Os números vêm de `src/impacto_negocio.py`, que lê `reports/modelo-final/teste_final.csv` e `teste_final.json` e grava `reports/impacto_negocio.csv`.

**Escopo.** Os cinco candidatos avaliados uma única vez no teste reservado por paciente (22.169 agendamentos, 4.497 faltas, prevalência 20,29%), todos com histórico de faltas. O modelo final é o **XGBoost**. Dummy entra só como referência de acaso.

## Hipóteses (não medidas neste dataset)

| Parâmetro | Valor | Observação |
|---|---|---|
| Efeito do contato (*e*) | 10%, 20% e 30% das faltas contatadas viram comparecimento ou reaproveitamento da vaga | Hipótese. O dataset não mede o efeito de um contato ativo. A diferença associada ao SMS (P6) **não** serve como valor de *e*. |
| Valor da vaga | US$ 37 | Fonte secundária (Ref. 4 do README); a fonte primária (Furtado et al., 2016) não foi localizada, por isso o valor é tratado como hipótese. |
| Custo por contato | R$ 1,50 | Hipótese do grupo. |
| Câmbio | R$ 4,00, 5,00 e 6,00 por US$ | **Hipótese de sensibilidade**, não cotação real. Necessária porque vaga e contato estão em moedas diferentes. |

Fórmulas: faltas capturadas = faltas entre os contatados; consultas recuperadas = capturadas × *e*; custo = contatos × custo por contato; custo líquido = recuperadas × valor da vaga − custo. Baseline: mesmo número de contatos escolhidos ao acaso (capturadas = contatos × 20,29%). Ponto de equilíbrio: o contato compensa quando precisão × *e* × valor da vaga > custo por contato.

## Política de contato

- **Top 20%** (K da seção 7): contata os 20% de maior risco, 4.434 agendamentos no teste. É a política comparada entre todos os modelos.
- **Limiar congelado do XGBoost** (0,657): escolhido nas previsões *out-of-fold* do desenvolvimento, antes de abrir o teste. No teste sinaliza 4.423 agendamentos (19,95%), praticamente o mesmo conjunto do top 20%.

A política de limiar ajustado pelo F1 (contato de 38% a 49% da agenda), usada na versão anterior deste documento, saiu: os números de teste dela vinham de um arquivo que não está mais na `main` e não existem para o modelo final. Se o grupo quiser analisar um contato mais amplo, é preciso calcular o limiar de F1 do XGBoost e avaliá-lo no teste.

## Resultado por 1.000 agendamentos (teste reservado)

| Modelo | Contatos | Faltas capturadas | Precisão | Recall | Capturadas ao acaso |
|---|---:|---:|---:|---:|---:|
| Logística | 200 | 68,8 | 34,4% | 33,9% | 40,6 |
| Árvore | 200 | 70,9 | 35,4% | 34,9% | 40,6 |
| Random Forest | 200 | 73,9 | 36,9% | 36,4% | 40,6 |
| **XGBoost (final)** | **200** | **74,2** | **37,1%** | **36,6%** | **40,6** |
| XGBoost, limiar congelado | 199 | 74,1 | 37,1% | 36,5% | 40,5 |

### Vagas recuperadas por 1.000 agendamentos (e = 10% / 20% / 30%)

| Modelo | Com o modelo | Contato ao acaso | Ganho sobre o acaso |
|---|---|---|---|
| Logística | 6,9 / 13,8 / 20,7 | 4,1 / 8,1 / 12,2 | 2,8 / 5,7 / 8,5 |
| Árvore | 7,1 / 14,2 / 21,3 | 4,1 / 8,1 / 12,2 | 3,0 / 6,1 / 9,1 |
| Random Forest | 7,4 / 14,8 / 22,2 | 4,1 / 8,1 / 12,2 | 3,3 / 6,7 / 10,0 |
| **XGBoost (final)** | **7,4 / 14,8 / 22,3** | 4,1 / 8,1 / 12,2 | **3,4 / 6,7 / 10,1** |

Custo de contato: R$ 300 por 1.000 agendamentos (200 contatos × R$ 1,50), igual para todos os modelos.

## Custo líquido (câmbio hipotético de R$ 5,00 por US$, valor da vaga ≈ R$ 185)

Em R$ por 1.000 agendamentos, para e = 10% / 20% / 30%:

| Modelo | Custo líquido | Ganho líquido sobre o contato aleatório (e = 20%) |
|---|---|---:|
| Logística | 973 / 2.247 / 3.520 | 1.046 |
| Árvore | 1.011 / 2.323 / 3.634 | 1.122 |
| Random Forest | 1.067 / 2.434 / 3.801 | 1.233 |
| **XGBoost (final)** | **1.073 / 2.445 / 3.818** | **1.244** |

No teste inteiro (22.169 agendamentos), o XGBoost com e = 20% dá R$ 54.214 de custo líquido, contra R$ 49.811 da logística. Os valores para câmbio de R$ 4,00 e R$ 6,00 estão em `reports/impacto_negocio.csv`.

## Ponto de equilíbrio

Menor *e* para o contato compensar:

| Modelo | Câmbio 4,0 | Câmbio 5,0 | Câmbio 6,0 |
|---|---:|---:|---:|
| Logística | 2,9% | 2,4% | 2,0% |
| Árvore | 2,9% | 2,3% | 1,9% |
| Random Forest | 2,7% | 2,2% | 1,8% |
| **XGBoost (final)** | **2,7%** | **2,2%** | **1,8%** |

Mesmo com *e* de 10%, o cenário mais pessimista, o retorno por real gasto fica entre 3,4 e 5,5 vezes. A conclusão depende do custo de R$ 1,50 e do valor da vaga, ambos hipóteses. Se o custo real incluir tempo de equipe, remarcação ou risco de sobreagendamento, o ponto de equilíbrio sobe e a conta deve ser refeita.

## O que isso significa

- **O contato ativo compensa em todos os cenários testados**, para qualquer modelo, enquanto *e* ficar acima do ponto de equilíbrio, que varia entre 1,8% (XGBoost, câmbio R$ 6) e 2,9% (logística, câmbio R$ 4).
- **A escolha do modelo pesa pouco no resultado.** O XGBoost recupera cerca de 1 vaga a mais por 1.000 agendamentos que a logística com e = 20% (14,8 contra 13,8), cerca de R$ 200 a mais de custo líquido por 1.000 agendamentos no câmbio de 5,0. Boa parte do valor vem de contatar quem está no topo do ranking, seja qual for o modelo. A diferença entre os modelos vem de uma única partição de teste, sem intervalo de confiança.
- **Contra o acaso, o ganho é grande:** com o mesmo custo, o XGBoost captura 74 faltas por 1.000 agendamentos contra 41 do contato aleatório (1,83 vez), o que dá 6,7 vagas a mais com e = 20%.
- **Cerca de 63% dos contatos vão a quem compareceria** (precisão de 37,1%). Esse custo está incluído na conta, mas pesa na experiência do paciente.

## Limitações

- *e* é hipótese. As vagas recuperadas escalam de forma linear com ele, então a ordem entre modelos não depende do valor escolhido, mas o valor absoluto sim.
- Uma consulta recuperada só tem valor se a vaga for preenchida (por reagendamento ou fila de espera). O cálculo assume que isso acontece.
- O cálculo não inclui custos de infraestrutura, tempo de equipe, remarcação, sobreagendamento nem o incômodo ao paciente.
- 35% dos agendamentos são do mesmo dia e quase não faltam (4,6%). Não foi verificado quantos deles entram no top 20%.
- O XGBoost usa histórico de faltas, mas só 28% dos agendamentos têm alguma consulta anterior na janela de 6 semanas. Em outro período ou unidade, o ganho do histórico pode ser diferente.
- Os escores do XGBoost usam `scale_pos_weight` e não são probabilidades calibradas; o corte por ranking (top 20%) não depende da calibração, mas um limiar fixo sim.
- Os valores são do teste reservado (cerca de 20% da base, 6 semanas de 2016, uma cidade). Não são previsão para outra unidade ou período.
- O teste reservado foi usado em relatórios anteriores (baselines e ajustes da S4/S5), além da avaliação única dos candidatos finais.
- O custo líquido usa câmbio hipotético; não é conversão real.
