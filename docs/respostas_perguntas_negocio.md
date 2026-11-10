# Respostas às perguntas de negócio 1 a 6 (issue #10)

Análise descritiva do dataset *Medical Appointment No Shows* após a limpeza de `src/data.py`, feita em `notebooks/05_perguntas_negocio.ipynb`. Tabelas completas em `reports/tabelas_perguntas_negocio.csv` e gráficos em `reports/figures/p1_*.png` a `p6_*.png`.

**Base analisada:** 109.889 agendamentos, com taxa geral de faltas de **20,10%**. Partindo do CSV bruto (110.527 linhas), foram removidas 6 linhas inválidas (1 idade negativa e 5 esperas negativas) e 632 duplicatas.

**Como ler:** todos os resultados são associações descritivas. Nenhum deles prova causa, e nenhum controla outras variáveis, exceto onde está dito. As perguntas 7 a 9 dependem do modelo e ficam para as etapas seguintes.

## Resumo

| Pergunta | Resposta curta |
|---|---|
| 1. Perfil (idade, sexo, bairro) | A idade importa bastante, o sexo não. Adolescentes e jovens faltam mais, idosos menos. Há diferença entre bairros |
| 2. Bolsa Família e saúde | Bolsa Família tem mais faltas. Hipertensão e diabetes têm menos, e alcoolismo não muda |
| 3. Tempo de espera | É a relação mais forte: 4,6% de faltas no mesmo dia contra cerca de 33% com mais de 30 dias de espera |
| 4. Dia da semana | Diferença pequena, com sexta-feira no topo |
| 5. Histórico de faltas | Efeito pequeno neste dataset |
| 6. SMS | Dentro de cada faixa de espera, quem recebeu SMS faltou menos, e a diferença cresce com a espera |

## P1. Taxa geral e perfil de quem falta

| Faixa etária | n | Taxa de falta |
|---|---:|---:|
| 0-12 | 20.952 | 20,41% |
| 13-17 | 6.303 | 26,64% |
| 18-29 | 16.621 | 24,60% |
| 30-44 | 21.852 | 21,68% |
| 45-59 | 23.085 | 17,77% |
| 60+ | 21.076 | 15,21% |

| Sexo | n | Taxa de falta |
|---|---:|---:|
| F | 71.409 | 20,23% |
| M | 38.480 | 19,86% |

- A taxa cai de forma contínua depois dos 18 anos. A faixa de 13 a 17 tem a maior taxa (26,64%) e a de 60+ tem a menor (15,21%), uma diferença de cerca de 11,4 pontos percentuais.
- O sexo não separa os grupos: a diferença é de 0,37 ponto percentual.
- **Bairros** (apenas os 54 com n ≥ 500; 27 ficaram de fora). Entre os 10 maiores, as taxas vão de 22,46% (Parque Moscoso) a 28,89% (Santos Dumont). Entre os 10 menores, vão de 15,76% (Santa Martha) a 17,71% (Redenção). A tabela completa dos extremos está no CSV.
- **Cuidado com bairros de n pequeno:** Santa Clara tem n = 501 e taxa de 26,35%, com margem de erro de cerca de ±3,9 pontos percentuais (IC 95%, cálculo aproximado). Os bairros com n grande, como Santos Dumont (n = 1.274, ±2,5 p.p.) e Jardim da Penha (n = 3.873, ±1,2 p.p.), são mais confiáveis. Tratar o ranking como descritivo.

## P2. Bolsa Família e condições de saúde

| Condição | Sem | Com | Diferença (com − sem) |
|---|---:|---:|---:|
| Bolsa Família | 19,71% (n = 99.089) | 23,68% (n = 10.800) | **+3,96 p.p.** |
| Hipertensão | 20,82% (n = 88.214) | 17,20% (n = 21.675) | **−3,62 p.p.** |
| Diabetes | 20,27% (n = 101.998) | 18,01% (n = 7.891) | **−2,26 p.p.** |
| Alcoolismo | 20,11% (n = 106.546) | 19,95% (n = 3.343) | −0,16 p.p. |

| Deficiências (contagem) | n | Taxa de falta |
|---|---:|---:|
| 0 | 107.672 | 20,15% |
| 1 | 2.021 | 17,62% |
| 2+ | 196 | 20,92% |

- Quem é beneficiário do Bolsa Família falta cerca de 4 pontos percentuais a mais.
- Hipertensos e diabéticos faltam menos. **Hipótese não testada:** isso pode refletir a idade, já que são grupos mais velhos e a P1 mostra que idosos faltam menos. Para afirmar, seria preciso comparar dentro de cada faixa etária.
- Alcoolismo não muda a taxa.
- Deficiências: o grupo 2+ tem só 196 registros, então a taxa de 20,92% é instável e não deve ser interpretada.

## P3. Tempo de espera

| Espera (dias) | n | Taxa de falta |
|---|---:|---:|
| 0 (mesmo dia) | 38.492 | 4,63% |
| 1-2 | 11.859 | 22,72% |
| 3-7 | 20.118 | 24,93% |
| 8-14 | 11.920 | 30,44% |
| 15-30 | 17.220 | 32,46% |
| 31+ | 10.280 | 32,89% |

- É a relação mais forte entre as analisadas. A taxa sobe de 4,63% no mesmo dia para 22,72% com 1 a 2 dias de espera, e continua subindo até cerca de 33% depois de 15 dias.
- **35,0% dos agendamentos são para o próprio dia** (38.492 de 109.889), e eles quase não faltam. Uma explicação possível é que sejam marcações de atendimento imediato, mas o dataset não permite confirmar isso.
- Para a ação de negócio, o lembrete ou a confirmação ativa tende a ser mais útil nos agendamentos com antecedência, e quase nada nos do mesmo dia.

## P4. Dia da semana da consulta

| Dia | n | Taxa de falta |
|---|---:|---:|
| Segunda | 22.570 | 20,52% |
| Terça | 25.506 | 19,99% |
| Quarta | 25.738 | 19,62% |
| Quinta | 17.172 | 19,28% |
| Sexta | 18.864 | 21,15% |
| Sábado | 39 | 23,08% |

- A variação entre os dias úteis é pequena: de 19,28% (quinta) a 21,15% (sexta), cerca de 1,9 ponto percentual.
- O sábado tem só 39 consultas e deve ser ignorado na interpretação.
- O dia da semana pode ser variável candidata, mas não é um fator forte.

## P5. Histórico de faltas do paciente

O histórico considera apenas consultas anteriores do mesmo paciente (`AppointmentDay` anterior ao `ScheduledDay` da linha atual), para não usar informação do futuro.

| Grupo | n | Taxa de falta |
|---|---:|---:|
| Sem histórico | 48.520 | 19,84% |
| Nunca faltou antes | 47.990 | 20,19% |
| Já faltou antes | 13.379 | 20,75% |

- Quem já faltou antes tem taxa 0,56 a 0,91 ponto percentual maior que os outros grupos. É uma diferença pequena.
- **Limitação importante:** as consultas do dataset cobrem só cerca de 6 semanas (29/04 a 08/06/2016), e 44% das linhas (48.520) não têm nenhuma consulta anterior dentro dessa janela. Com tão pouco histórico observável, o efeito do passado do paciente fica subestimado. Isso não contradiz a literatura, que aponta o histórico como fator relevante em bases mais longas (Referência 7), mas significa que aqui ele deve ser testado no modelo antes de ser dado como importante.

## P6. SMS, controlando o tempo de espera

Nenhum agendamento com 0 dias ou 1 a 2 dias de espera recebeu SMS, então a comparação só é possível a partir de 3 dias.

| Espera (dias) | Sem SMS | Com SMS | Diferença (com − sem) |
|---|---:|---:|---:|
| 3-7 | 26,51% (n = 8.570) | 23,76% (n = 11.548) | −2,75 p.p. |
| 8-14 | 33,77% (n = 4.916) | 28,11% (n = 7.004) | −5,66 p.p. |
| 15-30 | 36,63% (n = 6.700) | 29,80% (n = 10.520) | −6,83 p.p. |
| 31+ | 37,34% (n = 3.870) | 30,20% (n = 6.410) | −7,14 p.p. |

- Sem controlar a espera, o SMS parece piorar as faltas (27,57% com SMS contra 16,7% sem SMS, no dicionário de dados), mas isso acontece porque todos os SMS estão em agendamentos com 3 dias ou mais, que já faltam mais.
- Controlando a espera, **quem recebeu SMS faltou menos em todas as faixas**, e a diferença cresce de 2,75 para 7,14 pontos percentuais. No conjunto de 3 dias ou mais, a taxa é 27,57% com SMS contra cerca de 32,6% sem.
- **É associação, não efeito do SMS.** Os dados não mostram quem recebeu SMS e por quê (por exemplo, só quem tinha telefone cadastrado). Pacientes que recebem SMS podem ser diferentes dos que não recebem. Mesmo dentro de cada faixa, a faixa de 31+ dias é ampla.
- Para a métrica de negócio (seção 7 do README), essa diferença não substitui o efeito do contato ativo (*e*), que não foi medido neste dataset. Ela não deve ser usada como valor de *e* sem essa ressalva.

## Limpeza e comparação com o dicionário de dados

| Métrica | CSV bruto | Após limpeza |
|---|---:|---:|
| Linhas | 110.527 | 109.889 |
| Pacientes distintos | 62.299 | 62.298 |
| Taxa geral de faltas | 20,19% | 20,10% |
| Taxa de faltas no mesmo dia | 4,65% | 4,63% |
| Taxa com SMS | 27,57% | 27,57% |

Os números do bruto batem com o dicionário. O dicionário cita 618 duplicatas e a função atual remove 632. A diferença de 14 coincide com os 14 grupos em que o alvo diverge entre as linhas repetidas, o que sugere que a função ignora o alvo ao comparar. Falta o Matheus confirmar e registrar qual linha é mantida quando o alvo diverge.

## Implicações para as próximas etapas

- **Variáveis mais promissoras para o modelo:** dias de espera (forte), idade (forte), Bolsa Família (moderada) e bairro (moderada, com cuidado nos de n pequeno).
- **Variáveis fracas nesta análise:** sexo, alcoolismo, dia da semana e, nesta janela de dados, histórico de faltas.
- **Sobre os agendamentos do mesmo dia:** como quase não faltam, é esperado que o corte dos 20% de maior risco seja formado sobretudo por agendamentos com espera longa. Isso ainda precisa ser verificado com o modelo.
