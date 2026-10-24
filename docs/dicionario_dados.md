# Dicionário de Dados

Dataset: **Medical Appointment No Shows** (Kaggle), arquivo `KaggleV2-May-2016.csv`.
Primeira leitura feita em 2026-10-23 (issue #3). Os números abaixo são reproduzidos na seção 1 de `notebooks/01_eda.ipynb`.

## Visão geral

| Item | Valor |
|---|---|
| Linhas (agendamentos) | 110.527 |
| Colunas | 14 |
| Pacientes distintos | 62.299 (24.379 com mais de um agendamento; máximo de 88) |
| Valores nulos | 0 |
| Linhas duplicadas | 0 (mas 618 duplicatas se ignorarmos `AppointmentID`, ver abaixo) |
| Período de agendamento (`ScheduledDay`) | 10/11/2015 a 08/06/2016 |
| Período das consultas (`AppointmentDay`) | 29/04/2016 a 08/06/2016 (27 dias úteis) |
| Taxa de falta | 20,2% (22.319 `Yes` / 88.208 `No`) |

## Colunas

| Coluna | Tipo no CSV | Tipo correto | Significado | Valores | Problemas encontrados |
|---|---|---|---|---|---|
| `PatientId` | float64 | string / int64 | Identificador do paciente | 62.299 distintos | Lido como float (perde a "cara" de ID). 5 valores têm casas decimais (ex.: `93779.52927`), provavelmente erro de exportação. Não é feature: serve só para agrupar o histórico do paciente. |
| `AppointmentID` | int64 | int64 | Identificador do agendamento | Único por linha | Cresce com a data de agendamento (correlação 0,998 com `ScheduledDay`), então carrega informação temporal. Não usar como feature. |
| `Gender` | string | category | Sexo do paciente | `F` 71.840 (65%), `M` 38.687 (35%) | Sem problemas. Taxa de falta quase igual (F 20,3%, M 20,0%). |
| `ScheduledDay` | string | datetime (UTC) | Data e hora em que a consulta foi marcada | 06h–21h | Vem como texto ISO com `Z`. Em 5 linhas é **posterior** à consulta (ver "dias de espera"). |
| `AppointmentDay` | string | date | Data da consulta | 29/04 a 08/06/2016 | A hora é sempre `00:00:00`: só temos o dia, não o horário da consulta. Só 39 consultas em sábado e nenhuma em domingo. |
| `Age` | int64 | int64 | Idade em anos | -1 a 115 (mediana 37) | 1 idade negativa (-1). 3.539 com idade 0 (plausível: bebês). 7 linhas acima de 100 anos: 102 (2) e 115 (5); quatro das de 115 são do mesmo paciente, suspeito. |
| `Neighbourhood` | string | category | Bairro da unidade de saúde (Vitória-ES) | 81 bairros | Alta cardinalidade. Bairros com pouquíssimos registros (`PARQUE INDUSTRIAL` 1, `ILHAS OCEÂNICAS DE TRINDADE` 2, `AEROPORTO` 8) dão taxas de falta de 0% ou 100% que não significam nada. Tem acentos (cuidado com encoding). |
| `Scholarship` | int64 | bool | Paciente inscrito no Bolsa Família | 0: 99.666, 1: 10.861 (9,8%) | Nome em inglês engana ("bolsa de estudos"). Falta um pouco mais quem tem o benefício (23,7% vs 19,8%). |
| `Hipertension` | int64 | bool | Paciente hipertenso | 1: 21.801 (19,7%) | Erro de grafia no nome (`Hypertension`). |
| `Diabetes` | int64 | bool | Paciente diabético | 1: 7.943 (7,2%) | Sem problemas. |
| `Alcoholism` | int64 | bool | Paciente com alcoolismo | 1: 3.360 (3,0%) | Sem problemas. Taxa de falta idêntica à de quem não tem (20,1% vs 20,2%). |
| `Handcap` | int64 | int (0–4) | Número de deficiências do paciente | 0: 108.286, 1: 2.042, 2: 183, 3: 13, 4: 3 | Erro de grafia (`Handicap`). **Não é binária**, apesar de parecer: é uma contagem. Os níveis 3 e 4 têm poucas linhas. |
| `SMS_received` | int64 | bool | Paciente recebeu um ou mais SMS de lembrete | 1: 35.482 (32,1%) | Ver "armadilha do SMS" abaixo. |
| `No-show` | string | bool (alvo) | **Variável-alvo.** `Yes` = **faltou**, `No` = compareceu | `No` 88.208, `Yes` 22.319 | O nome confunde: "No-show = No" quer dizer que o paciente **foi**. O hífen no nome atrapalha no código (`df['No-show']`). Desbalanceada: ~20% de faltas. |

## Problemas que afetam as próximas etapas

1. **Dias de espera negativos.** Calculando `AppointmentDay - ScheduledDay` (por data), 5 linhas dão valor negativo (4 com -1 e 1 com -6). São erros de registro e devem ser removidas na limpeza (issue #7).
2. **Consultas no mesmo dia.** 38.563 agendamentos (35%) são para o próprio dia, e quase ninguém falta neles (4,6% contra 28,5% quando há espera). O tempo de espera tende a ser a variável mais forte do modelo.
3. **Armadilha do SMS.** Quem recebeu SMS falta *mais* (27,6% vs 16,7%), o que parece dizer que o SMS piora a situação. Mas nenhum agendamento com espera de até 2 dias recebeu SMS: o lembrete só é enviado para consultas marcadas com antecedência, que já faltam mais. Comparando só consultas com 3 ou mais dias de espera, quem recebeu SMS falta menos (27,6% vs 32,7%). A pergunta de negócio 2 do README precisa controlar o tempo de espera.
4. **Duplicatas escondidas.** Não há linhas idênticas, mas 618 linhas repetem tudo menos o `AppointmentID` (mesmo paciente, mesmo horário de marcação, mesma consulta). Em 14 desses grupos o alvo diverge (uma linha `Yes`, outra `No`). Precisamos decidir se são agendamentos duplos de verdade ou erro de sistema antes de remover.
5. **Idades inválidas.** Remover `Age < 0`; avaliar as idades acima de 100 (o paciente com 115 anos aparece 4 vezes).
6. **Janela curta.** As consultas cobrem só ~6 semanas (abril a junho de 2016). Sazonalidade (mês, férias) não pode ser estudada.
7. **Vazamento de dados.** `AppointmentID` e qualquer histórico do paciente calculado com consultas futuras vazam informação. As features de histórico (issue #9) só podem usar consultas anteriores à data da linha.
8. **Sem horário da consulta.** Como `AppointmentDay` não tem hora, não dá para usar o turno da consulta como feature, só o dia da semana.

## Sugestão de renomeação

Para facilitar o código, sugerimos padronizar os nomes na limpeza:

| Original | Novo |
|---|---|
| `PatientId` | `patient_id` |
| `AppointmentID` | `appointment_id` |
| `Gender` | `gender` |
| `ScheduledDay` | `scheduled_day` |
| `AppointmentDay` | `appointment_day` |
| `Age` | `age` |
| `Neighbourhood` | `neighbourhood` |
| `Scholarship` | `bolsa_familia` |
| `Hipertension` | `hypertension` |
| `Diabetes` | `diabetes` |
| `Alcoholism` | `alcoholism` |
| `Handcap` | `handicap` |
| `SMS_received` | `sms_received` |
| `No-show` | `no_show` (1 = faltou) |
