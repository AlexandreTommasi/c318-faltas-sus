# Agenda sem desperdício: previsão de faltas em consultas do SUS

Projeto da disciplina **C318 – Fundamentos de Machine Learning** (Inatel, Eng. de Software e Eng. de Computação), prof. Ricardo Augusto. Entregável da NP2.

> Status: versão 4.2, de 29/out/2026. Base: versão 4.1. Integra o README inicial do repositório, a auditoria das fontes e o dicionário de dados do Matheus (`docs/dicionario_dados.md`, issue #3). Os números do dataset vêm desse dicionário e serão reproduzidos no notebook `01_eda.ipynb`.

## 1. Equipe

Integrante
Frente principal

Alexandre Tommasi Alves
Coordenação e engenharia

Antonio Augusto D'Assumpção
Negócio e documentação

Matheus Vieira Honorio de Souza
Dados e análise exploratória

Fernando Puebla Stein
Modelagem e avaliação

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
- **Variável alvo:** `No-show` (no dataset, "Yes" = o paciente **faltou**, "No" = compareceu). Na limpeza será codificada como `no_show` (1 = faltou).
- **Desbalanceamento:** 22.319 faltas e 88.208 comparecimentos entre 110.527 agendamentos, ou seja, cerca de 20,2% de faltas, valor coerente com os 20,19% reportados na literatura (Referência 7). Acurácia sozinha não é uma boa métrica: um modelo que sempre prevê "compareceu" acertaria cerca de 80%.
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

## 6. Dataset

- **Fonte:** *Medical Appointment No Shows*, publicado no Kaggle por Joni Hoppen (Aquarela Advanced Analytics) (Referência 8). Arquivo utilizado: `KaggleV2-May-2016.csv`. A página do Kaggle foi consultada em 28/10/2026 e indica a versão 5, atualizada em 20/08/2017. A data em que o arquivo local foi baixado não foi registrada.
- **Como obter:** baixar pelo site do Kaggle ou pela CLI (`kaggle datasets download -d joniarroba/noshowappointments -p data/raw/ --unzip`) e deixar o CSV em `data/raw/` (diretório ignorado no git).

**Visão geral** (dicionário de dados, issue #3)

- **Linhas (agendamentos):** 110.527
- **Colunas:** 14
- **Pacientes distintos:** 62.299 (24.379 com mais de um agendamento)
- **Valores nulos:** 0
- **Período de agendamento (`ScheduledDay`):** 10/11/2015 a 08/06/2016
- **Período das consultas (`AppointmentDay`):** 29/04/2016 a 08/06/2016
- **Taxa de falta:** 20,2% (22.319 `Yes`, 88.208 `No`)

**Problemas identificados na primeira leitura** (a tratar na limpeza, issue #7)

- **Dias de espera negativos:** 5 linhas em que a consulta precede o agendamento, candidatas à remoção.
- **Idades inválidas ou suspeitas:** 1 linha com idade -1 (inválida) e 7 acima de 100 anos, entre elas 5 com 115, quatro do mesmo paciente, o que é suspeito.
- **Duplicatas escondidas:** 618 linhas repetem tudo menos `AppointmentID`; em 14 desses grupos o alvo diverge. É preciso decidir se são agendamentos duplos reais ou erro de sistema.
- **Consultas no mesmo dia:** cerca de 35% dos agendamentos são para o próprio dia, com taxa de falta bem menor (4,6% contra 28,5% quando há espera).
- **Identificadores:** `AppointmentID` cresce com a data de agendamento e não deve ser usado como variável.
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

Parâmetro
Valor adotado
Base

K
20% dos agendamentos de maior risco
Decisão do grupo (28/out)

*e* (efeito incremental)
Cenários de hipótese: 10%, 20% e 30%
Sem valor direto na literatura. Como ordem de grandeza, a Ref. 5 (21% para 19%) equivale a uma redução relativa de cerca de 10% das faltas, e a Ref. 7 (19,3% para 15,9%) a cerca de 18%, em contextos diferentes e sem grupo de controle na Ref. 5

Valor da vaga
Cerca de US$ 37 por consulta
Ref. 4, dado histórico de Vitória (2014–2015) de fonte primária não localizada

Custo por contato
R$ 1,50 por contato, hipótese de trabalho
Estimativa de referência para mensagem automatizada, não é custo observado. Avaliar também valores menores e maiores como sensibilidade

O valor da vaga está em dólares e o custo por contato em reais: use a mesma moeda ao simular o ponto de equilíbrio. Os cenários de *e*, o valor da vaga e o custo por contato são **hipóteses de trabalho**, não resultados do projeto. Qualquer número de consultas recuperadas deve ser apresentado como **estimativa condicionada a essas hipóteses**.

## 8. Cuidados metodológicos e limitações

- **Variável `SMS_received`:** fora do modelo, por ocorrer depois do agendamento. Usada apenas em análises exploratórias, controlando o tempo de espera.
- **Variáveis cadastrais** (`Age`, `Gender`, `Neighbourhood`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`): premissa de trabalho de que estão disponíveis no momento do agendamento.
- **Histórico de faltas:** só pode usar consultas anteriores e já concluídas na data do agendamento, para não vazar informação do futuro.
- **Estratégia de validação:** divisão principal por `PatientId`, para medir a generalização para pacientes novos e evitar o mesmo paciente em treino e teste, complementada por um teste temporal, limitado pela janela curta de dados (cerca de 6 semanas de consultas).
- **Métricas de avaliação:** PR-AUC, ROC-AUC, matriz de confusão, precisão e recall da classe de faltas, e precisão e recall no corte de 20%. Acurácia isolada é insuficiente.
- **Generalização:** dados de uma única cidade e de 2016, em janela curta e sem sazonalidade. Os resultados não representam o SUS como um todo.
- **Variáveis ausentes:** o dataset não traz motivo da falta, distância até a unidade, especialidade nem horário da consulta.
- **Comparabilidade:** a literatura reporta resultados heterogêneos com este dataset (AUC entre 0,68 e 0,88 e acurácia entre 78% e 95%, Referência 7), em condições diferentes. Os resultados do projeto não devem ser comparados com esses valores sem verificar se a validação é equivalente.
- **Fatores relevantes na literatura:** idade, histórico de faltas e intervalo entre agendamento e consulta (Referência 7) são candidatos a variáveis, e não resultado esperado.

## 9. Registro de decisões

Data
Decisão
Responsável

23/out
Estrutura do repositório e divisão inicial de pastas
Alexandre

23/out
Primeira leitura do dataset e dicionário de dados
Matheus

23/out
Enquadramento como classificação supervisionada
Grupo

28/out
`SMS_received` isolado do modelo preditivo
Líder e grupo

28/out
Validação principal por `PatientId`, com teste temporal como checagem extra
Fernando e grupo

28/out
Corte de contato fixado em K = 20%
Antonio

28/out
Definição da métrica de negócio, com efeito do contato e custo por contato como hipóteses
Antonio

em aberto
Padronização de nomes de colunas, tratamento de valores inválidos e duplicatas (issue #7)
Grupo

## 10. Estrutura do repositório e como rodar

```
c318-faltas-sus/
├── data/
│   ├── raw/            # dataset original (ignorado no git)
│   └── processed/      # dados tratados (ignorado no git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
├── src/
│   ├── data.py         # carregamento e limpeza
│   ├── features.py     # engenharia de atributos
│   ├── models.py       # treinamento e tuning
│   └── evaluate.py     # métricas e relatórios
├── docs/
│   ├── decisoes.md         # log de decisões do projeto
│   └── dicionario_dados.md # dicionário de dados (issue #3)
├── reports/figures/    # gráficos gerados
├── requirements.txt
└── README.md
```

```
git clone https://github.com/AlexandreTommasi/c318-faltas-sus.git
cd c318-faltas-sus
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## 11. Referências

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

## 12. Entrega das sprints 2 e 3

Tarefas do Fernando (#8 e #12): baseline ingênuo, regressão logística e
seleção inicial de atributos: [resultados, perguntas de negócio e limitações](docs/sprints-2-3.md).

```bash
python -m unittest discover -s tests -v
python -m src.analysis
```

O segundo comando lê o CSV original e gera métricas, tabelas e gráficos agregados
em `reports/sprints-2-3/`. Os notebooks 01 e 03 permitem acompanhar a análise.

## 13. Entrega das sprints 3 e 4

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
