# Agenda sem desperdício: previsão de faltas em consultas do SUS

Projeto da disciplina **C318 – Fundamentos de Machine Learning** (Inatel, Eng. de Software e Eng. da Computação), prof. Ricardo Augusto. Entregável da NP2.



## 1. Equipe

| Integrante | Frente principal |
|---|---|
| Alexandre Tommasi Alves | Coordenação e engenharia |
| Antonio Augusto D'Assumpção | Negócio e documentação |
| Matheus Vieira Honorio de Souza | Dados e análise exploratória |
| Fernando Puebla Stein | Modelagem e avaliação |

## 2. Contexto

Faltas sem aviso em consultas agendadas (absenteísmo) são um problema recorrente no SUS: a vaga fica ociosa, o recurso é desperdiçado e quem aguarda na fila espera mais. As taxas relatadas variam bastante conforme o local, o nível de atenção e a especialidade. Os números abaixo são exemplos de contextos específicos e **não devem ser lidos como uma taxa única para o SUS**.

**Taxas de absenteísmo relatadas**
- **Mundo e América do Sul:** média de 23% no mundo e de 27,8% na América do Sul, segundo a revisão sistemática de Dantas et al. (2018), reportada em Beltrame et al. (2019) (Referências 8 e 9). O resumo da Referência 1 repete esses valores sem citar a fonte.
- **Espírito Santo:** 38,6% das consultas e 32,1% dos exames especializados regulados na Região de Saúde Metropolitana do ES, em 2014–2016, a partir do sistema de regulação estadual (Referência 8). O escopo é a atenção especializada regulada, não necessariamente o mesmo das consultas do dataset.
- **Vitória (ES):** Salazar et al. (2022) relatam faltas de cerca de 30% das consultas em centros de saúde em 2014–2015, com custo médio de aproximadamente US$ 37 por consulta e perda estimada em cerca de US$ 17,5 milhões para o município (Referência 3). Esses números são históricos e estimados; não serão usados como validação do modelo.
- **Mauá (SP), exemplo local recente:** entre 30/jan e 21/fev/2026, 178 faltas em 862 ultrassonografias agendadas (20,65%) e 92 faltas em 923 tomografias (9,97%), segundo reportagem do Diário do Grande ABC (Referência 2). São exames, e não consultas, de um único município, em fonte jornalística.

**Intervenções e uso de modelos de previsão**
- Em uma unidade de São Paulo, a gestão das agendas por contato telefônico prévio, com confirmação de presença e liberação das vagas não confirmadas, junto de ações de conscientização da equipe, foi acompanhada por uma queda da taxa de 21% para 19% (2 pontos percentuais, 560 consultas), comparando abr–set/2015 com abr–set/2016, sem grupo de controle (Referência 4). O resultado sugere que a confirmação ativa pode ajudar, mas não permite atribuir a queda apenas ao contato telefônico.
- A revisão de Salazar et al. (2022) aponta que só um dos 24 estudos analisados validou o modelo em ambiente real. Nele, lembrar os 25% de pacientes de maior risco segundo o modelo reduziu as faltas de 19,3% para 15,9% (3,4 pontos percentuais), em exames de ressonância em Singapura (Referência 6, relatando Chong et al., 2020). Isso ilustra o uso pretendido aqui, **priorizar quem contatar**, mas o contexto é diferente do SUS.

## 3. Objetivos

### Objetivo de negócio

Reduzir consultas perdidas e liberar vagas, apoiando ações de **confirmação ativa** ou **reagendamento** direcionadas aos pacientes de maior risco.

### Objetivo de ciência de dados

Prever a falta (sim/não) de um paciente a uma consulta **no momento do agendamento**, estimando a probabilidade de falta para ranquear os agendamentos por risco.

## 4. Enquadramento do problema
- **Tipo:** aprendizagem supervisionada, classificação binária (falta / comparecimento).
- **Variável alvo:** `No-show` (no dataset, "Yes" = o paciente **faltou**, "No" = compareceu; atenção, o nome confunde). Na limpeza será codificada como `no_show` (1 = faltou).
- **Desbalanceamento:** 22.319 faltas e 88.208 comparecimentos entre 110.527 agendamentos, ou seja, cerca de 20,2% de faltas, valor coerente com os 20,19% reportados na literatura (Referência 6, Tabela 5). Acurácia sozinha não é uma boa métrica nesse cenário: um modelo que sempre prevê "compareceu" acertaria cerca de 80%.
- **Uso esperado do modelo:** produzir um score de risco, e não apenas um rótulo, para que a unidade escolha quantos pacientes contatar conforme sua capacidade.

## 5. Perguntas de negócio

**Perfil de quem falta**
1. Qual é a taxa geral de faltas e como ela varia por faixa etária, sexo e bairro?
2. O perfil socioeconômico (beneficiário do Bolsa Família) e as condições de saúde (hipertensão, diabetes, alcoolismo, deficiência) mudam a chance de falta?

**Comportamento do agendamento**
3. O tempo entre o agendamento e a consulta (dias de espera) influencia a falta?
4. O dia da semana da consulta importa?
5. Pacientes que já faltaram antes têm maior chance de faltar de novo?
6. Receber SMS está associado a menos faltas? A comparação precisa **controlar o tempo de espera**. Esta pergunta será tratada como análise observacional, sem concluir causalidade.

**Valor para a unidade**
7. Se o modelo indicar os 20% de agendamentos com maior risco, que fração das faltas reais ele captura?
8. Quantas consultas poderiam ser recuperadas com confirmação ativa, dada uma capacidade de contato limitada?
9. Em que ponto o custo de contatar um paciente deixa de compensar o custo de uma vaga perdida? Esta pergunta depende de custos reais da unidade, que não estão disponíveis no dataset.

## 6. Dataset
- **Fonte:** *Medical Appointment No Shows*, publicado no Kaggle por Joni Hoppen (Aquarela Advanced Analytics) (Referência 7). Página: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>. O arquivo usado é `KaggleV2-May-2016.csv`.
- **Como obter:** baixar pelo site do Kaggle (com login) ou pela CLI (`pip install kaggle`, com credenciais configuradas):
```bash
kaggle datasets download -d joniarroba/noshowappointments -p data/raw/ --unzip
```

O CSV deve ficar em `data/raw/` e **não é versionado** (ver `.gitignore`).

**Visão geral** (valores esperados para conferência no CSV)

| Item | Valor |
|---|---|
| Linhas (agendamentos) | 110.527 |
| Colunas | 14 |
| Pacientes distintos | 62.299 (24.379 com mais de um agendamento; máximo de 88 agendamentos) |
| Valores nulos | 0 |
| Linhas totalmente duplicadas | 0 (mas 618 repetem tudo menos `AppointmentID`) |
| Período de agendamento (`ScheduledDay`) | 10/11/2015 a 08/06/2016 |
| Período das consultas (`AppointmentDay`) | 29/04/2016 a 08/06/2016 (27 dias úteis) |
| Taxa de falta | 20,2% (22.319 `Yes`, 88.208 `No`) |

**Variáveis:** `PatientId`, `AppointmentID`, `Gender`, `ScheduledDay`, `AppointmentDay`, `Age`, `Neighbourhood` (81 bairros), `Scholarship` (inscrição no Bolsa Família, não bolsa de estudos), `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` (contagem de deficiências de 0 a 4, não binária), `SMS_received`, `No-show`.

**Pontos a verificar no CSV**

Os números abaixo vêm de uma auditoria anterior e serão reproduzidos antes da análise final. As decisões de tratamento já estão definidas:

- **Dias de espera negativos:** remover as linhas em que a consulta é anterior ao agendamento, pois são inconsistentes com o cenário de previsão.
- **Idades inválidas ou suspeitas:** remover idade negativa e idade acima de 100; manter idade 0, que pode representar bebês.
- **Duplicatas escondidas:** deduplicar registros com os mesmos atributos e o mesmo alvo; manter e documentar registros com alvo divergente, pois o dataset não permite determinar se são erros ou agendamentos distintos.
- **Consultas no mesmo dia:** calcular a proporção e a taxa de falta por faixa de espera; o tempo de espera será mantido como variável.
- **Identificadores:** excluir `AppointmentID` do modelo; usar `PatientId` somente para ordenar e construir histórico, nunca como preditor direto.
- **Bairros raros:** manter os bairros no dado bruto e agrupar categorias raras apenas durante a modelagem, se necessário para evitar estimativas instáveis.
- **Sem horário da consulta:** usar o dia da semana, mas não o turno, porque o dataset não informa o horário da consulta.
- **Documentação do dataset:** a Referência 4 aponta que falta documentação sobre como os dados foram coletados e pré-processados, além do forte desbalanceamento. A implementação deve padronizar os nomes para `patient_id`, `appointment_id`, `gender`, `scheduled_day`, `appointment_day`, `age`, `neighbourhood`, `bolsa_familia`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `sms_received`, `no_show` (1 = faltou).

## 7. Métrica de negócio

O cenário escolhido é uma unidade que consegue contatar até 20% dos agendamentos. Portanto, a métrica operacional principal será o **recall@20%**: a fração de todas as faltas reais capturada ao contatar os 20% de agendamentos com maior score de risco.
- **Resultado observado no teste (depende só do modelo):** dentre os K agendamentos de maior risco, quantas faltas reais foram capturadas (precisão e recall nesse corte).
- **Estimativa de impacto (depende de hipóteses):** consultas recuperadas ≈ faltas reais entre os pacientes contatados × efeito incremental do contato. Nem toda falta prevista é evitável, e não se deve contar como recuperada uma consulta que ocorreria mesmo sem contato.
- **Custo e capacidade:** custo de contato × número de contatos, comparado com o valor da vaga recuperada, limitado pela capacidade de contato da unidade (K).
- **Limite adotado:** `K = 20%` dos agendamentos, por ser um corte operacional simples e compatível com a taxa histórica de faltas. O efeito real do contato não será estimado com este dataset observacional.

## 8. Cuidados metodológicos e limitações

### Variáveis disponíveis no momento da previsão

O objetivo é prever a falta no momento do agendamento, então só podem entrar variáveis conhecidas nesse instante.

- `SMS_received` será excluída do modelo principal por potencial vazamento temporal: o SMS pode ser enviado depois do agendamento. Será analisada separadamente, controlando o tempo de espera, apenas como associação observacional.
- `ScheduledDay` e `AppointmentDay` serão usados para derivar dias de espera e dia da semana da consulta, desde que a consulta já esteja definida no ato do agendamento.
- `PatientId` será usado somente para ordenar e construir histórico de consultas anteriores; não será usado como preditor direto.
- `AppointmentID` será excluído por ser identificador e carregar informação temporal.
- O histórico de faltas usará apenas consultas anteriores já concluídas na data do agendamento da consulta prevista, para evitar vazamento do futuro.
- `Age`, `Gender`, `Neighbourhood`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism` e `Handcap` serão tratados como variáveis cadastrais disponíveis no agendamento.

### Estratégia de validação
- **Separar por `PatientId`:** será uma análise de sensibilidade para medir a capacidade de generalizar para pacientes nunca vistos.
- **Validação temporal:** será a estratégia principal: treinar em agendamentos anteriores e testar nos posteriores, reproduzindo o uso em consultas futuras. A janela de consultas do dataset é curta, o que limita a generalização.
- Uma divisão adicional por `PatientId` poderá ser reportada como análise de sensibilidade, mas não substituirá a validação temporal, porque o caso de uso inclui pacientes recorrentes.

### Métricas

Acurácia isolada é insuficiente. Serão avaliados precisão e recall da classe de faltas, matriz de confusão, PR-AUC, ROC-AUC e recall@20%. Nenhum resultado numérico foi obtido ainda porque os módulos e notebooks contêm apenas a estrutura inicial.

### Limitações
- **Generalização:** dados de uma única cidade e de 2016, em janela curta (cerca de 6 semanas de consultas) e sem sazonalidade. Os resultados não representam o SUS como um todo.
- **Variáveis ausentes:** o dataset não traz motivo da falta, distância até a unidade, especialidade nem horário da consulta, e não identifica explicitamente cancelamentos.
- **Dificuldade esperada e comparabilidade:** a revisão de Carreras-García et al. (2020) analisou 50 estudos. O resumo afirma que apenas dois superaram a taxa de comparecimento em acurácia e que um único obteve AUC acima de 0,9; o corpo do texto fala em cinco de 15 estudos que relatam ambos os valores, então a contagem exata é inconsistente no próprio artigo (Referência 5). Os resultados variam conforme dados, variáveis, métodos e critérios de avaliação, e não implicam que modelos de previsão de faltas não funcionem.
- **Resultados com este mesmo dataset:** os trabalhos que o usaram reportam valores heterogêneos e nem sempre comparáveis, com AUC entre 0,68 e 0,88 e acurácia entre 78% e 95%, em condições diferentes (Referência 6, Tabelas 5 e 7). A Referência 4 chega a atribuir AUC de 0,6 a um deles, e outros números desse artigo conflitam com os da Referência 5. Por isso, **os resultados do projeto não devem ser comparados com esses valores** sem verificar se a validação é equivalente.
- **Fatores relevantes na literatura:** a revisão de Salazar et al. (2022) destaca idade, histórico de faltas e intervalo entre agendamento e consulta como os fatores mais frequentes (Referência 6). Eles servem como **candidatos a variáveis**, e não como resultado esperado no dataset de Vitória.

## 9. Registro de decisões

| Data | Decisão | Responsável |
|---|---|---|
| 23/out | Estrutura do repositório: código reutilizável em `src/`, análises em `notebooks/` (`docs/decisoes.md`) | Alexandre |
| 23/out | Primeira leitura do dataset e dicionário de dados; o arquivo ainda não está neste checkout | Matheus |
| 25/out | Classificação supervisionada binária com score de risco | Grupo |
| 25/out | Remover esperas negativas e idades fora do intervalo plausível; manter idade 0 | Grupo |
| 25/out | Deduplicar registros com os mesmos atributos e alvo; manter conflitos para auditoria | Grupo |
| 25/out | Excluir `SMS_received` do modelo principal por potencial vazamento temporal | Grupo |
| 25/out | Usar validação temporal como avaliação principal | Grupo |
| 25/out | Usar recall@20% como métrica operacional principal | Grupo |

## 10. Estrutura do repositório e como rodar

```text
c318-faltas-sus/
|-- data/
|   |-- raw/            # dataset original (ignorado no git)
|   `-- processed/      # dados tratados (ignorado no git)
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocessamento.ipynb
|   |-- 03_modelagem.ipynb
|   `-- 04_avaliacao.ipynb
|-- src/
|   |-- data.py         # carregamento e limpeza
|   |-- features.py     # engenharia de atributos
|   |-- models.py       # treinamento e tuning
|   `-- evaluate.py     # métricas e relatórios
|-- docs/
|   `-- decisoes.md     # log de decisões do projeto
|-- reports/figures/    # gráficos gerados
|-- requirements.txt
`-- README.md
```

```bash
git clone https://github.com/AlexandreTommasi/c318-faltas-sus.git
cd c318-faltas-sus
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# baixar o dataset (ver seção 6) e abrir os notebooks
jupyter notebook
```

## 11. Referências

1. Fritz, A. G.; Saraiva, M.; Schimanoski, D. K.; Oliveira, G. B.; Rigo, V. S.; Falcão, F. A. O.; Schirmer, L. A. F.; Trindade, L. F.; Silva, B. *Vagas ociosas, cuidados interrompidos: o absenteísmo como barreira ao acesso em saúde*. Resumo simples, seção Medicina, Unijuí. Anais do 12° Congresso Internacional em Saúde (CISaúde - 2025). https://www.publicacoeseventos.unijui.edu.br/index.php/conintsau/article/download/27042/25594/63517



2. Diário do Grande ABC. *Carreta com especialistas realizou mais de 1.400 mil exames em Mauá*. 25/02/2026. https://www.dgabc.com.br/Noticia/4287157/carreta-com-especialistas-realizou-mais-de-1-4-mil-exames-em-maua

3. Salazar, L. H. A.; Leithardt, V. R. Q.; Parreira, W. D.; Fernandes, A. M. R.; Barbosa, J. L. V.; Correia, S. D. *Application of Machine Learning Techniques to Predict a Patient's No-Show in the Healthcare Sector*. Future Internet, 2022, 14(1), 3. https://doi.org/10.3390/fi14010003

4. *Estratégia para enfrentamento do absenteísmo em uma unidade de saúde do município de São Paulo*. 6º Simpósio Científico do CEJAM. Autores e ano não identificados na auditoria. https://simposio.cejam.org.br/documentos/trabalhos/6o-simposio-cientifico-cejam-estrategia-para-enfrentamento-do-absenteismo-em-uma-unidade-de-saude-do-municipio-de-sao-paulo-10.pdf

5. Carreras-García, D.; Delgado-Gómez, D.; Llorente-Fernández, F.; Arribas-Gil, A. *Patient No-Show Prediction: A Systematic Literature Review*. Entropy, 2020, 22(6), 675. https://doi.org/10.3390/e22060675

6. Salazar, L. H. A.; Parreira, W. D.; Fernandes, A. M. R.; Leithardt, V. R. Q. *No-Show in Medical Appointments with Machine Learning Techniques: A Systematic Literature Review*. Information, 2022, 13(11), 507. https://doi.org/10.3390/info13110507

7. Hoppen, J. (Aquarela Advanced Analytics). *Medical Appointment No Shows*. Kaggle. https://www.kaggle.com/datasets/joniarroba/noshowappointments

8. Beltrame, S. M.; Oliveira, A. E.; Santos, M. A. B.; Santos Neto, E. T. *Absenteísmo de usuários como fator de desperdício: desafio para sustentabilidade em sistema universal de saúde*. Saúde em Debate, 2019, 43(123), 1015–1030. https://doi.org/10.1590/0103-1104201912303

9. Dantas, L. F.; Fleck, J. L.; Cyrino Oliveira, F. L.; Hamacher, S. *No-shows in appointment scheduling – a systematic literature review*. Health Policy, 2018, 122(4), 412–421. https://doi.org/10.1016/j.healthpol.2018.02.002

## Pesquisa da primeira entrega (S1)

A [pesquisa de abordagens e métricas para classificação desbalanceada](docs/pesquisa_classificacao_desbalanceada.md)
registra a proposta de avaliação, os cuidados contra vazamento e as referências
localizadas no Kaggle, incluindo os limites da consulta. Corresponde à tarefa
de pesquisa de Fernando (#4); o baseline será desenvolvido na S2 (#8).

## Limitações

10. Santos, J. S. *Absenteísmo dos usuários em consultas e procedimentos especializados agendados no SUS: um estudo em um município baiano*. Dissertação (Mestrado), Universidade Federal da Bahia, 2008.
