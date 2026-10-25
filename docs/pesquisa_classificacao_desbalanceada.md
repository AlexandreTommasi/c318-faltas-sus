# Pesquisa de abordagens e métricas — primeira entrega (S1)

Responsável: Fernando Puebla Stein. Data: 25/10/2026. Issue: #4.

## Problema e classe de interesse

O projeto prevê faltas a consultas. A classe positiva será `No-show = Yes`
(faltou), codificada como 1; `No` (compareceu) será 0. Essa convenção deve
permanecer explícita nos gráficos, métricas e matrizes de confusão.

O README estima aproximadamente 20% de faltas. Essa proporção ainda precisa
ser conferida no CSV. Em um exemplo hipotético com 20% de faltas, prever que
todos comparecem gera 80% de acurácia e recall zero para faltas. Por isso,
acurácia isolada não serve como critério de seleção.

## Métricas propostas

| Métrica | Pergunta respondida | Uso e limitação |
|---|---|---|
| Precisão da classe falta | Entre os alertas, quantos eram faltas? | Indica a concentração de contatos em pacientes que faltariam; não demonstra que o contato evitaria a falta. |
| Recall da classe falta | Das faltas, quantas foram identificadas? | Mede cobertura; pode crescer com muitos falsos alertas. |
| F1 da classe falta | Como precisão e recall se equilibram? | Média harmônica; depende do limiar e atribui o mesmo peso aos dois objetivos. |
| Average Precision (AP) | O ranking concentra faltas nas maiores pontuações? | Proposta principal para comparação inicial; considerar a prevalência da classe positiva. |
| ROC-AUC | O modelo ordena uma falta acima de um comparecimento? | Complementar; não informa diretamente a precisão dos alertas na capacidade de contato disponível. |
| Acurácia balanceada | Qual a média do recall das duas classes? | Evita que a classe majoritária domine a acurácia. |
| Matriz de confusão | Quantos acertos e erros de cada tipo ocorreram? | Mostrar TN, FP, FN e TP com os rótulos explícitos. |

AP será calculada com `average_precision_score`, usando probabilidades da
classe 1. Ela resume a curva precisão-recall por uma soma ponderada dos
incrementos de recall; não é a área trapezoidal sob essa curva. Os relatórios
devem usar o nome **Average Precision**, evitando chamar ambas de PR-AUC.
Para pontuação constante, AP coincide com a prevalência positiva do conjunto
avaliado. ROC-AUC também deve usar probabilidades, e não rótulos previstos.

Proposta: comparar AP como métrica principal de ranking e apresentar precisão,
recall, F1, ROC-AUC, acurácia balanceada e matriz de confusão como complementos.
A escolha operacional do limiar depende do custo e da capacidade de contato,
a serem definidos pelo grupo. Esta é uma proposta de avaliação, não uma
decisão já validada por experimentos.

## Abordagens a comparar nas próximas entregas

| Abordagem | Papel no projeto | Cuidados |
|---|---|---|
| `DummyClassifier(strategy="prior")` | Referência ingênua que ignora os atributos e estima as frequências do treino. | Deve usar a mesma partição de avaliação dos demais modelos. |
| Regressão logística simples | Primeiro modelo supervisionado, de interpretação relativamente simples. | Padronizar atributos numéricos e codificar categorias dentro de um pipeline ajustado no treino. |
| Regressão logística com `class_weight="balanced"` | Avaliar pesos maiores para a classe minoritária. | Pode alterar precisão e recall; não pressupor melhora nem calibração das probabilidades. |
| Undersampling | Reduzir a classe majoritária no treino. | Descarta informação; nunca balancear validação/teste para simular desempenho real. |
| SMOTE / SMOTENC | Investigar sobreamostragem da classe minoritária. | Aplicar apenas no treino de cada fold; para dados mistos, estudar SMOTENC para não interpolar categorias como números contínuos. |
| Árvores, Random Forest e XGBoost | Comparar relações não lineares nas etapas posteriores. | Comparar com a mesma avaliação e controlar complexidade e sobreajuste. |

Não há evidência experimental nesta entrega de que uma dessas alternativas
seja superior. Ajuste de hiperparâmetros e de limiar deve ocorrer com dados de
treino/validação, mantendo o teste reservado para a avaliação final.

## Protocolo para evitar resultados otimistas

1. Confirmar rótulos, prevalência, duplicatas, valores inválidos e momento em
   que cada atributo está disponível.
2. Definir a pergunta de previsão: risco no agendamento ou perto da consulta.
   `SMS_received` pode ser posterior ao agendamento; não incluí-lo sem definir
   o instante da previsão. Sua associação com faltas não comprova efeito causal.
3. Excluir identificadores como atributos preditivos diretos. Preservar
   `PatientId` apenas para investigar repetição de pacientes e partições por grupo.
4. Para uma referência inicial, usar divisão estratificada com semente fixa.
   A repetição de pacientes entre treino e teste limita sua interpretação.
   Se o objetivo for generalizar para novos pacientes, usar separação por
   paciente; se for prever consultas futuras, avaliar uma separação temporal.
5. Ajustar imputação, normalização, codificação e eventual reamostragem apenas
   no treino, inclusive dentro de cada fold de validação cruzada.
6. Calcular as métricas sobre a distribuição original da avaliação. Registrar
   tamanho das partições, prevalência, semente, atributos e versões das bibliotecas.
7. Não interpretar faltas previstas como consultas recuperadas. Recuperação
   exige medir a efetividade da intervenção e o preenchimento da vaga.

## Notebooks de referência no Kaggle

Foram localizadas duas referências diretamente ligadas ao dataset:

- [Logistic Regression — Medical Appointment No-Show, Sangeet Raj Shrestha](https://www.kaggle.com/code/sangeetrajshrestha/logistic-regression-medical-appointment-no-show):
  a página identifica um notebook em R com foco em regressão logística.
  Serve como referência temática para o baseline.
- [EDA of No_Show_Appointments, Morningdewdelight](https://www.kaggle.com/code/morningdewdelight/eda-of-no-show-appointments):
  a página identifica um notebook em Python com etapas de limpeza e análise
  exploratória. Serve como referência temática para a investigação dos dados.

**Limite da consulta:** as páginas públicas consultadas expuseram os metadados,
mas não o código das células. Portanto, não foi possível auditar suas
partições, métricas, tratamento de vazamento ou resultados. Não foram
reproduzidos experimentos nem adotados resultados desses notebooks. A revisão
do código completo continua pendente de acesso às células; os critérios acima
permitem fazê-la sem assumir que os exemplos são metodologicamente corretos.

## Fontes técnicas

- [Scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html):
  definição das estratégias ingênuas, inclusive `prior`.
- [Scikit-learn — Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html):
  definição de AP e distinção da integração trapezoidal.
- [Scikit-learn — Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html):
  separação de treino/teste e pipelines para evitar vazamento.
- [Scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html):
  métricas de classificação e interpretação.
- [SMOTE — artigo original de Chawla et al.](https://arxiv.org/abs/1106.1813):
  proposta de geração de exemplos sintéticos da classe minoritária.

## Resultado desta entrega

Pesquisa e proposta metodológica documentadas. Não há modelos treinados,
métricas medidas ou comprovação de impacto operacional nesta etapa. A execução
do baseline fica para a S2 (#8); comparação de balanceamento e tuning seguem
o cronograma das próximas semanas.
