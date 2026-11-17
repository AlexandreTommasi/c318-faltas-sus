# Sprint 5 - Avaliação por curvas (#19)

Escopo da tarefa #19 do Matheus: curvas ROC, curvas precisão-recall e matrizes de
confusão das quatro estratégias da S4 (`unbalanced`, `class_weight`,
`undersampling`, `smote`), calculadas na validação cruzada do treino. Esta entrega
é local; não fecha nem comenta issues automaticamente.

## Status das tarefas

| Tarefa | Situação |
|---|---|
| #3 (S1) | fechada no GitHub |
| #7 (S2) | fechada no GitHub |
| #11 (S3, EDA) | implementada no repositório, issue aberta |
| #15 (S4, balanceamento) | implementada no repositório, issue aberta |
| #19 (S5, curvas) | esta entrega local, sem fechamento automático |

Tarefas de colegas (por exemplo #17 e #20) não fazem parte desta entrega.

## Reprodução

```bash
.venv/bin/python -m src.validation_curves   # agregados em reports/sprint-5/evaluation/
.venv/bin/python -m src.validation_plots    # figuras em reports/sprint-5/figures/
.venv/bin/python -m unittest discover -s tests -v
```

`src.validation_plots` lê somente os agregados congelados; não lê o dado bruto nem
treina modelos. `src.validation_curves` reutiliza `compare_strategies` com os mesmos
folds da S4 e não abre o teste reservado.

## Artefatos

`reports/sprint-5/evaluation/`:

- `fold_metrics.json`: 20 registros (4 estratégias x 5 folds), com as métricas exatas.
- `comparison.csv`: média e desvio por estratégia (mesmo esquema da S4).
- `metadata.json`: configuração, folds e limitações.
- `roc_curves.csv`: `strategy, fold, fpr, tpr`.
- `precision_recall_curves.csv`: `strategy, fold, recall, precision`.
- `confusion_matrices.csv`: `strategy, actual, predicted, count` (soma dos folds,
  4 linhas por estratégia).

Não há scores nem rótulos por indivíduo nos artefatos.

## Figuras

- [ROC](../reports/sprint-5/figures/roc.png)
- [Precisão-recall](../reports/sprint-5/figures/precision_recall.png)
- [Matrizes de confusão](../reports/sprint-5/figures/confusion_matrices.png)

O notebook [`notebooks/04_avaliacao.ipynb`](../notebooks/04_avaliacao.ipynb) apenas
lê os agregados e exibe as figuras.

## Leitura dos resultados

Médias dos cinco folds de validação (não são métricas do teste final):

| Estratégia | ROC-AUC | AP | Precisão (0,5) | Recall (0,5) |
|---|---:|---:|---:|---:|
| Sem balanceamento | 0,6584 | 0,3066 | 36,59% | 2,06% |
| Peso de classe | 0,6649 | 0,3073 | 31,31% | 55,73% |
| Subamostragem | 0,6644 | 0,3064 | 31,13% | 56,00% |
| SMOTENC | 0,6544 | 0,3031 | 30,57% | 55,69% |

As quatro estratégias têm AP próxima (0,3031 a 0,3073), acima da prevalência
aproximada de 20,06% no treino. Isso não demonstra superioridade estatística de
uma estratégia. No limiar fixo 0,5, o balanceamento aumenta o recall, mas também
os falsos positivos: sem balanceamento são 629 falsos positivos e 17.232 falsos
negativos; com peso de classe são 21.518 e 7.788, respectivamente, somando as
validações. Esse contraste depende do limiar e não equivale à política de contato
top-20%, nem comprova consultas recuperadas por uma intervenção.

As matrizes somam 87.720 agendamentos por estratégia, com 17.594 faltas reais;
cada linha do treino foi validada uma vez. Os 22.169 registros de teste continuam
reservados. A escolha do modelo final e o ajuste do limiar seguem fora da #19.

## Limitações

- Classe positiva: `no_show=1` (faltou). Validação por paciente no treino; o teste
  reservado não foi usado.
- Scores de modelos com `class_weight` ou reamostragem não são probabilidades
  calibradas.
- A matriz de confusão usa o limiar fixo 0,5, não otimizado; é diferente da regra
  top-20% do recall@20%.
- AP é average precision, não a área trapezoidal sob a curva.
- Desvio entre folds é desvio-padrão amostral, não intervalo de confiança.
- As curvas têm no máximo 501 pontos por fold, só para exibição; as métricas
  exatas foram calculadas antes da redução e não devem ser recalculadas a partir
  delas.
- Sem curva média nem curva agregada de todos os scores.
- Sem escolha de modelo final, ajuste de hiperparâmetros nem SHAP.
