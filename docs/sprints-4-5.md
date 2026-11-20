# Fernando — S4 e S5

Entrega limitada às issues #16 (regressão logística e árvore de decisão com
validação cruzada estratificada) e #20 (hiperparâmetros e limiar).

## Reprodução

Com as dependências instaladas e o CSV público do Kaggle em `data/raw/`:

```bash
python -m unittest discover -s tests -v
python -m src.model_selection --jobs 2
```

O notebook `notebooks/07_selecao_modelos.ipynb` apresenta os relatórios em
`reports/sprints-4-5/`. Nenhum dado individual é versionado.

## Protocolo

Usamos a limpeza e as features aprovadas na S3: idade, Bolsa Família,
hipertensão, diabetes, alcoolismo, deficiência, espera, gênero, bairro e dia
da semana. SMS, identificadores e datas brutas não entram no modelo. Mantemos
o experimento sem histórico do paciente para comparabilidade e para evitar
incorporar desfechos de outros folds na engenharia de atributos.

1. Reservamos 20% dos pacientes para teste com `GroupShuffleSplit`, seed 42,
   como na S2. Essa separação externa não é estratificada.
2. Dos pacientes restantes, reservamos 20% para validação independente do
   limiar. Nenhum paciente aparece em mais de um desses conjuntos.
3. No treino restante, usamos cinco folds com `StratifiedGroupKFold`, shuffle
   e seed 42. A estratificação é aproximada por causa dos grupos. Ambos os
   modelos usam os mesmos folds. Encoder e escala ficam dentro da pipeline,
   ajustados novamente em cada fold.
4. S4: registramos AP, ROC-AUC, precisão, recall e F1 de cada fold para
   logística com C=1 e árvore sem restrição de profundidade. Limiar 0,5.
5. S5: `GridSearchCV` seleciona pela média de AP (average precision):
   logística C em {0,01; 0,1; 1; 10}; árvore com profundidade em {3; 5; 8;
   ilimitada} e mínimo de amostras por folha em {20; 100; 300}. Não há
   balanceamento ou SMOTE nesta entrega, pois pertencem à tarefa #15.
6. Com cada melhor pipeline ajustada no treino da busca, escolhemos o limiar
   que maximiza F1 na validação independente. Avaliamos todos os valores
   distintos das probabilidades; empates favorecem o maior limiar. Não há
   novo treinamento após essa escolha.

O teste reservado (22.169 consultas, ~20% do total) **não é avaliado nesta
entrega**. Sua avaliação única cabe à tarefa de modelo final (#17), conforme
o guardrail do projeto.

**Nota de comparabilidade:** esta entrega reserva 20% adicionais do
desenvolvimento para validação independente do limiar antes do CV, de modo
que os folds de busca cobrem ~64% do total (contra ~80% nos folds do Matheus,
S4/#15). Os números de CV desta entrega não são diretamente comparáveis com
os daquela.

F1 é um critério técnico, sem pressupor custos ou eficácia de contato. As
métricas de ranking top 20% já existentes são mantidas: mudar o limiar não
altera AP, ROC-AUC ou ranking. O limiar F1 não impõe uma capacidade de contato
de 20%. As matrizes usam linhas reais e colunas preditas, na ordem [0, 1],
com 1 significando falta. Precisão sem positivos preditos é definida como zero.

## Resultados executados

Foram processadas 109.889 consultas: 70.443 no treino da busca, 17.277 na
validação de limiar e 22.169 reservadas para o teste final (prevalência de
faltas: 20,29%).

| Modelo | Melhor configuração | AP CV da busca | Limiar selecionado | F1 máximo na validação |
|---|---|---:|---:|---:|
| Logística | C=0,01 | 0,3057 | 0,1941 | 0,4150 |
| Árvore | profundidade ilimitada, folha mínima 300 | 0,3410 | 0,2129 | 0,4483 |

A AP da busca é uma estatística de seleção, sujeita a otimismo; o F1 máximo
é obtido na validação independente e não no treino. A avaliação externa no
teste reservado fica para a tarefa de modelo final (#17).

## Artefatos e limites

`s4_cv.csv` contém os cinco folds dos modelos sem tuning (média e desvio
padrão amostral são apresentados no notebook). `*_search.csv` registra todos
os candidatos e seus scores por fold; `*_thresholds.csv` registra até 501
pontos da curva precisão-recall na validação, incluindo o limiar escolhido.
`results.json` contém parâmetros, limiares, F1 máximo na validação, tamanhos
das partições e versões do ambiente. Métricas de teste não são versionadas.

A separação por paciente avalia
generalização para outros pacientes e não substitui validação temporal para
implantação futura. A coleta de uma cidade em 2016 limita generalização.
A escolha do modelo final, SHAP, curvas de avaliação e impacto de negócio
continuam fora desta entrega. Os baselines antigos usam mais dados para
ajuste (não reservavam validação do limiar); a comparação não isola apenas o
efeito de tuning.
