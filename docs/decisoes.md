# Log de Decisões do Projeto

Registro das principais decisões tomadas ao longo do desenvolvimento.

---

## Formato

Cada entrada segue o padrão:

**Data:** AAAA-MM-DD  
**Decisão:** O que foi decidido.  
**Motivação:** Por que essa escolha foi feita.  
**Alternativas consideradas:** O que mais foi avaliado e descartado.

---

## Decisões

### 2026-10-23 — Estrutura inicial do repositório

**Decisão:** Separar código reutilizável em `src/` e análises exploratórias em `notebooks/`.  
**Motivação:** Facilita reuso de funções entre notebooks e mantém os notebooks legíveis (sem código boilerplate repetido).  
**Alternativas consideradas:** Tudo em notebooks — descartado por dificultar manutenção e testes.

---

### 2026-10-28 — SMS_received fora do modelo

**Decisão:** Não usar `SMS_received` como variável preditora. A relação entre SMS e falta segue como análise exploratória (pergunta de negócio 6), controlando o tempo de espera.

**Motivação:** O objetivo é prever no momento do agendamento, e o SMS é enviado depois. O dicionário de dados mostra que nenhum agendamento com espera de até 2 dias recebeu SMS, então a variável também mistura o efeito do tempo de espera. Decisão do líder do grupo.

**Alternativas consideradas:** Usar como preditor — descartado por vazamento de informação posterior ao agendamento.

---

### 2026-10-28 — Estratégia de validação

**Decisão:** Validação principal com separação por `PatientId` (treino e teste sem pacientes em comum). Teste temporal como checagem adicional.

**Motivação:** Evita que o mesmo paciente apareça em treino e teste. O teste temporal fica limitado porque as consultas cobrem só cerca de 6 semanas (29/04 a 08/06/2016). Sugestão do Fernando, aceita pelo grupo.

**Alternativas consideradas:** Somente validação temporal — descartada pela janela curta. A confirmar com o professor.

---

### 2026-10-28 — Corte de contato K

**Decisão:** Avaliar a métrica de negócio contatando os 20% de agendamentos de maior risco (K = 20%). Cortes de 10% e 30% podem entrar como análise de sensibilidade.

**Motivação:** A capacidade real de contato é desconhecida; um único corte simplifica a definição e já é usado nas métricas do projeto (precisão e recall no top 20%).

**Alternativas consideradas:** Vários cortes como definição principal — descartado por complexidade desnecessária nesta etapa.

---

### 2026-10-28 — Definição da métrica de negócio

**Decisão:** Consultas recuperadas = faltas capturadas nos K agendamentos de maior risco × efeito incremental do contato (e). Resultado observado (faltas capturadas) separado da estimativa (consultas recuperadas e custo líquido). Comparação com contato aleatório de K pacientes. Detalhes e parâmetros na seção 7 do README.

**Motivação:** Evita contar como recuperada uma falta que não seria evitada pelo contato e deixa explícito o que vem do modelo e o que vem de hipótese. Cenários de e: 10%, 20% e 30%, como hipóteses sem valor direto na literatura. Valor da vaga: cerca de US$ 37 (Ref. 4, dado histórico de fonte secundária). Custo por contato ainda sem valor.

**Alternativas consideradas:** Fixar um único valor de e — descartado por falta de fonte. Levar e e o custo por contato ao professor na mentoria.

---

### 2026-10-31 — Limpeza do dataset (src/data.py)

**Decisão:** Remover linhas com `age < 0` (1 linha) e linhas em que `appointment_day < scheduled_day` por data (5 linhas com espera negativa). Remover duplicatas excluindo `appointment_id` e `no_show` do subset, mantendo a primeira ocorrência (632 linhas removidas; inclui os 14 grupos com alvo divergente, resolvidos corretamente). Shape final: 109.889 linhas × 14 colunas. Manter idades acima de 100 anos.

**Motivação:**  
- Idades negativas são erro de registro sem valor de imputação plausível.  
- Espera negativa (consulta marcada antes do agendamento) é erro de sistema; não representa comportamento real.  
- Excluir `no_show` do subset de deduplicação garante que grupos com alvo divergente (mesmos dados clínicos, `no_show` diferente) fiquem com uma única linha, evitando contradições no treino.  
- Idades acima de 100 são suspeitas (4 registros do mesmo paciente com 115 anos), mas há registros plausíveis (102 anos). Removê-las não é decisão de limpeza, e sim de domínio: documentamos e mantemos.

**Nota:** O CSV processado (`data/processed/consultas_limpas.csv`) perde os tipos (datas e `patient_id` voltam como string genérica, `no_show` volta como int64 via CSV). Quem reler o CSV deve chamar `convert_types()` novamente — a função é idempotente para `no_show` já numérico.

**Alternativas consideradas:** Imputar idade negativa pela mediana — descartado; há apenas 1 caso e a origem é desconhecida. Remover idades > 100 — descartado; preservadas para análise posterior. Incluir `no_show` no subset de deduplicação — descartado; deixa 14 grupos com alvo divergente no dataset.

---

### 2026-11-07 — Baseline e seleção inicial (Fernando, #8 e #12)

**Decisão:** Dummy e regressão logística com partição por paciente, seed 42, e checagem temporal em 25/05. Encoder e escala ajustados apenas no treino. Correlações de Spearman somente no treino; categóricas recebem one-hot. Identificadores, SMS, datas brutas e same_day ficam fora dos preditores.

**Motivação:** Preservar a avaliação e evitar informação posterior ao agendamento. Histórico do paciente permanece na tarefa #9 do Alexandre. EDA e respostas de negócio permanecem nas tarefas #11 e #10 do Matheus e Antonio.
