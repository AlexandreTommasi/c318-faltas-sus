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
