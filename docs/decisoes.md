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
