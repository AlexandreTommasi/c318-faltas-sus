"""Traduz as métricas do teste reservado em impacto operacional e financeiro.

Lê reports/modelo-final/teste_final.csv e teste_final.json (modelo final e
candidatos, todos avaliados uma única vez no mesmo teste reservado) e grava
reports/impacto_negocio.csv. Valor da vaga, custo por contato, câmbio e o
efeito do contato (e) são hipóteses, não medidas neste dataset.
"""

import csv
import json
import math
from pathlib import Path

CSV_PATH = Path("reports/modelo-final/teste_final.csv")
JSON_PATH = Path("reports/modelo-final/teste_final.json")
OUTPUT_PATH = Path("reports/impacto_negocio.csv")

FRACAO_CONTATO = 0.20
VALOR_VAGA_USD = 37.0
CUSTO_CONTATO_BRL = 1.50
CAMBIOS_BRL_POR_USD = (4.0, 5.0, 6.0)
EFEITOS = (0.10, 0.20, 0.30)
MODELO_FINAL = "xgboost"


def _sufixo(cambio: float) -> str:
    return str(cambio).replace(".", "_")


def _carregar():
    with CSV_PATH.open(encoding="utf-8", newline="") as stream:
        ranking = {row["candidate"]: row for row in csv.DictReader(stream)}
    matrizes = {
        item["candidate"]: item["confusion_matrix"]
        for item in json.loads(JSON_PATH.read_text(encoding="utf-8"))
    }
    return ranking, matrizes


def _totais(matrizes: dict) -> tuple[int, int]:
    """Retorna (n_teste, faltas) e confere que todos os modelos concordam."""
    totais = set()
    for (vn, fp), (fn, vp) in matrizes.values():
        totais.add((vn + fp + fn + vp, fn + vp))
    if len(totais) != 1:
        raise ValueError(f"Matrizes com totais diferentes: {totais}")
    return totais.pop()


def _politicas(ranking: dict, matrizes: dict, n_teste: int, faltas: int):
    contatos_top20 = math.ceil(FRACAO_CONTATO * n_teste)
    for nome, row in ranking.items():
        if nome == "dummy":
            continue
        capturadas = float(row["recall_at_20"]) * faltas
        esperado = float(row["precision_at_20"]) * contatos_top20
        if not math.isclose(capturadas, esperado, rel_tol=1e-6):
            raise ValueError(f"{nome}: precision@20 e recall@20 inconsistentes")
        yield nome, "top20", contatos_top20, capturadas
    (_, fp), (_, vp) = matrizes[MODELO_FINAL]
    yield MODELO_FINAL, "limiar_congelado", fp + vp, float(vp)


def build_rows(ranking: dict, matrizes: dict) -> list[dict]:
    n_teste, faltas = _totais(matrizes)
    prevalencia = faltas / n_teste
    linhas = []
    for modelo, corte, contatos, capturadas in _politicas(
        ranking, matrizes, n_teste, faltas
    ):
        aleatorio = contatos * prevalencia
        custo_total = contatos * CUSTO_CONTATO_BRL
        base = {
            "modelo": modelo,
            "corte": corte,
            "n_teste": n_teste,
            "faltas_teste": faltas,
            "prevalencia": prevalencia,
            "contatos": contatos,
            "capturadas": capturadas,
            "precisao": capturadas / contatos,
            "recall": capturadas / faltas,
            "capturadas_aleatorio": aleatorio,
            "contatos_por_1000": contatos / n_teste * 1000,
            "capturadas_por_1000": capturadas / n_teste * 1000,
            "capturadas_aleatorio_por_1000": aleatorio / n_teste * 1000,
            "custo_contato_total_brl": custo_total,
            "custo_contato_por_1000_brl": custo_total / n_teste * 1000,
            "valor_vaga_usd": VALOR_VAGA_USD,
            "custo_por_contato_brl": CUSTO_CONTATO_BRL,
        }
        for efeito in EFEITOS:
            linha = dict(base)
            linha["e"] = efeito
            linha["vagas_recuperadas"] = capturadas * efeito
            linha["vagas_recuperadas_aleatorio"] = aleatorio * efeito
            linha["ganho_sobre_aleatorio"] = (
                linha["vagas_recuperadas"] - linha["vagas_recuperadas_aleatorio"]
            )
            linha["vagas_recuperadas_por_1000"] = (
                linha["vagas_recuperadas"] / n_teste * 1000
            )
            linha["vagas_recuperadas_aleatorio_por_1000"] = (
                linha["vagas_recuperadas_aleatorio"] / n_teste * 1000
            )
            linha["ganho_sobre_aleatorio_por_1000"] = (
                linha["ganho_sobre_aleatorio"] / n_teste * 1000
            )
            for cambio in CAMBIOS_BRL_POR_USD:
                valor_vaga_brl = VALOR_VAGA_USD * cambio
                s = _sufixo(cambio)
                linha[f"custo_liquido_brl_cambio_{s}"] = (
                    linha["vagas_recuperadas"] * valor_vaga_brl - custo_total
                )
                linha[f"ganho_liquido_sobre_aleatorio_brl_cambio_{s}"] = (
                    linha["ganho_sobre_aleatorio"] * valor_vaga_brl
                )
            linhas.append(linha)
        for cambio in CAMBIOS_BRL_POR_USD:
            ponto = CUSTO_CONTATO_BRL / (
                (capturadas / contatos) * VALOR_VAGA_USD * cambio
            )
            for linha in linhas[-len(EFEITOS):]:
                linha[f"break_even_e_cambio_{_sufixo(cambio)}"] = ponto
    return linhas


def _imprimir(linhas: list[dict]) -> None:
    colunas = [
        "modelo", "corte", "e", "contatos", "capturadas", "precisao", "recall",
        "vagas_recuperadas_por_1000", "ganho_sobre_aleatorio_por_1000",
        "custo_liquido_brl_cambio_5_0", "break_even_e_cambio_5_0",
    ]
    print("| " + " | ".join(colunas) + " |")
    print("| " + " | ".join("---" for _ in colunas) + " |")
    for linha in linhas:
        print("| " + " | ".join(str(linha[c]) for c in colunas) + " |")


def main() -> None:
    ranking, matrizes = _carregar()
    linhas = build_rows(ranking, matrizes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as stream:
        escritor = csv.DictWriter(stream, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    _imprimir(linhas)


if __name__ == "__main__":
    main()
