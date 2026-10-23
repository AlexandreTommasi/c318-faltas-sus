"""Funções de carregamento e limpeza do dataset de faltas SUS."""

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Carrega o CSV original do Kaggle.

    Args:
        path: Caminho para KaggleV2-May-2016.csv.

    Returns:
        DataFrame com os dados brutos.
    """
    ...


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos, nomes de colunas e remove registros inválidos.

    Args:
        df: DataFrame bruto retornado por load_raw.

    Returns:
        DataFrame limpo.
    """
    ...


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Salva o DataFrame processado em formato Parquet.

    Args:
        df: DataFrame a ser salvo.
        path: Caminho de destino (ex.: data/processed/clean.parquet).
    """
    ...
