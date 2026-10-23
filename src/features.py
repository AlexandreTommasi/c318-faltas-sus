"""Engenharia de atributos para o modelo de faltas SUS."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria e transforma atributos a partir do DataFrame limpo.

    Inclui: tempo de espera (dias), faixas etárias, codificação
    de variáveis categóricas, etc.

    Args:
        df: DataFrame limpo retornado por data.clean.

    Returns:
        DataFrame com atributos prontos para modelagem.
    """
    ...


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica one-hot encoding ou ordinal encoding nas colunas categóricas.

    Args:
        df: DataFrame com atributos construídos.

    Returns:
        DataFrame com colunas numéricas.
    """
    ...


def split_features_target(df: pd.DataFrame, target_col: str = "No-show"):
    """Separa atributos (X) e variável-alvo (y).

    Args:
        df: DataFrame completo com atributos e target.
        target_col: Nome da coluna alvo.

    Returns:
        Tupla (X, y) como DataFrames/Series do pandas.
    """
    ...
