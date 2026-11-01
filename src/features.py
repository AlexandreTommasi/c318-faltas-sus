"""Engenharia de atributos para o modelo de faltas SUS."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria e transforma atributos a partir do DataFrame limpo.

    Atributos criados nesta etapa (issue #7):

    - ``waiting_days``: dias de espera entre a data do agendamento e a data
      da consulta (por dia, ignorando a hora de ``scheduled_day``).
    - ``appointment_weekday``: dia da semana da consulta (0 = segunda).

    Args:
        df: DataFrame limpo retornado por data.clean.

    Returns:
        DataFrame com atributos prontos para modelagem.
    """
    out = df.copy()
    out["waiting_days"] = (
        out["appointment_day"] - out["scheduled_day"].dt.normalize()
    ).dt.days
    out["appointment_weekday"] = out["appointment_day"].dt.dayofweek.astype("int8")
    return out


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
