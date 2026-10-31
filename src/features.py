"""Engenharia de atributos para o modelo de faltas SUS."""

import pandas as pd


# sms_received: posterior ao agendamento — vazamento de dados (ver decisoes.md).
# patient_id e appointment_id: identificadores, não features preditoras.
FEATURES_EXCLUIDAS = ["patient_id", "appointment_id", "sms_received"]


def add_waiting_days(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona 'waiting_days': dias entre o agendamento e a consulta (só datas)."""
    df = df.copy()
    df["waiting_days"] = (
        df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()
    ).dt.days
    return df


def add_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona 'weekday': dia da semana da consulta (0=segunda … 6=domingo)."""
    df = df.copy()
    df["weekday"] = df["appointment_day"].dt.dayofweek
    return df


def add_same_day(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona 'same_day': 1 se a consulta é no mesmo dia do agendamento."""
    df = df.copy()
    df["same_day"] = (
        df["appointment_day"].dt.normalize() == df["scheduled_day"].dt.normalize()
    ).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica add_waiting_days, add_weekday e add_same_day ao DataFrame limpo."""
    df = add_waiting_days(df)
    df = add_weekday(df)
    df = add_same_day(df)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica one-hot encoding ou ordinal encoding nas colunas categóricas.

    Args:
        df: DataFrame com atributos construídos.

    Returns:
        DataFrame com colunas numéricas.
    """
    ...


def split_features_target(df: pd.DataFrame, target_col: str = "no_show"):
    """Separa atributos (X) e variável-alvo (y).

    Args:
        df: DataFrame completo com atributos e target.
        target_col: Nome da coluna alvo.

    Returns:
        Tupla (X, y) como DataFrames/Series do pandas.
    """
    ...
