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


# Seleção explícita impede inclusão acidental de alvo, datas e identificadores.
NUMERIC_FEATURES = ["age", "bolsa_familia", "hypertension", "diabetes",
                    "alcoholism", "handicap", "waiting_days"]
CATEGORICAL_FEATURES = ["gender", "neighbourhood", "weekday"]
HISTORY_FEATURES = ["previous_appointments", "previous_no_shows", "previous_no_show_rate"]


def add_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Histórico disponível no agendamento; consulta só conclui ao fim do dia.

    Busca estritamente dias anteriores ao dia de agendamento, pois o dataset
    não informa o horário da consulta. Empates nunca observam o próprio alvo.
    A entrada pode estar fora de ordem e ter índices repetidos.
    """
    import numpy as np

    result = df.copy()
    counts = np.zeros(len(df), dtype=int)
    misses = np.zeros(len(df), dtype=int)
    # Índice posicional próprio preserva ordem e índice externos.
    work = df.reset_index(drop=True)
    for _, appointments in work.groupby("patient_id", sort=False):
        completed = appointments.groupby("appointment_day")["no_show"].agg(["count", "sum"]).sort_index()
        dates = completed.index.normalize().asi8
        positions = np.searchsorted(dates, appointments["scheduled_day"].dt.normalize().array.asi8, side="left")
        counts[appointments.index] = np.r_[0, completed["count"].cumsum().to_numpy()][positions]
        misses[appointments.index] = np.r_[0, completed["sum"].cumsum().to_numpy()][positions]
    result["previous_appointments"] = counts
    result["previous_no_shows"] = misses
    result["previous_no_show_rate"] = np.divide(misses, counts, out=np.zeros(len(df)), where=counts > 0)
    return result


def encode_categoricals(df: pd.DataFrame):
    """Retorna um ColumnTransformer não ajustado; fit apenas no treino."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in NUMERIC_FEATURES + HISTORY_FEATURES if c in df]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop")


def split_features_target(df: pd.DataFrame, target_col: str = "no_show"):
    """Separa somente atributos aprovados; same_day é redundante com espera."""
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    columns += [c for c in HISTORY_FEATURES if c in df]
    return df.loc[:, columns].copy(), df[target_col].copy()
