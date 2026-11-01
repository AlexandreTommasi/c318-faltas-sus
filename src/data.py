"""Funções de carregamento e limpeza do dataset de faltas SUS."""

from pathlib import Path

import pandas as pd

RENAME_COLUMNS = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "bolsa_familia",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}

BOOL_COLUMNS = ["bolsa_familia", "hypertension", "diabetes", "alcoholism", "sms_received"]

MAX_AGE = 100


def load_raw(path: str) -> pd.DataFrame:
    """Carrega o CSV original do Kaggle.

    Args:
        path: Caminho para KaggleV2-May-2016.csv.

    Returns:
        DataFrame com os dados brutos.
    """
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos, nomes de colunas e remove registros inválidos.

    Etapas (issue #7):

    - Renomeia colunas para o padrão snake_case do dicionário de dados.
    - Converte datas para datetime (`appointment_day` perde a hora, que é
      sempre 00:00:00) e `patient_id` para string.
    - Remove idades inválidas ou implausíveis (``age < 0`` ou ``age > 100``).
    - Remove linhas em que a consulta precede o agendamento
      (dias de espera negativos).
    - Remove agendamentos duplicados (mesma linha exceto `appointment_id`).
      Grupos em que o alvo diverge entre as duplicatas são removidos por
      inteiro, pois não há como saber qual rótulo é o correto.
    - Converte flags binárias para bool e o alvo `no_show` para 1/0.

    Args:
        df: DataFrame bruto retornado por load_raw.

    Returns:
        DataFrame limpo.
    """
    out = df.rename(columns=RENAME_COLUMNS).copy()

    out["scheduled_day"] = pd.to_datetime(out["scheduled_day"], utc=True).dt.tz_localize(None)
    out["appointment_day"] = (
        pd.to_datetime(out["appointment_day"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    out["patient_id"] = out["patient_id"].map(
        lambda v: str(int(v)) if float(v).is_integer() else str(v)
    )

    out = out[out["age"].between(0, MAX_AGE)]
    out = out[out["appointment_day"] >= out["scheduled_day"].dt.normalize()]

    for col in BOOL_COLUMNS:
        out[col] = out[col].astype(bool)
    out["gender"] = out["gender"].astype("category")
    out["neighbourhood"] = out["neighbourhood"].astype("category")
    out["no_show"] = out["no_show"].map({"Yes": 1, "No": 0}).astype("int8")

    key = [c for c in out.columns if c not in ("appointment_id", "no_show")]
    out = out[out.groupby(key)["no_show"].transform("nunique") == 1]
    out = out.drop_duplicates(subset=key + ["no_show"], keep="first")

    return out.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Salva o DataFrame processado em formato Parquet.

    Args:
        df: DataFrame a ser salvo.
        path: Caminho de destino (ex.: data/processed/clean.parquet).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
