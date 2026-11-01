"""Funções de carregamento e limpeza do dataset de faltas SUS."""

import pandas as pd


_RENAME_MAP = {
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


def load_raw(path: str) -> pd.DataFrame:
    """Lê o CSV bruto do Kaggle sem alterar nada."""
    return pd.read_csv(path, low_memory=False)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas conforme o dicionário de dados."""
    return df.rename(columns=_RENAME_MAP)


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos: patient_id → str, datas → datetime, no_show → int.

    patient_id: IDs inteiros viram string sem ".0"; os 5 IDs com decimais
    espúrios (ex.: 93779.52927) são preservados como string com o decimal,
    mantendo-os distintos de IDs inteiros reais.
    no_show: 1 = faltou (Yes), 0 = compareceu (No). Idempotente: se já for
    numérico (CSV processado relido), mantém como está.
    """
    df = df.copy()

    # patient_id: inteiros → sem ".0"; decimais reais → mantém decimal
    df["patient_id"] = df["patient_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x)
    )

    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)

    # idempotente: só converte se ainda não for inteiro (pandas 3.x usa StringDtype)
    if not pd.api.types.is_integer_dtype(df["no_show"]):
        df["no_show"] = (df["no_show"] == "Yes").astype(int)

    return df


def remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com idade negativa e espera negativa.

    Idades acima de 100 são mantidas (suspeitas mas não descartáveis;
    documentadas no dicionário de dados — 7 linhas, 4 do mesmo paciente).
    Espera negativa: appointment_day < scheduled_day (por data), 5 linhas.
    """
    df = df.copy()
    mask_age = df["age"] >= 0
    mask_wait = (
        df["appointment_day"].dt.normalize() >= df["scheduled_day"].dt.normalize()
    )
    return df[mask_age & mask_wait]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas ignorando appointment_id e no_show, mantendo a primeira.

    Excluir no_show do subset garante que nos 14 grupos com alvo divergente
    (mesmo paciente, mesma consulta, no_show diferente) reste apenas 1 linha,
    como descrito em docs/decisoes.md.
    """
    excluir = {"appointment_id", "no_show"}
    chave = [c for c in df.columns if c not in excluir]
    return df.drop_duplicates(subset=chave, keep="first")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica toda a pipeline de limpeza na ordem correta e imprime o progresso."""
    n0 = len(df)
    print(f"Linhas brutas:            {n0:>7}")

    df = rename_columns(df)
    df = convert_types(df)

    df = remove_invalid(df)
    n1 = len(df)
    print(f"Após remover inválidos:   {n1:>7}  (-{n0 - n1})")

    df = drop_duplicates(df)
    n2 = len(df)
    print(f"Após remover duplicatas:  {n2:>7}  (-{n1 - n2})")

    print(f"Total removido:           {n0 - n2:>7}  ({(n0 - n2) / n0:.2%})")
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Salva o DataFrame limpo em CSV."""
    df.to_csv(path, index=False)
    print(f"Salvo em: {path}")


# ---------------------------------------------------------------------------
# Ponto de entrada: python -m src.data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from src.features import build_features

    RAW_PATH = os.path.join("data", "raw", "KaggleV2-May-2016.csv")
    OUT_PATH = os.path.join("data", "processed", "consultas_limpas.csv")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = load_raw(RAW_PATH)
    df = clean(df)
    df = build_features(df)

    print(f"\nShape final: {df.shape}")
    taxa = df["no_show"].mean()
    print(f"Taxa de falta: {taxa:.1%} ({df['no_show'].sum()} de {len(df)})")

    save_processed(df, OUT_PATH)
