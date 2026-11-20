import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import clean, load_raw
from src.features import build_features
from src.models import split_by_patient


def prepare_eda(df):
    result = df.copy()
    result["age_group"] = pd.cut(
        result["age"], [-1, 12, 17, 29, 44, 59, np.inf],
        labels=["0–12", "13–17", "18–29", "30–44", "45–59", "60+"],
    )
    result["waiting_group"] = pd.cut(
        result["waiting_days"], [-1, 0, 2, 7, 14, 30, 60, np.inf],
        labels=["0", "1–2", "3–7", "8–14", "15–30", "31–60", "61+"],
    )
    return result


def rate_table(df, columns):
    table = df.groupby(columns, observed=True, dropna=False)["no_show"].agg(
        n="size", no_shows="sum", no_show_rate="mean",
    ).reset_index()
    table["share"] = table["n"] / len(df)
    return table


def run(raw_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = build_features(clean(load_raw(raw_path)))
    train, test = split_by_patient(df)
    eda = prepare_eda(df.iloc[train])
    variables = [
        "no_show", "age", "waiting_days", "age_group", "gender", "neighbourhood", "bolsa_familia",
        "hypertension", "diabetes", "alcoholism", "handicap", "waiting_group",
        "weekday", "sms_received",
    ]
    tables = {column: rate_table(eda, [column]) for column in variables}
    tables["sms_by_waiting"] = rate_table(eda, ["waiting_group", "sms_received"])
    summary = {
        "scope": "patient_split_training_only", "seed": 42,
        "clean_rows": len(df), "eda_rows": len(eda), "reserved_test_rows": len(test),
        "patients": int(eda["patient_id"].nunique()),
        "no_shows": int(eda["no_show"].sum()),
        "no_show_rate": float(eda["no_show"].mean()),
        "numeric_summary": eda[["age", "waiting_days"]].describe().to_dict(),
        "missing": eda.isna().sum().astype(int).to_dict(),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/KaggleV2-May-2016.csv")
    parser.add_argument("--output", default="reports/sprints-3-4/eda")
    args = parser.parse_args()
    summary, _ = run(args.raw, args.output)
    print(json.dumps(summary, indent=2))
