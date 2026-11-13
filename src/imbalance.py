import argparse
import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import clean, load_raw
from src.evaluate import compute_metrics
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features, split_features_target
from src.models import split_by_patient


STRATEGIES = ("unbalanced", "class_weight", "undersampling", "smote")
METRICS = (
    "pr_auc_ap", "roc_auc", "precision", "recall", "f1",
    "precision_at_20", "recall_at_20",
)


def make_strategy(strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    numeric = list(range(len(NUMERIC_FEATURES)))
    categorical = list(range(len(NUMERIC_FEATURES),
                             len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))
    before = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1), CATEGORICAL_FEATURES),
    ])
    sampler = {
        "unbalanced": "passthrough",
        "class_weight": "passthrough",
        "undersampling": RandomUnderSampler(random_state=42),
        "smote": SMOTENC(
            categorical_features=[NUMERIC_FEATURES.index(column) for column in
                                  ["bolsa_familia", "hypertension", "diabetes", "alcoholism", "handicap"]]
                                 + categorical,
            random_state=42, k_neighbors=5,
        ),
    }[strategy]
    after = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("before", before), ("sampler", sampler), ("after", after),
        ("model", LogisticRegression(max_iter=2000, random_state=42,
                                    class_weight="balanced" if strategy == "class_weight" else None)),
    ])


def compare_strategies(df, n_splits=5):
    train, test = split_by_patient(df)
    development = df.iloc[train].reset_index(drop=True)
    X, y = split_features_target(development)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y, groups=development["patient_id"]))
    records = []
    fold_metadata = []
    for fold, (fit, validation) in enumerate(folds, start=1):
        fit_groups = set(development.iloc[fit]["patient_id"])
        validation_groups = set(development.iloc[validation]["patient_id"])
        if fit_groups & validation_groups:
            raise ValueError("Pacientes compartilhados entre treino e validação.")
        if y.iloc[fit].nunique() != 2 or y.iloc[validation].nunique() != 2:
            raise ValueError("Cada partição precisa conter as duas classes.")
        if y.iloc[fit].value_counts().min() <= 5:
            raise ValueError("SMOTENC com k_neighbors=5 exige ao menos 6 exemplos por classe no treino.")
        fold_metadata.append({
            "fold": fold, "train_rows": len(fit), "validation_rows": len(validation),
            "train_patients": len(fit_groups), "validation_patients": len(validation_groups),
            "shared_patients": 0, "train_prevalence": float(y.iloc[fit].mean()),
            "validation_prevalence": float(y.iloc[validation].mean()),
        })
        for strategy in STRATEGIES:
            model = make_strategy(strategy)
            model.fit(X.iloc[fit], y.iloc[fit])
            metrics = compute_metrics(model, X.iloc[validation], y.iloc[validation])
            records.append({"strategy": strategy, "fold": fold, **metrics})
    results = pd.DataFrame(records)
    summary = results.groupby("strategy", sort=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    metadata = {
        "seed": 42, "cv": "StratifiedGroupKFold", "n_splits": n_splits,
        "development_rows": len(development), "reserved_test_rows": len(test),
        "test_evaluated": False, "threshold": 0.5, "contact_fraction": 0.2,
        "selection_metric": "pr_auc_ap_mean", "pr_auc_definition": "average_precision_score",
        "smote_variant": "SMOTENC before one-hot; categorical values are not interpolated",
        "smote_discrete_features": [
            "bolsa_familia", "hypertension", "diabetes", "alcoholism", "handicap",
        ] + CATEGORICAL_FEATURES,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "history_included": False,
        "folds": fold_metadata,
    }
    return results, summary, metadata


def run(raw_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = build_features(clean(load_raw(raw_path)))
    results, summary, metadata = compare_strategies(df)
    results.to_json(output / "fold_metrics.json", orient="records", indent=2)
    summary.to_csv(output / "comparison.csv")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return results, summary, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/KaggleV2-May-2016.csv")
    parser.add_argument("--output", default="reports/sprints-3-4/imbalance")
    args = parser.parse_args()
    _, summary, _ = run(args.raw, args.output)
    print(summary.to_string())
