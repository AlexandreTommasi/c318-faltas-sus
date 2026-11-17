"""Sprint 5 (#19): aggregate validation curves, without opening the reserved test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import clean, load_raw
from src.features import build_features
from src.imbalance import compare_strategies


MAX_CURVE_POINTS = 501


def curve_tables(y, scores, max_points=MAX_CURVE_POINTS):
    """Return display coordinates; exact AP/AUC are computed separately on all scores."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Curves require aligned vectors containing both binary classes.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and within [0, 1].")
    if max_points < 2:
        raise ValueError("At least two display points are required.")
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    tables = []
    for table in (
        pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        pd.DataFrame({"recall": recall, "precision": precision}),
    ):
        positions = np.unique(np.linspace(
            0, len(table) - 1, min(max_points, len(table)), dtype=int,
        ))
        tables.append(table.iloc[positions].reset_index(drop=True))
    return tuple(tables)


def evaluate_development(df, n_splits=5):
    """Evaluate the S4 strategies on their original grouped folds, one pass per fold."""
    roc_tables, pr_tables = [], []

    def collect(strategy, fold, y, scores):
        roc, pr = curve_tables(y, scores)
        for table, destination in ((roc, roc_tables), (pr, pr_tables)):
            table.insert(0, "fold", fold)
            table.insert(0, "strategy", strategy)
            destination.append(table)

    results, summary, metadata = compare_strategies(
        df, n_splits=n_splits, on_validation=collect,
    )
    matrices = []
    for strategy, group in results.groupby("strategy", sort=False):
        matrix = np.asarray(group["confusion_matrix"].tolist(), dtype=int).sum(axis=0)
        for actual in (0, 1):
            for predicted in (0, 1):
                matrices.append({
                    "strategy": strategy, "actual": actual, "predicted": predicted,
                    "count": int(matrix[actual, predicted]),
                })
    metadata.update({
        "issue": 19,
        "evaluation": "out-of-fold development validation, not final test evaluation",
        "positive_class": "no_show=1 (missed appointment)",
        "curve_scope": "one curve per strategy and validation fold; no pooled score curve",
        "curve_display": {
            "max_points_per_curve": MAX_CURVE_POINTS,
            "sampling": "evenly spaced curve indices, retaining both endpoints",
            "metrics_from_display_points": False,
        },
        "confusion_matrix": {
            "rows": "actual [0=attended, 1=missed]",
            "columns": "predicted [0=attended, 1=missed]",
            "aggregation": "sum of validation fold counts; each development row once",
            "decision_rule": "existing classifier predict at fixed threshold 0.5",
        },
        "uncertainty": "sample standard deviation across folds, not confidence intervals",
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "scores_calibrated": False,
    })
    return (
        results, summary, metadata, pd.concat(roc_tables, ignore_index=True),
        pd.concat(pr_tables, ignore_index=True), pd.DataFrame(matrices),
    )


def run(raw_path, output):
    """Freeze aggregate tables only; never persist labels or scores per appointment."""
    raw_path, output = Path(raw_path), Path(output)
    df = build_features(clean(load_raw(raw_path)))
    artifacts = evaluate_development(df)
    results, summary, metadata, roc, pr, matrices = artifacts
    metadata["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    results.to_json(output / "fold_metrics.json", orient="records", indent=2)
    summary.to_csv(output / "comparison.csv")
    roc.to_csv(output / "roc_curves.csv", index=False)
    pr.to_csv(output / "precision_recall_curves.csv", index=False)
    matrices.to_csv(output / "confusion_matrices.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/KaggleV2-May-2016.csv")
    parser.add_argument("--output", default="reports/sprint-5/evaluation")
    args = parser.parse_args()
    _, summary, _, _, _, _ = run(args.raw, args.output)
    print(summary.to_string())
