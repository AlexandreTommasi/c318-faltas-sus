"""Issues #13 e #17: Random Forest, XGBoost, ablação do histórico, modelo final.

CLI:
  python -m src.final_model --fase cv                             # Fase 1: CV candidatos
  python -m src.final_model --fase freeze --model M [--history]  # congela escolha
  python -m src.final_model --fase final                         # Fase 2: teste único
"""

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    add_patient_history,
    build_features,
)
from src.data import clean, load_raw
from src.models import split_by_patient


OUTPUT_DIR = Path("reports/modelo-final")

# Best hyperparams from Fernando's model_selection (reports/sprints-4-5/results.json)
_LOGISTIC_PARAMS = {"C": 0.01, "max_iter": 2000, "random_state": 42}
_TREE_PARAMS = {"max_depth": None, "min_samples_leaf": 300, "random_state": 42}

_RF_GRID = [
    {
        "n_estimators": 300,
        "min_samples_leaf": leaf,
        "max_features": feat,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    for leaf, feat in product([20, 100], ["sqrt", 0.5])
]

_XGB_GRID = [
    {
        "max_depth": depth,
        "learning_rate": lr,
        "n_estimators": 300,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
    }
    for depth, lr in product([4, 6], [0.05, 0.1])
]

_CANDIDATE_NAMES = ["dummy", "logistic", "decision_tree", "random_forest", "xgboost"]


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def get_feature_columns(use_history: bool = False) -> list:
    """Return ordered list of feature column names."""
    cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    if use_history:
        cols = cols + list(HISTORY_FEATURES)
    return cols


def make_preprocessor(use_history: bool = False) -> ColumnTransformer:
    """ColumnTransformer: scale numeric, OHE categorical. Must fit only on training data."""
    numeric = list(NUMERIC_FEATURES) + (list(HISTORY_FEATURES) if use_history else [])
    categorical = list(CATEGORICAL_FEATURES)
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Custom estimator: XGBoost with scale_pos_weight computed from y_train
# ---------------------------------------------------------------------------

class _AutoWeightXGB(BaseEstimator, ClassifierMixin):
    """XGBClassifier that computes scale_pos_weight from y_train at fit time.

    Not compatible with sklearn clone(): use _build_pipeline which creates
    fresh instances per fold.
    """

    def __init__(self, xgb_params=None):
        self.xgb_params = xgb_params or {}

    def fit(self, X, y, **kw):
        y_arr = np.asarray(y)
        neg = int((y_arr == 0).sum())
        pos = int((y_arr == 1).sum())
        spw = neg / pos if pos > 0 else 1.0
        self._clf = XGBClassifier(scale_pos_weight=spw, **self.xgb_params)
        self._clf.fit(X, y_arr, **kw)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def predict(self, X):
        return self._clf.predict(X)


# ---------------------------------------------------------------------------
# CV utilities
# ---------------------------------------------------------------------------

def _get_folds(development_df: pd.DataFrame, n_splits: int = 5) -> list:
    """StratifiedGroupKFold por paciente — mesmos parâmetros de imbalance.py."""
    y = development_df["no_show"]
    groups = development_df["patient_id"]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(splitter.split(development_df, y, groups))
    for train_idx, val_idx in folds:
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        if overlap:
            raise ValueError(f"Pacientes compartilhados entre folds: {len(overlap)}")
        if y.iloc[train_idx].nunique() != 2 or y.iloc[val_idx].nunique() != 2:
            raise ValueError("Cada fold deve conter as duas classes (0 e 1).")
    return folds


def _fold_metrics(y_val: np.ndarray, scores: np.ndarray) -> dict:
    """AP, ROC-AUC, precision@20%, recall@20% for one fold."""
    ap = float(average_precision_score(y_val, scores))
    roc = float(roc_auc_score(y_val, scores))
    k = max(1, int(np.ceil(0.2 * len(y_val))))
    thr = np.sort(scores)[-k]
    above = scores > thr
    tied = scores == thr
    captured = float(y_val[above].sum() + y_val[tied].mean() * (k - above.sum()))
    p20 = captured / k
    r20 = captured / y_val.sum() if y_val.sum() > 0 else 0.0
    return dict(ap=ap, roc_auc=roc, precision_at_20=p20, recall_at_20=r20)


def _build_pipeline(model_name: str, params: dict, use_history: bool) -> Pipeline:
    """Build a fresh Pipeline for the given model/params/history combo."""
    if model_name == "dummy":
        est = DummyClassifier(strategy="prior", random_state=42)
    elif model_name == "logistic":
        est = LogisticRegression(**params)
    elif model_name == "decision_tree":
        est = DecisionTreeClassifier(**params)
    elif model_name == "random_forest":
        est = RandomForestClassifier(**params)
    elif model_name == "xgboost":
        est = _AutoWeightXGB(xgb_params=params)
    else:
        raise ValueError(f"Candidato desconhecido: {model_name}")
    return Pipeline([
        ("preprocess", make_preprocessor(use_history)),
        ("model", est),
    ])


def _run_one_config(
    model_name: str,
    params: dict,
    use_history: bool,
    folds: list,
    dev_df: pd.DataFrame,
) -> tuple:
    """Run (model, params, use_history) on all folds.

    Returns (fold_rows, oof_scores, oof_labels).
    OOF scores kept in memory only; not saved here.
    """
    cols = get_feature_columns(use_history)
    X = dev_df[cols]
    y = dev_df["no_show"]
    fold_rows = []
    oof_scores = np.zeros(len(dev_df))
    for fold_i, (tr, va) in enumerate(folds, start=1):
        pipe = _build_pipeline(model_name, params, use_history)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pos_idx = list(pipe.classes_).index(1)
        scores = pipe.predict_proba(X.iloc[va])[:, pos_idx]
        metrics = _fold_metrics(np.asarray(y.iloc[va]), scores)
        fold_rows.append({"fold": fold_i, **metrics})
        oof_scores[va] = scores
    return fold_rows, oof_scores, np.asarray(y)


def _summarize(fold_rows: list) -> dict:
    keys = ["ap", "roc_auc", "precision_at_20", "recall_at_20"]
    result = {}
    for k in keys:
        vals = [r[k] for r in fold_rows]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals, ddof=1))
    return result


def _find_best_in_grid(
    model_name: str,
    grid: list,
    use_history: bool,
    folds: list,
    dev_df: pd.DataFrame,
) -> tuple:
    """Grid search over params: return (best_params, fold_rows, oof_scores, oof_labels)."""
    best_ap = -1.0
    best_result = None
    for params in grid:
        fold_rows, oof_sc, oof_lb = _run_one_config(model_name, params, use_history, folds, dev_df)
        mean_ap = float(np.mean([r["ap"] for r in fold_rows]))
        if mean_ap > best_ap:
            best_ap = mean_ap
            best_result = (params, fold_rows, oof_sc, oof_lb)
    return best_result


# ---------------------------------------------------------------------------
# Fase 1: CV de candidatos
# ---------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    output: Path = OUTPUT_DIR,
    n_splits: int = 5,
) -> tuple:
    """Fase 1 (#13): todos os candidatos nos mesmos 5 folds.

    Saves:
      cv_candidatos.csv        — média ± dp por (candidato, use_history), ordenada por AP
      cv_ablacao_historico.csv — ΔAP por candidato (com - sem histórico)
      cv_summary.json          — melhores params RF/XGBoost + metadados
      oof.npz                  — scores/rótulos OOF por candidato (gitignored)
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    train_idx, _test_idx = split_by_patient(df)
    dev_df = df.iloc[train_idx].reset_index(drop=True)
    folds = _get_folds(dev_df, n_splits)

    fixed = [
        ("dummy", {}),
        ("logistic", _LOGISTIC_PARAMS),
        ("decision_tree", _TREE_PARAMS),
    ]

    summary_rows = []
    oof_data = {}
    best_params_map = {}

    for use_history in [False, True]:
        for name, params in fixed:
            fold_rows, oof_sc, oof_lb = _run_one_config(
                name, params, use_history, folds, dev_df
            )
            s = _summarize(fold_rows)
            summary_rows.append({"candidate": name, "use_history": use_history, **s})
            oof_data[f"{name}__{int(use_history)}"] = (oof_sc, oof_lb)

        rf_params, rf_rows, rf_sc, rf_lb = _find_best_in_grid(
            "random_forest", _RF_GRID, use_history, folds, dev_df
        )
        s = _summarize(rf_rows)
        # Store best_params without noise keys for display
        rf_display = {k: v for k, v in rf_params.items() if k not in ("n_jobs", "random_state")}
        summary_rows.append({
            "candidate": "random_forest",
            "use_history": use_history,
            "best_params": json.dumps(rf_display, sort_keys=True),
            **s,
        })
        oof_data[f"random_forest__{int(use_history)}"] = (rf_sc, rf_lb)
        best_params_map[f"random_forest__{int(use_history)}"] = rf_params

        xgb_params, xgb_rows, xgb_sc, xgb_lb = _find_best_in_grid(
            "xgboost", _XGB_GRID, use_history, folds, dev_df
        )
        s = _summarize(xgb_rows)
        xgb_display = {k: v for k, v in xgb_params.items()
                       if k not in ("verbosity", "random_state", "eval_metric")}
        summary_rows.append({
            "candidate": "xgboost",
            "use_history": use_history,
            "best_params": json.dumps(xgb_display, sort_keys=True),
            **s,
        })
        oof_data[f"xgboost__{int(use_history)}"] = (xgb_sc, xgb_lb)
        best_params_map[f"xgboost__{int(use_history)}"] = xgb_params

    cv_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["use_history", "ap_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    cv_df.to_csv(output / "cv_candidatos.csv", index=False)

    ablation_rows = []
    for name in _CANDIDATE_NAMES:
        rows_no = cv_df[(cv_df.candidate == name) & ~cv_df.use_history]
        rows_yes = cv_df[(cv_df.candidate == name) & cv_df.use_history]
        if len(rows_no) and len(rows_yes):
            ap_no = float(rows_no.iloc[0]["ap_mean"])
            ap_yes = float(rows_yes.iloc[0]["ap_mean"])
            ablation_rows.append({
                "candidate": name,
                "ap_sem_historico": round(ap_no, 4),
                "ap_com_historico": round(ap_yes, 4),
                "delta_ap": round(ap_yes - ap_no, 4),
            })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(output / "cv_ablacao_historico.csv", index=False)

    cv_summary = {
        "n_splits": n_splits,
        "development_rows": len(dev_df),
        "features_no_history": get_feature_columns(False),
        "features_with_history": get_feature_columns(True),
        "best_params": {k: v for k, v in best_params_map.items()},
    }
    (output / "cv_summary.json").write_text(
        json.dumps(cv_summary, indent=2, default=str)
    )

    # Save OOF predictions (gitignored) for threshold selection in --fase freeze
    flat = {}
    for k, (sc, lb) in oof_data.items():
        flat[f"{k}__scores"] = sc
        flat[f"{k}__labels"] = lb
    np.savez_compressed(output / "oof.npz", **flat)

    return cv_df, ablation_df, oof_data


# ---------------------------------------------------------------------------
# Freeze: compute thresholds from OOF and lock the model choice
# ---------------------------------------------------------------------------

def run_freeze(
    model_name: str,
    use_history: bool,
    output: Path = OUTPUT_DIR,
) -> dict:
    """Compute thresholds from OOF and save selecao_congelada.json.

    Requires oof.npz and cv_summary.json (produced by --fase cv).
    """
    output = Path(output)
    oof_path = output / "oof.npz"
    summary_path = output / "cv_summary.json"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF não encontrado: {oof_path}. Execute --fase cv primeiro.")
    if not summary_path.exists():
        raise FileNotFoundError(f"Resumo não encontrado: {summary_path}. Execute --fase cv primeiro.")

    cv_summary = json.loads(summary_path.read_text())
    npz = np.load(oof_path)
    key = f"{model_name}__{int(use_history)}"
    scores = npz[f"{key}__scores"]
    labels = npz[f"{key}__labels"].astype(int)

    # Threshold (a): top-20% política de contato
    n = len(scores)
    k = max(1, int(np.ceil(0.2 * n)))
    threshold_top20 = float(np.sort(scores)[-k])

    # Threshold (b): maximiza F1
    from sklearn.metrics import precision_recall_curve
    precision_arr, recall_arr, thr_arr = precision_recall_curve(labels, scores)
    f1_arr = np.divide(
        2 * precision_arr[:-1] * recall_arr[:-1],
        precision_arr[:-1] + recall_arr[:-1],
        out=np.zeros(len(thr_arr)),
        where=(precision_arr[:-1] + recall_arr[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1_arr))
    threshold_f1 = float(thr_arr[best_idx])

    # Best params for this model
    if model_name == "dummy":
        params = {}
    elif model_name == "logistic":
        params = _LOGISTIC_PARAMS
    elif model_name == "decision_tree":
        params = _TREE_PARAMS
    else:
        params = cv_summary["best_params"].get(key, {})

    info = {
        "model": model_name,
        "use_history": use_history,
        "params": params,
        "limiar_top20": threshold_top20,
        "limiar_f1": threshold_f1,
        "limiar_principal": threshold_top20,
        "data_congelamento": datetime.now(timezone.utc).isoformat(),
        "oof_n_rows": int(n),
        "oof_ap": float(average_precision_score(labels, scores)),
        "nota": "limiar_principal = top-20% política de contato; limiar_f1 é alternativa",
    }
    frozen_path = output / "selecao_congelada.json"
    frozen_path.write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# Fase 2: avaliação única no teste
# ---------------------------------------------------------------------------

def run_final(
    df: pd.DataFrame,
    output: Path = OUTPUT_DIR,
) -> dict:
    """Fase 2 (#17): teste único, SHAP e análise de erros.

    Requires reports/modelo-final/selecao_congelada.json to exist.
    Saves: teste_final.csv, teste_final.json, shap_top10.csv,
           figures/shap_bar.png, figures/shap_beeswarm.png, error_analysis.csv
    """
    output = Path(output)
    frozen_path = output / "selecao_congelada.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"selecao_congelada.json não encontrado em {frozen_path}. "
            "Execute --fase freeze antes de --fase final."
        )

    frozen = json.loads(frozen_path.read_text())
    model_name = frozen["model"]
    use_history = bool(frozen["use_history"])
    params = frozen["params"]
    threshold = float(frozen["limiar_principal"])

    cv_summary = json.loads((output / "cv_summary.json").read_text())

    train_idx, test_idx = split_by_patient(df)
    dev_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Retrieve best params for RF/XGBoost (same use_history as final model)
    rf_params = cv_summary["best_params"].get(f"random_forest__{int(use_history)}", _RF_GRID[0])
    xgb_params = cv_summary["best_params"].get(f"xgboost__{int(use_history)}", _XGB_GRID[0])

    all_candidates = [
        ("dummy", {}),
        ("logistic", _LOGISTIC_PARAMS),
        ("decision_tree", _TREE_PARAMS),
        ("random_forest", rf_params),
        ("xgboost", xgb_params),
    ]

    cols = get_feature_columns(use_history)
    X_dev = dev_df[cols]
    y_dev = dev_df["no_show"]
    X_test = test_df[cols]
    y_test = np.asarray(test_df["no_show"])

    test_rows = []
    final_pipeline = None
    for name, cand_params in all_candidates:
        pipe = _build_pipeline(name, cand_params, use_history)
        pipe.fit(X_dev, y_dev)
        if name == model_name:
            final_pipeline = pipe
        pos_idx = list(pipe.classes_).index(1)
        scores = pipe.predict_proba(X_test)[:, pos_idx]

        metrics = _fold_metrics(y_test, scores)
        preds_thr = (scores >= threshold).astype(int)
        f1 = float(f1_score(y_test, preds_thr, zero_division=0))
        from sklearn.metrics import confusion_matrix as _cm
        conf = _cm(y_test, preds_thr, labels=[0, 1]).tolist()
        test_rows.append({
            "candidate": name,
            "use_history": use_history,
            **metrics,
            "f1_at_threshold": f1,
            "confusion_matrix": conf,
        })

    # Save test results (no individual scores)
    test_df_out = pd.DataFrame(test_rows)
    test_df_safe = test_df_out.drop(columns=["confusion_matrix"])
    test_df_safe.to_csv(output / "teste_final.csv", index=False)

    def _json_default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(type(obj))

    (output / "teste_final.json").write_text(
        json.dumps(test_rows, indent=2, default=_json_default)
    )

    # SHAP for the final model
    if model_name in ("random_forest", "xgboost", "decision_tree") and final_pipeline is not None:
        _run_shap(final_pipeline, dev_df, use_history, model_name, output)

    # Error analysis (aggregated only, no individual scores)
    _error_analysis(test_df, final_pipeline, use_history, threshold, output)

    return {"frozen": frozen, "test_results": test_rows}


def _run_shap(
    pipeline: Pipeline,
    dev_df: pd.DataFrame,
    use_history: bool,
    model_name: str,
    output: Path,
) -> None:
    """SHAP via TreeExplainer on a sample of development data."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)

    cols = get_feature_columns(use_history)
    X_dev = dev_df[cols]
    n_sample = min(3000, len(X_dev))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_dev), n_sample, replace=False)
    X_sample = X_dev.iloc[idx]

    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X_sample)

    numeric_cols = list(NUMERIC_FEATURES) + (list(HISTORY_FEATURES) if use_history else [])
    ohe = preprocessor.named_transformers_["categorical"]
    ohe_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = numeric_cols + ohe_names

    model_step = pipeline.named_steps["model"]
    actual_model = model_step._clf if isinstance(model_step, _AutoWeightXGB) else model_step

    # ColumnTransformer with OHE returns a sparse matrix; convert to dense for SHAP
    if hasattr(X_transformed, "toarray"):
        X_arr = X_transformed.toarray()
    else:
        X_arr = np.asarray(X_transformed)

    explainer = shap.TreeExplainer(actual_model)
    sv = explainer.shap_values(X_arr, check_additivity=False)

    # For RF binary: sv is list[2 arrays]; for XGBoost: single array
    shap_vals = sv[1] if isinstance(sv, list) else sv

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top10_idx = np.argsort(mean_abs)[-10:][::-1]
    top10_names = [feature_names[i] for i in top10_idx]
    top10_vals = mean_abs[top10_idx]

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(10), top10_vals[::-1])
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_bar.png", dpi=150)
    plt.close(fig)

    # Beeswarm using summary_plot (stable across shap versions)
    X_sample_df = pd.DataFrame(X_arr, columns=feature_names)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample_df, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150)
    plt.close("all")

    top10_df = pd.DataFrame({
        "rank": range(1, 11),
        "feature": top10_names,
        "mean_abs_shap": top10_vals,
    })
    top10_df.to_csv(output / "shap_top10.csv", index=False)


def _error_analysis(
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    use_history: bool,
    threshold: float,
    output: Path,
) -> None:
    """FP/FN rates per group — aggregated only, no individual scores saved."""
    if pipeline is None:
        return

    cols = get_feature_columns(use_history)
    X_test = test_df[cols]
    y_test = np.asarray(test_df["no_show"])

    pos_idx = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(X_test)[:, pos_idx]
    preds = (scores >= threshold).astype(int)

    analysis = pd.DataFrame({
        "age": test_df["age"].to_numpy(),
        "waiting_days": test_df["waiting_days"].to_numpy(),
        "weekday": test_df["weekday"].to_numpy(),
        "y_true": y_test,
        "y_pred": preds,
    })
    if use_history and "has_history" in test_df.columns:
        analysis["has_history"] = test_df["has_history"].to_numpy()

    analysis["fp"] = ((y_test == 0) & (preds == 1)).astype(int)
    analysis["fn"] = ((y_test == 1) & (preds == 0)).astype(int)
    analysis["tn"] = ((y_test == 0) & (preds == 0)).astype(int)
    analysis["tp"] = ((y_test == 1) & (preds == 1)).astype(int)

    analysis["waiting_bin"] = pd.cut(
        analysis["waiting_days"],
        bins=[-1, 0, 7, 30, 90, 10000],
        labels=["0d", "1-7d", "8-30d", "31-90d", ">90d"],
    )
    analysis["age_bin"] = pd.cut(
        analysis["age"],
        bins=[-1, 17, 39, 59, 200],
        labels=["0-17", "18-39", "40-59", "60+"],
    )

    rows = []
    dimensions = [("waiting_days", "waiting_bin"), ("age", "age_bin"), ("weekday", "weekday")]
    if use_history and "has_history" in analysis.columns:
        dimensions.append(("has_history", "has_history"))

    for dim_name, col in dimensions:
        grouped = analysis.groupby(col, observed=True)[["fp", "fn", "tp", "tn"]].sum()
        denom_fp = (grouped["fp"] + grouped["tn"]).clip(lower=1)
        denom_fn = (grouped["fn"] + grouped["tp"]).clip(lower=1)
        grouped["fp_rate"] = grouped["fp"] / denom_fp
        grouped["fn_rate"] = grouped["fn"] / denom_fn
        for grp, row in grouped.iterrows():
            rows.append({
                "dimension": dim_name,
                "group": str(grp),
                "n_fp": int(row["fp"]),
                "n_fn": int(row["fn"]),
                "n_tp": int(row["tp"]),
                "n_tn": int(row["tn"]),
                "fp_rate": round(float(row["fp_rate"]), 4),
                "fn_rate": round(float(row["fn_rate"]), 4),
            })

    pd.DataFrame(rows).to_csv(output / "error_analysis.csv", index=False)


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers
# ---------------------------------------------------------------------------

def _make_preprocessor_explicit(
    numeric_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    """ColumnTransformer for explicit column lists (used in sensitivity CV loops)."""
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), list(numeric_cols)))
    if categorical_cols:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols))
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _group_rare_neighbourhoods(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    threshold: int = 100,
) -> tuple:
    """Replace neighbourhoods with < threshold rows in train_df with 'OUTROS'.

    Applied inside each fold so that the grouping uses only fold-train counts
    (no leakage from validation). Val rows whose neighbourhood was not seen in
    train are also mapped to 'OUTROS'.
    """
    counts = train_df["neighbourhood"].value_counts()
    mapping = {n: (n if c >= threshold else "OUTROS") for n, c in counts.items()}
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["neighbourhood"] = train_df["neighbourhood"].map(mapping).fillna("OUTROS")
    val_df["neighbourhood"] = val_df["neighbourhood"].map(mapping).fillna("OUTROS")
    return train_df, val_df


def run_neighbourhood_sensitivity(
    df: pd.DataFrame,
    output: Path = OUTPUT_DIR,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Sensibilidade de AP ao bairro em CV (nunca toca o teste).

    Variantes avaliadas nos mesmos 5 folds do modelo final:
    - modelo_final: AP de referência do cv_candidatos.csv
    - sem_neighbourhood: exclui neighbourhood das features
    - bairros_agrupados: bairros com < 100 agendamentos no treino do fold → 'OUTROS'

    Salva: reports/modelo-final/sensibilidade_bairro.csv
    """
    frozen_path = output / "selecao_congelada.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"selecao_congelada.json não encontrado: {frozen_path}")

    frozen = json.loads(frozen_path.read_text())
    xgb_params = frozen["params"]

    train_idx, _test_idx = split_by_patient(df)
    dev_df = df.iloc[train_idx].reset_index(drop=True)
    folds = _get_folds(dev_df, n_splits=n_splits)

    # Read reference AP from saved CV results
    ref_ap_mean, ref_ap_std = 0.3722, 0.0058
    cv_path = output / "cv_candidatos.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        mask = (cv_df["candidate"] == "xgboost") & (cv_df["use_history"].astype(bool))
        if mask.any():
            row = cv_df[mask].iloc[0]
            ref_ap_mean = round(float(row["ap_mean"]), 4)
            ref_ap_std = round(float(row["ap_std"]), 4)

    base_numeric = list(NUMERIC_FEATURES) + list(HISTORY_FEATURES)
    all_cats = list(CATEGORICAL_FEATURES)
    no_neigh_cats = [c for c in all_cats if c != "neighbourhood"]

    variants = [
        # (name, group_rare, numeric_cols, categorical_cols)
        ("sem_neighbourhood", False, base_numeric, no_neigh_cats),
        ("bairros_agrupados", True,  base_numeric, all_cats),
    ]

    result_rows = [
        {"variante": "modelo_final", "ap_mean": ref_ap_mean, "ap_std": ref_ap_std}
    ]

    for variant_name, group_rare, num_cols, cat_cols in variants:
        feature_cols = num_cols + cat_cols
        fold_aps = []
        for train_fold_idx, val_fold_idx in folds:
            train_fold = dev_df.iloc[train_fold_idx].copy()
            val_fold = dev_df.iloc[val_fold_idx].copy()

            if group_rare:
                train_fold, val_fold = _group_rare_neighbourhoods(train_fold, val_fold)

            X_train = train_fold[feature_cols]
            y_train = train_fold["no_show"].to_numpy()
            X_val = val_fold[feature_cols]
            y_val = val_fold["no_show"].to_numpy()

            pipe = Pipeline([
                ("preprocess", _make_preprocessor_explicit(num_cols, cat_cols)),
                ("model", _AutoWeightXGB(xgb_params=xgb_params)),
            ])
            pipe.fit(X_train, y_train)

            pos_idx = list(pipe.classes_).index(1)
            scores = pipe.predict_proba(X_val)[:, pos_idx]
            fold_aps.append(_fold_metrics(y_val, scores)["ap"])

        arr = np.array(fold_aps)
        result_rows.append({
            "variante": variant_name,
            "ap_mean": round(float(arr.mean()), 4),
            "ap_std": round(float(arr.std(ddof=1)), 4),
        })

    sens_df = pd.DataFrame(result_rows)
    sens_df.to_csv(output / "sensibilidade_bairro.csv", index=False)
    return sens_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_shap_by_variable(
    df: pd.DataFrame,
    output: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Recalcula SHAP no desenvolvimento e agrega por variável original.

    Nunca toca o teste. Salva:
      shap_por_variavel.csv  — importância agregada por variável
      figures/shap_por_variavel.png — gráfico de barras ordenado
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap as shap_lib

    output = Path(output)
    frozen_path = output / "selecao_congelada.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"selecao_congelada.json não encontrado: {frozen_path}")

    frozen = json.loads(frozen_path.read_text())
    model_name = frozen["model"]
    use_history = bool(frozen["use_history"])
    params = frozen["params"]

    train_idx, _test_idx = split_by_patient(df)
    dev_df = df.iloc[train_idx].reset_index(drop=True)

    cols = get_feature_columns(use_history)
    X_dev = dev_df[cols]
    y_dev = dev_df["no_show"]

    pipe = _build_pipeline(model_name, params, use_history)
    pipe.fit(X_dev, y_dev)

    n_sample = min(3000, len(X_dev))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_dev), n_sample, replace=False)
    X_sample = X_dev.iloc[idx]

    preprocessor = pipe.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_arr = X_transformed.toarray()
    else:
        X_arr = np.asarray(X_transformed)

    numeric_cols = list(NUMERIC_FEATURES) + (list(HISTORY_FEATURES) if use_history else [])
    ohe = preprocessor.named_transformers_["categorical"]
    ohe_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = numeric_cols + ohe_names

    model_step = pipe.named_steps["model"]
    actual_model = model_step._clf if isinstance(model_step, _AutoWeightXGB) else model_step

    explainer = shap_lib.TreeExplainer(actual_model)
    sv = explainer.shap_values(X_arr, check_additivity=False)
    shap_vals = sv[1] if isinstance(sv, list) else sv

    # Aggregate: per row, sum SHAP values WITH SIGN across all OHE columns of the
    # same original variable; then compute mean(|per-row sum|) across rows.
    # For numeric/history features there is one column → same formula degenerates
    # to mean(|SHAP|) for that single column.
    all_original = numeric_cols + list(CATEGORICAL_FEATURES)
    var_importance: dict = {}
    for var in all_original:
        if var in numeric_cols:
            if var in feature_names:
                i = feature_names.index(var)
                # single column: sum-per-row = the value itself
                var_importance[var] = float(np.abs(shap_vals[:, i]).mean())
        else:
            prefix = f"{var}_"
            indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
            if indices:
                # Step 1: for each row, sum SHAP with sign across OHE columns
                per_row = shap_vals[:, indices].sum(axis=1)  # shape (n_sample,)
                # Step 2: mean of absolute value across rows
                var_importance[var] = float(np.abs(per_row).mean())

    sorted_items = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
    shap_var_df = pd.DataFrame([
        {"rank": i + 1, "variable": var, "mean_abs_shap_agregado": round(val, 6)}
        for i, (var, val) in enumerate(sorted_items)
    ])
    shap_var_df.to_csv(output / "shap_por_variavel.csv", index=False)

    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)
    names = [v for v, _ in sorted_items]
    vals = [val for _, val in sorted_items]
    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.42)))
    ax.barh(range(len(names)), vals[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Soma de mean |SHAP| (pós-OHE, amostra 3000)")
    ax.set_title(f"Importância por variável original — {model_name}")
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_por_variavel.png", dpi=150)
    plt.close(fig)

    return shap_var_df


def add_falta_rate_to_error_analysis(output: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Adiciona taxa de falta real a error_analysis.csv (aritmética sobre contagens)."""
    path = output / "error_analysis.csv"
    df = pd.read_csv(path)
    df["n_total"] = df["n_fp"] + df["n_fn"] + df["n_tp"] + df["n_tn"]
    df["n_faltas"] = df["n_fn"] + df["n_tp"]
    df["falta_rate"] = (df["n_faltas"] / df["n_total"].clip(lower=1)).round(4)
    df.to_csv(path, index=False)
    return df


def _load_df(raw_path: str) -> pd.DataFrame:
    df = build_features(clean(load_raw(raw_path)))
    df = add_patient_history(df)
    return df


def _print_cv_summary(cv_df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    no_hist = cv_df[~cv_df.use_history][
        ["candidate", "ap_mean", "ap_std", "roc_auc_mean",
         "precision_at_20_mean", "recall_at_20_mean", "best_params"]
    ].copy()
    print("\n=== CV sem histórico (sorted by AP) ===")
    print(no_hist.to_string(index=False))

    with_hist = cv_df[cv_df.use_history][
        ["candidate", "ap_mean", "ap_std", "roc_auc_mean",
         "precision_at_20_mean", "recall_at_20_mean", "best_params"]
    ].copy()
    print("\n=== CV com histórico (sorted by AP) ===")
    print(with_hist.to_string(index=False))

    print("\n=== Ablação do histórico (ΔAP) ===")
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fase",
        choices=["cv", "freeze", "final", "shap", "sensibilidade"],
        required=True,
        help="cv: Fase 1 validação cruzada; freeze: congela escolha; final: Fase 2 teste; "
             "shap: agrega SHAP por variável original; sensibilidade: análise de bairro em CV",
    )
    parser.add_argument("--raw", default="data/raw/KaggleV2-May-2016.csv")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--model", default=None, help="Para --fase freeze: nome do modelo escolhido")
    parser.add_argument("--history", action="store_true", help="Para --fase freeze: usar histórico")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output)

    if args.fase == "cv":
        df = _load_df(args.raw)
        cv_df, ablation_df, _ = run_cv(df, output=out, n_splits=args.splits)
        _print_cv_summary(cv_df, ablation_df)
        print(f"\nArtefatos salvos em: {out}")

    elif args.fase == "freeze":
        if args.model is None:
            parser.error("--fase freeze requer --model MODEL")
        info = run_freeze(args.model, args.history, output=out)
        print(json.dumps(info, indent=2))

    elif args.fase == "final":
        df = _load_df(args.raw)
        result = run_final(df, output=out)
        rows = result["test_results"]
        print("\n=== Avaliação no teste reservado ===")
        safe = [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in rows]
        print(pd.DataFrame(safe).to_string(index=False))
        print(f"\nArtefatos salvos em: {out}")

    elif args.fase == "shap":
        df = _load_df(args.raw)
        shap_var_df = run_shap_by_variable(df, output=out)
        print("\n=== SHAP por variável original ===")
        print(shap_var_df.to_string(index=False))
        ea_df = add_falta_rate_to_error_analysis(output=out)
        print("\n=== error_analysis.csv atualizado com falta_rate ===")
        print(ea_df.to_string(index=False))
        print(f"\nArtefatos salvos em: {out}")

    elif args.fase == "sensibilidade":
        df = _load_df(args.raw)
        sens_df = run_neighbourhood_sensitivity(df, output=out, n_splits=args.splits)
        print("\n=== Sensibilidade ao bairro (CV, 5 folds) ===")
        print(sens_df.to_string(index=False))
        print(f"\nArtefatos salvos em: {out}")
