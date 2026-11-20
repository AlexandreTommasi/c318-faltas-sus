"""S4 (#16) e S5 (#20): python -m src.model_selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.metrics import precision_recall_curve, make_scorer, precision_score

from src.data import clean, load_raw
from src.features import build_features, split_features_target, encode_categoricals
from src.models import split_by_patient
from src.evaluate import compute_metrics


def patient_folds(X, y, groups, n_splits=5):
    """Estratificação aproximada, sem compartilhar pacientes entre folds."""
    folds = list(StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    ).split(X, y, groups))
    for train, valid in folds:
        if set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise ValueError("Pacientes compartilhados entre folds")
        if y.iloc[train].nunique() != 2 or y.iloc[valid].nunique() != 2:
            raise ValueError("Cada fold precisa conter ambas as classes")
    return folds


def select_threshold(y, scores):
    """Maximiza F1 na validação; empates favorecem o maior limiar."""
    y, scores = np.asarray(y), np.asarray(scores)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("A validação precisa conter as duas classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilidades devem estar entre zero e um")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    table = pd.DataFrame(dict(threshold=thresholds, precision=precision[:-1],
                              recall=recall[:-1], f1=f1[:-1]))
    best = table.sort_values(['f1', 'threshold'], ascending=False).iloc[0]
    return float(best.threshold), table


class ThresholdClassifier:
    """Aplica limiar fixo a uma pipeline já ajustada, sem novo treinamento."""
    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = threshold
        self.classes_ = estimator.classes_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        scores = self.predict_proba(X)[:, list(self.classes_).index(1)]
        return (scores >= self.threshold).astype(int)


def run(df, output, n_splits=5, n_jobs=1, evaluate_test=False):
    """Reserva teste por paciente e validação independente para o limiar.

    O ajuste final usa somente o treino de busca: não refaz fit depois de
    calibrar o limiar, preservando a escala das probabilidades selecionadas.
    O teste reservado não é avaliado por padrão (evaluate_test=False);
    sua avaliação cabe à tarefa de modelo final (#17).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    development, test = split_by_patient(df)
    development_df = df.iloc[development].reset_index(drop=True)
    fit, validation = split_by_patient(development_df)
    X, y = split_features_target(development_df)
    X_fit, y_fit = X.iloc[fit], y.iloc[fit]
    groups = development_df.patient_id.iloc[fit]
    folds = patient_folds(X_fit, y_fit, groups, n_splits)
    scoring = dict(ap='average_precision', roc_auc='roc_auc',
                   precision=make_scorer(precision_score, zero_division=0),
                   recall='recall', f1='f1')
    candidates = {
        'logistic': (LogisticRegression(max_iter=2000, random_state=42),
                     {'model__C': [0.01, 0.1, 1.0, 10.0]}),
        'decision_tree': (DecisionTreeClassifier(random_state=42),
                          {'model__max_depth': [3, 5, 8, None],
                           'model__min_samples_leaf': [20, 100, 300]}),
    }
    if evaluate_test:
        X_test, y_test = split_features_target(df.iloc[test])
    results, cv_rows = {}, []
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([('preprocess', encode_categoricals(X_fit)),
                             ('model', estimator)])
        baseline = cross_validate(pipeline, X_fit, y_fit, cv=folds,
                                  scoring=scoring, n_jobs=n_jobs, error_score='raise')
        for fold in range(n_splits):
            cv_rows.append(dict(model=name, fold=fold + 1,
                                **{metric: float(baseline['test_' + metric][fold])
                                   for metric in scoring}))
        search = GridSearchCV(pipeline, grid, scoring=scoring, refit='ap',
                              cv=folds, n_jobs=n_jobs, error_score='raise')
        search.fit(X_fit, y_fit)
        pd.DataFrame(search.cv_results_).to_csv(output / f'{name}_search.csv', index=False)
        model = search.best_estimator_
        scores = model.predict_proba(X.iloc[validation])[:, list(model.classes_).index(1)]
        threshold, curve = select_threshold(y.iloc[validation], scores)
        curve.to_csv(output / f'{name}_thresholds.csv', index=False)
        entry = {
            'best_params': search.best_params_,
            'best_cv_ap': float(search.best_score_),
            'threshold': threshold,
            'validation_best_f1': float(curve.f1.max()),
        }
        if evaluate_test:
            entry['test_default'] = compute_metrics(model, X_test, y_test)
            entry['test_adjusted'] = compute_metrics(
                ThresholdClassifier(model, threshold), X_test, y_test)
        results[name] = entry
    pd.DataFrame(cv_rows).to_csv(output / 's4_cv.csv', index=False)
    results['metadata'] = dict(seed=42, folds=n_splits, rows=len(df),
                               fit_rows=len(fit), validation_rows=len(validation),
                               test_rows=len(test), sklearn=sklearn.__version__,
                               numpy=np.__version__, pandas=pd.__version__,
                               features=list(X.columns),
                               threshold_objective='F1 on independent validation',
                               search_objective='average_precision',
                               history_used=False)
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/KaggleV2-May-2016.csv')
    parser.add_argument('--output', default='reports/sprints-4-5')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    df = build_features(clean(load_raw(args.raw)))
    print(json.dumps(run(df, args.output, n_jobs=args.jobs), indent=2))
