"""Métricas de avaliação e geração de relatórios."""

import pandas as pd
from sklearn.base import BaseEstimator


def compute_metrics(model: BaseEstimator, X_test, y_test) -> dict:
    """Calcula métricas de classificação para o conjunto de teste.

    Inclui: acurácia, precisão, recall, F1, ROC-AUC e PR-AUC.

    Args:
        model: Modelo treinado.
        X_test: Atributos de teste.
        y_test: Rótulos verdadeiros de teste.

    Returns:
        Dicionário com nome da métrica → valor.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix)

    y = np.asarray(y_test)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    result = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "pr_auc_ap": float(average_precision_score(y, scores)) if y.sum() else None,
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "n_test": len(y),
        "prevalence": float(y.mean()),
    }
    k = max(1, int(np.ceil(0.2 * len(y))))
    # Empates recebem o valor esperado sob desempate aleatório, sem favorecer ordem.
    threshold = np.sort(scores)[-k]
    above, tied = scores > threshold, scores == threshold
    captured = float(y[above].sum() + y[tied].mean() * (k - above.sum()))
    result.update(contacts=k, captured_no_shows=captured,
                  precision_at_20=captured / k,
                  recall_at_20=captured / y.sum() if y.sum() else 0.0,
                  random_expected_no_shows=float(k * y.mean()))
    return result


def plot_confusion_matrix(model: BaseEstimator, X_test, y_test, save_path: str = None):
    """Exibe e opcionalmente salva a matriz de confusão.

    Args:
        model: Modelo treinado.
        X_test: Atributos de teste.
        y_test: Rótulos verdadeiros.
        save_path: Se fornecido, salva a figura nesse caminho.
    """
    ...


def plot_roc_curve(model: BaseEstimator, X_test, y_test, save_path: str = None):
    """Plota a curva ROC do modelo.

    Args:
        model: Modelo treinado com predict_proba.
        X_test: Atributos de teste.
        y_test: Rótulos verdadeiros.
        save_path: Se fornecido, salva a figura nesse caminho.
    """
    ...


def compare_models(results: dict) -> pd.DataFrame:
    """Consolida métricas de múltiplos modelos em uma tabela comparativa.

    Args:
        results: Dicionário {nome_do_modelo: dict_de_metricas}.

    Returns:
        DataFrame com modelos nas linhas e métricas nas colunas.
    """
    ...
