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
    ...


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
