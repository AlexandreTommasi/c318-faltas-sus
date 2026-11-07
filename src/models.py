"""Treinamento, tuning e persistência de modelos."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def get_baseline() -> BaseEstimator:
    """Retorna um classificador de linha de base (DummyClassifier).

    Returns:
        Instância do classificador não treinada.
    """
    return DummyClassifier(strategy="prior", random_state=42)


def get_logistic_regression() -> BaseEstimator:
    """Retorna uma Regressão Logística configurada para o problema.

    Returns:
        Instância do classificador não treinada.
    """
    return LogisticRegression(max_iter=2000, random_state=42)


def get_random_forest() -> BaseEstimator:
    """Retorna um Random Forest configurado para o problema.

    Returns:
        Instância do classificador não treinada.
    """
    ...


def get_xgboost() -> BaseEstimator:
    """Retorna um XGBClassifier configurado para o problema.

    Returns:
        Instância do classificador não treinada.
    """
    ...


def train(model: BaseEstimator, X_train, y_train) -> BaseEstimator:
    """Treina o modelo e retorna a instância ajustada.

    Args:
        model: Classificador scikit-learn compatível.
        X_train: Atributos de treino.
        y_train: Rótulos de treino.

    Returns:
        Modelo treinado.
    """
    return model.fit(X_train, y_train)


def save_model(model: BaseEstimator, path: str) -> None:
    """Serializa o modelo em disco com joblib.

    Args:
        model: Modelo treinado.
        path: Caminho de destino (ex.: models/rf.joblib).
    """
    ...


def load_model(path: str) -> BaseEstimator:
    """Carrega um modelo serializado do disco.

    Args:
        path: Caminho do arquivo joblib.

    Returns:
        Modelo carregado.
    """
    ...


def split_by_patient(df, test_size=0.2):
    """Partição reprodutível por paciente; sem pacientes compartilhados."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return next(splitter.split(df, groups=df["patient_id"]))


def split_temporal(df, cutoff="2016-05-25"):
    """Treino com desfechos conhecidos antes do corte; teste agendado após.

    Consultas pendentes no corte ficam fora do treino. Pacientes podem se
    repetir nesta checagem, que representa o uso futuro na mesma população.
    """
    import numpy as np
    import pandas as pd

    boundary = pd.Timestamp(cutoff, tz="UTC")
    return (np.flatnonzero(df["appointment_day"] < boundary),
            np.flatnonzero(df["scheduled_day"] >= boundary))


def run_baselines(df, train_indices, test_indices):
    """Pré-processamento ajustado no treino para ambos os classificadores."""
    from sklearn.pipeline import make_pipeline
    from src.features import encode_categoricals, split_features_target
    from src.evaluate import compute_metrics

    X, y = split_features_target(df)
    results = {}
    for name, model in [("dummy", get_baseline()), ("logistic", get_logistic_regression())]:
        pipeline = make_pipeline(encode_categoricals(X), model)
        pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
        results[name] = compute_metrics(pipeline, X.iloc[test_indices], y.iloc[test_indices])
    return results
