"""Treinamento, tuning e persistência de modelos."""

from sklearn.base import BaseEstimator


def get_baseline() -> BaseEstimator:
    """Retorna um classificador de linha de base (DummyClassifier).

    Returns:
        Instância do classificador não treinada.
    """
    ...


def get_logistic_regression() -> BaseEstimator:
    """Retorna uma Regressão Logística configurada para o problema.

    Returns:
        Instância do classificador não treinada.
    """
    ...


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
    ...


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
