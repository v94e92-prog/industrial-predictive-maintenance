"""Модели: у всех score() возвращает число, чем больше, тем подозрительнее."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DutyRule:
    """Базовое инженерное правило: доля работы под нагрузкой за последний час.

    Порог задан заранее: компрессор не должен час подряд работать под нагрузкой.
    """

    name = "Правило: нагрузка за час"
    needs_failures = False
    fixed_threshold = 0.9
    columns = ("duty_1h",)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DutyRule:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return X["duty_1h"].to_numpy()


class LowPressureSignal:
    """То, что на оборудовании уже есть: штатный сигнал LPS о падении давления ниже 7 бар."""

    name = "Штатный сигнал LPS"
    needs_failures = False
    fixed_threshold = 0.01
    columns = ("lps_1h",)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LowPressureSignal:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return X["lps_1h"].to_numpy()


class RobustZ:
    """Наибольшее отклонение признака от нормы в единицах медианного разброса."""

    name = "Статистика: robust z"
    needs_failures = False
    fixed_threshold = None
    columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RobustZ:
        healthy = X[y == 0]
        self.median_ = healthy.median()
        mad = (healthy - self.median_).abs().median()
        # Запасная оценка разброса для признаков, у которых MAD равен нулю (например, LPS).
        self.scale_ = (1.4826 * mad).where(mad > 1e-9, healthy.std()).replace(0, 1.0)
        return self

    def deviations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Отклонения по каждому признаку: из них видно, что именно вызвало тревогу."""
        return ((X - self.median_) / self.scale_).abs()

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.deviations(X).max(axis=1).to_numpy()


class Forest:
    """Isolation Forest, обученный на периодах без зарегистрированных отказов."""

    name = "Isolation Forest"
    needs_failures = False
    fixed_threshold = None
    columns = None

    def __init__(self, seed: int = 42) -> None:
        self.model = IsolationForest(
            n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forest:
        self.model.fit(X[y == 0])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X)


class LogReg:
    """Логистическая регрессия на размеченных отказах. Нужен хотя бы один отказ в обучении."""

    name = "Логистическая регрессия"
    needs_failures = True
    fixed_threshold = None
    columns = None

    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogReg:
        self.model.fit(X, y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(X)


MODELS = (LowPressureSignal, DutyRule, RobustZ, Forest, LogReg)
