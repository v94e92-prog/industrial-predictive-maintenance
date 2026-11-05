"""Оценка по событиям и проверка с движением вперёд по времени.

Заводу важны не строки, а события: пойман ли отказ, насколько рано пришла тревога
и сколько раз система тревожила зря.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm.data import Failure
from pdm.features import BARS_PER_HOUR

HORIZON = pd.Timedelta(hours=6)
# Тревогу считаем относящейся к отказу, если она пришла не раньше чем за сутки до него.
MATCH_BEFORE = pd.Timedelta(hours=24)
MATCH_AFTER = pd.Timedelta(hours=6)
CONFIRM_BARS = 6  # оценка должна держаться выше порога 30 минут подряд
MERGE_GAP = pd.Timedelta(hours=2)
FOLD_STARTS = tuple(
    pd.Timestamp(d)
    for d in ("2020-04-01", "2020-05-01", "2020-06-01", "2020-07-01", "2020-08-01")
)


def make_labels(index: pd.DatetimeIndex, failures: tuple[Failure, ...],
                horizon: pd.Timedelta = HORIZON) -> pd.Series:
    """1, если отказ идёт сейчас или начнётся в пределах горизонта."""
    y = pd.Series(0, index=index, dtype=int)
    for f in failures:
        y[(index >= f.start - horizon) & (index <= f.end)] = 1
    return y


def confirmed_alarms(scores: pd.Series, threshold: float,
                     confirm_bars: int = CONFIRM_BARS) -> pd.Series:
    """Тревога только если оценка выше порога confirm_bars баров подряд. NaN прерывает серию."""
    above = (scores >= threshold).astype(float).where(scores.notna())
    return above.rolling(confirm_bars, min_periods=confirm_bars).min().fillna(0).astype(bool)


def alarm_episodes(alarms: pd.Series, merge_gap: pd.Timedelta = MERGE_GAP) -> list[tuple]:
    """Склеивает тревоги в эпизоды: начало и конец каждого."""
    times = alarms.index[alarms.to_numpy()]
    if len(times) == 0:
        return []
    episodes, start, previous = [], times[0], times[0]
    for t in times[1:]:
        if t - previous > merge_gap:
            episodes.append((start, previous))
            start = t
        previous = t
    episodes.append((start, previous))
    return episodes


@dataclass
class EventResult:
    """Результат модели по событиям за весь проверяемый период."""

    delays_min: dict[int, float | None]  # отказ -> минуты от начала до тревоги (минус = заранее)
    false_episodes: list[tuple]
    eval_days: float
    total_episodes: int = 0  # все эпизоды тревог: на каждый нужен осмотр

    @property
    def caught(self) -> int:
        return sum(v is not None for v in self.delays_min.values())

    @property
    def false_per_30d(self) -> float:
        return len(self.false_episodes) / self.eval_days * 30 if self.eval_days else 0.0


def score_events(alarms: pd.Series, failures: tuple[Failure, ...]) -> EventResult:
    """Сопоставляет тревоги с отказами. Лишними считаются эпизоды вне окон всех отказов."""
    delays: dict[int, float | None] = {}
    for number, f in enumerate(failures, 1):
        window = alarms[(alarms.index >= f.start - MATCH_BEFORE) & (alarms.index <= f.end)]
        fired = window.index[window.to_numpy()]
        delays[number] = (fired[0] - f.start).total_seconds() / 60 if len(fired) else None

    def related(episode: tuple) -> bool:
        first, last = episode
        return any(
            first <= f.end + MATCH_AFTER and last >= f.start - MATCH_BEFORE for f in failures
        )

    episodes = alarm_episodes(alarms)
    false = [e for e in episodes if not related(e)]
    return EventResult(delays, false, len(alarms) / (BARS_PER_HOUR * 24), len(episodes))


def walk_forward(features: pd.DataFrame, failures: tuple[Failure, ...], model_cls,
                 quantile: float = 0.995,
                 fold_starts: tuple[pd.Timestamp, ...] = FOLD_STARTS,
                 confirm_bars: int = CONFIRM_BARS) -> tuple[pd.Series, dict]:
    """Каждый месяц обучаемся только на прошлом и проверяем на следующем месяце.

    Метки в обучении обрезаны на горизонт до границы, чтобы не подсмотреть будущий отказ.
    Порог берём из оценок на здоровых барах обучения, тестовые данные его не трогают.
    Возвращает ряд тревог и сведения о месяцах, где модель обучить было нельзя.
    """
    y_all = make_labels(features.index, failures)
    # Модель получает только свои признаки: правилу незачем ждать суточного окна.
    if model_cls.columns is not None:
        features = features[list(model_cls.columns)]
    valid = features.notna().all(axis=1)
    pieces, skipped = [], []
    ends = (*fold_starts[1:], features.index[-1] + pd.Timedelta(minutes=5))
    for start, end in zip(fold_starts, ends):
        cutoff = start - HORIZON
        train = valid & (features.index < cutoff)
        X_train, y_train = features[train], y_all[train]
        test_index = features.index[(features.index >= start) & (features.index < end)]
        if model_cls.needs_failures and y_train.sum() == 0:
            skipped.append(str(start.date()))
            pieces.append(pd.Series(False, index=test_index))
            continue
        model = model_cls().fit(X_train, y_train)
        threshold = model_cls.fixed_threshold
        if threshold is None:
            threshold = float(np.quantile(model.score(X_train[y_train == 0]), quantile))
        test_valid = test_index[valid.loc[test_index].to_numpy()]
        scores = pd.Series(np.nan, index=test_index)
        scores.loc[test_valid] = model.score(features.loc[test_valid])
        pieces.append(confirmed_alarms(scores, threshold, confirm_bars))
    return pd.concat(pieces), {"skipped_folds": skipped}
