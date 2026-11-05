"""Агрегация телеметрии в 5-минутные бары, контроль качества и признаки.

Все признаки используют только прошлое. Бар с меткой t содержит данные из (t-5 мин, t],
а окна признаков заканчиваются на этом баре, поэтому в момент t всё уже известно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR = "5min"
BARS_PER_HOUR = 12
# Из описания датасета: около 0 А выключен, 4 А без нагрузки, 7 А под нагрузкой, 9 А пуск.
LOAD_CURRENT_A = 5.0
OFF_CURRENT_A = 1.0
# Замеров идёт примерно по 30 на бар. Если их меньше 10, бар считаем пустым.
MIN_SAMPLES_PER_BAR = 10


def to_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Сводит сырые замеры в бары на равномерной сетке, пропуски времени остаются NaN."""
    df = raw.set_index("timestamp")
    work = pd.DataFrame(
        {
            "duty": (df["Motor_current"] > LOAD_CURRENT_A).astype(float),
            "off": (df["Motor_current"] < OFF_CURRENT_A).astype(float),
            "dv": df["DV_pressure"],
            "tp3": df["TP3"],
            "oil": df["Oil_temperature"],
            "lps": df["LPS"],
            "tank_gap": (df["TP3"] - df["Reservoirs"]).abs(),
        }
    )
    resampler = work.resample(BAR, closed="right", label="right")
    bars = resampler.mean()
    bars["tp3_min"] = resampler["tp3"].min()
    bars["n"] = resampler["tp3"].size()

    stds = df[["TP3", "Oil_temperature", "Motor_current"]].resample(
        BAR, closed="right", label="right"
    ).std()
    # Все три аналоговых сигнала идеально постоянны: живой датчик так себя не ведёт.
    bars["frozen"] = (stds < 1e-9).all(axis=1) & (bars["n"] >= MIN_SAMPLES_PER_BAR)
    bars["no_data"] = bars["n"] < MIN_SAMPLES_PER_BAR
    return bars


def quality_report(bars: pd.DataFrame) -> dict[str, float]:
    """Сводка качества данных: сколько времени датчики молчали или были залипшими."""
    hours = 1 / BARS_PER_HOUR
    return {
        "total_hours": round(len(bars) * hours, 1),
        "no_data_hours": round(float(bars["no_data"].sum()) * hours, 1),
        "frozen_hours": round(float(bars["frozen"].sum()) * hours, 1),
    }


def _trailing(s: pd.Series, hours: int, how: str = "mean") -> pd.Series:
    """Окно по прошлому. Если данных меньше половины окна, результат NaN."""
    window = hours * BARS_PER_HOUR
    return getattr(s.rolling(window, min_periods=window // 2), how)()


def build_features(bars: pd.DataFrame, mask_frozen: bool = True) -> pd.DataFrame:
    """Строит признаки. Бары без данных и с залипшими датчиками считаются пропусками.

    mask_frozen=False оставляет залипшие показания: нужно, чтобы измерить пользу фильтра.
    """
    bad = bars["no_data"] | (bars["frozen"] if mask_frozen else False)
    clean = bars[["duty", "off", "dv", "tp3", "tp3_min", "oil", "lps", "tank_gap"]].mask(bad)

    features = pd.DataFrame(index=bars.index)
    for hours in (1, 6):
        features[f"duty_{hours}h"] = _trailing(clean["duty"], hours)
        features[f"off_{hours}h"] = _trailing(clean["off"], hours)
        features[f"dv_{hours}h"] = _trailing(clean["dv"], hours)
        features[f"tp3_{hours}h"] = _trailing(clean["tp3"], hours)
        features[f"oil_{hours}h"] = _trailing(clean["oil"], hours)
    features["tp3_min_1h"] = _trailing(clean["tp3_min"], 1, "min")
    features["lps_1h"] = _trailing(clean["lps"], 1)
    features["tank_gap_1h"] = _trailing(clean["tank_gap"], 1)
    # Отклонение от собственной нормы за сутки: уровень зависит от сезона и режима.
    features["duty_vs_24h"] = features["duty_1h"] - _trailing(clean["duty"], 24, "median")
    features["oil_vs_24h"] = features["oil_1h"] - _trailing(clean["oil"], 24, "median")
    # Текущий бар плохой: оценку не выдаём, даже если окно ещё заполнено.
    features[bad.to_numpy()] = np.nan
    return features
