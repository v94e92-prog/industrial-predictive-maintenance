"""Загрузка MetroPT-3 и таблица отказов."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current", "COMP", "DV_eletric", "Towers",
    "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses",
)


@dataclass(frozen=True)
class Failure:
    """Отказ из отчёта компании: интервал, когда оборудование было неисправно."""

    start: pd.Timestamp
    end: pd.Timestamp
    kind: str = "air leak"


# Таблица из описания датасета (Data Description_Metro.pdf). В источнике второй отказ
# тоже подписан #1, а дата его обслуживания (30 апреля) раньше самого отказа, поэтому
# колонку обслуживания мы не используем. Отказы нумеруем по порядку.
METROPT3_FAILURES = tuple(
    Failure(pd.Timestamp(start), pd.Timestamp(end))
    for start, end in (
        ("2020-04-18 00:00", "2020-04-18 23:59"),
        ("2020-05-29 23:30", "2020-05-30 06:00"),
        ("2020-06-05 10:00", "2020-06-07 14:30"),
        ("2020-07-15 14:30", "2020-07-15 19:00"),
    )
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет сырые данные: нужные колонки, порядок времени, отсутствие пропусков."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В данных нет колонок: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("Колонка timestamp должна иметь тип даты и времени")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Метки времени должны идти по возрастанию")
    if df["timestamp"].duplicated().any():
        raise ValueError("В данных есть повторяющиеся метки времени")
    nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
    if nulls.any():
        raise ValueError(f"В данных есть пропуски: {nulls[nulls > 0].to_dict()}")
    return df


def load_metropt3(path: str | Path) -> pd.DataFrame:
    """Читает CSV или Parquet с MetroPT-3 и проверяет его."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    return validate_raw(df)
