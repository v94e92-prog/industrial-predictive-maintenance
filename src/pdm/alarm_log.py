"""Журнал тревог: когда сработало, с чем связано и какие показатели вышли за норму."""
from __future__ import annotations

import pandas as pd

from pdm.data import Failure
from pdm.evaluation import MATCH_AFTER, MATCH_BEFORE, alarm_episodes, make_labels
from pdm.models import RobustZ

# Только понятные человеку показатели, с единицами измерения.
READABLE = {
    "duty_1h": ("доля часа под нагрузкой", ""),
    "off_1h": ("доля часа в выключенном состоянии", ""),
    "dv_1h": ("давление DV", " бар"),
    "tp3_1h": ("давление TP3", " бар"),
    "oil_1h": ("температура масла", " °C"),
    "oil_vs_24h": ("масло относительно суточной нормы", " °C"),
    "lps_1h": ("доля часа с давлением ниже 7 бар", ""),
    "tank_gap_1h": ("разница давления TP3 и резервуаров", " бар"),
}


def explain(features: pd.DataFrame, moment: pd.Timestamp, failures: tuple[Failure, ...],
            top: int = 3) -> list[str]:
    """Что сильнее всего отклонилось от нормы в момент тревоги.

    Норма считается только по прошлому и без уже известных на тот момент отказов.
    """
    columns = list(READABLE)
    past = features.loc[features.index < moment, columns].dropna()
    known = tuple(f for f in failures if f.end < moment)
    model = RobustZ().fit(past, make_labels(past.index, known))
    row = features.loc[[moment], columns]
    strongest = model.deviations(row).iloc[0].dropna().sort_values(ascending=False).head(top)
    reasons = []
    for column in strongest.index:
        title, unit = READABLE[column]
        text = f"{title}: {row[column].iloc[0]:.2f}{unit} при норме {model.median_[column]:.2f}{unit}"
        reasons.append(text.replace(".", ","))
    return reasons


def build_alarm_log(alarms: pd.Series, features: pd.DataFrame,
                    failures: tuple[Failure, ...]) -> list[dict]:
    """Сводит тревоги в журнал. Эпизод вне окон отказов помечается как неподтверждённый."""
    log = []
    for start, end in alarm_episodes(alarms):
        failure_number = None
        for number, f in enumerate(failures, 1):
            if start <= f.end + MATCH_AFTER and end >= f.start - MATCH_BEFORE:
                failure_number = number
                break
        log.append(
            {
                "start": start,
                "end": end,
                "hours": (end - start).total_seconds() / 3600,
                "failure": failure_number,
                "reasons": explain(features, start, failures),
            }
        )
    return log
