"""Тесты на то, что может незаметно испортить результат: подсмотр будущего и подсчёт событий."""
import numpy as np
import pandas as pd
import pytest

from pdm.data import REQUIRED_COLUMNS, Failure, validate_raw
from pdm.evaluation import (
    alarm_episodes,
    confirmed_alarms,
    make_labels,
    score_events,
    walk_forward,
)
from pdm.features import build_features, to_bars
from pdm.models import DutyRule, LogReg, RobustZ


def make_raw(hours: int = 72, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Живая телеметрия с шагом 10 секунд: компрессор с шумом работает циклами."""
    rng = np.random.default_rng(seed)
    n = hours * 360
    current = np.where(np.arange(n) % 360 < 40, 7.0, 0.05) + rng.normal(0, 0.02, n)
    raw = pd.DataFrame({c: rng.normal(1, 0.01, n) for c in REQUIRED_COLUMNS if c != "timestamp"})
    raw["timestamp"] = pd.date_range(start, periods=n, freq="10s")
    raw["Motor_current"] = current
    raw["TP3"] = 9 + rng.normal(0, 0.05, n)
    raw["Reservoirs"] = raw["TP3"]
    raw["Oil_temperature"] = 60 + rng.normal(0, 0.5, n)
    raw["DV_pressure"] = rng.normal(0, 0.01, n)
    raw["LPS"] = 0.0
    return raw


def hourly_index(hours: int = 48) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=hours * 12, freq="5min")


def test_future_data_does_not_change_past_features():
    raw = make_raw()
    cutoff = raw["timestamp"].iloc[len(raw) // 2]
    full = build_features(to_bars(raw))
    changed = raw.copy()
    later = changed["timestamp"] > cutoff
    changed.loc[later, "Motor_current"] = 9.0
    changed.loc[later, "TP3"] = 1.0
    other = build_features(to_bars(changed))
    known = full.index <= cutoff
    pd.testing.assert_frame_equal(full[known], other[known])


def test_bar_contains_only_data_up_to_its_label():
    raw = make_raw(hours=1)
    bars = to_bars(raw)
    label = bars.index[3]
    inside = raw[(raw["timestamp"] > label - pd.Timedelta(minutes=5)) & (raw["timestamp"] <= label)]
    assert bars.loc[label, "n"] == len(inside)


def test_frozen_sensors_are_flagged_and_produce_no_features():
    raw = make_raw()
    stuck = raw["timestamp"].between("2020-01-02 00:00", "2020-01-02 06:00")
    for column, value in (("TP3", 8.25), ("Oil_temperature", 49.45), ("Motor_current", 0.04)):
        raw.loc[stuck, column] = value
    bars = to_bars(raw)
    assert bars.loc["2020-01-02 03:00", "frozen"]
    assert not bars.loc["2020-01-01 12:00", "frozen"]
    assert build_features(bars).loc["2020-01-02 03:00"].isna().all()


def test_gap_in_data_produces_no_features():
    raw = make_raw()
    raw = raw[~raw["timestamp"].between("2020-01-02 00:00", "2020-01-02 06:00")]
    features = build_features(to_bars(raw))
    assert features.loc["2020-01-02 03:00"].isna().all()


def test_labels_cover_horizon_and_failure_only():
    index = hourly_index()
    failure = Failure(pd.Timestamp("2020-01-02 00:00"), pd.Timestamp("2020-01-02 02:00"))
    y = make_labels(index, (failure,), horizon=pd.Timedelta(hours=1))
    assert y.loc["2020-01-01 22:55"] == 0
    assert y.loc["2020-01-01 23:00"] == 1
    assert y.loc["2020-01-02 02:00"] == 1
    assert y.loc["2020-01-02 02:05"] == 0


def test_alarm_needs_confirmation_and_nan_breaks_series():
    index = hourly_index(4)
    scores = pd.Series(0.0, index=index)
    scores.iloc[10:15] = 1.0  # 5 баров: мало
    scores.iloc[30:36] = 1.0  # 6 баров: достаточно
    alarms = confirmed_alarms(scores, 0.5, confirm_bars=6)
    assert alarms.iloc[:30].sum() == 0
    assert alarms.iloc[35]
    scores.iloc[33] = np.nan
    assert confirmed_alarms(scores, 0.5, confirm_bars=6).sum() == 0


def test_episodes_merge_when_close_and_split_when_far():
    index = hourly_index(24)
    alarms = pd.Series(False, index=index)
    alarms.iloc[[10, 11, 20, 200]] = True
    episodes = alarm_episodes(alarms, merge_gap=pd.Timedelta(hours=2))
    assert episodes == [(index[10], index[20]), (index[200], index[200])]


def test_event_scoring_counts_delay_misses_and_false_alarms():
    index = hourly_index(240)
    alarms = pd.Series(False, index=index)
    caught = Failure(pd.Timestamp("2020-01-03 00:00"), pd.Timestamp("2020-01-03 05:00"))
    missed = Failure(pd.Timestamp("2020-01-06 00:00"), pd.Timestamp("2020-01-06 05:00"))
    alarms.loc["2020-01-03 01:00"] = True
    alarms.loc["2020-01-09 12:00"] = True
    result = score_events(alarms, (caught, missed))
    assert result.delays_min == {1: 60.0, 2: None}
    assert result.caught == 1
    assert result.false_episodes == [(pd.Timestamp("2020-01-09 12:00"),) * 2]
    assert result.false_per_30d == pytest.approx(3.0)


def synthetic_features() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2020-06-30", freq="5min")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"duty_1h": rng.uniform(0, 0.3, len(index)), "other": rng.normal(0, 1, len(index))},
        index=index,
    )


def test_walk_forward_does_not_use_test_data_for_training(monkeypatch):
    features = synthetic_features()
    starts = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01"))
    seen = []
    original = RobustZ.fit

    def spy(self, X, y):
        seen.append(X.index.max())
        return original(self, X, y)

    monkeypatch.setattr(RobustZ, "fit", spy)
    walk_forward(features, (), RobustZ, fold_starts=starts)
    assert seen[0] < starts[0]
    assert seen[1] < starts[1]


def test_walk_forward_skips_supervised_model_without_past_failures():
    features = synthetic_features()
    failure = Failure(pd.Timestamp("2020-03-15"), pd.Timestamp("2020-03-16"))
    starts = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01"))
    _, info = walk_forward(features, (failure,), LogReg, fold_starts=starts)
    assert info["skipped_folds"] == ["2020-03-01"]


def test_rule_fires_on_continuous_load():
    features = synthetic_features()
    failure = Failure(pd.Timestamp("2020-03-15 00:00"), pd.Timestamp("2020-03-15 12:00"))
    features.loc[failure.start : failure.end, "duty_1h"] = 1.0
    starts = (pd.Timestamp("2020-03-01"),)
    alarms, _ = walk_forward(features, (failure,), DutyRule, fold_starts=starts)
    result = score_events(alarms, (failure,))
    assert result.delays_min[1] == 25.0
    assert result.false_episodes == []


def test_validate_raw_rejects_bad_data():
    raw = make_raw(hours=1)
    with pytest.raises(ValueError, match="колонок"):
        validate_raw(raw.drop(columns=["TP3"]))
    with pytest.raises(ValueError, match="возрастанию"):
        validate_raw(raw.iloc[::-1])
    broken = raw.copy()
    broken.loc[0, "TP3"] = np.nan
    with pytest.raises(ValueError, match="пропуски"):
        validate_raw(broken)
