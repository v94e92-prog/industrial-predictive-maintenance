"""Тесты цены ошибок, доверительных интервалов и журнала тревог."""
import numpy as np
import pandas as pd
import pytest

from pdm.alarm_log import READABLE, build_alarm_log, explain
from pdm.data import Failure
from pdm.economics import (
    Costs,
    break_even_ratio,
    detection_interval,
    false_alarm_interval,
    hours_saved,
    net_value,
)
from pdm.evaluation import EventResult, make_labels, walk_forward
from pdm.models import DutyRule

FAILURES = (
    Failure(pd.Timestamp("2020-01-10 00:00"), pd.Timestamp("2020-01-10 10:00")),
    Failure(pd.Timestamp("2020-01-20 00:00"), pd.Timestamp("2020-01-20 02:00")),
    Failure(pd.Timestamp("2020-01-25 00:00"), pd.Timestamp("2020-01-25 10:00")),
)


def make_result(delays, total_episodes=3) -> EventResult:
    return EventResult(dict(enumerate(delays, 1)), [], 30.0, total_episodes)


def test_detection_interval_matches_known_values():
    # При 4 из 4 нижняя граница равна 0,025^(1/4), верхняя равна 1
    low, high = detection_interval(4, 4)
    assert low == pytest.approx(0.025 ** 0.25, abs=1e-6)
    assert high == 1.0
    low, high = detection_interval(0, 4)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** 0.25, abs=1e-6)


def test_detection_interval_requires_failures():
    with pytest.raises(ValueError):
        detection_interval(0, 0)


def test_false_alarm_interval_contains_rate_and_handles_zero():
    low, high = false_alarm_interval(3, 150)
    assert low < 3 / 150 * 30 < high
    low, high = false_alarm_interval(0, 150)
    assert low == 0
    assert high == pytest.approx(-np.log(0.025) / 150 * 30, abs=1e-6)


def test_hours_saved_counts_delay_reaction_and_misses():
    result = make_result([60.0, 90.0, None])
    assert hours_saved(result, FAILURES, 2.0) == {1: 7.0, 2: 0.0, 3: 0.0}


def test_early_alarm_saves_no_more_than_failure_duration():
    result = make_result([-120.0, None, None])
    assert hours_saved(result, FAILURES, 0.0)[1] == 10.0


def test_negative_reaction_is_rejected():
    with pytest.raises(ValueError):
        hours_saved(make_result([60.0, None, None]), FAILURES, -1.0)


def test_break_even_ratio_and_no_savings():
    assert break_even_ratio(7, 70.0) == pytest.approx(0.1)
    assert break_even_ratio(7, 0.0) == float("inf")


def test_net_value_subtracts_every_inspection():
    result = make_result([60.0, None, None], total_episodes=5)
    value = net_value(result, FAILURES, Costs(inspection=100, loss_per_hour=50, reaction_hours=1))
    assert value["saved_hours"] == 8.0
    assert value["benefit"] == 400
    assert value["inspection_cost"] == 500
    assert value["net"] == -100


def log_features() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2020-01-31", freq="5min")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {column: rng.normal(1, 0.05, len(index)) for column in READABLE}, index=index
    )
    frame["duty_1h"] = rng.uniform(0.05, 0.15, len(index))
    frame.loc[FAILURES[0].start : FAILURES[0].end, "duty_1h"] = 1.0
    return frame


def test_explain_names_the_deviating_signal_first():
    features = log_features()
    moment = FAILURES[0].start + pd.Timedelta(hours=1)
    reasons = explain(features, moment, FAILURES)
    assert reasons[0].startswith("доля часа под нагрузкой: 1,00")


def test_explain_uses_only_the_past():
    features = log_features()
    moment = FAILURES[0].start + pd.Timedelta(hours=1)
    changed = features.copy()
    changed.loc[changed.index > moment] = 500.0
    assert explain(features, moment, FAILURES) == explain(changed, moment, FAILURES)


def test_alarm_log_links_failures_and_marks_unconfirmed():
    features = log_features()
    alarms = pd.Series(False, index=features.index)
    alarms.loc[FAILURES[0].start + pd.Timedelta(hours=1)] = True
    alarms.loc[pd.Timestamp("2020-01-15 12:00")] = True
    log = build_alarm_log(alarms, features, FAILURES)
    assert [entry["failure"] for entry in log] == [1, None]
    assert len(log[0]["reasons"]) == 3


def test_stricter_confirmation_delays_alarm():
    features = log_features()[["duty_1h"]]
    start = (pd.Timestamp("2020-01-05"),)
    fast, _ = walk_forward(features, FAILURES, DutyRule, fold_starts=start, confirm_bars=3)
    slow, _ = walk_forward(features, FAILURES, DutyRule, fold_starts=start, confirm_bars=12)
    assert fast.idxmax() < slow.idxmax()
    assert make_labels(features.index, FAILURES).sum() > 0
