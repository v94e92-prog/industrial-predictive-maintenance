"""Цена ошибок и точность оценок: то, по чему завод решает, нужна ли ему система.

Настоящих цен в открытых данных нет, поэтому главный результат здесь не сумма, а порог
окупаемости: во сколько раз час незамеченного отказа должен быть дороже одного осмотра.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta, chi2

from pdm.data import Failure
from pdm.evaluation import EventResult


def detection_interval(caught: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Точный интервал Клоппера-Пирсона для доли пойманных отказов."""
    if total <= 0:
        raise ValueError("Нужен хотя бы один отказ")
    alpha = 1 - level
    low = 0.0 if caught == 0 else float(beta.ppf(alpha / 2, caught, total - caught + 1))
    high = 1.0 if caught == total else float(beta.ppf(1 - alpha / 2, caught + 1, total - caught))
    return low, high


def false_alarm_interval(count: int, days: float, level: float = 0.95) -> tuple[float, float]:
    """Точный пуассоновский интервал для числа лишних тревог за 30 дней."""
    if days <= 0:
        raise ValueError("Период должен быть больше нуля")
    alpha = 1 - level
    low = 0.0 if count == 0 else float(chi2.ppf(alpha / 2, 2 * count)) / 2
    high = float(chi2.ppf(1 - alpha / 2, 2 * count + 2)) / 2
    return low / days * 30, high / days * 30


def hours_saved(result: EventResult, failures: tuple[Failure, ...],
                reaction_hours: float) -> dict[int, float]:
    """Сколько часов отказа удалось бы сократить, если действовать сразу после тревоги.

    Предположение: без системы отказ длился бы столько же, сколько в отчёте компании.
    Часы считаются от первой тревоги плюс время реакции. Если тревога пришла заранее,
    раньше начала отказа сокращать нечего. Пропущенный отказ не даёт экономии.
    """
    if reaction_hours < 0:
        raise ValueError("Время реакции не может быть отрицательным")
    saved = {}
    for number, failure in enumerate(failures, 1):
        delay = result.delays_min[number]
        if delay is None:
            saved[number] = 0.0
            continue
        duration = (failure.end - failure.start).total_seconds() / 3600
        saved[number] = max(0.0, duration - max(delay, 0) / 60 - reaction_hours)
    return saved


def break_even_ratio(inspections: int, saved_hours: float) -> float:
    """Порог окупаемости: цена часа незамеченного отказа в единицах цены одного осмотра."""
    if saved_hours <= 0:
        return float("inf")
    return inspections / saved_hours


@dataclass(frozen=True)
class Costs:
    """Цены заказчика. Их нужно получить на самом предприятии."""

    inspection: float
    loss_per_hour: float
    reaction_hours: float = 4.0


def net_value(result: EventResult, failures: tuple[Failure, ...], costs: Costs) -> dict[str, float]:
    """Экономия от сокращения отказов минус стоимость всех осмотров за период."""
    saved = sum(hours_saved(result, failures, costs.reaction_hours).values())
    benefit = saved * costs.loss_per_hour
    spent = result.total_episodes * costs.inspection
    return {
        "saved_hours": saved,
        "benefit": benefit,
        "inspection_cost": spent,
        "net": benefit - spent,
    }
