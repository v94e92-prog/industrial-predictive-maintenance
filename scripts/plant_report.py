"""Отчёт для завода на данных MetroPT-3.

python scripts/plant_report.py ПУТЬ_К_ДАННЫМ [--inspection N --loss-per-hour N --reaction-hours N]

Цены указывать не обязательно: без них отчёт покажет только порог окупаемости.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pdm.alarm_log import build_alarm_log
from pdm.data import METROPT3_FAILURES, load_metropt3
from pdm.economics import (
    Costs,
    break_even_ratio,
    detection_interval,
    false_alarm_interval,
    hours_saved,
    net_value,
)
from pdm.evaluation import FOLD_STARTS, score_events, walk_forward
from pdm.features import build_features, quality_report, to_bars
from pdm.models import MODELS, DutyRule

REPORTS = Path(__file__).resolve().parent.parent / "reports"
FAILURES = METROPT3_FAILURES
REACTIONS = (0.0, 2.0, 4.0, 8.0)
THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
CONFIRMS = ((3, "15 мин"), (6, "30 мин"), (12, "60 мин"))


def num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def delays_text(result) -> str:
    return ", ".join("нет" if v is None else f"{v:.0f}" for v in result.delays_min.values())


def sensitivity(features) -> list[dict]:
    """Как результат правила зависит от порога и времени подтверждения."""
    rows = []
    for threshold in THRESHOLDS:
        rule = type("Rule", (DutyRule,), {"fixed_threshold": threshold})
        for bars, label in CONFIRMS:
            alarms, _ = walk_forward(features, FAILURES, rule, confirm_bars=bars)
            result = score_events(alarms, FAILURES)
            rows.append(
                {
                    "threshold": threshold,
                    "confirm": label,
                    "caught": result.caught,
                    "delays_min": result.delays_min,
                    "false": len(result.false_episodes),
                }
            )
    return rows


def compare_models(features) -> list[dict]:
    """Порог окупаемости каждого подхода при реакции 4 часа."""
    rows = []
    for model_cls in MODELS:
        alarms, info = walk_forward(features, FAILURES, model_cls)
        result = score_events(alarms, FAILURES)
        saved = sum(hours_saved(result, FAILURES, 4.0).values())
        rows.append(
            {
                "name": model_cls.name,
                "caught": result.caught,
                "inspections": result.total_episodes,
                "saved": saved,
                "ratio": break_even_ratio(result.total_episodes, saved),
                "skipped": info["skipped_folds"],
            }
        )
    return rows


def render(data: dict) -> str:
    rule = data["rule"]
    lines = [
        "# Отчёт о проверке на данных компрессора MetroPT-3",
        "",
        ("Период проверки: апрель–август 2020 года. Подход: простое правило, тревога при работе "
        "под нагрузкой более 90% часа, подтверждённой 30 минут подряд."),
        "",
        "## Коротко",
        "",
        (f"- **Поймано отказов:** {rule['caught']} из {len(FAILURES)}. Из-за малого числа отказов "
        f"настоящая доля может быть от {num(rule['detection'][0] * 100, 0)}% до "
        f"{num(rule['detection'][1] * 100, 0)}% (95% доверительный интервал)."),
        (f"- **Задержка:** {delays_text(rule['result'])} минут после начала отказа. "
        "Заранее система не предупреждает."),
        (f"- **Лишние тревоги:** {rule['false']} за {num(rule['days'], 0)} дн., это "
        f"{num(rule['false_rate'])} за 30 дней (интервал {num(rule['false_ci'][0])}–"
        f"{num(rule['false_ci'][1])}). В эти эпизоды компрессор вёл себя как при отказе, но в "
        "отчётах компании их нет, поэтому это могут быть и незарегистрированные проблемы."),
        (f"- **Система не видела оборудование {num(rule['blind_share'] * 100, 0)}% времени:** "
        "данных не было или датчики залипли. Часть этого времени поезд мог просто не работать, "
        "но отказ в такой период система не заметит."),
        "",
        "## Журнал тревог",
        "",
        "| Начало | Длительность, ч | Связь с отказом | Что вышло за норму |",
        "|---|---|---|---|",
    ]
    for entry in data["log"]:
        link = f"отказ №{entry['failure']}" if entry["failure"] else "нет в отчётах"
        lines.append(
            f"| {entry['start']:%d.%m %H:%M} | {num(entry['hours'])} | {link} | "
            f"{'; '.join(entry['reasons'])} |"
        )

    lines += [
        "",
        "## Окупаемость",
        "",
        ("Настоящих цен в открытых данных нет. Таблица показывает порог: система окупается, если "
        "час незамеченного отказа дороже одного осмотра хотя бы во столько раз. Предположение: "
        "без системы отказ длился бы столько же, сколько в отчёте компании."),
        "",
        "| Время от тревоги до устранения, ч | Сэкономлено часов отказа | Порог окупаемости |",
        "|---|---|---|",
    ]
    for row in data["reactions"]:
        lines.append(
            f"| {num(row['reaction'], 0)} | {num(row['saved'])} | {num(row['ratio'], 2)} |"
        )
    lines += [
        "",
        (f"Осмотров за период: {rule['inspections']}. Если часть лишних тревог окажется настоящими "
        "проблемами, окупаемость будет лучше. Если без системы отказ заметили бы быстрее, чем в "
        "отчётах, хуже."),
    ]
    if data["costs"]:
        value, costs = data["costs"]
        lines += [
            "",
            (f"**Расчёт по указанным ценам:** осмотр {money(costs.inspection)}, час отказа "
            f"{money(costs.loss_per_hour)}, реакция {num(costs.reaction_hours, 0)} ч. Экономия "
            f"{money(value['benefit'])}, осмотры {money(value['inspection_cost'])}, итог "
            f"**{money(value['net'])}** за период."),
        ]

    lines += [
        "",
        "## Сравнение подходов при реакции 4 часа",
        "",
        "| Подход | Поймано | Осмотров | Сэкономлено часов | Порог окупаемости |",
        "|---|---|---|---|---|",
    ]
    for row in data["models"]:
        ratio = "нет экономии" if row["ratio"] == float("inf") else num(row["ratio"], 2)
        note = " (без отказа №1: не на чем учиться)" if row["skipped"] else ""
        lines.append(
            f"| {row['name']}{note} | {row['caught']} из {len(FAILURES)} | {row['inspections']} | "
            f"{num(row['saved'])} | {ratio} |"
        )

    lines += [
        "",
        ("Сигнал LPS срабатывает очень часто, поэтому его совпадения с отказами, скорее всего, "
        "случайны, а сэкономленные часы завышены."),
        "",
        "## Устойчивость настроек правила",
        "",
        "| Порог | Подтверждение | Поймано | Задержка, мин (№1–№4) | Лишних |",
        "|---|---|---|---|---|",
    ]
    for row in data["sensitivity"]:
        delays = ", ".join("нет" if v is None else f"{v:.0f}" for v in row["delays_min"].values())
        lines.append(
            f"| {num(row['threshold'], 2)} | {row['confirm']} | {row['caught']} | {delays} | "
            f"{row['false']} |"
        )

    quality = data["quality"]
    lines += [
        "",
        "## Качество данных за всё время",
        "",
        f"- Всего: {num(quality['total_hours'], 0)} ч.",
        f"- Нет данных: {num(quality['no_data_hours'], 0)} ч.",
        f"- Датчики залипли: {num(quality['frozen_hours'], 0)} ч.",
        "",
        "## Что это не доказывает",
        "",
        ("Отказов всего 4, компрессор один, настройки правила выбраны после просмотра этих же "
        "данных. Для настоящего заказчика проверку нужно повторить на его оборудовании и с его "
        "ценами."),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--inspection", type=float, help="цена одного осмотра")
    parser.add_argument("--loss-per-hour", type=float, help="цена часа незамеченного отказа")
    parser.add_argument("--reaction-hours", type=float, default=4.0)
    parser.add_argument("--out", default=str(REPORTS / "plant_report.md"))
    args = parser.parse_args()
    if (args.inspection is None) != (args.loss_per_hour is None):
        parser.error("Цены указываются вместе: --inspection и --loss-per-hour")

    bars = to_bars(load_metropt3(args.data))
    features = build_features(bars)
    alarms, _ = walk_forward(features, FAILURES, DutyRule)
    result = score_events(alarms, FAILURES)
    evaluated = bars[bars.index >= FOLD_STARTS[0]]

    rule = {
        "result": result,
        "caught": result.caught,
        "detection": detection_interval(result.caught, len(FAILURES)),
        "false": len(result.false_episodes),
        "days": result.eval_days,
        "false_rate": result.false_per_30d,
        "false_ci": false_alarm_interval(len(result.false_episodes), result.eval_days),
        "inspections": result.total_episodes,
        "blind_share": float((evaluated["frozen"] | evaluated["no_data"]).mean()),
    }
    reactions = []
    for reaction in REACTIONS:
        saved = sum(hours_saved(result, FAILURES, reaction).values())
        reactions.append(
            {
                "reaction": reaction,
                "saved": saved,
                "ratio": break_even_ratio(result.total_episodes, saved),
            }
        )
    costs = None
    if args.inspection is not None:
        priced = Costs(args.inspection, args.loss_per_hour, args.reaction_hours)
        costs = (net_value(result, FAILURES, priced), priced)

    data = {
        "rule": rule,
        "log": build_alarm_log(alarms, features, FAILURES),
        "reactions": reactions,
        "models": compare_models(features),
        "sensitivity": sensitivity(features),
        "quality": quality_report(bars),
        "costs": costs,
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    summary = {
        "rule": {k: v for k, v in rule.items() if k != "result"},
        "reactions": reactions,
        "models": data["models"],
        "sensitivity": data["sensitivity"],
    }
    summary["models"] = [
        {**row, "ratio": None if row["ratio"] == float("inf") else row["ratio"]}
        for row in summary["models"]
    ]
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(render(data))


if __name__ == "__main__":
    main()
