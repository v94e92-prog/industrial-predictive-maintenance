"""Полный прогон на MetroPT-3: python scripts/run_metropt3.py ПУТЬ_К_ДАННЫМ"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pdm.data import METROPT3_FAILURES, load_metropt3
from pdm.evaluation import score_events, walk_forward
from pdm.features import build_features, quality_report, to_bars
from pdm.models import MODELS, DutyRule

REPORTS = Path(__file__).resolve().parent.parent / "reports"
QUANTILES = (0.99, 0.995, 0.999)


def main(path: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bars = to_bars(load_metropt3(path))
    features = build_features(bars)
    report = {"quality": quality_report(bars), "models": {}}
    print("Качество данных:", report["quality"])

    for model_cls in MODELS:
        fixed = model_cls.fixed_threshold is not None
        for quantile in (QUANTILES[1],) if fixed else QUANTILES:
            alarms, info = walk_forward(features, METROPT3_FAILURES, model_cls, quantile=quantile)
            result = score_events(alarms, METROPT3_FAILURES)
            key = model_cls.name if fixed else f"{model_cls.name} q={quantile}"
            report["models"][key] = {
                "delays_min": result.delays_min,
                "caught": result.caught,
                "false_per_30d": round(result.false_per_30d, 1),
                "false_count": len(result.false_episodes),
                "false_episodes": [[str(a), str(b)] for a, b in result.false_episodes],
                **info,
            }
            print(
                f"{key:40s} поймано {result.caught}/4, задержки, мин: {result.delays_min}, "
                f"лишних: {len(result.false_episodes)} ({result.false_per_30d:.1f} за 30 дн) "
                f"{info['skipped_folds'] or ''}"
            )

    # Что было бы без фильтра залипших датчиков
    alarms, _ = walk_forward(build_features(bars, mask_frozen=False), METROPT3_FAILURES, DutyRule)
    unfiltered = score_events(alarms, METROPT3_FAILURES)
    report["rule_without_frozen_filter"] = {
        "delays_min": unfiltered.delays_min,
        "false_count": len(unfiltered.false_episodes),
        "false_episodes": [[str(a), str(b)] for a, b in unfiltered.false_episodes],
    }
    print("Правило без фильтра залипания:", report["rule_without_frozen_filter"])
    print("Лишние эпизоды правила:", report["models"][DutyRule.name]["false_episodes"])

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Сохранено:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/metropt3.parquet")
