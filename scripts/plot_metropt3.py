"""График для README: python scripts/plot_metropt3.py ПУТЬ_К_ДАННЫМ"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pdm.data import METROPT3_FAILURES, load_metropt3
from pdm.evaluation import score_events, walk_forward
from pdm.features import build_features, to_bars
from pdm.models import DutyRule

OUT = Path(__file__).resolve().parent.parent / "reports" / "rule_on_events.png"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
LINE, ZONE, ALARM = "#2a78d6", "#f6d9cd", "#eb6834"
PAD = pd.Timedelta(hours=18)


def main(path: str) -> None:
    features = build_features(to_bars(load_metropt3(path)))
    alarms, _ = walk_forward(features, METROPT3_FAILURES, DutyRule)
    result = score_events(alarms, METROPT3_FAILURES)

    panels = [
        (f"Отказ {i}: тревога через {result.delays_min[i]:.0f} мин", f.start, f.end, True)
        for i, f in enumerate(METROPT3_FAILURES, 1)
    ]
    panels += [
        (f"Нет в отчётах: {a:%d.%m}", a, b, False) for a, b in result.false_episodes
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), sharey=True, facecolor=SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (title, start, end, is_failure) in zip(axes.flat, panels):
        window = (features.index >= start - PAD) & (features.index <= end + PAD)
        ax.set_facecolor(SURFACE)
        if is_failure:
            ax.axvspan(start, end, color=ZONE, lw=0)
        ax.plot(features.index[window], features["duty_1h"][window], color=LINE, lw=2)
        ax.axhline(DutyRule.fixed_threshold, color=MUTED, lw=1, ls=(0, (4, 3)))
        fired = alarms.index[alarms.to_numpy()]
        first = fired[(fired >= start - PAD) & (fired <= end + PAD)]
        if len(first):
            ax.plot(first[0], 1.03, marker="v", ms=9, color=ALARM, clip_on=False, mew=1.5,
                    mec=SURFACE)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 1.08)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="y", color=GRID, lw=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    for ax in axes[:, 0]:
        ax.set_ylabel("Доля часа под нагрузкой", color=MUTED, fontsize=10)

    legend = axes.flat[-1]
    legend.set_visible(True)
    legend.axis("off")
    handles = [
        plt.Line2D([], [], color=LINE, lw=2, label="Нагрузка за последний час"),
        plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="Порог правила, 0,9"),
        plt.Rectangle((0, 0), 1, 1, color=ZONE, label="Отказ по отчёту компании"),
        plt.Line2D([], [], color=ALARM, marker="v", ms=9, ls="", label="Первая тревога"),
    ]
    legend.legend(handles=handles, loc="center left", frameon=False, fontsize=10,
                  labelcolor=INK)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print("Сохранено:", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/metropt3.parquet")
