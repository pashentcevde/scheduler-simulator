#!/usr/bin/env python3
"""Сводка по нескольким парам прогонов A/B, снятым на разных сидах.

Использование:
    python3 python/aggregate_seeds.py runs/A1:runs/B1 runs/A2:runs/B2 ...

Для каждой метрики печатает среднее по A, среднее по B, средний относительный
сдвиг, разброс между сидами и знаковый тест (на скольких сидах B оказалась
лучше). Смысл в том, чтобы не выдавать за результат разницу в 1-2%, которая
на другом сиде разворачивается в обратную сторону.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# метрика -> (подпись, направление: -1 меньше лучше, +1 больше лучше)
METRICS = [
    ("jobs_completed", "завершено задач", +1),
    ("throughput_per_sim_hour", "пропускная, задач/сим-час", +1),
    ("wait_p90_sim_min", "ожидание p90, сим-мин", -1),
    ("wait_p99_sim_min", "ожидание p99, сим-мин", -1),
    ("wait_mean_sim_min", "ожидание среднее, сим-мин", -1),
    ("queue_depth_mean", "глубина очереди", -1),
    ("gpu_utilization_mean", "утилизация GPU", +1),
    ("preemptions", "вытеснений", -1),
    ("cost_overhead_pct", "перерасход к оптимуму, %", -1),
    ("wasted_vram_gb_hours", "впустую VRAM, ГБ·ч", -1),
    ("oom_jobs", "OOM", -1),
    ("efficiency_score_weighted", "efficiency (взвеш.)", +1),
]


def load(run: str) -> dict:
    return json.loads((Path(run) / "metrics.json").read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{x:.3g}"


def main() -> None:
    pairs = []
    for arg in sys.argv[1:]:
        a, _, b = arg.partition(":")
        if not b:
            raise SystemExit(f"ожидал пару вида runsA:runsB, получил {arg!r}")
        pairs.append((load(a), load(b)))
    if not pairs:
        raise SystemExit(__doc__)

    name_a = pairs[0][0]["placement"]
    name_b = pairs[0][1]["placement"]
    print(f"\n{name_a} против {name_b}, сидов: {len(pairs)}\n")
    head = (
        f"{'метрика':28s}{name_a:>12s}{name_b:>12s}"
        f"{'Δ сред.':>10s}{'разброс Δ':>12s}{'B лучше':>10s}"
    )
    print(head)
    print("-" * len(head))

    for key, label, direction in METRICS:
        deltas, va, vb, wins = [], [], [], 0
        for ma, mb in pairs:
            xa, xb = ma.get(key), mb.get(key)
            if xa is None or xb is None:
                continue
            va.append(xa)
            vb.append(xb)
            if xa:
                deltas.append(100 * (xb - xa) / abs(xa))
            if (xb - xa) * direction > 0:
                wins += 1
        if not va:
            continue
        spread = (
            f"±{statistics.stdev(deltas):.1f}%" if len(deltas) > 1 else "—"
        )
        mean_d = f"{statistics.fmean(deltas):+.1f}%" if deltas else "—"
        # если знак сдвига не устойчив, помечаем результат как шум
        mark = "" if wins in (0, len(pairs)) else "  (?)"
        print(
            f"{label:28s}{fmt(statistics.fmean(va)):>12s}"
            f"{fmt(statistics.fmean(vb)):>12s}{mean_d:>10s}{spread:>12s}"
            f"{f'{wins}/{len(pairs)}':>10s}{mark}"
        )

    print(
        "\n(?) — знак сдвига не одинаков на всех сидах: результат не отличим "
        "от шума,\n    для вывода нужно больше прогонов."
    )


if __name__ == "__main__":
    main()
