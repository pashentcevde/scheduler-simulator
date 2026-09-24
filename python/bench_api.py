#!/usr/bin/env python3
"""Замер латентности опроса кластера: обычный kubectl против kubectl proxy.

Показывает, какой --interval снапшотера реально достижим. Правило: такт
опроса должен занимать не больше ~30% интервала, иначе снапшотер начнёт
отставать и тики поедут (это видно в timeseries.jsonl как разъезжающийся ts).

    python3 python/bench_api.py
    python3 python/bench_api.py --repeat 50
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab import L_LAB, L_POOL, kubectl_json  # noqa: E402

# ровно то, что снапшотер делает на каждом такте
TICK = (
    ("get", "pods,jobs", "-A", "-l", f"{L_LAB}=true"),
    ("get", "clusterqueues,workloads", "-A"),
)


def measure(repeat: int) -> list[float]:
    samples = []
    for _ in range(repeat):
        started = time.time()
        items = 0
        for call in TICK:
            items += len(kubectl_json(*call).get("items", []))
        samples.append(time.time() - started)
    return samples


def report(label: str, samples: list[float]) -> float:
    p50 = statistics.median(samples)
    p90 = sorted(samples)[int(0.9 * len(samples))]
    print(
        f"{label:24s} p50 {1000 * p50:7.1f} мс   p90 {1000 * p90:7.1f} мс   "
        f"макс {1000 * max(samples):7.1f} мс"
    )
    return p50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    # прогрев: первый вызов поднимает прокси и делает discovery
    kubectl_json("get", "nodes", "-l", L_POOL)

    os.environ["KUBE_FAST"] = "0"
    slow = report("обычный kubectl", measure(args.repeat))

    os.environ["KUBE_FAST"] = "1"
    fast = report("через kubectl proxy", measure(args.repeat))

    os.environ["KUBE_LIST_FROM_CACHE"] = "1"
    cached = report("proxy + watch-кэш", measure(args.repeat))
    os.environ.pop("KUBE_LIST_FROM_CACHE")

    best = min(fast, cached)
    print(f"\nускорение: ×{slow / best:.1f}")
    print(
        f"достижимый SNAPSHOT_INTERVAL: {max(0.05, round(best / 0.3, 2))} с "
        f"(было {max(0.05, round(slow / 0.3, 2))} с)"
    )
    print(
        "\nПроверьте, что число объектов в ответах совпадает с выводом "
        "kubectl — если быстрый путь молча вернул пустой список, ускорение "
        "будет отличным, а прогон бессмысленным."
    )


if __name__ == "__main__":
    main()
