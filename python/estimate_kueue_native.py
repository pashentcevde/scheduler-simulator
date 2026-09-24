#!/usr/bin/env python3
"""
Офлайн-оценка режима kueue-native.

    python3 python/estimate_kueue_native.py config/scenarios/s01_steady_mixed.yaml

Зачем нужно. В режиме kueue-native карта не указывается, и какой флейвор
достанется задаче, до прогона неизвестно — это решает сам kueue. Но механику
его выбора можно воспроизвести приближённо и оценить, чем такой режим
заканчивается, не занимая кластер.

Что воспроизводится. Kueue перебирает флейворы в порядке объявления в
resourceGroups (от дешёвых к дорогим) и берёт первый, по которому ЕСТЬ
СВОБОДНАЯ КВОТА. Признак «влезает ли модель в память карты» в этом решении
не участвует вообще — kueue про VRAM ничего не знает. Здесь считается
упрощённо: вместо квот очередей берётся физическая ёмкость пула (это
оптимистично для kueue: реальные квоты меньше и отправляют задачи на дорогие
карты чаще), а число карт — то, которое запросил бы пользователь, потому что
kueue подставляет флейвор, но не меняет `nvidia.com/gpu: N` в поде.

Что НЕ воспроизводится: приоритеты, вытеснение, заимствование между
очередями и фрагментация внутри нод. Поэтому число — оценка порядка величины,
а не предсказание. Точный ответ даёт прогон в кластере.
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import ModelCatalog, PoolConfig, load_yaml  # noqa: E402


def build_plan_via_submit(scenario: str, pool: str, models: str, tenants: str) -> list[dict]:
    """Построить план тем же генератором, что и обычный прогон."""
    out = Path(tempfile.mkdtemp())
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "submit.py"),
            "--scenario", scenario,
            "--placement", "manual",
            "--run-dir", str(out),
            "--pool", pool,
            "--models", models,
            "--tenants", tenants,
            "--dry-run",
        ],
        capture_output=True,
        check=True,
    )
    return [json.loads(l) for l in (out / "jobs.jsonl").open(encoding="utf-8")]


def simulate(jobs: list[dict], cfg: PoolConfig, mode: str) -> dict:
    """Дискретно-событийная прокрутка: кто куда поедет и кто упадёт по OOM.

    mode='kueue-native' — флейвор выбирает kueue: первый по порядку, где есть
                          свободные карты; число карт из запроса пользователя.
    mode='manual'       — карта и число карт заданы пользователем.
    mode='cheapest'     — минимально достаточная карта по оценке VRAM.
    """
    usable = cfg.defaults["usable_vram_fraction"]
    order = cfg.sorted_by_cost()
    free = {p.name: float(p.total_gpus) for p in cfg.pools}
    by_name = {p.name: p for p in cfg.pools}

    events: list[tuple[float, str, float]] = []   # (время, пул, освобождается карт)
    dist: Counter = Counter()
    oom = 0
    queued = 0
    cost = 0.0

    for job in sorted(jobs, key=lambda j: j["submit_at_sim_min"]):
        t = job["submit_at_sim_min"]
        while events and events[0][0] <= t:
            _, pool_name, gpus = heapq.heappop(events)
            free[pool_name] += gpus

        if mode == "kueue-native":
            gpus = job["user_gpus"]
            pool = next((p for p in order if free[p.name] >= gpus), None)
            if pool is None:
                # свободных карт нет нигде: задача ждёт, а потом всё равно
                # получит первый освободившийся флейвор
                queued += 1
                while events:
                    _, pool_name, released = heapq.heappop(events)
                    free[pool_name] += released
                    if free[pool_name] >= gpus:
                        pool = by_name[pool_name]
                        t = max(t, _)
                        break
                if pool is None:
                    pool = order[-1]
        elif mode == "manual":
            pool, gpus = by_name[job["user_pool"]], job["user_gpus"]
        else:
            pool, gpus = by_name[job["min_pool"]], job["min_gpus"]

        dist[pool.name] += 1
        duration_min = job["sim_minutes"]

        if job["actual_vram_gb"] > gpus * pool.vram_gb * usable:
            # OOM: карта занята только на время загрузки модели
            oom += 1
            duration_min *= job.get("startup_frac", 0.05)
        else:
            cost += gpus * pool.relative_cost * duration_min / 60.0

        free[pool.name] -= gpus
        heapq.heappush(events, (t + duration_min, pool.name, float(gpus)))

    return {
        "mode": mode,
        "jobs": len(jobs),
        "oom": oom,
        "oom_pct": round(100 * oom / max(1, len(jobs)), 1),
        "cost": round(cost),
        "waited": queued,
        "dist": dict(dist),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--tenants", default="config/tenants.yaml")
    args = ap.parse_args()

    cfg = PoolConfig.load(args.pool)

    for scn in args.scenarios:
        name = Path(scn).stem
        jobs = build_plan_via_submit(scn, args.pool, args.models, args.tenants)
        print(f"\n=== {name}  ({len(jobs)} задач, {cfg.total_gpus} GPU)")
        print(f"{'режим':14} {'OOM':>5} {'OOM %':>7} {'стоимость':>10}   распределение по картам")
        for mode in ("kueue-native", "manual", "cheapest"):
            r = simulate(jobs, cfg, mode)
            dist = " ".join(f"{k}={v}" for k, v in sorted(r["dist"].items()))
            print(f"{mode:14} {r['oom']:5} {r['oom_pct']:6.1f}% {r['cost']:10}   {dist}")


if __name__ == "__main__":
    main()
