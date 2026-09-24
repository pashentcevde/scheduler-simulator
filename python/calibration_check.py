#!/usr/bin/env python3
"""
Проверка калибровки confidence.

    # офлайн, по планам сценариев — кластер не нужен
    python3 python/calibration_check.py --scenarios config/scenarios/s01_steady_mixed.yaml \
                                                    config/scenarios/s09_oom_storm.yaml

    # по результатам реальных прогонов
    python3 python/calibration_check.py --runs runs/s01_steady_mixed-cheapest-*

Что проверяется. Планировщик приписывает каждому размещению число
`confidence` — вероятность того, что задача поместится в выбранную карту.
Скрипт раскладывает размещения по корзинам этой вероятности и сравнивает
предсказание с тем, что произошло на самом деле. Если метрика калибрована,
то в корзине 0.7 должны доезжать примерно 70% задач.

ДВА РЕЖИМА И ЧТО ИМЕННО КАЖДЫЙ ДОКАЗЫВАЕТ — разница существенная.

  --scenarios (офлайн). Берутся планы прогонов: там есть и решение
    планировщика с его confidence, и ground truth (actual-vram-gb).
    Факт падения вычисляется арифметически: actual > gpus * vram * usable.
    Это проверка ВНУТРЕННЕЙ СОГЛАСОВАННОСТИ — что аналитическая формула
    вероятности, округления и учёт полезной доли памяти не разъезжаются с
    тем, как стенд порождает нагрузку. Обе стороны опираются на одно и то же
    распределение U(lo, hi), поэтому совпадение здесь НЕ доказывает, что
    модель соответствует реальности. Оно доказывает, что в коде нет ошибки.

  --runs (по прогонам). Факт берётся из oom.jsonl — то есть из решений,
    принятых моделью GPU уже после того, как задача прошла через kueue,
    kube-scheduler и реально куда-то приехала. Это более сильная проверка:
    она захватывает и фрагментацию, и вытеснение, и повторные постановки.
    Но и она проверяет стенд, а не природу: настоящая валидация возможна
    только на статистике реальных запусков.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import PoolConfig  # noqa: E402

POLICIES = ("manual", "cheapest", "queue-aware", "kueue-aware")


def jobs_from_scenario(scenario: str, policy: str, cfg_paths: dict) -> list[dict]:
    """Построить план сценария под заданной политикой (в кластер ничего не идёт)."""
    out = Path(tempfile.mkdtemp())
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "submit.py"),
            "--scenario", scenario,
            "--placement", policy,
            "--run-dir", str(out),
            "--pool", cfg_paths["pool"],
            "--models", cfg_paths["models"],
            "--tenants", cfg_paths["tenants"],
            "--dry-run",
        ],
        capture_output=True,
        check=True,
    )
    return [json.loads(l) for l in (out / "jobs.jsonl").open(encoding="utf-8")]


def outcomes_offline(jobs: list[dict], cfg: PoolConfig) -> list[tuple[float, bool]]:
    """(предсказанная вероятность, упала ли задача) — факт считается арифметически."""
    usable_frac = cfg.defaults["usable_vram_fraction"]
    pools = {p.name: p for p in cfg.pools}
    out = []
    for j in jobs:
        conf = j.get("confidence")
        if conf is None:          # kueue-native: карта на момент подачи неизвестна
            continue
        pool = pools.get(j["requested_pool"])
        if pool is None:
            continue
        usable = j["gpus"] * pool.vram_gb * usable_frac
        out.append((float(conf), j["actual_vram_gb"] > usable))
    return out


def outcomes_from_run(run_dir: Path) -> list[tuple[float, bool]]:
    """(предсказанная вероятность, упала ли задача) — факт из oom.jsonl прогона."""
    jobs_path = run_dir / "jobs.jsonl"
    if not jobs_path.exists():
        raise SystemExit(f"{run_dir}: нет jobs.jsonl")
    oom_path = run_dir / "oom.jsonl"
    fallen = set()
    if oom_path.exists():
        for line in oom_path.open(encoding="utf-8"):
            if line.strip():
                fallen.add(json.loads(line)["job_id"])
    out = []
    for line in jobs_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("confidence") is None:
            continue
        out.append((float(j["confidence"]), j["job_id"] in fallen))
    return out


def report(pairs: list[tuple[float, bool]], bucket: float = 0.1) -> None:
    if not pairs:
        raise SystemExit("нет размещений с определённой confidence")

    buckets: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for conf, fell in pairs:
        key = round(round(conf / bucket) * bucket, 2)
        buckets[key][0] += 1
        buckets[key][1] += int(fell)

    print(
        f"\n{'confidence':>11} {'задач':>7} {'упало':>7} "
        f"{'факт. успех':>12} {'предсказано':>12} {'разница':>9}"
    )
    print("─" * 62)
    for key in sorted(buckets):
        n, fell = buckets[key]
        actual = 1 - fell / n
        print(
            f"{key:11.1f} {n:7} {fell:7} {actual:11.0%} {key:11.0%} "
            f"{actual - key:+8.0%}"
        )

    total = len(pairs)
    fell_total = sum(f for _, f in pairs)
    expected_ok = sum(c for c, _ in pairs)
    print("─" * 62)
    print(
        f"всего {total} размещений; упало {fell_total} ({fell_total / total:.1%}), "
        f"ожидалось {total - expected_ok:.0f} ({1 - expected_ok / total:.1%})"
    )
    err = abs((total - expected_ok) / total - fell_total / total)
    verdict = "калибровка в порядке" if err < 0.02 else "РАСХОЖДЕНИЕ, разбираться"
    print(f"расхождение по суммарной доле падений: {err:.1%} — {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument(
        "--policies",
        nargs="*",
        default=list(POLICIES),
        help="какие политики прогнать в офлайн-режиме",
    )
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--tenants", default="config/tenants.yaml")
    ap.add_argument("--bucket", type=float, default=0.1)
    args = ap.parse_args()

    if not args.scenarios and not args.runs:
        raise SystemExit("укажи --scenarios ... либо --runs ...")

    cfg = PoolConfig.load(args.pool)
    paths = {"pool": args.pool, "models": args.models, "tenants": args.tenants}
    pairs: list[tuple[float, bool]] = []

    for scenario in args.scenarios:
        for policy in args.policies:
            jobs = jobs_from_scenario(scenario, policy, paths)
            got = outcomes_offline(jobs, cfg)
            pairs += got
            print(f"[план] {Path(scenario).stem:22} {policy:12} {len(got):5} размещений")

    for run in args.runs:
        got = outcomes_from_run(Path(run))
        pairs += got
        print(f"[прогон] {Path(run).name:34} {len(got):5} размещений")

    report(pairs, args.bucket)


if __name__ == "__main__":
    main()
