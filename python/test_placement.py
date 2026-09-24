#!/usr/bin/env python3
"""
Проверка политик размещения без кластера.

    python3 python/test_placement.py            # всё
    python3 python/test_placement.py --only budget

Три части:

  budget   бюджет срочности по приоритету в queue-aware. Проверяется на
           модельных состояниях из отчёта (таблица 6): решения должны
           совпасть с опубликованными. До этой правки код бюджета по
           приоритету не знал и воспроизводил только колонку cheapest.

  quota    арифметика квотного запаса. Состояния ClusterQueue собираются
           из настоящего manifests/03-cohort-and-queues.yaml, к нему
           дописывается status — то есть проверяется тот же код разбора,
           который работает с живым кластером.

  kueue    решения kueue-aware там, где картина по kueue расходится с
           картиной по подам. Это главный аргумент за отдельный источник
           состояния: в обоих случаях счёт по подам даёт неверный ответ.

Возвращает ненулевой код, если хоть одна проверка не сошлась.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kueue_state import build_state  # noqa: E402
from lab import ROOT, PoolConfig  # noqa: E402
from recommender import (  # noqa: E402
    CheapestFitPolicy,
    ClusterState,
    JobRequest,
    KueueAwarePolicy,
    QueueAwarePolicy,
    UncertaintyModel,
)

PRIORITY_VALUES = {"production": 1000, "high": 500, "normal": 100, "low": 10}
CFG = PoolConfig.load(ROOT / "config" / "pool.yaml")
FLAVOR_OF = {p.name: p.flavor for p in CFG.pools}
PRODUCT_OF = {p.name: p.product for p in CFG.pools}
CAPACITY = {p.flavor: float(p.total_gpus) for p in CFG.pools}

failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:<52} {got}" + ("" if ok else f"  ожидалось {want}"))
    if not ok:
        failures.append(label)


def request(vram: float, priority: str = "normal", queue: str = "team-a") -> JobRequest:
    return JobRequest(
        model_id="test",
        params_b=7,
        precision="fp16",
        ctx_tokens=8192,
        vram_gb=vram,
        tenant=queue[-1],
        queue=queue,
        priority_class=priority,
        priority_value=PRIORITY_VALUES[priority],
    )


def pods_state(free: dict[str, float]) -> ClusterState:
    """Состояние, как его видит сборщик по подам: только свободные карты."""
    return ClusterState(
        free_gpu_by_flavor={f: float(free.get(f, CAPACITY[f])) for f in CAPACITY},
        total_gpu_by_flavor=dict(CAPACITY),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Бюджет срочности: воспроизведение таблицы 6 отчёта
# ──────────────────────────────────────────────────────────────────────────────


def test_budget() -> None:
    print("\nБюджет срочности по приоритету (таблица 6 отчёта)")
    print("  состояние кластера            задача   cheapest  production  normal  low")

    cheapest = CheapestFitPolicy(CFG)
    qa = QueueAwarePolicy(CFG)

    all_free = {f: CAPACITY[f] for f in CAPACITY}
    cases = [
        ("всё свободно", 11.0, all_free, ("t4×1", "t4×1", "t4×1", "t4×1")),
        ("T4 заняты", 11.0, {**all_free, "gpu-t4": 0}, ("t4×1", "l4×1", "l4×1", "t4×1")),
        # В отчёте эта строка описана как "A100-80 занят на 67%". Одной этой
        # занятости мало: пока свободен A100-40, вариант a100-40×2 стоит 7.0
        # против 6.0 у a100-80 и при нулевой занятости выигрывает у обоих.
        # Опубликованные решения получаются, когда занят и он, — то есть
        # состояние в отчёте описано неполно.
        (
            "A100-80 занят на 67%, A100-40 занят",
            62.0,
            {**all_free, "gpu-a100-80": 2, "gpu-a100-40": 0},
            ("a100-80×1", "h100×1", "h100×1", "a100-80×1"),
        ),
        (
            "свободны только дорогие",
            11.0,
            {f: 0 for f in CAPACITY} | {"gpu-a100-80": 6, "gpu-h100": 8},
            ("t4×1", "a100-80×1", "t4×1", "t4×1"),
        ),
        ("всё занято", 11.0, {f: 0 for f in CAPACITY}, ("t4×1", "t4×1", "t4×1", "t4×1")),
    ]

    for label, vram, free, want in cases:
        st = pods_state(free)
        got = [
            f"{cheapest.recommend(request(vram), st).pool}×"
            f"{cheapest.recommend(request(vram), st).gpus}"
        ]
        for prio in ("production", "normal", "low"):
            p = qa.recommend(request(vram, prio), st)
            got.append(f"{p.pool}×{p.gpus}")
        check(f"{label} / {vram:.0f}ГБ", tuple(got), want)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Квотный запас
# ──────────────────────────────────────────────────────────────────────────────


def cluster_queues(usage: dict[str, dict[str, float]]) -> list[dict]:
    """Настоящие ClusterQueue из манифеста + синтетический status.

    Квоты, лимиты заимствования и выдачи берутся из того же файла, который
    применяется в кластер, — проверяется реальная раскладка, а не выдуманная.
    """
    docs = [
        d
        for d in yaml.safe_load_all(
            (ROOT / "manifests" / "03-cohort-and-queues.yaml").read_text(encoding="utf-8")
        )
        if d and d.get("kind") == "ClusterQueue"
    ]
    for cq in docs:
        used = usage.get(cq["metadata"]["name"], {})
        cq["status"] = {
            "pendingWorkloads": 0,
            "flavorsReservation": [
                {
                    "name": flavor,
                    "resources": [{"name": "nvidia.com/gpu", "total": str(int(n))}],
                }
                for flavor, n in used.items()
            ],
        }
    return docs


def workload(queue: str, pool: str, gpus: int, priority: str) -> dict:
    """Ожидающий допуска Workload: карта задана nodeSelector'ом, как у нас
    её задают и пользователь, и рекомендатель."""
    return {
        "kind": "Workload",
        "metadata": {"name": f"wl-{queue}-{pool}", "namespace": queue},
        "spec": {
            "queueName": queue,
            "priority": PRIORITY_VALUES[priority],
            "podSets": [
                {
                    "count": 1,
                    "template": {
                        "spec": {
                            "nodeSelector": {"nvidia.com/gpu.product": PRODUCT_OF[pool]},
                            "containers": [
                                {"resources": {"requests": {"nvidia.com/gpu": str(gpus)}}}
                            ],
                        }
                    },
                }
            ],
        },
        "status": {},
    }


def kueue_state(usage, pending=()) -> ClusterState:
    return build_state(
        cluster_queues(usage),
        list(pending),
        {p.product: p.flavor for p in CFG.pools},
    )


def test_quota() -> None:
    print("\nКвотный запас: свободно по квоте против свободно физически")

    # team-b и team-c выбрали свой номинал по T4 (3 и 5), team-a не запускала
    # ничего. Физически свободны 2 карты — номинал team-a, у которой
    # lendingLimit по T4 равен нулю.
    st = kueue_state({"team-b": {"gpu-t4": 3}, "team-c": {"gpu-t4": 5}})
    check("физически свободно T4", st.free_gpu_by_flavor["gpu-t4"], 2.0)
    check("team-b: доступно T4 по квоте", st.headroom("team-b", "gpu-t4").free, 0.0)
    check("team-a: доступно T4 по квоте", st.headroom("team-a", "gpu-t4").free, 2.0)

    # team-b заняла 5 карт A100-40 при номинале 3: одну одолжила у team-a,
    # одну у team-c. Физически свободных нет, но два своих номинала team-a
    # вернёт вытеснением.
    st = kueue_state({"team-b": {"gpu-a100-40": 5}, "team-c": {"gpu-a100-40": 1}})
    check("физически свободно A100-40", st.free_gpu_by_flavor["gpu-a100-40"], 0.0)
    hr = st.headroom("team-a", "gpu-a100-40")
    check("team-a: A100-40 доступно сейчас", hr.free, 0.0)
    check("team-a: A100-40 вернётся вытеснением", hr.reclaim, 2.0)
    check("team-b: A100-40 больше не занять", st.headroom("team-b", "gpu-a100-40").free, 0.0)

    # Потолок: team-c по H100 имеет номинал 1 и borrowingLimit 7, но соседи
    # отдают только 1 (team-a) + 2 (team-b) — 70B в fp16 на 4 картах у неё
    # проходит впритык и только при полностью пустом пуле.
    st = kueue_state({})
    check("team-c: потолок по H100", st.headroom("team-c", "gpu-h100").limit, 8.0)
    check("team-c: доступно H100 сейчас", st.headroom("team-c", "gpu-h100").free, 4.0)

    # Глубина очереди: считаем только тех, кто нас не пропустит
    pending = [
        workload("team-b", "l4", 2, "production"),
        workload("team-c", "l4", 1, "low"),
    ]
    st = kueue_state({}, pending)
    check("впереди low-задачи", st.ahead_gpus("gpu-l4", PRIORITY_VALUES["low"]), 3.0)
    check("впереди high-задачи", st.ahead_gpus("gpu-l4", PRIORITY_VALUES["high"]), 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Решения kueue-aware
# ──────────────────────────────────────────────────────────────────────────────


def test_kueue_aware() -> None:
    print("\nkueue-aware против queue-aware там, где подов недостаточно")
    qa = QueueAwarePolicy(CFG, UncertaintyModel())
    ka = KueueAwarePolicy(CFG, UncertaintyModel())

    # ── Случай 1: карты простаивают, но заперты чужой квотой ────────────────
    # T4: team-b выбрала весь свой номинал (3), team-c — свой (5). Свободны
    # 2 карты, но это номинал team-a, у которой lendingLimit по T4 равен нулю.
    # L4 загружен так же (свободны 2), но там свободное принадлежит самой
    # team-b. По подам оба флейвора выглядят одинаково, T4 вдвое дешевле —
    # и queue-aware уводит задачу к картам, которых ей не дадут.
    usage = {
        "team-a": {"gpu-l4": 2, "gpu-a100-40": 2},
        "team-b": {"gpu-t4": 3, "gpu-l4": 2},
        "team-c": {"gpu-t4": 5, "gpu-l4": 4, "gpu-a100-40": 1},
    }
    ks = kueue_state(usage)
    ps = pods_state({"gpu-t4": 2, "gpu-l4": 2, "gpu-a100-40": 3})
    req = request(11.0, "normal", "team-b")
    a, b = qa.recommend(req, ps), ka.recommend(req, ks)
    check("queue-aware отправляет на запертые T4", f"{a.pool}×{a.gpus}", "t4×1")
    check("kueue-aware: по квоте T4 недоступен", ks.headroom("team-b", "gpu-t4").free, 0.0)
    check("kueue-aware выбирает доступное", f"{b.pool}×{b.gpus}", "l4×1")

    # ── Случай 2: карты заняты, но заняты нашим номиналом ───────────────────
    # A100-80 (6 карт, номинал team-a — 4) держат team-b и team-c, крупные
    # флейворы заняты целиком. Физически свободного нет нигде, но свои
    # четыре карты team-a вернёт вытеснением: reclaimWithinCohort: Any.
    usage = {
        "team-b": {"gpu-a100-80": 4, "gpu-a100-40": 4, "gpu-h100": 4},
        "team-c": {"gpu-a100-80": 2, "gpu-a100-40": 2, "gpu-h100": 4},
    }
    ks = kueue_state(usage)
    ps = pods_state({"gpu-a100-40": 0, "gpu-a100-80": 0, "gpu-h100": 0})
    req = request(70.0, "production", "team-a")
    a, b = qa.recommend(req, ps), ka.recommend(req, ks)
    check("team-a: A100-80 вернётся вытеснением", ks.headroom("team-a", "gpu-a100-80").reclaim, 4.0)
    check("queue-aware просто встаёт в очередь", "вытеснением" in a.reason, False)
    check("kueue-aware забирает свою квоту", f"{b.pool}×{b.gpus}", "a100-80×1")
    check("и говорит, почему", "вытеснением" in b.reason, True)
    print(f"       └ {b.reason}")

    # Та же картина, но задача низкоприоритетная: вытеснять ради batch-задачи
    # нечего, и бюджет 1.5× не пускает её на карту дороже.
    b = ka.recommend(request(70.0, "low", "team-a"), ks)
    check("low не вытесняет, а встаёт в очередь", "стартовать негде" in b.reason, True)

    # ── Случай 3: свободно везде, но перед L4 стоит очередь ─────────────────
    # 20 ГБ одинаково хорошо ложатся на l4×1 и на t4×2 — оба стоят 2.0.
    # Разницу видно только по очереди, которой в подах нет вообще.
    req = request(20.0, "normal", "team-b")
    quiet = ka.recommend(req, kueue_state({}))
    check("пустой кластер: одна карта лучше двух", f"{quiet.pool}×{quiet.gpus}", "l4×1")
    busy = ka.recommend(req, kueue_state({}, [workload("team-c", "l4", 8, "production")]))
    check("перед L4 очередь на 8 карт", busy.pool != "l4", True)
    check("уходим на равный по цене t4×2", f"{busy.pool}×{busy.gpus}", "t4×2")
    print(f"       └ {busy.reason}")

    # Очередь из низкоприоритетных задач нас не блокирует: BestEffortFIFO
    # пропустит production вперёд, и обходить L4 незачем.
    low_q = ka.recommend(
        request(20.0, "production", "team-b"),
        kueue_state({}, [workload("team-c", "l4", 8, "low")]),
    )
    check("очередь из low прод не тормозит", f"{low_q.pool}×{low_q.gpus}", "l4×1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["budget", "quota", "kueue"])
    args = ap.parse_args()

    if args.only in (None, "budget"):
        test_budget()
    if args.only in (None, "quota"):
        test_quota()
    if args.only in (None, "kueue"):
        test_kueue_aware()

    print()
    if failures:
        print(f"не сошлось: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("все проверки прошли")


if __name__ == "__main__":
    main()
