#!/usr/bin/env python3
"""
Анализ прогона: считает качественные метрики работы кластера.

    python3 python/analyze.py runs/<run>                  # отчёт по одному прогону
    python3 python/analyze.py --compare runs/<A> runs/<B> # сравнение двух

Что считается (и почему это важно для сравнения политик размещения):

  throughput          — сколько задач кластер реально прожевал за прогон.
                        Прямая метрика из слайда "N задач / время".
  wait_p50/p90/p99    — время от подачи задачи до старта её пода, отдельно по
                        приоритетам. Показывает, работает ли приоритизация.
  gpu_hours_allocated — интеграл занятых карт по времени. Отношение к
                        gpu_hours_available даёт утилизацию.
  idle_gpu_hours      — простой железа: то, за что платят и не используют.
  borrowed_gpu_hours  — сколько GPU-часов команды взяли друг у друга
                        сверх своей номинальной квоты.
  starved_jobs        — задачи, которые так и не стартовали к концу прогона.
  preemptions         — сколько раз kueue вытеснял workload'ы.
  frag_free_on_busy   — среднее число свободных карт на уже занятых нодах
                        (фрагментация).
  cost_units / waste  — условная стоимость GPU-часов и та её часть, которую
                        съел завышенный класс карты.
  policy_changed_pct  — доля задач, где рекомендатель решил иначе, чем
                        пользователь. При --placement manual всегда 0.
  confidence_mean     — средняя вероятность того, что задача поместится
                        в выбранную карту (калиброванная, см. recommender.py).
  risky_placements_pct— доля размещений с риском падения выше 15%.
  oom_jobs            — задачи, убитые нехваткой VRAM: карта оказалась меньше,
                        чем реально потребовала модель.
  oom_wasted_gpu_hours— GPU-часы, потраченные впустую до момента падения.
  efficiency_score    — скоринг недоутилизации из "Задачи #3":
                        max(доля занятой памяти, доля потреблённой мощности).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_runtime import efficiency_scores  # noqa: E402
from lab import PoolConfig  # noqa: E402

BAR = "▁▂▃▄▅▆▇█"


# ──────────────────────────────────────────────────────────────────────────────
# загрузка
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def sparkline(values: list[float], width: int = 60, hi: float | None = None) -> str:
    """hi задаётся явно, когда нужно сравнить две серии в одном масштабе."""
    if not values:
        return ""
    step = max(1, len(values) // width)
    sampled = [
        statistics.fmean(values[i : i + step]) for i in range(0, len(values), step)
    ][:width]
    lo = 0.0
    hi = hi if hi else (max(sampled) or 1.0)
    return "".join(BAR[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in sampled)


# ──────────────────────────────────────────────────────────────────────────────
# расчёт метрик
# ──────────────────────────────────────────────────────────────────────────────


def _gpu_hours_by_pool(jobs: list[dict], time_scale: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for j in jobs:
        hours = (j["run_s"] or 0) * time_scale / 3600.0
        out[j["requested_pool"]] = round(
            out.get(j["requested_pool"], 0.0) + j["gpus"] * hours, 2
        )
    return out


def analyze(run_dir: Path, pool_cfg: PoolConfig) -> dict:
    ts = load_jsonl(run_dir / "timeseries.jsonl")
    planned = {j["job_id"]: j for j in load_jsonl(run_dir / "jobs.jsonl")}
    observed = load_json(run_dir / "jobs_observed.json", {})
    workloads = load_json(run_dir / "workloads.json", {})
    oom_events = load_jsonl(run_dir / "oom.jsonl")
    scenario = load_json(run_dir / "scenario.json", {})
    meta = load_json(run_dir / "meta.json", {})

    if not ts:
        raise SystemExit(f"{run_dir}: нет timeseries.jsonl — прогон не собирался?")

    time_scale = float(scenario.get("time_scale", 1.0))
    interval = float(meta.get("interval", 1.0))
    capacity = float(meta.get("gpu_capacity") or ts[0]["gpu"]["capacity"] or 1)
    pools = {p.name: p for p in pool_cfg.pools}

    wall_span = ts[-1]["ts"] - ts[0]["ts"]
    sim_hours = wall_span * time_scale / 3600.0 if wall_span else 0.0

    # ── по задачам ──────────────────────────────────────────────────────────
    jobs = []
    for jid, plan in planned.items():
        obs = observed.get(jid, {})
        created = obs.get("created_ts")
        # pod_start_ts — из статуса пода, разрешение 1 с независимо от такта
        # снапшотера. first_seen_running_ts — момент такта, он квантует
        # ожидание шагом --interval (при ×120 это 2 сим-минуты, из-за чего
        # медиана ожидания вырождается в пол измерения). start_ts — это
        # startTime Job'а: он ловит момент снятия suspend, но не задержку
        # kube-scheduler'а. Порядок предпочтения — от точного к грубому.
        started = (
            obs.get("pod_start_ts")
            or obs.get("first_seen_running_ts")
            or obs.get("start_ts")
        )
        finished = obs.get("completion_ts")
        # Длительность работы считаем от ПОСЛЕДНЕГО старта пода. У вытесненной
        # задачи kueue создаёт новый под, и finished - первый_старт включает
        # всё ожидание между попытками: на этих прогонах это раздувало run_s
        # на 20-37%, а у самих вытесненных задач — в 2-3 раза.
        run_started = (
            obs.get("last_pod_start_ts")
            or obs.get("pod_start_ts")
            or obs.get("first_seen_running_ts")
            or obs.get("start_ts")
        )
        # при kueue-native карта в плане не указана ("auto") — берём ту,
        # на которой задача фактически оказалась
        effective_pool = plan["requested_pool"]
        if effective_pool == "auto":
            effective_pool = obs.get("actual_pool") or "auto"
        pool = pools.get(effective_pool)
        min_pool = pools.get(plan["min_pool"])
        rec = {
            "job_id": jid,
            "tenant": plan["tenant"],
            "priority_class": plan["priority_class"],
            "requested_pool": effective_pool,
            "declared_pool": plan["requested_pool"],
            "min_pool": plan["min_pool"],
            "gpus": plan["gpus"],
            "min_gpus": plan["min_gpus"],
            "vram_gb": plan["vram_gb"],
            "oversized": plan["oversized"],
            "user_pool": plan.get("user_pool", plan["requested_pool"]),
            "user_gpus": plan.get("user_gpus", plan["gpus"]),
            "confidence": plan.get("confidence"),
            "attempt": plan.get("attempt", 1),
            "actual_vram_gb": plan.get("actual_vram_gb", 0.0),
            "policy_changed": plan.get("requested_pool") != plan.get("user_pool")
            or plan.get("gpus") != plan.get("user_gpus"),
            "planned_duration_s": plan["sim_minutes"] * 60 / time_scale,
            "submitted": created,
            "started": started,
            "finished": finished,
            "wait_s": (started - created) if (started and created) else None,
            "run_s": (finished - run_started) if (finished and run_started) else None,
            # сколько раз задача стартовала: >1 означает вытеснение с потерей
            # прогресса (чекпоинтов нет)
            "attempts": int(obs.get("attempts", 1) or 1),
            "completed": bool(finished),
            "started_flag": bool(started),
            "vram_provided_gb": (pool.vram_gb * plan["gpus"]) if pool else 0.0,
            "cost_rate": (pool.relative_cost * plan["gpus"]) if pool else 0.0,
            "min_cost_rate": (min_pool.relative_cost * plan["min_gpus"])
            if min_pool
            else 0.0,
        }
        jobs.append(rec)

    oom_by_job = {e["job_id"]: e for e in oom_events}
    for j in jobs:
        e = oom_by_job.get(j["job_id"])
        j["oom"] = e is not None
        j["oom_overshoot_gb"] = e["overshoot_gb"] if e else 0.0
        j["oom_wasted_s"] = e["wasted_seconds"] if e else 0.0

    submitted_jobs = [j for j in jobs if j["submitted"]]
    completed = [j for j in jobs if j["completed"]]
    started_jobs = [j for j in jobs if j["started_flag"]]
    starved = [j for j in submitted_jobs if not j["started_flag"]]

    waits = [j["wait_s"] * time_scale / 60.0 for j in started_jobs if j["wait_s"] is not None]

    def waits_for(pred) -> list[float]:
        return [
            j["wait_s"] * time_scale / 60.0
            for j in started_jobs
            if j["wait_s"] is not None and pred(j)
        ]

    # ── утилизация ──────────────────────────────────────────────────────────
    util_series = [r["gpu"]["utilization"] for r in ts]
    alloc_series = [r["gpu"]["allocated"] for r in ts]
    pending_series = [
        r["jobs"]["suspended"] + r["pods"]["unscheduled"] for r in ts
    ]
    frag_series = [r["gpu"]["free_on_busy_nodes"] for r in ts]

    gpu_hours_allocated = sum(alloc_series) * interval * time_scale / 3600.0
    gpu_hours_available = capacity * sim_hours

    borrowed_series = [
        sum(q.get("gpu_borrowed", 0.0) for q in r.get("queues", {}).values()) for r in ts
    ]
    borrowed_gpu_hours = sum(borrowed_series) * interval * time_scale / 3600.0

    # ── стоимость и перерасход класса карты ─────────────────────────────────
    cost_units = 0.0
    wasted_vram_gb_hours = 0.0
    for j in completed:
        hours = (j["run_s"] or 0) * time_scale / 3600.0
        cost_units += j["cost_rate"] * hours
        wasted_vram_gb_hours += max(0.0, j["vram_provided_gb"] - j["vram_gb"]) * hours

    # Оптимум и «цена решений» считаются по ПЛАНУ, а не по факту. Иначе они
    # зависят от того, сколько задач успело завершиться и сколько раз их
    # вытесняли, и у одного и того же потока задач получаются разные значения
    # в разных прогонах — сравнивать cost_overhead_pct становится нельзя.
    optimal_cost_units = 0.0
    planned_cost_units = 0.0
    for j in jobs:
        hours = j["planned_duration_s"] * time_scale / 3600.0
        optimal_cost_units += j["min_cost_rate"] * hours
        planned_cost_units += j["cost_rate"] * hours

    oom_jobs = [j for j in jobs if j["oom"]]
    oom_wasted_gpu_hours = sum(
        j["gpus"] * j["oom_wasted_s"] * time_scale / 3600.0 for j in oom_jobs
    )
    retries = len([j for j in jobs if int(j.get("attempt", 1) or 1) > 1])
    efficiency = efficiency_scores(run_dir / "usage.jsonl", interval, time_scale)

    known_conf = [j["confidence"] for j in jobs if j["confidence"] is not None]

    # ── OOM против уверенности планировщика ────────────────────────────────
    # Вопрос, на который отвечает блок: годится ли confidence как сигнал для
    # AdmissionCheck. Смотрим на него с двух сторон:
    #   recall    — какую долю падений порог поймал бы;
    #   precision — какую долю отклонённых задач мы отклонили бы зря.
    RISK_THRESHOLD = 0.85
    conf_known = [j for j in jobs if j["confidence"] is not None]
    oom_conf = [j["confidence"] for j in conf_known if j["oom"]]
    risky = [j for j in conf_known if j["confidence"] < RISK_THRESHOLD]
    risky_oom = [j for j in risky if j["oom"]]
    safe_oom = [
        j for j in conf_known if j["confidence"] >= RISK_THRESHOLD and j["oom"]
    ]

    # калибровка по корзинам: обещанная вероятность против фактической
    conf_buckets: dict[str, dict] = {}
    for j in conf_known:
        key = f"{round(j['confidence'], 1):.1f}"
        b = conf_buckets.setdefault(key, {"jobs": 0, "oom": 0})
        b["jobs"] += 1
        b["oom"] += int(j["oom"])
    for key, b in conf_buckets.items():
        b["actual_success"] = round(1 - b["oom"] / b["jobs"], 3)
        b["predicted_success"] = float(key)
    conf_buckets = dict(sorted(conf_buckets.items()))

    oom_vs_confidence = {
        "threshold": RISK_THRESHOLD,
        "oom_confidence_mean": round(statistics.fmean(oom_conf), 3)
        if oom_conf
        else None,
        "oom_confidence_median": round(statistics.median(oom_conf), 3)
        if oom_conf
        else None,
        "oom_confidence_min": round(min(oom_conf), 3) if oom_conf else None,
        "oom_confidence_max": round(max(oom_conf), 3) if oom_conf else None,
        "ok_confidence_mean": round(
            statistics.fmean([j["confidence"] for j in conf_known if not j["oom"]]), 3
        )
        if len(conf_known) > len(oom_conf)
        else None,
        # сколько падений было бы отсечено порогом
        "oom_below_threshold": len(risky_oom),
        "oom_above_threshold": len(safe_oom),
        "oom_recall_at_threshold": round(100 * len(risky_oom) / len(oom_conf), 1)
        if oom_conf
        else None,
        # цена этого отсечения
        "risky_placements": len(risky),
        "risky_precision_pct": round(100 * len(risky_oom) / len(risky), 1)
        if risky
        else None,
        "false_rejects": len(risky) - len(risky_oom),
        # сводная калибровка
        "predicted_oom_pct": round(
            100 * sum(1 - j["confidence"] for j in conf_known) / len(conf_known), 1
        )
        if conf_known
        else None,
        "actual_oom_pct": round(100 * len(oom_conf) / len(conf_known), 1)
        if conf_known
        else None,
        "buckets": conf_buckets,
    }

    preemptions = sum(int(w.get("evictions", 0) or 0) for w in workloads.values())
    eviction_reasons: dict[str, int] = {}
    for w in workloads.values():
        for r in w.get("eviction_reasons", []) or []:
            eviction_reasons[r] = eviction_reasons.get(r, 0) + 1

    per_tenant = {}
    for tenant in sorted({j["tenant"] for j in jobs}):
        sub = [j for j in jobs if j["tenant"] == tenant]
        done = [j for j in sub if j["completed"]]
        w = waits_for(lambda j, t=tenant: j["tenant"] == t)
        per_tenant[tenant] = {
            "submitted": len(sub),
            "completed": len(done),
            "starved": len([j for j in sub if j["submitted"] and not j["started_flag"]]),
            "wait_p50_sim_min": round(quantile(w, 0.5), 2) if w else None,
            "wait_p90_sim_min": round(quantile(w, 0.9), 2) if w else None,
            "wait_total_sim_hours": round(sum(w) / 60.0, 2) if w else None,
            "gpu_hours": round(
                sum(
                    j["gpus"] * (j["run_s"] or 0) * time_scale / 3600.0 for j in done
                ),
                2,
            ),
        }

    per_priority = {}
    for prio in sorted({j["priority_class"] for j in jobs}):
        w = waits_for(lambda j, p=prio: j["priority_class"] == p)
        sub = [j for j in jobs if j["priority_class"] == prio]
        per_priority[prio] = {
            "submitted": len(sub),
            "completed": len([j for j in sub if j["completed"]]),
            "wait_p50_sim_min": round(quantile(w, 0.5), 2) if w else None,
            "wait_p90_sim_min": round(quantile(w, 0.9), 2) if w else None,
            "wait_p99_sim_min": round(quantile(w, 0.99), 2) if w else None,
            "wait_total_sim_hours": round(sum(w) / 60.0, 2) if w else None,
        }

    return {
        "run": run_dir.name,
        "scenario": scenario.get("name"),
        "placement": scenario.get("placement", scenario.get("mode", "manual")),
        "time_scale": time_scale,
        "wall_seconds": round(wall_span, 1),
        "sim_hours": round(sim_hours, 3),
        "gpu_capacity": capacity,
        "jobs_submitted": len(submitted_jobs),
        "jobs_started": len(started_jobs),
        "jobs_completed": len(completed),
        "starved_jobs": len(starved),
        "throughput_per_sim_hour": round(len(completed) / sim_hours, 2)
        if sim_hours
        else 0.0,
        "wait_p50_sim_min": round(quantile(waits, 0.5), 2) if waits else None,
        "wait_p90_sim_min": round(quantile(waits, 0.9), 2) if waits else None,
        "wait_p99_sim_min": round(quantile(waits, 0.99), 2) if waits else None,
        "wait_mean_sim_min": round(statistics.fmean(waits), 2) if waits else None,
        # суммарное ожидание по всем задачам. В отличие от квантилей учитывает
        # сразу и сколько задач ждало, и как долго, поэтому не зависит от
        # того, в какую точку распределения попал очередной перцентиль
        "wait_total_sim_hours": round(sum(waits) / 60.0, 2) if waits else None,
        "wait_over_threshold": {
            str(t): len([w for w in waits if w > t])
            for t in (5, 15, 30, 60, 120)
        },
        "wait_hours_per_completed_job": round(
            sum(waits) / 60.0 / len(completed), 4
        )
        if waits and completed
        else None,
        "gpu_utilization_mean": round(statistics.fmean(util_series), 4),
        "gpu_utilization_p90": round(quantile(util_series, 0.9), 4),
        "gpu_hours_allocated": round(gpu_hours_allocated, 2),
        "gpu_hours_available": round(gpu_hours_available, 2),
        "idle_gpu_hours": round(max(0.0, gpu_hours_available - gpu_hours_allocated), 2),
        "borrowed_gpu_hours": round(borrowed_gpu_hours, 2),
        "queue_depth_mean": round(statistics.fmean(pending_series), 2),
        "queue_depth_max": max(pending_series) if pending_series else 0,
        "frag_free_on_busy_mean": round(statistics.fmean(frag_series), 2),
        "preemptions": preemptions,
        "eviction_reasons": eviction_reasons,
        "cost_units": round(cost_units, 1),
        "optimal_cost_units": round(optimal_cost_units, 1),
        # цена того, что выбрал планировщик, если бы всё отработало по плану:
        # чистая характеристика политики, без примеси вытеснений и OOM
        "planned_cost_units": round(planned_cost_units, 1),
        # разница между фактической и плановой ценой — это переделанная после
        # вытеснений работа (чекпоинтов нет, прогресс теряется целиком)
        "preemption_wasted_cost_units": round(
            max(0.0, cost_units - planned_cost_units), 1
        ),
        "cost_overhead_pct": round(
            100 * (planned_cost_units / optimal_cost_units - 1), 1
        )
        if optimal_cost_units
        else 0.0,
        "wasted_vram_gb_hours": round(wasted_vram_gb_hours, 1),
        "oversized_jobs_pct": round(
            100 * len([j for j in jobs if j["oversized"]]) / max(1, len(jobs)), 1
        ),
        "oom_jobs": len(oom_jobs),
        "oom_pct": round(100 * len(oom_jobs) / max(1, len(submitted_jobs)), 1),
        "oom_wasted_gpu_hours": round(oom_wasted_gpu_hours, 2),
        "oom_retries": retries,
        "oom_vs_confidence": oom_vs_confidence,
        "efficiency_score_mean": efficiency.get("efficiency_score_mean"),
        "memory_score_mean": efficiency.get("memory_score_mean"),
        "power_score_mean": efficiency.get("power_score_mean"),
        "underutilized_gpu_hours": efficiency.get("wasted_gpu_hours"),
        "efficiency_score_weighted": efficiency.get("efficiency_score_weighted"),
        "policy_changed_pct": round(
            100 * len([j for j in jobs if j["policy_changed"]]) / max(1, len(jobs)), 1
        ),
        "confidence_mean": round(statistics.fmean(known_conf), 3) if known_conf else None,
        "risky_placements_pct": round(
            100 * len([c for c in known_conf if c < 0.85]) / len(known_conf), 1
        )
        if known_conf
        else None,
        "gpu_hours_by_pool": _gpu_hours_by_pool(completed, time_scale),
        "per_tenant": per_tenant,
        "per_priority": per_priority,
        "_series": {
            "utilization": util_series,
            "queue_depth": pending_series,
            "borrowed": borrowed_series,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# вывод
# ──────────────────────────────────────────────────────────────────────────────


def fmt(v, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}{unit}"
    return f"{v}{unit}"


def report_text(m: dict) -> str:
    lines = []
    a = lines.append
    a(f"# Прогон {m['run']}")
    a("")
    a(f"сценарий **{m['scenario']}**, политика размещения **{m['placement']}**, "
      f"сжатие времени ×{m['time_scale']:g}")
    a(f"реального времени {m['wall_seconds']:.0f}s = "
      f"{m['sim_hours']:.1f} симулированных часов, "
      f"{m['gpu_capacity']:.0f} GPU в кластере")
    a("")
    a("## Пропускная способность")
    a("")
    a(f"| подано | стартовало | завершено | не стартовало | задач/сим-час |")
    a(f"|---:|---:|---:|---:|---:|")
    a(f"| {m['jobs_submitted']} | {m['jobs_started']} | {m['jobs_completed']} "
      f"| {m['starved_jobs']} | {m['throughput_per_sim_hour']} |")
    a("")
    a("## Время ожидания (сим-минуты, от подачи до старта пода)")
    a("")
    a("| p50 | p90 | p99 | среднее |")
    a("|---:|---:|---:|---:|")
    a(f"| {fmt(m['wait_p50_sim_min'])} | {fmt(m['wait_p90_sim_min'])} "
      f"| {fmt(m['wait_p99_sim_min'])} | {fmt(m['wait_mean_sim_min'])} |")
    a("")
    if m.get("wait_total_sim_hours") is not None:
        a(f"Суммарно кластер продержал задачи в очереди "
          f"**{m['wait_total_sim_hours']} сим-часов** "
          f"({m['wait_hours_per_completed_job']} сим-часа на завершённую "
          f"задачу). В отличие от квантилей эта величина учитывает сразу и "
          f"сколько задач ждало, и как долго, поэтому не зависит от того, в "
          f"какую точку распределения попал очередной перцентиль.")
        a("")
        a("| ждали дольше | задач |")
        a("|---|---:|")
        for t in (5, 15, 30, 60, 120):
            n = m["wait_over_threshold"].get(str(t))
            if n is not None:
                a(f"| {t} сим-мин | {n} |")
        a("")
    a("### По приоритетам")
    a("")
    a("| приоритет | подано | завершено | p50 | p90 | p99 | ожидание всего, сим-ч |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for prio, v in m["per_priority"].items():
        a(f"| {prio} | {v['submitted']} | {v['completed']} "
          f"| {fmt(v['wait_p50_sim_min'])} | {fmt(v['wait_p90_sim_min'])} "
          f"| {fmt(v['wait_p99_sim_min'])} | {fmt(v.get('wait_total_sim_hours'))} |")
    a("")
    a("### По командам")
    a("")
    a("| команда | подано | завершено | не стартовало | p50 | p90 "
      "| ожидание всего, сим-ч | GPU-часов |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tenant, v in m["per_tenant"].items():
        a(f"| {tenant} | {v['submitted']} | {v['completed']} | {v['starved']} "
          f"| {fmt(v['wait_p50_sim_min'])} | {fmt(v['wait_p90_sim_min'])} "
          f"| {fmt(v.get('wait_total_sim_hours'))} | {v['gpu_hours']} |")
    a("")
    a("## Утилизация")
    a("")
    a(f"- средняя занятость GPU: **{100 * m['gpu_utilization_mean']:.1f}%** "
      f"(p90 {100 * m['gpu_utilization_p90']:.1f}%)")
    a(f"- GPU-часов занято / доступно: "
      f"**{m['gpu_hours_allocated']} / {m['gpu_hours_available']}**")
    a(f"- простой: **{m['idle_gpu_hours']} GPU-часов**")
    a(f"- заимствовано между очередями: **{m['borrowed_gpu_hours']} GPU-часов**")
    a(f"- средняя глубина очереди: {m['queue_depth_mean']} (максимум {m['queue_depth_max']})")
    a(f"- фрагментация (свободных карт на занятых нодах, в среднем): "
      f"{m['frag_free_on_busy_mean']}")
    a(f"- вытеснений: {m['preemptions']} {m['eviction_reasons'] or ''}")
    a("")
    a("## Выбор карты")
    a("")
    a(f"- решений рекомендателя, отличных от выбора пользователя: "
      f"**{m['policy_changed_pct']}%**")
    if m.get("confidence_mean") is None:
        a("- вероятность вместимости не определена (карту выбирал kueue)")
    else:
        a(f"- средняя вероятность вместимости: {m['confidence_mean']}; "
          f"размещений с риском падения выше 15%: {m['risky_placements_pct']}%")
    a(f"- GPU-часы по типам карт: {m['gpu_hours_by_pool']}")
    a("")
    a("### OOM: карта оказалась меньше, чем нужно модели")
    a("")
    a(f"- задач убито по нехватке VRAM: **{m['oom_jobs']}** ({m['oom_pct']}% от поданных)")
    a(f"- впустую потрачено до падения: **{m['oom_wasted_gpu_hours']} GPU-часов**")
    a(f"- повторных постановок после OOM: {m['oom_retries']}")
    ovc = m.get("oom_vs_confidence") or {}
    if ovc.get("oom_confidence_mean") is not None:
        a("")
        a(f"**Уверенность у упавших задач** (порог AdmissionCheck "
          f"{ovc['threshold']}):")
        a("")
        a(f"- средняя confidence у упавших: **{ovc['oom_confidence_mean']}** "
          f"(медиана {ovc['oom_confidence_median']}, "
          f"разброс {ovc['oom_confidence_min']}…{ovc['oom_confidence_max']})")
        a(f"- у выживших: {ovc['ok_confidence_mean']}")
        a(f"- падений ниже порога: **{ovc['oom_below_threshold']}** из "
          f"{ovc['oom_below_threshold'] + ovc['oom_above_threshold']} "
          f"(порог поймал бы {ovc['oom_recall_at_threshold']}% падений)")
        a(f"- цена: отклонили бы {ovc['risky_placements']} задач, из них "
          f"зря — {ovc['false_rejects']} "
          f"(точность порога {ovc['risky_precision_pct']}%)")
        a(f"- калибровка: модель обещала {ovc['predicted_oom_pct']}% падений, "
          f"фактически {ovc['actual_oom_pct']}%")
        if ovc.get("buckets"):
            a("")
            a("| confidence | задач | упало | факт. успех | предсказано |")
            a("|---:|---:|---:|---:|---:|")
            for key, b in ovc["buckets"].items():
                a(f"| {key} | {b['jobs']} | {b['oom']} | "
                  f"{100 * b['actual_success']:.0f}% | "
                  f"{100 * b['predicted_success']:.0f}% |")
    a("")
    a("### Недоутилизация (скоринг из задачи #3)")
    a("")
    if m.get("efficiency_score_mean") is None:
        a("- модель потребления не запускалась (нет usage.jsonl)")
    else:
        a(f"- memory_score {m['memory_score_mean']}, "
          f"power_score {m['power_score_mean']}, "
          f"efficiency = max(...) = **{m['efficiency_score_mean']}**")
        if m.get("efficiency_score_weighted") is not None:
            a(f"- взвешенно по GPU-часам: **{m['efficiency_score_weighted']}** "
              f"(невзвешенное среднее выше завышают короткие задачи)")
        a(f"- зарезервировано, но не использовано: "
          f"**{m['underutilized_gpu_hours']} GPU-часов**")
    a("")
    a("### Перерасход относительно минимально достаточной карты")
    a("")
    a(f"- условная стоимость прогона: **{m['cost_units']}** "
      f"при оптимальном подборе карт было бы **{m['optimal_cost_units']}** "
      f"(+{m['cost_overhead_pct']}%)")
    a(f"- впустую зарезервировано VRAM: **{m['wasted_vram_gb_hours']} ГБ·час**")
    a(f"- доля задач с завышенным классом карты: {m['oversized_jobs_pct']}%")
    a("")
    a("## Динамика")
    a("")
    a("```")
    a(f"утилизация  {sparkline(m['_series']['utilization'])}")
    a(f"очередь     {sparkline(m['_series']['queue_depth'])}")
    a(f"заимствов.  {sparkline(m['_series']['borrowed'])}")
    a("```")
    return "\n".join(lines)


COMPARE_ROWS = [
    ("задач завершено", "jobs_completed", "{:.0f}", "+"),
    ("задач не стартовало", "starved_jobs", "{:.0f}", "-"),
    ("пропускная способность, задач/сим-час", "throughput_per_sim_hour", "{:.2f}", "+"),
    ("ожидание p50, сим-мин", "wait_p50_sim_min", "{:.1f}", "-"),
    ("ожидание p90, сим-мин", "wait_p90_sim_min", "{:.1f}", "-"),
    ("ожидание p99, сим-мин", "wait_p99_sim_min", "{:.1f}", "-"),
    # интегральная метрика: не зависит от того, куда попал перцентиль
    ("ожидание всего, сим-часов", "wait_total_sim_hours", "{:.1f}", "-"),
    ("ожидание на задачу, сим-часов", "wait_hours_per_completed_job", "{:.3f}", "-"),
    ("средняя утилизация GPU", "gpu_utilization_mean", "{:.3f}", "+"),
    ("GPU-часов занято", "gpu_hours_allocated", "{:.1f}", "+"),
    ("простой, GPU-часов", "idle_gpu_hours", "{:.1f}", "-"),
    ("заимствовано, GPU-часов", "borrowed_gpu_hours", "{:.1f}", "+"),
    ("глубина очереди, средняя", "queue_depth_mean", "{:.1f}", "-"),
    ("вытеснений", "preemptions", "{:.0f}", "="),
    ("условная стоимость GPU-часов", "cost_units", "{:.0f}", "-"),
    ("перерасход к оптимуму, %", "cost_overhead_pct", "{:.1f}", "-"),
    ("впустую зарезервировано VRAM, ГБ·час", "wasted_vram_gb_hours", "{:.0f}", "-"),
    ("задач убито по OOM", "oom_jobs", "{:.0f}", "-"),
    ("GPU-часов сожжено на OOM", "oom_wasted_gpu_hours", "{:.1f}", "-"),
    ("efficiency_score (задача #3)", "efficiency_score_mean", "{:.3f}", "+"),
    ("недоутилизировано, GPU-часов", "underutilized_gpu_hours", "{:.1f}", "-"),
]


def report_compare(a: dict, b: dict) -> str:
    lines = []
    out = lines.append
    out(f"# Сравнение политик: {a['placement']} ({a['run']}) "
        f"vs {b['placement']} ({b['run']})")
    out("")
    if a.get("scenario") != b.get("scenario"):
        out(f"> ВНИМАНИЕ: сценарии разные ({a.get('scenario')} и {b.get('scenario')}), "
            f"сравнение некорректно")
        out("")
    out(f"| метрика | {a['placement']} | {b['placement']} | Δ |")
    out("|---|---:|---:|---:|")
    for title, key, tpl, better in COMPARE_ROWS:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            out(f"| {title} | {fmt(va)} | {fmt(vb)} | — |")
            continue
        if va == 0:
            delta = "—" if vb == 0 else "+∞"
        else:
            pct = 100 * (vb - va) / abs(va)
            mark = ""
            if better == "+":
                mark = " ✅" if pct > 1 else (" ❌" if pct < -1 else "")
            elif better == "-":
                mark = " ✅" if pct < -1 else (" ❌" if pct > 1 else "")
            delta = f"{pct:+.1f}%{mark}"
        out(f"| {title} | {tpl.format(va)} | {tpl.format(vb)} | {delta} |")
    out("")
    out("## Ожидание по приоритетам (p90, сим-мин)")
    out("")
    prios = sorted(set(a["per_priority"]) | set(b["per_priority"]))
    out(f"| приоритет | {a['placement']} | {b['placement']} |")
    out("|---|---:|---:|")
    for p in prios:
        out(f"| {p} | {fmt(a['per_priority'].get(p, {}).get('wait_p90_sim_min'))} "
            f"| {fmt(b['per_priority'].get(p, {}).get('wait_p90_sim_min'))} |")
    out("")
    out("## Завершено задач по командам")
    out("")
    tenants = sorted(set(a["per_tenant"]) | set(b["per_tenant"]))
    out(f"| команда | {a['placement']} | {b['placement']} |")
    out("|---|---:|---:|")
    for t in tenants:
        out(f"| {t} | {a['per_tenant'].get(t, {}).get('completed', 0)} "
            f"| {b['per_tenant'].get(t, {}).get('completed', 0)} |")
    out("")
    out("```")
    shared = max(
        max(a["_series"]["utilization"], default=0),
        max(b["_series"]["utilization"], default=0),
    )
    out(f"утилизация {a['placement']:11} {sparkline(a['_series']['utilization'], hi=shared)}")
    out(f"утилизация {b['placement']:11} {sparkline(b['_series']['utilization'], hi=shared)}")
    out("```")
    return "\n".join(lines)


def write_csv(run_dir: Path, m: dict) -> None:
    import csv

    with (run_dir / "utilization.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tick", "utilization", "queue_depth", "borrowed_gpu"])
        s = m["_series"]
        for i in range(len(s["utilization"])):
            w.writerow(
                [i, s["utilization"][i], s["queue_depth"][i], s["borrowed"][i]]
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="каталоги прогонов")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument("--json", action="store_true", help="печатать метрики как JSON")
    args = ap.parse_args()

    cfg = PoolConfig.load(args.pool)

    if args.compare:
        a = analyze(Path(args.compare[0]), cfg)
        b = analyze(Path(args.compare[1]), cfg)
        text = report_compare(a, b)
        print(text)
        out = Path(args.compare[1]) / "compare.md"
        out.write_text(text, encoding="utf-8")
        print(f"\n[analyze] отчёт сохранён: {out}", file=sys.stderr)
        return

    if not args.runs:
        raise SystemExit("укажи каталог прогона или --compare A B")

    for run in args.runs:
        run_dir = Path(run)
        m = analyze(run_dir, cfg)
        clean = {k: v for k, v in m.items() if not k.startswith("_")}
        (run_dir / "metrics.json").write_text(
            json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        write_csv(run_dir, m)
        if args.json:
            print(json.dumps(clean, ensure_ascii=False, indent=1))
        else:
            text = report_text(m)
            print(text)
            (run_dir / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
