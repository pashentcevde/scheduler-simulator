#!/usr/bin/env python3

#Генератор нагрузки: подаёт в кластер задачи по сценарию.


from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import (  # noqa: E402
    A_JOBID,
    JobSpec,
    ModelCatalog,
    PoolConfig,
    apply_docs,
    cheapest_placement,
    eprint,
    load_yaml,
    oversized_placement,
    poisson_gap,
    kubectl_json,
    undersized_placement,
    render_job,
    weighted_choice,
)
from recommender import (  # noqa: E402
    ClusterStateSource,
    JobRequest,
    Placement,
    UncertaintyModel,
    build_policy,
    all_policies,
    next_tier_placement,
)
from kueue_state import KueueStateSource

NAMESPACES = {"a": "team-a", "b": "team-b", "c": "team-c"}
QUEUES = {"a": "team-a", "b": "team-b", "c": "team-c"}
PRIORITY_VALUES = {"production": 1000, "high": 500, "normal": 100, "low": 10}


def resolve_tenants(scenario: dict, tenants_path: str) -> list[dict]:
    """Развернуть tenants_preset + tenant_overrides в список команд"""
    if scenario.get("tenants"):
        return scenario["tenants"]
    presets = load_yaml(tenants_path)["presets"]
    preset_name = scenario.get("tenants_preset", "standard")
    if preset_name not in presets:
        raise SystemExit(
            f"неизвестный пресет {preset_name!r}, доступны: {', '.join(presets)}"
        )
    tenants = [dict(t) for t in presets[preset_name]]
    for tid, patch in (scenario.get("tenant_overrides") or {}).items():
        for t in tenants:
            if t["id"] == tid:
                t.update(patch)
                break
        else:
            raise SystemExit(f"tenant_overrides: нет команды {tid!r} в пресете")
    return tenants


def _sanitize_run_id(name: str) -> str:
    """Имя прогона попадает в имена Job'ов, поэтому приводим его к DNS-1123."""
    clean = "".join(c if (c.isalnum() and c.isascii()) else "-" for c in name.lower())
    clean = clean.strip("-")[-40:].strip("-")
    return clean or "run"


def rate_multiplier(phases: list[dict], tenant: str, sim_min: float) -> float:
    if not phases:
        return 1.0
    for ph in phases:
        if ph["from_sim_minute"] <= sim_min < ph["to_sim_minute"]:
            return float(ph.get("rates", {}).get(tenant, 1.0))
    return 1.0


def build_plan(
    scenario: dict, cfg: PoolConfig, cat: ModelCatalog, run_id: str, tenants: list[dict]
) -> list[dict]:
    """Детерминированно построить список задач с временами подачи"""
    rng = random.Random(scenario.get("seed", 0))
    rng_oversize = random.Random(scenario.get("seed", 0) + 10_000_019)
    rng_truth = random.Random(scenario.get("seed", 0) + 20_000_003)
    horizon = float(scenario["duration_sim_minutes"])
    max_jobs = int(scenario.get("max_jobs", 10_000))
    phases = scenario.get("phases", []) or []
    profiles = {p["name"]: p for p in cat.job_profiles}

    events: list[tuple[float, dict]] = []

    for tenant_cfg in tenants:
        tenant = tenant_cfg["id"]
        base_rate_per_min = float(tenant_cfg["arrival_per_sim_hour"]) / 60.0
        model_mix = tenant_cfg.get("model_mix") or {
            m["id"]: 1.0 for m in cat.models
        }
        profile_mix = tenant_cfg.get("profile_mix") or {
            p["name"]: p["weight"] for p in cat.job_profiles
        }
        prio_mix = tenant_cfg["priority_mix"]
        oversize_prob = float(tenant_cfg.get("oversize_prob", 0.0))
        undersize_prob = float(tenant_cfg.get("undersize_prob", 0.0))

        t = 0.0
        while t < horizon:
            mult = rate_multiplier(phases, tenant, t)
            rate = base_rate_per_min * mult
            if rate <= 0:
                # проматываем до конца текущей фазы
                t = next(
                    (
                        ph["to_sim_minute"]
                        for ph in phases
                        if ph["from_sim_minute"] <= t < ph["to_sim_minute"]
                    ),
                    horizon,
                )
                continue
            t += poisson_gap(rng, rate)
            if t >= horizon:
                break

            model_id = weighted_choice(rng, model_mix)
            model = cat.by_id(model_id)
            precision = weighted_choice(rng, model["precisions"])
            ctx = int(weighted_choice(rng, {str(k): v for k, v in model["contexts"].items()}))
            vram = cat.vram_estimate_gb(model_id, precision, ctx)

            min_pool, min_gpus = cheapest_placement(cfg, vram)
            # бросок делаем всегда (даже при oversize_prob=0), чтобы поток
            # rng_oversize не зависел от значения вероятности
            roll = rng_oversize.random()
            alt_pool, alt_gpus = oversized_placement(cfg, vram, min_pool, rng_oversize)
            under_pool, under_gpus = undersized_placement(
                cfg, min_pool, min_gpus, rng_oversize
            )
            if roll < oversize_prob:
                user_pool, user_gpus = alt_pool, alt_gpus
            elif roll > 1.0 - undersize_prob:
                user_pool, user_gpus = under_pool, under_gpus
            else:
                user_pool, user_gpus = min_pool, min_gpus
            oversized = user_pool.relative_cost > min_pool.relative_cost
            undersized = user_pool.relative_cost < min_pool.relative_cost or (
                user_pool.name == min_pool.name and user_gpus < min_gpus
            )

            profile_name = weighted_choice(rng, profile_mix)
            prof = profiles[profile_name]
            lo, hi = prof["duration_min"]
            sim_minutes = rng.uniform(float(lo), float(hi))

            # ground truth: сколько задача съест на самом деле и как сильно
            # будет грузить карту. Рекомендатель этого не видит.
            raw_vram = cat.raw_vram_gb(model_id, precision, ctx)
            noise_lo, noise_hi = cat.actual_vram_noise
            actual_vram = round(raw_vram * rng_truth.uniform(noise_lo, noise_hi), 2)
            ulo, uhi = prof["gpu_util_pct"]
            util_pct = round(rng_truth.uniform(float(ulo), float(uhi)), 1)
            slo, shi = prof["startup_frac"]
            startup_frac = round(rng_truth.uniform(float(slo), float(shi)), 3)

            events.append(
                (
                    t,
                    {
                        "tenant": tenant,
                        "model_id": model_id,
                        "precision": precision,
                        "ctx_tokens": ctx,
                        "vram_gb": vram,
                        "min_pool": min_pool.name,
                        "min_gpus": min_gpus,
                        "user_pool": user_pool.name,
                        "user_gpus": user_gpus,
                        "profile": profile_name,
                        "sim_minutes": round(sim_minutes, 2),
                        "priority_class": weighted_choice(rng, prio_mix),
                        "submit_at_sim_min": round(t, 3),
                        "oversized": oversized,
                        "undersized": undersized,
                        "actual_vram_gb": actual_vram,
                        "util_pct": util_pct,
                        "startup_frac": startup_frac,
                    },
                )
            )

    events.sort(key=lambda e: e[0])
    events = events[:max_jobs]

    plan = []
    for i, (_, ev) in enumerate(events):
        ev = dict(ev)
        ev["seq"] = i
        suffix = run_id[-6:].strip("-") or "0"
        ev["job_id"] = f"{ev['tenant']}-{i:04d}-{suffix}"
        ev["scenario"] = scenario["name"]
        plan.append(ev)
    return plan


def make_request(ev: dict, cat: ModelCatalog) -> JobRequest:
    """Превратить строку плана в запрос к рекомендателю"""
    model = cat.by_id(ev["model_id"])
    return JobRequest(
        model_id=ev["model_id"],
        params_b=model["params_b"],
        precision=ev["precision"],
        ctx_tokens=ev["ctx_tokens"],
        vram_gb=ev["vram_gb"],
        tenant=ev["tenant"],
        queue=QUEUES[ev["tenant"]],
        priority_class=ev["priority_class"],
        priority_value=PRIORITY_VALUES.get(ev["priority_class"], 100),
        user_pool=ev["user_pool"],
        user_gpus=ev["user_gpus"],
    )


def make_spec(
    ev: dict,
    scenario: dict,
    run_id: str,
    cat: ModelCatalog,
    policy,
    state,
    attempt: int = 1,
    job_id: str | None = None,
    force_pool: tuple[str, int] | None = None,
) -> JobSpec:
    """Применить политику размещения и собрать JobSpec.

    Вызывается в момент подачи, а не при построении плана: политика
    queue-aware смотрит на текущее состояние кластера.
    """
    time_scale = float(scenario["time_scale"])
    request = make_request(ev, cat)
    if force_pool:  # повторная постановка после OOM: карта задана явно
        pool_name, gpus = force_pool
        placement = Placement(
            pool=pool_name,
            gpus=gpus,
            confidence=None,
            policy=f"{policy.name}+retry",
            reason=f"повтор после OOM, попытка {attempt}",
        )
    else:
        placement = policy.recommend(request, state)
    duration_seconds = round(ev["sim_minutes"] * 60.0 / time_scale, 3)
    return JobSpec(
        job_id=job_id or ev["job_id"],
        tenant=ev["tenant"],
        namespace=NAMESPACES[ev["tenant"]],
        queue=QUEUES[ev["tenant"]],
        priority_class=ev["priority_class"],
        model_id=ev["model_id"],
        precision=ev["precision"],
        ctx_tokens=ev["ctx_tokens"],
        profile=ev["profile"],
        vram_gb=ev["vram_gb"],
        min_pool=ev["min_pool"],
        min_gpus=ev["min_gpus"],
        requested_pool=placement.pool,
        gpus=placement.gpus,
        user_pool=ev["user_pool"],
        user_gpus=ev["user_gpus"],
        placement_policy=placement.policy,
        confidence=placement.confidence,
        reason=placement.reason,
        duration_seconds=duration_seconds,
        sim_minutes=ev["sim_minutes"],
        submit_at_sim_min=ev["submit_at_sim_min"],
        scenario=ev["scenario"],
        run_id=run_id,
        oversized=ev["oversized"],
        actual_vram_gb=ev["actual_vram_gb"],
        util_pct=ev["util_pct"],
        startup_frac=ev["startup_frac"],
        attempt=attempt,
    )


def _find_oom_jobs(seen: set[str]) -> list[dict]:
    """Найти поды, убитые моделью GPU по OOM, и вернуть их задачи"""
    out = []
    try:
        pods = kubectl_json("get", "pods", "-A", "-l", "lab.skalar.ai/lab=true")
    except RuntimeError:
        return out
    for pod in pods.get("items", []):
        status = pod.get("status", {})
        if status.get("phase") != "Failed" or status.get("reason") != "OOMKilled":
            continue
        annot = pod["metadata"].get("annotations", {})
        job_id = annot.get(A_JOBID)
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        gpus = 0.0
        for c in pod["spec"].get("containers", []):
            gpus += float(
                c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0
            )
        out.append(
            {
                "base_id": job_id.split("-r")[0],
                "pool": pod["metadata"]["labels"].get("lab.skalar.ai/requested-pool"),
                "gpus": int(gpus),
            }
        )
    return out


def summarize_plan(plan: list[dict], cfg: PoolConfig, scenario: dict) -> str:
    """Сводка по плану — то, что пользователи собираются попросить.

    Считается по выбору пользователя (user_pool), потому что политика
    размещения применяется позже, уже в момент подачи.
    """
    time_scale = float(scenario["time_scale"])
    by_tenant: dict[str, int] = {}
    by_pool: dict[str, int] = {}
    gpu_seconds = 0.0
    optimal_seconds = 0.0
    oversized = 0
    undersized = 0
    for ev in plan:
        dur = ev["sim_minutes"] * 60.0 / time_scale
        by_tenant[ev["tenant"]] = by_tenant.get(ev["tenant"], 0) + 1
        by_pool[ev["user_pool"]] = by_pool.get(ev["user_pool"], 0) + 1
        gpu_seconds += ev["user_gpus"] * dur
        optimal_seconds += ev["min_gpus"] * dur
        oversized += int(ev["oversized"])
        undersized += int(ev.get("undersized", False))
    horizon = float(scenario["duration_sim_minutes"]) * 60.0 / time_scale
    capacity = cfg.total_gpus * horizon if horizon else 1
    return (
        f"задач: {len(plan)}  "
        f"по командам: {by_tenant}  "
        f"по картам (выбор пользователя): {by_pool}\n"
        f"  запрошено GPU-секунд: {gpu_seconds:.0f} "
        f"(при минимально достаточных картах было бы {optimal_seconds:.0f}); "
        f"ёмкость кластера за горизонт: {capacity:.0f} "
        f"(нагрузка ≈ {100 * gpu_seconds / capacity:.0f}% от пика)\n"
        f"  задач с завышенным классом карты: {oversized} "
        f"({100 * oversized / max(1, len(plan)):.0f}%), "
        f"с заниженным: {undersized} "
        f"({100 * undersized / max(1, len(plan)):.0f}%)"
    )


def summarize_applied(specs: list[JobSpec]) -> str:
    by_pool: dict[str, int] = {}
    changed = 0
    conf_low = 0
    known = 0
    for sp in specs:
        by_pool[sp.requested_pool] = by_pool.get(sp.requested_pool, 0) + 1
        if sp.requested_pool != sp.user_pool or sp.gpus != sp.user_gpus:
            changed += 1
        if sp.confidence is not None:
            known += 1
            # 0.85 — тот порог, на который ориентирован AdmissionCheck:
            # риск падения выше 15%
            if sp.confidence < 0.85:
                conf_low += 1
    return (
        f"фактически размещено по картам: {by_pool}\n"
        f"  решений, отличных от выбора пользователя: {changed}/{len(specs)}; "
        f"с риском падения выше 15% (confidence < 0.85): {conf_low}/{known}"
    )


# Минимум тактов снапшотера на самую короткую задачу. Меньше — модель GPU не
# успевает ни снять потребление, ни поймать OOM: задача «мигает» между двумя
# опросами и выглядит успешной.
MIN_TICKS_PER_JOB = 8


def _check_time_resolution(plan: list[dict], time_scale: float) -> None:
    """Предупредить, если сжатие времени съело разрешение по коротким задачам.

    Это самая тихая из возможных поломок: при слишком большом time_scale
    метрики не ломаются, а улучшаются — OOM перестают находиться, короткие
    задачи выпадают из скоринга, и прогон выглядит удачным.
    """
    if not plan:
        return
    try:
        interval = float(os.environ.get("SNAPSHOT_INTERVAL") or 1.0)
    except ValueError:
        interval = 1.0
    shortest = min(ev["sim_minutes"] for ev in plan)
    seconds = shortest * 60.0 / time_scale
    ticks = seconds / interval if interval > 0 else 0.0
    eprint(
        f"[submit] самая короткая задача: {shortest:.1f} сим-мин = "
        f"{seconds:.2f}с реальных = {ticks:.1f} тактов "
        f"(интервал {interval}с, time_scale {time_scale:g})"
    )
    if ticks < MIN_TICKS_PER_JOB:
        eprint(
            f"[submit] ВНИМАНИЕ: нужно хотя бы {MIN_TICKS_PER_JOB} тактов. "
            f"Уменьши time_scale до "
            f"{shortest * 60.0 / (MIN_TICKS_PER_JOB * interval):.0f}, "
            f"или опусти SNAPSHOT_INTERVAL до "
            f"{seconds / MIN_TICKS_PER_JOB:.2f}с, или подними нижнюю границу "
            f"duration_min у самого короткого профиля до "
            f"{MIN_TICKS_PER_JOB * interval * time_scale / 60.0:.0f} сим-мин."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument(
        "--placement",
        choices=sorted(all_policies()),
        default="manual",
        help="кто выбирает карту: никто (kueue-native), пользователь (manual) "
        "или рекомендатель (cheapest / queue-aware)",
    )
    ap.add_argument(
        "--state-source",
        choices=["auto", "pods", "kueue"],
        default="auto",
        help="откуда политика узнаёт занятость кластера: по подам на нодах "
        "(pods) или по квотам и очередям kueue (kueue). auto — как просит "
        "сама политика",
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--plan", help="переиспользовать план из другого прогона (для сравнения политик)"
    )
    ap.add_argument(
        "--no-live-state",
        action="store_true",
        help="не опрашивать кластер: политика queue-aware будет считать его пустым",
    )
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--tenants", default="config/tenants.yaml")
    ap.add_argument("--dry-run", action="store_true", help="только построить план")
    ap.add_argument("--seed", type=int, help="переопределить seed сценария")
    ap.add_argument(
        "--state-ttl-sim-min",
        type=float,
        default=5.0,
        help="как долго (в СИМУЛИРОВАННЫХ минутах) политика переиспользует "
        "снимок состояния кластера. В реальные секунды пересчитывается по "
        "time_scale, чтобы ускорение прогона не меняло поведение политики",
    )
    ap.add_argument(
        "--oversize-prob",
        type=float,
        help="переопределить oversize_prob у всех команд (0 = идеальный подбор карты)",
    )
    ap.add_argument(
        "--oom-retry-limit",
        type=int,
        help="сколько раз переставлять упавшую по OOM задачу на карту классом "
        "выше (по умолчанию берётся из сценария, 0 = не переставлять)",
    )
    ap.add_argument(
        "--max-exec-factor",
        type=float,
        default=0.0,
        help="проставить kueue.x-k8s.io/max-exec-time-seconds = factor*длительность "
        "(0 = не проставлять). Понадобится для backfill из задачи #2.",
    )
    args = ap.parse_args()

    scenario = load_yaml(args.scenario)
    if args.seed is not None:
        scenario["seed"] = args.seed
        scenario["name"] = f"{scenario['name']}-seed{args.seed}"
    tenants = resolve_tenants(scenario, args.tenants)
    if args.oversize_prob is not None:
        for t in tenants:
            t["oversize_prob"] = args.oversize_prob
        scenario["name"] = f"{scenario['name']}-ovs{args.oversize_prob:g}"
    retry_limit = (
        args.oom_retry_limit
        if args.oom_retry_limit is not None
        else int(scenario.get("oom_retry_limit", 0))
    )

    cfg = PoolConfig.load(args.pool)
    cat = ModelCatalog.load(args.models)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = _sanitize_run_id(run_dir.name)

    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        eprint(f"[submit] план переиспользован из {args.plan}")
    else:
        plan = build_plan(scenario, cfg, cat, run_id, tenants)

    time_scale = float(scenario["time_scale"])
    _check_time_resolution(plan, time_scale)
    policy = build_policy(args.placement, cfg, UncertaintyModel.from_catalog(cat))
    
    # Какой сборщик состояния поднять. Политика объявляет свою потребность
    # атрибутом state_source; kueue-aware и quota-aware читают квоты, всем
    # остальным достаточно подов. Флаг --state-source перекрывает выбор:
    # это же способ прогнать queue-aware на данных kueue и увидеть, сколько
    # даёт один только переход на другой источник, без смены целевой функции.
    # TTL кэша состояния кластера задан в РЕАЛЬНЫХ секундах, а решение
    # политики зависит от того, насколько свежий снимок она видит в
    # симулированном времени. При time_scale 480 дефолтные 3 секунды — это
    # 24 сим-минуты, за которые подаётся 4-5 задач: все они видят один и тот
    # же снимок и дружно уезжают в один пул. Держим окно постоянным в
    # симулированном времени, чтобы ускорение прогона не меняло поведение
    # планировщика.
    state_ttl = args.state_ttl_sim_min * 60.0 / time_scale
    eprint(
        f"[submit] кэш состояния кластера: {args.state_ttl_sim_min:g} сим-мин "
        f"= {state_ttl:.3f}с реальных"
    )

    wanted = args.state_source
    if wanted == "auto":
        wanted = getattr(policy, "state_source", "pods")
    live = not (args.no_live_state or args.dry_run)
    if wanted == "kueue":
        state_source = KueueStateSource(cfg, live=live, ttl=state_ttl)
    else:
        state_source = ClusterStateSource(cfg, live=live, ttl=state_ttl)
    eprint(f"[submit] источник состояния кластера: {wanted}")

    (run_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (run_dir / "scenario.json").write_text(
        json.dumps(
            {
                **scenario,
                "tenants": tenants,
                "placement": args.placement,
                "oom_retry_limit": retry_limit,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    eprint(f"[submit] сценарий {scenario['name']} / политика размещения {args.placement}")
    eprint("[submit] " + summarize_plan(plan, cfg, scenario))

    if args.dry_run:
        specs = [
            make_spec(ev, scenario, run_id, cat, policy, state_source.get())
            for ev in plan
        ]
        with (run_dir / "jobs.jsonl").open("w", encoding="utf-8") as fh:
            for sp in specs:
                fh.write(json.dumps(sp.to_dict(), ensure_ascii=False) + "\n")
        eprint("[submit] " + summarize_applied(specs))
        eprint("[submit] --dry-run: в кластер ничего не отправлено")
        return

    wall_horizon = float(scenario["duration_sim_minutes"]) * 60.0 / time_scale
    eprint(f"[submit] подача займёт ~{wall_horizon / 60:.1f} мин реального времени")

    t0 = time.time()
    submitted = 0
    idx = 0
    batch: list[dict] = []
    batch_deadline = 0.0
    specs: list[JobSpec] = []
    jobs_file = (run_dir / "jobs.jsonl").open("w", encoding="utf-8")

    plan_by_id = {ev["job_id"]: ev for ev in plan}
    attempts: dict[str, int] = {}      # базовый job_id -> сколько раз ставили
    seen_failed: set[str] = set()
    retries = 0
    next_retry_check = 0.0

    def flush_spec(spec: JobSpec) -> None:
        specs.append(spec)
        jobs_file.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")
        batch.append(render_job(spec, cfg, args.max_exec_factor))

    while idx < len(plan) or (retry_limit and time.time() - t0 < wall_horizon + 60):
        now = time.time() - t0

        # Повторная постановка задач, упавших по OOM. Ровно то поведение,
        # которое описано в постановке: пользователь ставит ту же задачу
        # заново, взяв карту побольше, и тем самым греет кластер вхолостую.
        if retry_limit and now >= next_retry_check:
            next_retry_check = now + 3.0
            for spec in _find_oom_jobs(seen_failed):
                base_id = spec["base_id"]
                ev = plan_by_id.get(base_id)
                if ev is None:
                    continue
                attempt = attempts.get(base_id, 1) + 1
                if attempt > retry_limit + 1:
                    continue
                attempts[base_id] = attempt
                pool_name, gpus = next_tier_placement(
                    cfg, spec["pool"], spec["gpus"], ev["vram_gb"]
                )
                flush_spec(
                    make_spec(
                        ev, scenario, run_id, cat, policy, state_source.get(),
                        attempt=attempt,
                        job_id=f"{base_id}-r{attempt}",
                        force_pool=(pool_name, gpus),
                    )
                )
                retries += 1
                if batch_deadline == 0.0:
                    batch_deadline = now + 0.5

        if idx >= len(plan):
            if batch:
                apply_docs(batch)
                submitted += len(batch)
                batch, batch_deadline = [], 0.0
            time.sleep(1.0)
            continue

        ev = plan[idx]
        due = ev["submit_at_sim_min"] * 60.0 / time_scale
        if due <= now:
            # решение о размещении принимается здесь и сейчас: политике
            # queue-aware нужна актуальная картина занятости кластера
            flush_spec(make_spec(ev, scenario, run_id, cat, policy, state_source.get()))
            idx += 1
            if batch_deadline == 0.0:
                batch_deadline = now + 0.5
            continue

        # накопленное отправляем пачкой — так меньше вызовов kubectl
        if batch and (now >= batch_deadline or len(batch) >= 25):
            apply_docs(batch)
            submitted += len(batch)
            jobs_file.flush()
            eprint(
                f"[submit] t={now:6.1f}s sim={now * time_scale / 60:7.1f}min "
                f"подано {submitted}/{len(plan)}"
            )
            batch, batch_deadline = [], 0.0
            continue

        time.sleep(min(0.25, max(0.01, due - now)))

    if batch:
        apply_docs(batch)
        submitted += len(batch)
    jobs_file.close()

    eprint("[submit] " + summarize_applied(specs))
    if retry_limit:
        eprint(f"[submit] повторных постановок после OOM: {retries}")
    eprint(f"[submit] готово: подано {submitted} задач за {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
