#!/usr/bin/env python3

#Снапшотер состояния кластера.


from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_runtime import GpuRuntime  # noqa: E402
from lab import (  # noqa: E402
    PoolConfig,
    A_JOBID,
    A_VRAM,
    A_DURATION,
    GPU_RES,
    L_LAB,
    L_MIN_POOL,
    L_MODEL,
    L_PLACEMENT,
    L_POOL,
    L_PROFILE,
    L_REQ_POOL,
    L_TENANT,
    eprint,
    kubectl_json,
    now_iso,
    parse_quantity,
    parse_ts,
)

RUNNING_PHASES = ("Running",)
_stop = False


def _handle_sigint(*_a):
    global _stop
    _stop = True
    eprint("\n[snapshot] получен сигнал остановки, дописываю файлы…")


def fetch_nodes() -> dict:
    data = kubectl_json("get", "nodes", "-l", L_POOL)
    nodes = {}
    for n in data.get("items", []):
        labels = n["metadata"].get("labels", {})
        alloc = n.get("status", {}).get("allocatable", {})
        nodes[n["metadata"]["name"]] = {
            "pool": labels.get(L_POOL),
            "product": labels.get("nvidia.com/gpu.product"),
            "gpu_capacity": int(parse_quantity(alloc.get(GPU_RES, 0))),
            "cpu_capacity": parse_quantity(alloc.get("cpu", 0)),
        }
    return nodes


def pod_gpus(pod: dict) -> float:
    total = 0.0
    for c in pod.get("spec", {}).get("containers", []):
        req = c.get("resources", {}).get("requests", {})
        total += parse_quantity(req.get(GPU_RES, 0))
    return total


def cond_time(obj: dict, ctype: str) -> float | None:
    for c in obj.get("status", {}).get("conditions", []) or []:
        if c.get("type") == ctype and c.get("status") == "True":
            return parse_ts(c.get("lastTransitionTime"))
    return None


class Collector:
    def __init__(
        self,
        run_dir: Path,
        interval: float,
        cfg: PoolConfig | None = None,
        enforce_oom: bool = True,
        slow_every: int = 1,
    ):
        self.run_dir = run_dir
        self.interval = interval
        self.nodes = fetch_nodes()
        self.runtime: GpuRuntime | None = None
        if cfg is not None:
            self.runtime = GpuRuntime(
                cfg=cfg, run_dir=run_dir, enforce_oom=enforce_oom
            )
            self.runtime.load_nodes(
                kubectl_json("get", "nodes", "-l", L_POOL).get("items", [])
            )
        self.jobs: dict[str, dict] = {}
        self.workloads: dict[str, dict] = {}
        self.t0 = time.time()
        self.ticks = 0
        # Модель GPU (сэмплы потребления и OOM) смотрит только на поды, и
        # именно она требует частого такта. Job'ы, ClusterQueue и Workload
        # нужны для очередей и вытеснений, а там разрешение в секунду и так
        # избыточно — их достаточно опрашивать реже. Это убирает 3 из 4
        # запросов с горячего пути.
        self.slow_every = max(1, int(slow_every))
        self._slow_cache: tuple[list, list, list] = ([], [], [])

    # ── сбор ────────────────────────────────────────────────────────────────
    def tick(self) -> dict:
        ts = time.time()
        elapsed = ts - self.t0

        # горячий путь: поды нужны каждый такт — на них живёт модель GPU
        pods = kubectl_json(
            "get", "pods", "-A", "-l", f"{L_LAB}=true"
        ).get("items", [])

        # холодный путь: очереди и вытеснения. Разрешение в статусе всё равно
        # секундное, поэтому опрашиваем раз в slow_every тактов и держим
        # slow_every * interval около секунды.
        if self.ticks % self.slow_every == 0:
            jobs = kubectl_json(
                "get", "jobs", "-A", "-l", f"{L_LAB}=true"
            ).get("items", [])
            kueue = kubectl_json("get", "clusterqueues,workloads", "-A")
            cqs = [i for i in kueue.get("items", []) if i["kind"] == "ClusterQueue"]
            wls = [i for i in kueue.get("items", []) if i["kind"] == "Workload"]
            self._slow_cache = (jobs, cqs, wls)
        else:
            jobs, cqs, wls = self._slow_cache

        rec = {
            "ts": ts,
            "iso": now_iso(),
            "t": round(elapsed, 3),
        }
        if self.runtime is not None:
            rec["gpu_runtime"] = self.runtime.tick(pods, ts)
        rec.update(self._pods_snapshot(pods, ts))
        rec.update(self._jobs_snapshot(jobs, ts))
        rec["queues"] = self._queues_snapshot(cqs)
        self._track_workloads(wls, ts)
        rec["workloads"] = {
            "total": len(wls),
            "admitted": sum(1 for w in wls if cond_time(w, "Admitted")),
            "finished": sum(1 for w in wls if cond_time(w, "Finished")),
            "evicted_events": sum(
                len(w.get("status", {}).get("schedulingStats", {}).get("evictions", []) or [])
                for w in wls
            ),
        }
        return rec

    def _pods_snapshot(self, pods: list[dict], ts: float) -> dict:
        gpu_by_node: dict[str, float] = defaultdict(float)
        gpu_by_pool: dict[str, float] = defaultdict(float)
        gpu_by_tenant: dict[str, float] = defaultdict(float)
        phase_count: dict[str, int] = defaultdict(int)
        unscheduled = 0
        unscheduled_gpus = 0.0

        for p in pods:
            phase = p.get("status", {}).get("phase", "Unknown")
            phase_count[phase] += 1
            labels = p["metadata"].get("labels", {})
            node = p.get("spec", {}).get("nodeName")
            g = pod_gpus(p)
            if phase in RUNNING_PHASES and node:
                gpu_by_node[node] += g
                pool = self.nodes.get(node, {}).get("pool", labels.get(L_POOL, "?"))
                gpu_by_pool[pool] += g
                gpu_by_tenant[labels.get(L_TENANT, "?")] += g
            elif phase == "Pending" and not node:
                unscheduled += 1
                unscheduled_gpus += g

            # фиксируем момент старта пода. first_seen_running_ts — это такт
            # наблюдения, его разрешение равно --interval; pod_start_ts берётся
            # из статуса пода и не зависит от того, как часто мы опрашиваем
            # кластер. Ожидание надо считать по второму.
            jid = p["metadata"].get("annotations", {}).get(A_JOBID)
            if jid and phase in RUNNING_PHASES:
                job = self.jobs.setdefault(jid, {})
                job.setdefault("first_seen_running_ts", ts)
                pod_start = p.get("status", {}).get("startTime")
                if pod_start:
                    started_at = parse_ts(pod_start)
                    # первый старт — для ожидания в очереди
                    job.setdefault("pod_start_ts", started_at)
                    # последний старт — для длительности работы. Вытесненная
                    # задача получает новый под, и без этого completion_ts
                    # минус первый старт включает всё ожидание между
                    # попытками и выдаёт его за время счёта.
                    job["last_pod_start_ts"] = started_at
                    pods_seen = job.setdefault("pod_starts", {})
                    pods_seen[p["metadata"]["name"]] = started_at
                    job["attempts"] = len(pods_seen)
                job.setdefault("node", node)
                # при политике kueue-native карта заранее неизвестна: какой
                # флейвор выбрал kueue, видно только по ноде, куда уехал под
                job.setdefault("actual_pool", self.nodes.get(node, {}).get("pool"))

        allocated = sum(gpu_by_node.values())
        capacity = sum(n["gpu_capacity"] for n in self.nodes.values())

        # фрагментация: свободные карты на нодах, где уже что-то крутится
        free_on_busy = 0.0
        free_total = 0.0
        for name, meta in self.nodes.items():
            free = meta["gpu_capacity"] - gpu_by_node.get(name, 0.0)
            free_total += free
            if gpu_by_node.get(name, 0.0) > 0:
                free_on_busy += free

        return {
            "gpu": {
                "capacity": capacity,
                "allocated": allocated,
                "utilization": round(allocated / capacity, 4) if capacity else 0.0,
                "by_pool": dict(gpu_by_pool),
                "by_tenant": dict(gpu_by_tenant),
                "free_total": free_total,
                "free_on_busy_nodes": free_on_busy,
            },
            "pods": {
                "by_phase": dict(phase_count),
                "unscheduled": unscheduled,
                "unscheduled_gpus": unscheduled_gpus,
            },
        }

    def _jobs_snapshot(self, jobs: list[dict], ts: float) -> dict:
        active = suspended = succeeded = failed = 0
        pending_gpus = 0.0
        for j in jobs:
            meta = j["metadata"]
            labels = meta.get("labels", {})
            annot = meta.get("annotations", {})
            jid = annot.get(A_JOBID, meta["name"])
            st = j.get("status", {})
            spec = j.get("spec", {})

            rec = self.jobs.setdefault(jid, {})
            rec.setdefault("name", meta["name"])
            rec.setdefault("namespace", meta["namespace"])
            rec.setdefault("tenant", labels.get(L_TENANT))
            rec.setdefault("model", labels.get(L_MODEL))
            rec.setdefault("profile", labels.get(L_PROFILE))
            rec.setdefault("requested_pool", labels.get(L_REQ_POOL))
            rec.setdefault("placement_policy", labels.get(L_PLACEMENT))
            rec.setdefault("min_pool", labels.get(L_MIN_POOL))
            rec.setdefault("vram_gb", float(annot.get(A_VRAM, 0) or 0))
            rec.setdefault("planned_duration_s", float(annot.get(A_DURATION, 0) or 0))
            rec.setdefault("gpus", _job_gpus(j))
            rec.setdefault("created_ts", parse_ts(meta.get("creationTimestamp")))
            rec.setdefault("first_seen_ts", ts)

            if st.get("startTime") and "start_ts" not in rec:
                rec["start_ts"] = parse_ts(st["startTime"])
            if st.get("completionTime") and "completion_ts" not in rec:
                rec["completion_ts"] = parse_ts(st["completionTime"])
            if st.get("succeeded"):
                rec["succeeded"] = True
                succeeded += 1
            elif st.get("failed"):
                rec["failed"] = True
                failed += 1
            elif spec.get("suspend"):
                suspended += 1
                pending_gpus += rec["gpus"]
            else:
                active += 1

        return {
            "jobs": {
                "total": len(jobs),
                "active": active,
                "suspended": suspended,
                "succeeded": succeeded,
                "failed": failed,
                "suspended_gpus": pending_gpus,
            }
        }

    def _queues_snapshot(self, cqs: list[dict]) -> dict:
        out = {}
        for cq in cqs:
            name = cq["metadata"]["name"]
            st = cq.get("status", {})
            usage = {}
            for fu in st.get("flavorsUsage", []) or []:
                for r in fu.get("resources", []):
                    if r["name"] == GPU_RES:
                        usage[fu["name"]] = {
                            "used": parse_quantity(r.get("total")),
                            "borrowed": parse_quantity(r.get("borrowed")),
                        }
            reserved = {}
            for fu in st.get("flavorsReservation", []) or []:
                for r in fu.get("resources", []):
                    if r["name"] == GPU_RES:
                        reserved[fu["name"]] = parse_quantity(r.get("total"))
            out[name] = {
                "pending": st.get("pendingWorkloads", 0),
                "admitted": st.get("admittedWorkloads", 0),
                "reserving": st.get("reservingWorkloads", 0),
                "gpu_used": sum(v["used"] for v in usage.values()),
                "gpu_borrowed": sum(v["borrowed"] for v in usage.values()),
                "by_flavor": usage,
                "reserved_by_flavor": reserved,
            }
        return out

    def _track_workloads(self, wls: list[dict], ts: float) -> None:
        for w in wls:
            meta = w["metadata"]
            key = f"{meta['namespace']}/{meta['name']}"
            spec = w.get("spec", {})
            st = w.get("status", {})
            rec = self.workloads.setdefault(key, {})
            rec.setdefault("namespace", meta["namespace"])
            rec.setdefault("name", meta["name"])
            rec.setdefault("job_id", _owner_name(meta))
            rec.setdefault("queue", spec.get("queueName"))
            rec.setdefault("priority", spec.get("priority"))
            rec.setdefault(
                "priority_class", (spec.get("priorityClassRef") or {}).get("name")
            )
            rec.setdefault("created_ts", parse_ts(meta.get("creationTimestamp")))
            rec.setdefault("first_seen_ts", ts)
            rec["gpus"] = _workload_gpus(spec)

            adm = st.get("admission") or {}
            if adm:
                rec["cluster_queue"] = adm.get("clusterQueue")
                flavors = set()
                for psa in adm.get("podSetAssignments", []) or []:
                    for res, fl in (psa.get("flavors") or {}).items():
                        if res == GPU_RES:
                            flavors.add(fl)
                if flavors:
                    rec["flavor"] = sorted(flavors)[0]

            for ctype, field in (
                ("QuotaReserved", "quota_reserved_ts"),
                ("Admitted", "admitted_ts"),
                ("Finished", "finished_ts"),
                ("Evicted", "last_evicted_ts"),
                ("Preempted", "last_preempted_ts"),
            ):
                t = cond_time(w, ctype)
                if t and field not in rec:
                    rec[field] = t
                elif t and ctype in ("Evicted", "Preempted"):
                    rec[field] = t

            evictions = (st.get("schedulingStats") or {}).get("evictions") or []
            rec["evictions"] = sum(int(e.get("count", 1)) for e in evictions)
            rec["eviction_reasons"] = sorted(
                {
                    f"{e.get('reason')}/{e.get('underlyingCause', '')}".rstrip("/")
                    for e in evictions
                }
            )

    # ── файлы ───────────────────────────────────────────────────────────────
    def flush(self) -> None:
        (self.run_dir / "workloads.json").write_text(
            json.dumps(self.workloads, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (self.run_dir / "jobs_observed.json").write_text(
            json.dumps(self.jobs, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (self.run_dir / "nodes.json").write_text(
            json.dumps(self.nodes, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def _job_gpus(job: dict) -> float:
    tmpl = job.get("spec", {}).get("template", {}).get("spec", {})
    total = 0.0
    for c in tmpl.get("containers", []):
        total += parse_quantity(
            c.get("resources", {}).get("requests", {}).get(GPU_RES, 0)
        )
    return total * int(job.get("spec", {}).get("parallelism", 1) or 1)


def _workload_gpus(spec: dict) -> float:
    total = 0.0
    for ps in spec.get("podSets", []) or []:
        count = int(ps.get("count", 1) or 1)
        for c in ps.get("template", {}).get("spec", {}).get("containers", []):
            total += count * parse_quantity(
                c.get("resources", {}).get("requests", {}).get(GPU_RES, 0)
            )
    return total


def _owner_name(meta: dict) -> str | None:
    for o in meta.get("ownerReferences", []) or []:
        if o.get("kind") == "Job":
            return o.get("name")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument(
        "--slow-every",
        type=int,
        default=1,
        help="опрашивать Job/ClusterQueue/Workload раз в N тактов (поды — "
        "каждый такт). N*interval держи около 1 с: чаще не нужно, в статусе "
        "всё равно секундное разрешение, реже — поедет учёт вытеснений",
    )
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument(
        "--no-gpu-model",
        action="store_true",
        help="не крутить модель VRAM/потребления (без неё OOM невозможен)",
    )
    ap.add_argument("--duration", type=float, default=0.0, help="секунд, 0 = без лимита")
    ap.add_argument(
        "--until-drained",
        type=float,
        default=0.0,
        help="остановиться, когда N секунд подряд нет ни активных, ни ожидающих задач",
    )
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = None if args.no_gpu_model else PoolConfig.load(args.pool)
    col = Collector(
        run_dir, args.interval, cfg=cfg, slow_every=args.slow_every
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "interval": args.interval,
                "slow_every": args.slow_every,
                "started_iso": now_iso(),
                "started_ts": col.t0,
                "gpu_capacity": sum(n["gpu_capacity"] for n in col.nodes.values()),
                "nodes": len(col.nodes),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    eprint(
        f"[snapshot] {len(col.nodes)} нод, "
        f"{sum(n['gpu_capacity'] for n in col.nodes.values())} GPU; "
        f"интервал {args.interval}s -> {run_dir}"
    )
    if col.runtime is not None:
        eprint("[snapshot] модель GPU включена: контроль VRAM + учёт потребления")

    ts_file = (run_dir / "timeseries.jsonl").open("w", encoding="utf-8")
    idle_since: float | None = None
    slow_ticks = 0
    tick_costs: list[float] = []

    try:
        while not _stop:
            started = time.time()
            try:
                rec = col.tick()
            except RuntimeError as exc:
                eprint(f"[snapshot] ошибка опроса: {exc}")
                time.sleep(args.interval)
                continue

            ts_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ts_file.flush()
            col.ticks += 1
            if col.ticks % 30 == 0:
                col.flush()
                oom = rec.get("gpu_runtime", {}).get("oom_total", 0)
                eprint(
                    f"[snapshot] t={rec['t']:7.1f}s "
                    f"GPU {rec['gpu']['allocated']:.0f}/{rec['gpu']['capacity']} "
                    f"({100 * rec['gpu']['utilization']:.0f}%) "
                    f"jobs act={rec['jobs']['active']} susp={rec['jobs']['suspended']} "
                    f"done={rec['jobs']['succeeded']} oom={oom}"
                )

            busy = (
                rec["jobs"]["active"] > 0
                or rec["jobs"]["suspended"] > 0
                or rec["pods"]["unscheduled"] > 0
            )
            if args.until_drained:
                if busy or rec["jobs"]["total"] == 0:
                    idle_since = None
                else:
                    idle_since = idle_since or started
                    if started - idle_since >= args.until_drained:
                        eprint("[snapshot] кластер разгрузился, останавливаюсь")
                        break

            if args.duration and (started - col.t0) >= args.duration:
                break

            took = time.time() - started
            tick_costs.append(took)
            # Предупреждаем не только когда такт вылез за интервал, но и когда
            # съедает больше половины: при таком запасе первый же всплеск
            # нагрузки на apiserver начнёт двигать тики, и разметка времени
            # поедет незаметно.
            if took > 0.5 * args.interval:
                slow_ticks += 1
                if slow_ticks in (5, 50, 500):
                    level = "не укладывается в" if took > args.interval else "близок к"
                    eprint(
                        f"[snapshot] опрос {level} интервал "
                        f"({took:.3f}s при интервале {args.interval}s, "
                        f"случай {slow_ticks}-й). Увеличь --interval или "
                        f"--slow-every, либо включи kubectl proxy (KUBE_FAST=1)"
                    )
            time.sleep(max(0.0, args.interval - took))
    finally:
        col.flush()
        if col.runtime is not None:
            col.runtime.close()
            eprint(f"[snapshot] OOM за прогон: {len(col.runtime.oom_events)}")
        ts_file.close()
        eprint(f"[snapshot] записано {col.ticks} снимков в {run_dir}")
        if tick_costs:
            ordered = sorted(tick_costs)
            eprint(
                f"[snapshot] цена такта: медиана "
                f"{1000 * ordered[len(ordered) // 2]:.0f} мс, "
                f"p90 {1000 * ordered[int(0.9 * len(ordered))]:.0f} мс, "
                f"максимум {1000 * ordered[-1]:.0f} мс "
                f"при интервале {1000 * args.interval:.0f} мс"
            )


if __name__ == "__main__":
    main()
