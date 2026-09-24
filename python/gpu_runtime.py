#модель физики

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from lab import (
    A_ACTUAL_VRAM,
    A_DURATION,
    A_JOBID,
    A_STARTUP_FRAC,
    A_UTIL_PCT,
    GPU_RES,
    PRODUCT_LABEL,
    PoolConfig,
    eprint,
    kubectl,
    now_iso,
    parse_quantity,
    parse_ts,
)


@dataclass
class NodeInfo:
    name: str
    pool: str
    product: str
    vram_per_gpu: float
    tdp_w: float
    idle_w: float


@dataclass
class PodRuntime:
    """Состояние одного работающего пода глазами модели GPU."""

    job_id: str
    namespace: str
    pod: str
    node: str
    pool: str
    gpus: float
    allocated_vram_gb: float   # сколько памяти реально выделено под задачу
    actual_vram_gb: float      # сколько ей нужно на самом деле
    util_pct: float            # утилизация на установившемся режиме
    startup_frac: float        # доля времени на прогрев
    duration_s: float
    started_ts: float
    oom: bool = False
    oom_ts: float | None = None
    samples: int = 0
    vram_gb_seconds: float = 0.0
    watt_seconds: float = 0.0
    util_seconds: float = 0.0


def _pod_start_ts(pod: dict, fallback: float) -> float:
    """Момент старта пода из его статуса; такт наблюдения — как запасной."""
    raw = pod.get("status", {}).get("startTime")
    if not raw:
        return fallback
    try:
        return parse_ts(raw)
    except Exception:  # noqa: BLE001 — испорченный timestamp не должен ронять прогон
        return fallback


@dataclass
class GpuRuntime:
    cfg: PoolConfig
    run_dir: Path
    enforce_oom: bool = True
    seed: int = 0

    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    pods: dict[str, PodRuntime] = field(default_factory=dict)
    oom_events: list[dict] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))
    _usage_file: object = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._usage_file = (self.run_dir / "usage.jsonl").open("w", encoding="utf-8")

    #ноды
    def load_nodes(self, node_items: list[dict]) -> None:
        by_product = {p.product: p for p in self.cfg.pools}
        for n in node_items:
            labels = n["metadata"].get("labels", {})
            product = labels.get(PRODUCT_LABEL)
            pool = by_product.get(product)
            if not pool:
                continue
            self.nodes[n["metadata"]["name"]] = NodeInfo(
                name=n["metadata"]["name"],
                pool=pool.name,
                product=product,
                vram_per_gpu=pool.vram_gb,
                tdp_w=pool.tdp_w,
                idle_w=pool.idle_w,
            )

    #такт
    def tick(self, pod_items: list[dict], ts: float) -> dict:
        """Обработать текущий список подов"""
        usable = self.cfg.defaults.get("usable_vram_fraction", 0.92)
        live: set[str] = set()
        killed_now: list[str] = []

        for pod in pod_items:
            if pod.get("status", {}).get("phase") != "Running":
                continue
            meta = pod["metadata"]
            key = f"{meta['namespace']}/{meta['name']}"
            live.add(key)
            annot = meta.get("annotations", {})
            node_name = pod.get("spec", {}).get("nodeName")
            node = self.nodes.get(node_name)
            if node is None:
                continue

            rt = self.pods.get(key)
            if rt is None:
                gpus = 0.0
                for c in pod["spec"].get("containers", []):
                    gpus += parse_quantity(
                        c.get("resources", {}).get("requests", {}).get(GPU_RES, 0)
                    )
                rt = PodRuntime(
                    job_id=annot.get(A_JOBID, meta["name"]),
                    namespace=meta["namespace"],
                    pod=meta["name"],
                    node=node_name,
                    pool=node.pool,
                    gpus=gpus,
                    allocated_vram_gb=round(gpus * node.vram_per_gpu * usable, 2),
                    actual_vram_gb=float(annot.get(A_ACTUAL_VRAM, 0) or 0),
                    util_pct=float(annot.get(A_UTIL_PCT, 50) or 50),
                    startup_frac=float(annot.get(A_STARTUP_FRAC, 0.05) or 0.05),
                    duration_s=float(annot.get(A_DURATION, 60) or 60),
                    # фаза прогрева должна отсчитываться от реального старта
                    # пода, а не от такта, на котором мы его заметили: иначе
                    # при крупном --interval короткие задачи «стартуют» позже,
                    # чем на самом деле, и OOM ловится с опозданием
                    started_ts=_pod_start_ts(pod, ts),
                )
                self.pods[key] = rt

            if rt.oom:
                continue

            elapsed = ts - rt.started_ts
            phase_frac = elapsed / rt.duration_s if rt.duration_s else 1.0

            # OOM случается не мгновенно: сначала процесс поднимается и грузит
            # веса, и только упёршись в предел памяти — падает
            loading_done = phase_frac >= min(0.5, rt.startup_frac)
            if (
                self.enforce_oom
                and loading_done
                and rt.actual_vram_gb > rt.allocated_vram_gb
            ):
                self._kill_oom(rt, ts)
                killed_now.append(key)
                continue

            self._sample(rt, node, ts, phase_frac)

        # поды, которых больше нет в списке, считаем завершёнными
        for key in list(self.pods):
            if key not in live:
                self.pods.pop(key, None)

        return {
            "tracked": len(self.pods),
            "oom_total": len(self.oom_events),
            "oom_now": len(killed_now),
        }

    #OOM
    def _kill_oom(self, rt: PodRuntime, ts: float) -> None:
        rt.oom = True
        rt.oom_ts = ts
        overshoot = rt.actual_vram_gb - rt.allocated_vram_gb
        event = {
            "ts": ts,
            "iso": now_iso(),
            "job_id": rt.job_id,
            "namespace": rt.namespace,
            "pod": rt.pod,
            "node": rt.node,
            "pool": rt.pool,
            "gpus": rt.gpus,
            "allocated_vram_gb": rt.allocated_vram_gb,
            "actual_vram_gb": rt.actual_vram_gb,
            "overshoot_gb": round(overshoot, 2),
            "wasted_seconds": round(ts - rt.started_ts, 1),
        }
        self.oom_events.append(event)
        (self.run_dir / "oom.jsonl").open("a", encoding="utf-8").write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )

        patch = json.dumps(
            {
                "status": {
                    "phase": "Failed",
                    "reason": "OOMKilled",
                    "message": (
                        f"нужно {rt.actual_vram_gb:.1f}ГБ VRAM, "
                        f"выделено {rt.allocated_vram_gb:.1f}ГБ "
                        f"({rt.pool}×{rt.gpus:.0f})"
                    ),
                    "containerStatuses": [
                        {
                            "name": "server",
                            "image": "vllm/vllm-openai:kwok-fake",
                            "ready": False,
                            "started": False,
                            "restartCount": 0,
                            "state": {
                                "terminated": {
                                    "exitCode": 137,
                                    "reason": "OOMKilled",
                                    "finishedAt": now_iso().replace("+00:00", "Z"),
                                }
                            },
                        }
                    ],
                }
            }
        )
        try:
            kubectl(
                "patch",
                "pod",
                rt.pod,
                "-n",
                rt.namespace,
                "--subresource=status",
                "--type=merge",
                "-p",
                patch,
            )
        except Exception as exc:  # noqa: BLE001 — прогон важнее одного патча
            eprint(f"[gpu-runtime] не смог пометить {rt.pod} как OOMKilled: {exc}")

    #потребление
    def _sample(self, rt: PodRuntime, node: NodeInfo, ts: float, phase_frac: float) -> None:
        """Потребление памяти и мощности"""
        if phase_frac < rt.startup_frac:
            ramp = phase_frac / rt.startup_frac if rt.startup_frac else 1.0
            vram = rt.actual_vram_gb * (0.15 + 0.85 * ramp)
            util = rt.util_pct * 0.1 * ramp
        else:
            jitter = 1.0 + self._rng.uniform(-0.08, 0.08)
            vram = rt.actual_vram_gb * min(1.0, 0.95 * jitter + 0.05)
            util = max(0.0, min(100.0, rt.util_pct * jitter))

        watts = rt.gpus * (node.idle_w + (node.tdp_w - node.idle_w) * util / 100.0)

        rt.samples += 1
        rt.vram_gb_seconds += vram
        rt.watt_seconds += watts
        rt.util_seconds += util

        self._usage_file.write(
            json.dumps(
                {
                    "ts": ts,
                    "job_id": rt.job_id,
                    "pod": rt.pod,
                    "node": rt.node,
                    "pool": rt.pool,
                    "gpus": rt.gpus,
                    "vram_gb": round(vram, 2),
                    "vram_allocated_gb": rt.allocated_vram_gb,
                    "watts": round(watts, 1),
                    "watts_reserved": round(rt.gpus * node.tdp_w, 1),
                    "util_pct": round(util, 1),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def close(self) -> None:
        if self._usage_file:
            self._usage_file.flush()
            self._usage_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Скоринг
# ──────────────────────────────────────────────────────────────────────────────


def efficiency_scores(
    usage_path: Path, interval: float, time_scale: float = 1.0
) -> dict:
    """interval — шаг снапшотера в РЕАЛЬНЫХ секундах, time_scale — сжатие
    времени. GPU-часы считаются в симулированном времени, поэтому без
    time_scale они занижены ровно в time_scale раз."""
    if not usage_path.exists():
        return {}

    per_job: dict[str, dict] = {}
    for line in usage_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        rec = per_job.setdefault(
            s["job_id"],
            {
                "samples": 0,
                "vram": 0.0,
                "vram_alloc": 0.0,
                "watts": 0.0,
                "watts_res": 0.0,
                "util": 0.0,
                "gpus": s["gpus"],
                "pool": s["pool"],
            },
        )
        rec["samples"] += 1
        rec["vram"] += s["vram_gb"]
        rec["vram_alloc"] += s["vram_allocated_gb"]
        rec["watts"] += s["watts"]
        rec["watts_res"] += s["watts_reserved"]
        rec["util"] += s["util_pct"]

    jobs = {}
    total_gpu_hours = wasted_gpu_hours = 0.0
    for job_id, r in per_job.items():
        if not r["samples"]:
            continue
        mem = r["vram"] / r["vram_alloc"] if r["vram_alloc"] else 0.0
        power = r["watts"] / r["watts_res"] if r["watts_res"] else 0.0
        eff = max(mem, power)
        gpu_hours = r["gpus"] * r["samples"] * interval * time_scale / 3600.0
        jobs[job_id] = {
            "memory_score": round(mem, 3),
            "power_score": round(power, 3),
            "util_mean_pct": round(r["util"] / r["samples"], 1),
            "efficiency_score": round(eff, 3),
            "gpu_hours": round(gpu_hours, 4),
            "wasted_gpu_hours": round(gpu_hours * (1 - eff), 4),
            "pool": r["pool"],
        }
        total_gpu_hours += gpu_hours
        wasted_gpu_hours += gpu_hours * (1 - eff)

    if not jobs:
        return {}

    return {
        "jobs": jobs,
        "memory_score_mean": round(
            sum(j["memory_score"] for j in jobs.values()) / len(jobs), 3
        ),
        "power_score_mean": round(
            sum(j["power_score"] for j in jobs.values()) / len(jobs), 3
        ),
        "efficiency_score_mean": round(
            sum(j["efficiency_score"] for j in jobs.values()) / len(jobs), 3
        ),
        # то же, но взвешенно по GPU-часам: пятиминутный smoke не должен
        # весить столько же, сколько четырёхчасовой long-serving
        "efficiency_score_weighted": round(
            sum(j["efficiency_score"] * j["gpu_hours"] for j in jobs.values())
            / total_gpu_hours,
            3,
        )
        if total_gpu_hours
        else None,
        "observed_gpu_hours": round(total_gpu_hours, 2),
        "wasted_gpu_hours": round(wasted_gpu_hours, 2),
        "worst_offenders": sorted(
            (
                {"job_id": k, **v}
                for k, v in jobs.items()
                if v["gpu_hours"] > 0
            ),
            key=lambda x: -x["wasted_gpu_hours"],
        )[:10],
    }
