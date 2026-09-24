"""
Общая библиотека стенда
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Метки/аннотации стенда
LAB = "lab.skalar.ai"
L_LAB = f"{LAB}/lab"
L_TENANT = f"{LAB}/tenant"
L_MODE = f"{LAB}/mode"
L_POOL = f"{LAB}/gpu-pool"
L_PLACEMENT = f"{LAB}/placement"
L_SCENARIO = f"{LAB}/scenario"
L_RUN = f"{LAB}/run"
L_MODEL = f"{LAB}/model"
L_PROFILE = f"{LAB}/profile"
L_REQ_POOL = f"{LAB}/requested-pool"
L_MIN_POOL = f"{LAB}/min-pool"
A_VRAM = f"{LAB}/vram-estimate-gb"
A_DURATION = f"{LAB}/planned-duration-seconds"
A_SIM_DURATION = f"{LAB}/planned-sim-minutes"
A_JOBID = f"{LAB}/job-id"
A_USER_POOL = f"{LAB}/user-requested-pool"

# то, что известно стенду, но не видно рекомендателю
A_ACTUAL_VRAM = f"{LAB}/actual-vram-gb"
A_UTIL_PCT = f"{LAB}/actual-util-pct"
A_STARTUP_FRAC = f"{LAB}/startup-frac"
A_ATTEMPT = f"{LAB}/attempt"

# метки планировщика
A_REC_POOL = "scheduler.skalar.ai/recommended-gpu-product"
A_REC_CONF = "scheduler.skalar.ai/recommendation-confidence"
A_REC_POLICY = "scheduler.skalar.ai/recommendation-policy"
A_REC_REASON = "scheduler.skalar.ai/recommendation-reason"

KUEUE_QUEUE = "kueue.x-k8s.io/queue-name"
KUEUE_PRIO = "kueue.x-k8s.io/priority-class"
KUEUE_MAXEXEC = "kueue.x-k8s.io/max-exec-time-seconds"

KWOK_DELAY = "pod-complete.stage.kwok.x-k8s.io/delay"
KWOK_JITTER = "pod-complete.stage.kwok.x-k8s.io/jitter-delay"
KWOK_TOLERATION = {
    "key": "kwok.x-k8s.io/node",
    "operator": "Equal",
    "value": "fake",
    "effect": "NoSchedule",
}

GPU_RES = "nvidia.com/gpu"

# Псевдо-пул: карта не выбрана, флейвор подберёт сам kueue перебором
# в порядке объявления в resourceGroups (от дешёвых к дорогим).
AUTO_POOL = "auto"
PRODUCT_LABEL = "nvidia.com/gpu.product"

# Заглушка kwok
IMAGE = "vllm/vllm-openai:kwok-fake"


# ──────────────────────────────────────────────────────────────────────────────
# kubectl
# ──────────────────────────────────────────────────────────────────────────────


def kube_context() -> str:
    """Контекст kubectl. Пусто = текущий контекст пользователя."""
    return os.environ.get("KUBE_CONTEXT", "")
    

def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    ctx = kube_context()
    cmd = ["kubectl", *(["--context", ctx] if ctx else []), *args]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # нет kubectl в PATH
        raise RuntimeError(f"не смог запустить kubectl: {exc}") from exc
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"kubectl {' '.join(args)} -> rc={proc.returncode}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def kubectl_json(*args: str) -> dict:
    """`kubectl get ... -o json`, по возможности через локальный kubectl proxy.

    Быстрый путь убирает форк процесса и TLS-хендшейк на каждом вызове — это
    то, что упирает такт снапшотера в ~1 секунду. Если прокси недоступен или
    форма команды непривычная, молча откатываемся на subprocess, поэтому
    поведение снаружи не меняется.
    """
    if os.environ.get("KUBE_FAST", "1") != "0":
        try:
            from kube_fast import try_get_json  # noqa: PLC0415 — ленивый импорт

            fast = try_get_json(args)
        except ImportError:
            fast = None
        if fast is not None:
            return fast
    out = kubectl(*args, "-o", "json")
    return json.loads(out) if out.strip() else {}


class NoAliasDumper(yaml.SafeDumper):
    """Без YAML-якорей"""

    def ignore_aliases(self, data):
        return True


def dump_yaml(obj: dict) -> str:
    return yaml.dump(obj, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True)


def apply_docs(docs: Iterable[dict]) -> None:
    """kubectl apply нескольких объектов одним вызовом."""
    payload = "\n---\n".join(dump_yaml(d) for d in docs)
    if not payload.strip():
        return
    kubectl("apply", "-f", "-", stdin=payload)


# ──────────────────────────────────────────────────────────────────────────────
# Конфиги
# ──────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Pool:
    name: str
    product: str
    flavor: str
    vram_gb: float
    tdp_w: float
    idle_w: float
    relative_cost: float
    nodes: int
    gpus_per_node: int

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node


@dataclass
class PoolConfig:
    defaults: dict
    pools: list[Pool]

    @classmethod
    def load(cls, path: str | Path = ROOT / "config" / "pool.yaml") -> "PoolConfig":
        raw = load_yaml(path)
        pools = [Pool(**p) for p in raw["pools"]]
        return cls(defaults=raw["defaults"], pools=pools)

    def by_name(self, name: str) -> Pool:
        for p in self.pools:
            if p.name == name:
                return p
        raise KeyError(name)

    def sorted_by_cost(self) -> list[Pool]:
        return sorted(self.pools, key=lambda p: (p.relative_cost, p.vram_gb))

    @property
    def total_gpus(self) -> int:
        return sum(p.total_gpus for p in self.pools)


# ──────────────────────────────────────────────────────────────────────────────
# Модель потребления VRAM
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelCatalog:
    runtime_overhead_gb: float
    safety_margin: float
    bytes_per_param: dict[str, float]
    actual_vram_noise: tuple[float, float]
    models: list[dict]
    job_profiles: list[dict]

    @classmethod
    def load(cls, path: str | Path = ROOT / "config" / "models.yaml") -> "ModelCatalog":
        raw = load_yaml(path)
        return cls(
            runtime_overhead_gb=raw["runtime_overhead_gb"],
            safety_margin=raw["safety_margin"],
            bytes_per_param=raw["bytes_per_param"],
            actual_vram_noise=tuple(raw.get("actual_vram_noise", [1.0, 1.0])),
            models=raw["models"],
            job_profiles=raw["job_profiles"],
        )

    def by_id(self, model_id: str) -> dict:
        for m in self.models:
            if m["id"] == model_id:
                return m
        raise KeyError(model_id)

    def raw_vram_gb(self, model_id: str, precision: str, ctx_tokens: int) -> float:
        m = self.by_id(model_id)
        weights = m["params_b"] * self.bytes_per_param[precision]
        kv = m["kv_gb_per_1k_ctx"] * (ctx_tokens / 1000.0)
        return weights + kv + self.runtime_overhead_gb

    def vram_estimate_gb(self, model_id: str, precision: str, ctx_tokens: int) -> float:
        return round(
            self.raw_vram_gb(model_id, precision, ctx_tokens)
            * (1.0 + self.safety_margin),
            2,
        )


USABLE_VRAM_FRACTION = 0.92
ALLOWED_TP = (1, 2, 4, 8)


def fit_on_pool(pool: Pool, vram_gb: float) -> int | None:
    """Минимальное число карт этого пула (степень двойки, в пределах ноды),
    на которое влезает vram_gb. None — не влезает."""
    for tp in ALLOWED_TP:
        if tp > pool.gpus_per_node:
            break
        if tp * pool.vram_gb * USABLE_VRAM_FRACTION >= vram_gb:
            return tp
    return None


def cheapest_placement(cfg: PoolConfig, vram_gb: float) -> tuple[Pool, int]:
    """Самый дешёвый вариант размещения: минимизируем count * relative_cost."""
    best: tuple[float, int, Pool] | None = None
    for pool in cfg.sorted_by_cost():
        tp = fit_on_pool(pool, vram_gb)
        if tp is None:
            continue
        cost = tp * pool.relative_cost
        if best is None or (cost, tp) < (best[0], best[1]):
            best = (cost, tp, pool)
    if best is None:
        # не влезает никуда — берём самый крупный пул на максимуме TP
        pool = max(cfg.pools, key=lambda p: p.vram_gb * min(p.gpus_per_node, 8))
        return pool, min(pool.gpus_per_node, 8)
    return best[2], best[1]


def undersized_placement(
    cfg: PoolConfig, min_pool: Pool, min_gpus: int, rng
) -> tuple[Pool, int]:
    """Пользователь недооценил задачу"""
    cheaper = [p for p in cfg.sorted_by_cost() if p.relative_cost < min_pool.relative_cost]
    if cheaper:
        pool = rng.choice(cheaper[-2:]) 
        return pool, min(min_gpus, pool.gpus_per_node)
    if min_gpus > 1:
        return min_pool, min_gpus // 2
    return min_pool, min_gpus


def oversized_placement(
    cfg: PoolConfig, vram_gb: float, min_pool: Pool, rng
) -> tuple[Pool, int]:
    """Пользователь просит карту побольше, с запасом"""
    candidates = [
        p
        for p in cfg.sorted_by_cost()
        if p.relative_cost > min_pool.relative_cost and fit_on_pool(p, vram_gb)
    ]
    if not candidates:
        return min_pool, fit_on_pool(min_pool, vram_gb) or 1
    weights = [1.0 / (i + 1) for i in range(len(candidates))]
    pool = rng.choices(candidates, weights=weights, k=1)[0]
    return pool, fit_on_pool(pool, vram_gb) or 1


# ──────────────────────────────────────────────────────────────────────────────
# Рендер Job'а
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class JobSpec:
    job_id: str
    tenant: str
    namespace: str
    queue: str
    priority_class: str
    model_id: str
    precision: str
    ctx_tokens: int
    profile: str
    vram_gb: float
    min_pool: str
    min_gpus: int
    requested_pool: str
    gpus: int
    user_pool: str
    user_gpus: int
    placement_policy: str
    confidence: float | None
    reason: str
    actual_vram_gb: float
    util_pct: float
    startup_frac: float
    attempt: int
    duration_seconds: int
    sim_minutes: float
    submit_at_sim_min: float
    scenario: str
    run_id: str
    oversized: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _kwok_duration(seconds: float) -> str:
    """Длительность для аннотаций kwok, в миллисекундах.

    Целые секунды не годятся: при большом time_scale короткие профили
    (smoke — 5-20 сим-минут) схлопываются в 0-2 секунды и теряют разброс.
    """
    return f"{max(1, int(round(seconds * 1000)))}ms"


def render_job(spec: JobSpec, cfg: PoolConfig, max_exec_factor: float = 0.0) -> dict:
    """Собрать манифест batch/v1 Job для kueue"""
    auto = spec.requested_pool == AUTO_POOL
    pool = None if auto else cfg.by_name(spec.requested_pool)
    d = cfg.defaults
    cpu = d["cpu_per_gpu"] * spec.gpus
    mem = f"{d['memory_gi_per_gpu'] * spec.gpus}Gi"

    labels = {
        L_LAB: "true",
        L_TENANT: spec.tenant,
        L_MODE: "kueue",
        L_PLACEMENT: spec.placement_policy,
        L_SCENARIO: spec.scenario,
        L_RUN: spec.run_id,
        L_MODEL: spec.model_id,
        L_PROFILE: spec.profile,
        L_REQ_POOL: spec.requested_pool,
        L_MIN_POOL: spec.min_pool,
        L_POOL: spec.requested_pool,
    }
    annotations = {
        A_VRAM: str(spec.vram_gb),
        A_DURATION: str(spec.duration_seconds),
        A_SIM_DURATION: str(round(spec.sim_minutes, 1)),
        A_JOBID: spec.job_id,
        A_USER_POOL: spec.user_pool,
        A_ACTUAL_VRAM: str(spec.actual_vram_gb),
        A_UTIL_PCT: str(round(spec.util_pct, 1)),
        A_STARTUP_FRAC: str(round(spec.startup_frac, 3)),
        A_ATTEMPT: str(spec.attempt),
        A_REC_POOL: AUTO_POOL if auto else pool.product,
        A_REC_CONF: "n/a" if spec.confidence is None else f"{spec.confidence:.3f}",
        A_REC_POLICY: spec.placement_policy,
        A_REC_REASON: spec.reason,
    }

    node_selector = {} if auto else {PRODUCT_LABEL: pool.product}

    pod_labels = dict(labels)

    labels[KUEUE_QUEUE] = spec.queue
    labels[KUEUE_PRIO] = spec.priority_class
    if max_exec_factor > 0:
        labels[KUEUE_MAXEXEC] = str(int(spec.duration_seconds * max_exec_factor) + 60)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": spec.job_id,
            "namespace": spec.namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "parallelism": 1,
            "completions": 1,
            "backoffLimit": 0,
            "suspend": True,
            "template": {
                "metadata": {
                    "labels": pod_labels,
                    "annotations": {
                        **annotations,
                        # столько под будет "работать" под управлением kwok.
                        # jitter обязателен и должен быть равен delay: иначе
                        # kwok берёт jitterDurationMilliseconds из Stage
                        # (63000) и разыгрывает длительность как
                        # U(delay, 63s), обрезая всё, что длиннее
                        KWOK_DELAY: _kwok_duration(spec.duration_seconds),
                        KWOK_JITTER: _kwok_duration(spec.duration_seconds),
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    **({} if auto else {"nodeSelector": node_selector}),
                    "tolerations": [KWOK_TOLERATION],
                    "containers": [
                        {
                            "name": "server",
                            "image": IMAGE,
                            "args": [
                                f"--model={spec.model_id}",
                                f"--max-model-len={spec.ctx_tokens}",
                                f"--quantization={spec.precision}",
                                f"--tensor-parallel-size={spec.gpus}",
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": str(cpu),
                                    "memory": mem,
                                    GPU_RES: str(spec.gpus),
                                },
                                "limits": {
                                    "cpu": str(cpu),
                                    "memory": mem,
                                    GPU_RES: str(spec.gpus),
                                },
                            },
                        }
                    ],
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Прочее
# ──────────────────────────────────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    v = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


def parse_quantity(q: Any) -> float:
    """Достаточная для стенда реализация разбора k8s quantity."""
    if q is None:
        return 0.0
    if isinstance(q, (int, float)):
        return float(q)
    s = str(q).strip()
    if not s:
        return 0.0
    multipliers = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "Pi": 1024**5,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
        "m": 1e-3,
    }
    for suffix in ("Ki", "Mi", "Gi", "Ti", "Pi", "k", "M", "G", "T", "P", "m"):
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * multipliers[suffix]
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def weighted_choice(rng, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = [float(v) for v in mapping.values()]
    return rng.choices(keys, weights=weights, k=1)[0]


def poisson_gap(rng, rate_per_second: float) -> float:
    """Интервал до следующего события пуассоновского потока."""
    if rate_per_second <= 0:
        return math.inf
    return rng.expovariate(rate_per_second)


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)
