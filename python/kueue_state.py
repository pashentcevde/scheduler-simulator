#!/usr/bin/env python3
"""
Состояние кластера, собранное по kueue, а не по физическим подам.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import ( 
    GPU_RES,
    PRODUCT_LABEL,
    PoolConfig,
    eprint,
    kubectl_json,
    parse_quantity,
)
from recommender import ClusterState, Headroom 

# ключ для ожидающих workload'ов, у которых флейвор не задан: в режиме
UNKNOWN_FLAVOR = "auto"


# ──────────────────────────────────────────────────────────────────────────────
# Разбор объектов kueue
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FlavorQuota:
    """Квота одной очереди по одному флейвору"""

    nominal: float = 0.0
    borrowing_limit: float | None = None  # None — kueue не ограничивает
    lending_limit: float | None = None    # None — отдаём весь свободный номинал

    @property
    def ceiling(self) -> float:
        """Потолок: сколько очередь может занять при идеально пустой когорте"""
        if self.borrowing_limit is None:
            return math.inf
        return self.nominal + self.borrowing_limit

    def lendable(self, used: float) -> float:
        """Сколько из своего простаивающего номинала очередь готова отдать"""
        free = max(0.0, self.nominal - used)
        limit = self.nominal if self.lending_limit is None else self.lending_limit
        return min(free, limit)


@dataclass
class QueueQuota:
    name: str
    cohort: str | None = None
    flavors: dict[str, FlavorQuota] = field(default_factory=dict)


def parse_cluster_queues(items: list[dict]) -> dict[str, QueueQuota]:
    """Квоты по GPU из spec.resourceGroups"""
    out: dict[str, QueueQuota] = {}
    for cq in items:
        name = cq["metadata"]["name"]
        spec = cq.get("spec", {}) or {}
        # v1beta2: cohortName; v1beta1 назывался cohort — читаем оба
        cohort = spec.get("cohortName") or spec.get("cohort")
        q = QueueQuota(name=name, cohort=cohort)
        for group in spec.get("resourceGroups", []) or []:
            for flavor in group.get("flavors", []) or []:
                for res in flavor.get("resources", []) or []:
                    if res.get("name") != GPU_RES:
                        continue
                    bl = res.get("borrowingLimit")
                    ll = res.get("lendingLimit")
                    q.flavors[flavor["name"]] = FlavorQuota(
                        nominal=parse_quantity(res.get("nominalQuota")),
                        borrowing_limit=None if bl is None else parse_quantity(bl),
                        lending_limit=None if ll is None else parse_quantity(ll),
                    )
        out[name] = q
    return out


def parse_usage(items: list[dict], prefer: str = "reservation") -> dict[str, dict[str, float]]:
    """Сколько карт каждая очередь держит по каждому флейвору.

    prefer='reservation' — flavorsReservation: квота зарезервирована, даже
    если под ещё не поехал. Это то, что kueue вычтет из доступного соседям,
    и потому именно это интересует рекомендателя.
    prefer='usage' — flavorsUsage: только допущенное.
    """
    first = "flavorsReservation" if prefer == "reservation" else "flavorsUsage"
    second = "flavorsUsage" if prefer == "reservation" else "flavorsReservation"
    out: dict[str, dict[str, float]] = {}
    for cq in items:
        st = cq.get("status", {}) or {}
        rows = st.get(first) or st.get(second) or []
        per_flavor: dict[str, float] = {}
        for fu in rows:
            for r in fu.get("resources", []) or []:
                if r.get("name") == GPU_RES:
                    per_flavor[fu["name"]] = parse_quantity(r.get("total"))
        out[cq["metadata"]["name"]] = per_flavor
    return out


def parse_pending_workloads(
    items: list[dict], product_to_flavor: dict[str, str]
) -> dict[str, list[tuple[float, float]]]:
    """Глубина очереди в картах: флейвор -> [(приоритет, карт), ...].

    Ожидающим считается workload без status.admission и без условия Finished.
    Целевой флейвор берётся из nodeSelector'а пода: именно так и пользователь,
    и рекомендатель просят конкретную карту. Если селектора нет (kueue-native),
    запись уходит в UNKNOWN_FLAVOR — какой флейвор ей достанется, до допуска
    не знает никто.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for w in items:
        status = w.get("status", {}) or {}
        if status.get("admission"):
            continue
        if any(
            c.get("type") in ("Finished", "Admitted") and c.get("status") == "True"
            for c in status.get("conditions", []) or []
        ):
            continue
        spec = w.get("spec", {}) or {}
        priority = float(spec.get("priority", 0) or 0)
        for ps in spec.get("podSets", []) or []:
            count = float(ps.get("count", 1) or 1)
            pod_spec = (ps.get("template", {}) or {}).get("spec", {}) or {}
            gpus = 0.0
            for c in pod_spec.get("containers", []) or []:
                req = (c.get("resources", {}) or {}).get("requests", {}) or {}
                gpus += parse_quantity(req.get(GPU_RES, 0))
            if gpus <= 0:
                continue
            product = (pod_spec.get("nodeSelector") or {}).get(PRODUCT_LABEL)
            flavor = product_to_flavor.get(product, UNKNOWN_FLAVOR)
            out.setdefault(flavor, []).append((priority, gpus * count))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Квотный запас
# ──────────────────────────────────────────────────────────────────────────────


def compute_headroom(
    quotas: dict[str, QueueQuota],
    usage: dict[str, dict[str, float]],
    queue: str,
    flavor: str,
    cohort_free: float,
) -> Headroom:
    """Сколько карт очередь может занять по этому флейвору.

        своё свободное   = nominalQuota - занято нами
        можно занять     = своё свободное + min(остаток borrowingLimit,
                                                сколько соседи готовы отдать)
        из этого сейчас  = min(можно занять, физически свободно в когорте)
        остальное        = наш номинал, который держат соседи; вернётся
                           вытеснением

    Последняя строка и есть то, ради чего всё это считается. По подам такая
    карта неотличима от чужой занятой, хотя kueue отдаст её по первому
    требованию: reclaimWithinCohort забирает свою квоту у кого угодно.
    """
    q = quotas.get(queue)
    if q is None or flavor not in q.flavors:
        # очередь не покрывает этот флейвор — карты недостижимы в принципе
        return Headroom(free=0.0, reclaim=0.0, limit=0.0, used=0.0)

    fq = q.flavors[flavor]
    used = usage.get(queue, {}).get(flavor, 0.0)
    own_free = max(0.0, fq.nominal - used)
    borrowed_now = max(0.0, used - fq.nominal)

    if fq.borrowing_limit is None:
        borrow_room = math.inf
    else:
        borrow_room = max(0.0, fq.borrowing_limit - borrowed_now)

    lendable = 0.0
    others_borrowed = 0.0
    for other, oq in quotas.items():
        if other == queue or oq.cohort != q.cohort or oq.cohort is None:
            continue
        ofq = oq.flavors.get(flavor)
        if ofq is None:
            continue
        oused = usage.get(other, {}).get(flavor, 0.0)
        lendable += ofq.lendable(oused)
        # сколько сосед взял сверх своего номинала — то есть из общего
        # простаивающего номинала когорты, часть которого наша
        others_borrowed += max(0.0, oused - ofq.nominal)

    total = own_free + min(borrow_room, lendable)
    free = min(total, max(0.0, cohort_free))

    # Вытеснением можно вернуть только своё: не больше свободного номинала
    # и не больше того, что соседи реально одолжили. Заимствованное сверх
    # номинала так не возвращается — на него у kueue другое правило
    # (borrowWithinCohort, только LowerPriority), и рассчитывать на него
    # рекомендатель не станет.
    reclaim = min(own_free, others_borrowed, max(0.0, total - free))

    return Headroom(
        free=free,
        reclaim=reclaim,
        limit=fq.ceiling if fq.ceiling != math.inf else fq.nominal + lendable,
        used=used,
    )


def build_state(
    cqs: list[dict],
    workloads: list[dict],
    product_to_flavor: dict[str, str],
    prefer: str = "reservation",
) -> ClusterState:
    """Собрать ClusterState из объектов kueue."""
    quotas = parse_cluster_queues(cqs)
    usage = parse_usage(cqs, prefer=prefer)

    flavors = sorted({f for q in quotas.values() for f in q.flavors})
    capacity = {
        f: sum(q.flavors[f].nominal for q in quotas.values() if f in q.flavors)
        for f in flavors
    }
    used_total = {
        f: sum(u.get(f, 0.0) for u in usage.values()) for f in flavors
    }
    cohort_free = {f: max(0.0, capacity[f] - used_total[f]) for f in flavors}

    headroom = {
        name: {
            f: compute_headroom(quotas, usage, name, f, cohort_free[f]) for f in flavors
        }
        for name in quotas
    }

    pending_counts = {
        cq["metadata"]["name"]: (cq.get("status", {}) or {}).get("pendingWorkloads", 0)
        for cq in cqs
    }

    return ClusterState(
        free_gpu_by_flavor=cohort_free,
        total_gpu_by_flavor=capacity,
        pending_by_queue=pending_counts,
        fetched_at=time.time(),
        source="kueue",
        headroom_by_queue=headroom,
        pending_gpu_by_flavor=parse_pending_workloads(workloads, product_to_flavor),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Источник состояния
# ──────────────────────────────────────────────────────────────────────────────


class KueueStateSource:
    """Кэширующий поставщик состояния, читающий kueue.

    Интерфейс тот же, что у ClusterStateSource: .get() -> ClusterState.
    Один вызов kubectl на TTL — clusterqueues и workloads забираются вместе.
    """

    def __init__(
        self,
        cfg: PoolConfig,
        ttl: float = 3.0,
        live: bool = True,
        prefer: str = "reservation",
    ):
        self.cfg = cfg
        self.ttl = ttl
        self.live = live
        self.prefer = prefer
        self.product_to_flavor = {p.product: p.flavor for p in cfg.pools}
        self._cache = ClusterState()
        self._static = ClusterState(
            total_gpu_by_flavor={p.flavor: float(p.total_gpus) for p in cfg.pools},
            free_gpu_by_flavor={p.flavor: float(p.total_gpus) for p in cfg.pools},
            source="static",
        )
        self._warned = False

    def get(self) -> ClusterState:
        if not self.live:
            return self._static
        now = time.time()
        if now - self._cache.fetched_at < self.ttl:
            return self._cache
        try:
            self._cache = self._fetch()
        except (RuntimeError, KeyError, ValueError) as exc:
            # кластер недоступен или отвечает не тем — не роняем подачу
            if not self._warned:
                eprint(f"[kueue-state] состояние недоступно ({exc}), "
                       f"работаем как с пустым кластером")
                self._warned = True
            return self._static
        return self._cache

    def _fetch(self) -> ClusterState:
        data = kubectl_json("get", "clusterqueues,workloads", "-A")
        items = data.get("items", [])
        cqs = [i for i in items if i.get("kind") == "ClusterQueue"]
        wls = [i for i in items if i.get("kind") == "Workload"]
        if not cqs:
            raise RuntimeError("ни одной ClusterQueue не найдено")
        return build_state(cqs, wls, self.product_to_flavor, prefer=self.prefer)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: показать, что видит рекомендатель
# ──────────────────────────────────────────────────────────────────────────────


def render(state: ClusterState) -> str:
    flavors = sorted(state.total_gpu_by_flavor)
    lines = [
        f"источник: {state.source}",
        "",
        "Ёмкость и занятость когорты (по kueue):",
        f"  {'флейвор':<14} {'всего':>6} {'свободно':>9} {'занято':>8} {'в очереди':>10}",
    ]
    for f in flavors:
        total = state.total_gpu_by_flavor[f]
        free = state.free_gpu_by_flavor.get(f, 0.0)
        ahead = state.ahead_gpus(f, 0)
        lines.append(
            f"  {f:<14} {total:>6.0f} {free:>9.0f} {100 * state.pressure(f):>7.0f}% {ahead:>10.0f}"
        )
    if UNKNOWN_FLAVOR in state.pending_gpu_by_flavor:
        n = sum(g for _, g in state.pending_gpu_by_flavor[UNKNOWN_FLAVOR])
        lines.append(f"  (+ {n:.0f} карт ждут без указания флейвора — kueue-native)")

    lines += ["", "Квотный запас по очередям (доступно сейчас / вернётся вытеснением / потолок):"]
    for queue in sorted(state.headroom_by_queue):
        lines.append(f"  {queue}")
        for f in flavors:
            hr = state.headroom(queue, f)
            mark = "  ← карты простаивают, но заняты чужой квотой" if (
                hr.free == 0 and state.free_gpu_by_flavor.get(f, 0) > 0
            ) else ""
            mark = mark or ("  ← своё, вернётся вытеснением" if hr.reclaim > 0 else "")
            lines.append(
                f"    {f:<14} занято {hr.used:>4.0f}  доступно {hr.free:>4.0f}  "
                f"reclaim {hr.reclaim:>4.0f}  потолок {hr.limit:>4.0f}{mark}"
            )
    if state.pending_by_queue:
        lines += ["", "Ожидают допуска (workload'ов): " + ", ".join(
            f"{q}={n}" for q, n in sorted(state.pending_by_queue.items())
        )]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument(
        "--fixture",
        help="json с полями clusterqueues/workloads вместо обращения к кластеру "
        "(формат — вывод `kubectl get clusterqueues,workloads -A -o json`)",
    )
    ap.add_argument(
        "--usage",
        choices=["reservation", "usage"],
        default="reservation",
        help="что считать занятым: зарезервированное (по умолчанию) или "
        "только допущенное",
    )
    ap.add_argument("--json", action="store_true", help="машинный вывод")
    args = ap.parse_args()

    cfg = PoolConfig.load(args.pool)
    p2f = {p.product: p.flavor for p in cfg.pools}

    if args.fixture:
        data = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
        items = data.get("items", [])
        cqs = [i for i in items if i.get("kind") == "ClusterQueue"]
        wls = [i for i in items if i.get("kind") == "Workload"]
        state = build_state(cqs, wls, p2f, prefer=args.usage)
    else:
        state = KueueStateSource(cfg, ttl=0.0, prefer=args.usage).get()

    if args.json:
        print(json.dumps(
            {
                "source": state.source,
                "capacity": state.total_gpu_by_flavor,
                "free": state.free_gpu_by_flavor,
                "pending_workloads": state.pending_by_queue,
                "headroom": {
                    q: {f: vars(h) for f, h in per.items()}
                    for q, per in state.headroom_by_queue.items()
                },
            },
            ensure_ascii=False,
            indent=1,
        ))
    else:
        print(render(state))


if __name__ == "__main__":
    main()
