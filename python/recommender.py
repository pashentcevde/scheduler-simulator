"""
  ManualPolicy — карту выбрал пользователь, планировщик не вмешивается.
  
  CheapestFitPolicy — оценить VRAM, отфильтровать заведомо неподходящие карты,
                      взять самый дешёвый вариант размещения.
                      
  QueueAwarePolicy — то же плюс штраф за занятость флейвора. из нескольких
                     подходящих вариантов выбирается не просто самый дешёвый,
                     а тот, где задача быстрее поедет.
                      
  KueueAwarePolicy — занятость и глубина очереди берутся из kueue
                     (квоты, заимствование, pending workloads), а не из
                     физических подов.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from lab import (
    AUTO_POOL,
    GPU_RES,
    ALLOWED_TP,
    USABLE_VRAM_FRACTION,
    Pool,
    PoolConfig,
    fit_on_pool,
    kubectl_json,
    parse_quantity,
)


# ──────────────────────────────────────────────────────────────────────────────
# Данные
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class JobRequest:
    """Всё, что известно о задаче в момент подачи"""

    model_id: str
    params_b: float
    precision: str
    ctx_tokens: int
    vram_gb: float
    tenant: str
    queue: str
    priority_class: str
    priority_value: int
    # что попросил пользователь (может отсутствовать, если он не указал карту)
    user_pool: str | None = None
    user_gpus: int | None = None


@dataclass
class Placement:
    pool: str
    gpus: int
    confidence: float | None   # вероятность вместимости; None — не определена
    policy: str
    reason: str


@dataclass
class UncertaintyModel:
    """Модель неопределённости оценки VRAM"""

    safety_margin: float = 0.10
    noise_lo: float = 0.85
    noise_hi: float = 1.25

    @classmethod
    def from_catalog(cls, catalog) -> "UncertaintyModel":
        lo, hi = catalog.actual_vram_noise
        return cls(
            safety_margin=catalog.safety_margin,
            noise_lo=float(lo),
            noise_hi=float(hi),
        )

    def fit_probability(self, usable_gb: float, declared_gb: float) -> float:
        """Вероятность того, что задача поместится в выделенную память. """
        if declared_gb <= 0:
            return 0.5
        threshold = usable_gb * (1.0 + self.safety_margin) / declared_gb
        span = self.noise_hi - self.noise_lo
        if span <= 0:
            return 1.0 if threshold >= self.noise_hi else 0.0
        return max(0.0, min(1.0, (threshold - self.noise_lo) / span))


@dataclass
class Headroom:
    """Сколько карт этого флейвора очередь может занять прямо сейчас.

    free      можно занять, никого не трогая: свободный номинал плюс то,
              что соседи по когорте готовы одолжить (borrowingLimit против
              их lendingLimit), в пределах физически незанятого.
    reclaim   свой номинал, который прямо сейчас занят соседями. Физически
              карта занята, но по квоте она наша: kueue отберёт её обратно
              вытеснением (reclaimWithinCohort). Стоит это чужой потерянной
              работы, поэтому в решении учитывается со штрафом.
    limit     квотный потолок: nominalQuota + borrowingLimit. Выше него
              очередь не поднимется, сколько бы карт ни простаивало.
    """

    free: float = 0.0
    reclaim: float = 0.0
    limit: float = 0.0
    used: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.reclaim


@dataclass
class ClusterState:
    """Снимок загрузки кластера, каким его видит рекомендатель"""

    free_gpu_by_flavor: dict[str, float] = field(default_factory=dict)
    total_gpu_by_flavor: dict[str, float] = field(default_factory=dict)
    pending_by_queue: dict[str, int] = field(default_factory=dict)
    fetched_at: float = 0.0
    
    # картина по kueue
    source: str = "pods"
    # очередь -> флейвор -> сколько карт очередь может занять
    headroom_by_queue: dict[str, dict[str, Headroom]] = field(default_factory=dict)
    # флейвор -> [(приоритет, карт), ...] по ещё не допущенным workload'ам
    pending_gpu_by_flavor: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )

    def pressure(self, flavor: str) -> float:
        """0.0 — флейвор свободен, 1.0 — занят целиком."""
        total = self.total_gpu_by_flavor.get(flavor, 0.0)
        if total <= 0:
            return 1.0
        free = max(0.0, self.free_gpu_by_flavor.get(flavor, 0.0))
        return max(0.0, min(1.0, 1.0 - free / total))
        
    def headroom(self, queue: str, flavor: str) -> Headroom:
        """Квотный запас очереди по флейвору. Если источник состояния 
        про квоты ничего не знает, деградируем до физической доступности
        """
        hr = self.headroom_by_queue.get(queue)
        if hr is None:
            free = max(0.0, self.free_gpu_by_flavor.get(flavor, 0.0))
            total = self.total_gpu_by_flavor.get(flavor, 0.0)
            return Headroom(free=free, reclaim=0.0, limit=total, used=total - free)
        return hr.get(flavor, Headroom())

    def ahead_gpus(self, flavor: str, priority_value: float) -> float:
        """Сколько карт этого флейвора уже просят те, кто нас не пропустит.
        Считаются ожидающие допуска workload'ы с приоритетом не ниже нашего
        """
        return sum(
            gpus
            for prio, gpus in self.pending_gpu_by_flavor.get(flavor, ())
            if prio >= priority_value
        )

    def queue_depth(self, flavor: str, priority_value: float) -> float:
        """Глубина очереди по флейвору от-но его ёмкости.
        0.0 — впереди никого, 1.0 — впереди задачи ровно на весь пул
        """
        total = self.total_gpu_by_flavor.get(flavor, 0.0)
        if total <= 0:
            return 0.0
        return self.ahead_gpus(flavor, priority_value) / total


class Policy(Protocol):
    name: str

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement: ...


# ──────────────────────────────────────────────────────────────────────────────
# Сбор состояния кластера
# ──────────────────────────────────────────────────────────────────────────────


class ClusterStateSource:
    """Кэширующий поставщик состояния кластера.
    Занятость считается по подам на нодах, а не по квотам kueue
    """

    def __init__(self, cfg: PoolConfig, ttl: float = 1.0, live: bool = True):
        self.cfg = cfg
        self.ttl = ttl
        self.live = live
        self._cache = ClusterState()
        self._static = ClusterState(
            total_gpu_by_flavor={p.flavor: float(p.total_gpus) for p in cfg.pools},
            free_gpu_by_flavor={p.flavor: float(p.total_gpus) for p in cfg.pools},
        )

    def get(self) -> ClusterState:
        if not self.live:
            return self._static
        now = time.time()
        if now - self._cache.fetched_at < self.ttl:
            return self._cache
        try:
            self._cache = self._fetch()
        except RuntimeError:
            # кластер недоступен — не роняем подачу, работаем как без состояния
            return self._static
        return self._cache

    def _fetch(self) -> ClusterState:
        flavor_of_product = {p.product: p.flavor for p in self.cfg.pools}
        total = {p.flavor: float(p.total_gpus) for p in self.cfg.pools}
        used: dict[str, float] = {f: 0.0 for f in total}

        pods = kubectl_json("get", "pods", "-A", "-l", "lab.skalar.ai/lab=true")
        node_flavor: dict[str, str] = {}
        nodes = kubectl_json("get", "nodes", "-l", "lab.skalar.ai/gpu-pool")
        for n in nodes.get("items", []):
            product = n["metadata"]["labels"].get("nvidia.com/gpu.product")
            if product in flavor_of_product:
                node_flavor[n["metadata"]["name"]] = flavor_of_product[product]

        for pod in pods.get("items", []):
            if pod.get("status", {}).get("phase") != "Running":
                continue
            node = pod.get("spec", {}).get("nodeName")
            flavor = node_flavor.get(node)
            if not flavor:
                continue
            for c in pod["spec"].get("containers", []):
                used[flavor] += parse_quantity(
                    c.get("resources", {}).get("requests", {}).get(GPU_RES, 0)
                )

        pending: dict[str, int] = {}
        cqs = kubectl_json("get", "clusterqueues")
        for cq in cqs.get("items", []):
            pending[cq["metadata"]["name"]] = cq.get("status", {}).get(
                "pendingWorkloads", 0
            )

        return ClusterState(
            free_gpu_by_flavor={f: total[f] - used[f] for f in total},
            total_gpu_by_flavor=total,
            pending_by_queue=pending,
            fetched_at=time.time(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Политики
# ──────────────────────────────────────────────────────────────────────────────


# Бюджет срочности: во сколько раз дороже минимально достаточного варианта
# политика готова заплатить за немедленный старт
PRIORITY_BUDGET = {"production": 8.0, "high": 6.0, "normal": 4.0, "low": 1.5}
DEFAULT_BUDGET = 4.0


def budget_ratio(priority_class: str, override: float | None = None) -> float:
    if override is not None:
        return override
    return PRIORITY_BUDGET.get(priority_class, DEFAULT_BUDGET)


def _confidence(
    pool: Pool, gpus: int, vram_gb: float, unc: UncertaintyModel | None = None
) -> float:
    """Вероятность того, что задача поместится в выбранную конфигурацию"""
    usable = gpus * pool.vram_gb * USABLE_VRAM_FRACTION
    if vram_gb <= 0:
        return 0.5
    return round((unc or UncertaintyModel()).fit_probability(usable, vram_gb), 3)


def _candidates(cfg: PoolConfig, vram_gb: float) -> list[tuple[Pool, int, float]]:
    """Все допустимые варианты размещения: (пул, число карт, условная цена)."""
    out = []
    for pool in cfg.pools:
        tp = fit_on_pool(pool, vram_gb)
        if tp is None:
            continue
        out.append((pool, tp, tp * pool.relative_cost))
    return out


class KueueNativePolicy:
    """Никакого подбора: решение отдано механизму флейворов kueue"""

    name = "kueue-native"

    def __init__(self, cfg: PoolConfig, uncertainty: UncertaintyModel | None = None):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        gpus = request.user_gpus or 1
        return Placement(
            pool=AUTO_POOL,
            gpus=gpus,
            # карта станет известна только после допуска, поэтому вероятность
            # вместимости в момент подачи не определена
            confidence=None,
            policy=self.name,
            reason=f"карта не задана, флейвор подберёт kueue; {gpus} карт(ы) "
            f"из запроса пользователя",
        )


class ManualPolicy:
    """Карту выбирает пользователь."""

    name = "manual"

    def __init__(self, cfg: PoolConfig, uncertainty: UncertaintyModel | None = None):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        if request.user_pool and request.user_gpus:
            pool = self.cfg.by_name(request.user_pool)
            return Placement(
                pool=request.user_pool,
                gpus=request.user_gpus,
                confidence=_confidence(pool, request.user_gpus, request.vram_gb, self.unc),
                policy=self.name,
                reason="выбор пользователя",
            )
        # пользователь ничего не указал — деваться некуда, считаем эвристикой
        return CheapestFitPolicy(self.cfg, self.unc).recommend(request, state)


class CheapestFitPolicy:
    """самый дешёвый вариант, куда задача влезает"""

    name = "cheapest"

    def __init__(self, cfg: PoolConfig, uncertainty: UncertaintyModel | None = None):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            pool = max(
                self.cfg.pools, key=lambda p: p.vram_gb * min(p.gpus_per_node, max(ALLOWED_TP))
            )
            gpus = min(pool.gpus_per_node, max(ALLOWED_TP))
            return Placement(
                pool=pool.name,
                gpus=gpus,
                confidence=0.3,
                policy=self.name,
                reason=f"оценка {request.vram_gb:.0f}ГБ не влезает ни в один пул, "
                f"берём максимум",
            )
        pool, gpus, cost = min(cands, key=lambda c: (c[2], c[1], c[0].vram_gb))
        return Placement(
            pool=pool.name,
            gpus=gpus,
            confidence=_confidence(pool, gpus, request.vram_gb, self.unc),
            policy=self.name,
            reason=f"оценка {request.vram_gb:.0f}ГБ, дешевле всего {pool.name}×{gpus} "
            f"(цена {cost:.1f})",
        )


class QueueAwarePolicy:
    """Дешевизна плюс доступность прямо сейчас.

    Чисто стоимостная эвристика сгоняет всю мелочь на дешёвые карты и оставляет
    дорогие простаивать: задача формально размещена оптимально с точки зрения утилизации, но стоит в
    очереди. Здесь решение принимается в два шага:

      1. если есть варианты, куда задача влезает прямо сейчас (свободных карт
         хватает), рассматриваются только они.
      2. внутри этой группы цена домножается на штраф за занятость флейвора,
         чтобы не занимать последнюю свободную карту дефицитного пула.
         
    Так же вводится бюджет срочности: платить за немедленный старт
    больше, чем в n против самого дешёвого варианта, планировщик не станет. 
    Множитель зависит от приоритета задачи (PRIORITY_BUDGET):
    проду разрешено уехать на карту сильно дороже минимально
    достаточной, лишь бы стартовать сейчас, а low-задача обязана дождаться
    дешёвой. если задан параметр `max_cost_ratio`, он перекрывает приоритет одним общим значением.

    Если свободного ничего нет, выбирается самый дешёвый вариант — задача всё
    равно встанет в очередь, поэтому логично поставить ее в дешевую
    """

    name = "queue-aware"

    def __init__(
        self,
        cfg: PoolConfig,
        uncertainty: UncertaintyModel | None = None,
        congestion_weight: float = 1.5,
        max_cost_ratio: float | None = None,
    ):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()
        self.congestion_weight = congestion_weight
        self.max_cost_ratio = max_cost_ratio
        self._fallback = CheapestFitPolicy(cfg, uncertainty)

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)

        cheapest_pool, cheapest_gpus, cheapest_cost = min(
            cands, key=lambda c: (c[2], c[1])
        )
        ratio = budget_ratio(request.priority_class, self.max_cost_ratio)
        budget = cheapest_cost * ratio

        available = [
            c
            for c in cands
            if state.free_gpu_by_flavor.get(c[0].flavor, 0.0) >= c[1]
            and c[2] <= budget
        ]

        if not available:
            p = self._fallback.recommend(request, state)
            p.policy = self.name
            p.reason += (
                f"; в бюджет {budget:.1f} ({request.priority_class}, ×{ratio:g}) "
                f"и в свободные карты не уложились, ждём в очереди"
            )
            return p

        def score(c):
            pool, gpus, cost = c
            return cost * (1.0 + self.congestion_weight * state.pressure(pool.flavor))

        pool, gpus, cost = min(available, key=lambda c: (score(c), c[2], c[1]))
        free = state.free_gpu_by_flavor.get(pool.flavor, 0.0)
        note = ""
        if pool.name != cheapest_pool.name or gpus != cheapest_gpus:
            note = (
                f"; вместо {cheapest_pool.name}×{cheapest_gpus} — там занятость "
                f"{100 * state.pressure(cheapest_pool.flavor):.0f}%, "
                f"свободно {state.free_gpu_by_flavor.get(cheapest_pool.flavor, 0):.0f}"
            )
        return Placement(
            pool=pool.name,
            gpus=gpus,
            confidence=_confidence(pool, gpus, request.vram_gb, self.unc),
            policy=self.name,
            reason=f"оценка {request.vram_gb:.0f}ГБ, {pool.name}×{gpus} стартует сразу "
            f"(свободно {free:.0f}, занятость "
            f"{100 * state.pressure(pool.flavor):.0f}%), бюджет {budget:.1f} "
            f"({request.priority_class}){note}",
        )
        
        
class KueueAwarePolicy:
    """Занятость и очередь — по kueue, а не по физическим подам.
    Решение принимается так:

      1. бюджет срочности по приоритету отсекает слишком дорогие варианты
      2. если что-то доступно по квоте без вытеснения — выбираем среди этого,
         минимизируя цену со штрафами за занятость и за глубину очереди;
      3. если нет, но есть вариант, доступный через reclaim своего номинала,
         берём его со штрафом reclaim_penalty: вытеснение стоит чужой работы,
         но своё мы имеем право забрать. Для low этот путь закрыт;
      4. если не доступно ничего — задача всё равно встанет в очередь; выбираем
         очередь покороче в пределах бюджета, а не просто самую дешёвую.
    """

    name = "kueue-aware"
    state_source = "kueue"

    # кому разрешено рассчитывать на возврат своей квоты вытеснением
    RECLAIM_ALLOWED = ("production", "high", "normal")

    def __init__(
        self,
        cfg: PoolConfig,
        uncertainty: UncertaintyModel | None = None,
        congestion_weight: float = 1.0,
        queue_weight: float = 1.2,
        reclaim_penalty: float = 0.5,
        max_cost_ratio: float | None = None,
    ):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()
        self.congestion_weight = congestion_weight
        self.queue_weight = queue_weight
        self.reclaim_penalty = reclaim_penalty
        self.max_cost_ratio = max_cost_ratio
        self._fallback = CheapestFitPolicy(cfg, uncertainty)

    def _score(self, cost: float, flavor: str, state: ClusterState, prio: float) -> float:
        """Цена с двумя штрафами: за занятость и за очередь впереди нас"""
        return (
            cost
            * (1.0 + self.congestion_weight * state.pressure(flavor))
            * (1.0 + self.queue_weight * state.queue_depth(flavor, prio))
        )

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)

        prio = request.priority_value
        cheapest_cost = min(c[2] for c in cands)
        ratio = budget_ratio(request.priority_class, self.max_cost_ratio)
        budget = cheapest_cost * ratio
        affordable = [c for c in cands if c[2] <= budget]

        now: list[tuple] = []       # доступно по квоте, никого не трогая
        by_reclaim: list[tuple] = []  # доступно через возврат своей квоты
        for pool, gpus, cost in affordable:
            hr = state.headroom(request.queue, pool.flavor)
            if hr.free >= gpus:
                now.append((pool, gpus, cost, hr))
            elif hr.total >= gpus:
                by_reclaim.append((pool, gpus, cost, hr))

        if now:
            pool, gpus, cost, hr = min(
                now, key=lambda c: (self._score(c[2], c[0].flavor, state, prio), c[2], c[1])
            )
            depth = state.ahead_gpus(pool.flavor, prio)
            reason = (
                f"оценка {request.vram_gb:.0f}ГБ, {pool.name}×{gpus}: по квоте "
                f"{request.queue} доступно {hr.free:.0f} карт "
                f"(занято по kueue {100 * state.pressure(pool.flavor):.0f}%, "
                f"впереди в очереди {depth:.0f} карт), бюджет {budget:.1f} "
                f"({request.priority_class})"
            )
            reason += self._why_not_cheapest(cands, pool, gpus, request, state, prio)
            return self._out(pool, gpus, request, reason)

        if by_reclaim and request.priority_class in self.RECLAIM_ALLOWED:
            pool, gpus, cost, hr = min(
                by_reclaim,
                key=lambda c: (
                    self._score(c[2], c[0].flavor, state, prio) * (1.0 + self.reclaim_penalty),
                    c[2],
                    c[1],
                ),
            )
            reason = (
                f"оценка {request.vram_gb:.0f}ГБ, {pool.name}×{gpus}: свободных карт "
                f"нет, но {hr.reclaim:.0f} из номинала {request.queue} заняты "
                f"соседями по когорте — kueue вернёт их вытеснением "
                f"(приоритет {request.priority_class})"
            )
            return self._out(pool, gpus, request, reason)

        # Стартовать сейчас негде. Задача встаёт в очередь — выбираем, в какую:
        # самая дешёвая очередь не обязательно самая короткая.
        pool, gpus, cost = min(
            affordable or cands,
            key=lambda c: (
                c[2] * (1.0 + self.queue_weight * state.queue_depth(c[0].flavor, prio)),
                c[2],
                c[1],
            ),
        )
        hr = state.headroom(request.queue, pool.flavor)
        reason = (
            f"оценка {request.vram_gb:.0f}ГБ, стартовать негде: по квоте "
            f"{request.queue} доступно {hr.total:.0f} карт при потолке "
            f"{hr.limit:.0f}. Встаём в самую короткую очередь в пределах "
            f"бюджета {budget:.1f} — {pool.name}×{gpus}, впереди "
            f"{state.ahead_gpus(pool.flavor, prio):.0f} карт"
        )
        return self._out(pool, gpus, request, reason)

    def _why_not_cheapest(
        self,
        cands: list,
        chosen: Pool,
        chosen_gpus: int,
        request: JobRequest,
        state: ClusterState,
        prio: float,
    ) -> str:
        """Если взят не самый дешёвый вариант — сказать, что было не так"""
        pool, gpus, _ = min(cands, key=lambda c: (c[2], c[1]))
        if pool.name == chosen.name and gpus == chosen_gpus:
            return ""
        hr = state.headroom(request.queue, pool.flavor)
        ahead = state.ahead_gpus(pool.flavor, prio)
        if hr.free < gpus:
            why = f"доступно по квоте {hr.free:.0f} при потолке {hr.limit:.0f}"
        elif ahead:
            why = f"впереди в очереди {ahead:.0f} карт"
        else:
            why = f"занят на {100 * state.pressure(pool.flavor):.0f}%"
        return f"; вместо {pool.name}×{gpus} — там {why}"

    def _out(self, pool: Pool, gpus: int, request: JobRequest, reason: str) -> Placement:
        return Placement(
            pool=pool.name,
            gpus=gpus,
            confidence=_confidence(pool, gpus, request.vram_gb, self.unc),
            policy=self.name,
            reason=reason,
        )


def next_tier_placement(
    cfg: PoolConfig, current_pool: str, current_gpus: int, vram_gb: float
) -> tuple[str, int]:
    """Куда переставить задачу, упавшую по OOM.

    Пользователь после OOM не занимается тонкой настройкой — он берёт
    следующий по мощности вариант. Сначала пробуем удвоить число карт
    в том же пуле, если нода это позволяет, иначе переходим тиром выше.
    """
    try:
        pool = cfg.by_name(current_pool)
    except KeyError:
        pool = cfg.sorted_by_cost()[0]

    if current_gpus * 2 <= pool.gpus_per_node:
        return pool.name, current_gpus * 2

    for nxt in cfg.sorted_by_cost():
        if nxt.relative_cost <= pool.relative_cost:
            continue
        tp = fit_on_pool(nxt, vram_gb) or 1
        return nxt.name, tp

    biggest = cfg.sorted_by_cost()[-1]
    return biggest.name, min(biggest.gpus_per_node, 8)


POLICIES = {
    "kueue-native": KueueNativePolicy,
    "manual": ManualPolicy,
    "cheapest": CheapestFitPolicy,
    "queue-aware": QueueAwarePolicy,
    "kueue-aware": KueueAwarePolicy,
}


def all_policies() -> dict[str, type]:
    out = dict(POLICIES)
    return out


def build_policy(
    name: str, cfg: PoolConfig, uncertainty: UncertaintyModel | None = None
) -> Policy:
    known = all_policies()
    if name not in known:
        raise SystemExit(
            f"неизвестная политика {name!r}, доступны: {', '.join(known)}"
        )
    return known[name](cfg, uncertainty)
