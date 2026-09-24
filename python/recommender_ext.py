"""
Вариации эвристического планировщика поверх recommender.py.

Каждая политика — отдельная гипотеза о том, чего не хватает cheapest
и queue-aware. Все они укладываются в тот же интерфейс recommend(request,
state) и меряются офлайн через python/simulate.py.

  queue-aware-prio   бюджет на срочность зависит от приоритета — то, что
                     описано в отчёте, но в коде отсутствует
  quota-aware        "свободно" считается не физически, а в пределах квоты
                     и лимитов заимствования очереди
  risk-aware         не берём вариант, у которого вероятность вместимости
                     ниже порога: платим тиром выше, но не платим OOM'ом
  duration-aware     стоимость считается за всю задачу, а не за GPU-час:
                     дорогая карта для пятиминутной задачи почти бесплатна
  wait-cost          единая целевая функция cost + цена_ожидания * ожидание
                     вместо "фильтр + штраф"
  packing-aware      при равной цене предпочитаем пул, где меньше
                     фрагментация: не занимаем последнюю пустую 4-карточную
                     ноду однокарточной задачей
  reserving          антистадный слой: политика помнит свои собственные
                     решения за последние N минут и вычитает их из
                     свободных карт
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lab import ALLOWED_TP, USABLE_VRAM_FRACTION, Pool, PoolConfig, fit_on_pool
from recommender import (
    CheapestFitPolicy,
    ClusterState,
    JobRequest,
    Placement,
    QueueAwarePolicy,
    UncertaintyModel,
    _candidates,
    _confidence,
)

# бюджет срочности: во сколько раз дороже минимума готовы платить за
# немедленный старт (значения — из отчёта, раздел 4.3)
PRIORITY_BUDGET = {"production": 8.0, "high": 6.0, "normal": 4.0, "low": 1.5}

# сколько условных единиц стоит одна сим-минута ожидания задачи
# (для wait-cost); прод ждать не должен, batch — обязан
WAIT_PRICE = {"production": 0.30, "high": 0.15, "normal": 0.05, "low": 0.005}


def _all_candidates(cfg: PoolConfig, vram_gb: float) -> list[tuple[Pool, int, float]]:
    """В отличие от _candidates — все допустимые числа карт, а не только
    минимальное. Нужно, как только появится зависимость времени от карты:
    "2 карты вдвое быстрее" станет отдельным вариантом."""
    out = []
    for pool in cfg.pools:
        base = fit_on_pool(pool, vram_gb)
        if base is None:
            continue
        for tp in ALLOWED_TP:
            if tp < base or tp > pool.gpus_per_node:
                continue
            out.append((pool, tp, tp * pool.relative_cost))
    return out


def _free(state: ClusterState, flavor: str) -> float:
    return float(state.free_gpu_by_flavor.get(flavor, 0.0))


def _headroom(state: ClusterState, flavor: str) -> float:
    """Сколько карт очередь реально может занять сейчас: минимум из
    физически свободных и квотного потолка. Если симулятор/сборщик состояния
    квоту не отдал — деградируем до физической доступности."""
    hr = getattr(state, "quota_headroom", None)
    phys = _free(state, flavor)
    if not hr:
        return phys
    return min(phys, float(hr.get(flavor, 0.0)))


class _Base:
    def __init__(self, cfg: PoolConfig, uncertainty: UncertaintyModel | None = None):
        self.cfg = cfg
        self.unc = uncertainty or UncertaintyModel()
        self._fallback = CheapestFitPolicy(cfg, uncertainty)

    def _out(self, pool, gpus, vram, reason) -> Placement:
        return Placement(
            pool=pool.name,
            gpus=gpus,
            confidence=_confidence(pool, gpus, vram, self.unc),
            policy=self.name,
            reason=reason,
        )


class QueueAwarePrioPolicy(_Base):
    """queue-aware, но бюджет срочности зависит от приоритета задачи."""

    name = "queue-aware-prio"

    def __init__(self, cfg, uncertainty=None, congestion_weight: float = 1.5):
        super().__init__(cfg, uncertainty)
        self.congestion_weight = congestion_weight

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        cheapest = min(cands, key=lambda c: (c[2], c[1]))
        budget = cheapest[2] * PRIORITY_BUDGET.get(request.priority_class, 4.0)
        avail = [c for c in cands if _free(state, c[0].flavor) >= c[1] and c[2] <= budget]
        if not avail:
            p = self._fallback.recommend(request, state)
            p.policy = self.name
            p.reason += "; в бюджет и в свободные карты не уложились, ждём"
            return p
        pool, gpus, cost = min(
            avail,
            key=lambda c: (c[2] * (1 + self.congestion_weight * state.pressure(c[0].flavor)), c[2], c[1]),
        )
        return self._out(
            pool, gpus, request.vram_gb,
            f"{pool.name}×{gpus}, бюджет {budget:.1f} ({request.priority_class})",
        )


class QuotaAwarePolicy(_Base):
    """Свободно = физически свободно И укладывается в квотный потолок очереди.

    Именно этого не хватает queue-aware: карта может простаивать, но быть
    недоступной команде, потому что чужая номинальная квота не отдаётся.
    """

    name = "quota-aware"

    def __init__(self, cfg, uncertainty=None, congestion_weight: float = 1.5):
        super().__init__(cfg, uncertainty)
        self.congestion_weight = congestion_weight

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        cheapest = min(cands, key=lambda c: (c[2], c[1]))
        budget = cheapest[2] * PRIORITY_BUDGET.get(request.priority_class, 4.0)
        avail = [c for c in cands if _headroom(state, c[0].flavor) >= c[1] and c[2] <= budget]
        if not avail:
            p = self._fallback.recommend(request, state)
            p.policy = self.name
            p.reason += "; ни один флейвор не доступен в пределах квоты"
            return p
        pool, gpus, cost = min(
            avail,
            key=lambda c: (c[2] * (1 + self.congestion_weight * state.pressure(c[0].flavor)), c[2], c[1]),
        )
        return self._out(pool, gpus, request.vram_gb, f"{pool.name}×{gpus} доступен в пределах квоты")


class RiskAwarePolicy(_Base):
    """Отсекаем варианты с высоким риском OOM.

    Прямая реализация правила из постановки ("не допускать задачу, если
    confidence ниже порога"), только применённая не в AdmissionCheck,
    а в самом выборе: вместо отказа берём следующий вариант.
    Порог зависит от приоритета: прод падать не должен, batch может.
    """

    name = "risk-aware"
    THRESHOLD = {"production": 0.98, "high": 0.95, "normal": 0.85, "low": 0.7}

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        thr = self.THRESHOLD.get(request.priority_class, 0.85)
        safe = [
            c for c in cands
            if self.unc.fit_probability(
                c[1] * c[0].vram_gb * USABLE_VRAM_FRACTION, request.vram_gb
            ) >= thr
        ]
        pool, gpus, cost = min(safe or cands, key=lambda c: (c[2], c[1]))
        note = "" if safe else "; безопасного варианта нет, берём максимум доступного"
        return self._out(pool, gpus, request.vram_gb, f"{pool.name}×{gpus}, порог риска {thr}{note}")


class DurationAwarePolicy(_Base):
    """Стоимость — за всю задачу, а не за GPU-час.

    Пятиминутный smoke на H100 стоит дешевле, чем 6-часовой эндпоинт на L4.
    Поэтому бюджет срочности имеет смысл считать в абсолютных единицах:
    короткой задаче можно дать дорогую карту, длинной — нельзя.
    Длительность здесь берётся из оценки (в реальной системе — из истории
    запусков пользователя, это же признак для будущей ML-модели).
    """

    name = "duration-aware"

    def __init__(self, cfg, uncertainty=None, overspend_abs: float = 6.0):
        super().__init__(cfg, uncertainty)
        self.overspend_abs = overspend_abs  # усл. единиц, готовых переплатить

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        hours = max(0.1, getattr(request, "expected_hours", 1.0))
        cheapest = min(cands, key=lambda c: (c[2], c[1]))
        budget_abs = cheapest[2] * hours + self.overspend_abs * PRIORITY_BUDGET.get(
            request.priority_class, 4.0
        ) / 8.0
        avail = [c for c in cands if _headroom(state, c[0].flavor) >= c[1] and c[2] * hours <= budget_abs]
        if not avail:
            p = self._fallback.recommend(request, state)
            p.policy = self.name
            p.reason += "; переплата за срочность не окупается, ждём дешёвую"
            return p
        pool, gpus, _ = min(avail, key=lambda c: (c[2], c[1]))
        return self._out(pool, gpus, request.vram_gb, f"{pool.name}×{gpus}, бюджет {budget_abs:.1f} на задачу")


class WaitCostPolicy(_Base):
    """Одна целевая функция вместо "фильтр + штраф".

    score = стоимость_задачи + цена_минуты_ожидания * ожидаемое_ожидание.
    Ожидание оценивается грубо: свободно сейчас — ноль, иначе тем больше,
    чем выше занятость флейвора. Ценность в том, что нет разрыва между
    "влезаю сейчас" и "не влезаю": между ними непрерывный переход.
    """

    name = "wait-cost"

    def __init__(self, cfg, uncertainty=None, wait_scale: float = 45.0):
        super().__init__(cfg, uncertainty)
        self.wait_scale = wait_scale  # сим-минут ожидания при полностью занятом пуле

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        price = WAIT_PRICE.get(request.priority_class, 0.05)
        hours = max(0.1, getattr(request, "expected_hours", 1.0))

        def score(c):
            pool, gpus, cost = c
            if _headroom(state, pool.flavor) >= gpus:
                wait = 0.0
            else:
                wait = self.wait_scale * (0.5 + state.pressure(pool.flavor))
            return cost * hours + price * wait

        pool, gpus, cost = min(cands, key=lambda c: (score(c), c[2], c[1]))
        return self._out(
            pool, gpus, request.vram_gb,
            f"{pool.name}×{gpus}, score {score((pool, gpus, cost)):.2f}",
        )


class PackingAwarePolicy(_Base):
    """При прочих равных не занимаем дефицитную по нодам конфигурацию.

    Однокарточная задача, севшая на пустую H100-ноду, отнимает возможность
    запустить 70B: четырёхкарточная задача помещается только на пустую ноду.
    Поэтому к цене добавляется штраф за "крупноузловые" пулы для мелких задач.
    """

    name = "packing-aware"

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)

        def score(c):
            pool, gpus, cost = c
            waste = pool.gpus_per_node / max(1, gpus)  # во сколько раз нода крупнее задачи
            return cost * (1.0 + 0.25 * (waste - 1.0)) * (
                1.0 + 1.5 * state.pressure(pool.flavor)
            )

        free_ok = [c for c in cands if _headroom(state, c[0].flavor) >= c[1]]
        pool, gpus, cost = min(free_ok or cands, key=lambda c: (score(c), c[2], c[1]))
        return self._out(pool, gpus, request.vram_gb, f"{pool.name}×{gpus} с учётом фрагментации")


class ReservingPolicy(_Base):
    """Антистадный слой поверх queue-aware.

    Состояние кластера обновляется не мгновенно: пачка задач, поданных
    за одну секунду, видит одни и те же свободные карты и дружно уезжает
    в один флейвор (сценарий s10). Политика ведёт собственный счётчик
    уже выданных, но ещё не запущенных рекомендаций и вычитает их
    из свободных карт.
    """

    name = "reserving"

    def __init__(self, cfg, uncertainty=None, congestion_weight: float = 1.5):
        super().__init__(cfg, uncertainty)
        self.congestion_weight = congestion_weight
        self._reserved: dict[str, float] = {}
        self._last_state_at: float = -1.0

    def recommend(self, request: JobRequest, state: ClusterState) -> Placement:
        # новый снимок состояния — резервы обнуляются, они уже в нём учтены
        if state.fetched_at != self._last_state_at:
            self._last_state_at = state.fetched_at
            self._reserved.clear()

        cands = _candidates(self.cfg, request.vram_gb)
        if not cands:
            return self._fallback.recommend(request, state)
        cheapest = min(cands, key=lambda c: (c[2], c[1]))
        budget = cheapest[2] * PRIORITY_BUDGET.get(request.priority_class, 4.0)

        def free_now(flavor: str) -> float:
            return _headroom(state, flavor) - self._reserved.get(flavor, 0.0)

        avail = [c for c in cands if free_now(c[0].flavor) >= c[1] and c[2] <= budget]
        if not avail:
            p = self._fallback.recommend(request, state)
            p.policy = self.name
            p.reason += "; с учётом уже выданных рекомендаций свободного нет"
            return p
        pool, gpus, cost = min(
            avail,
            key=lambda c: (c[2] * (1 + self.congestion_weight * state.pressure(c[0].flavor)), c[2], c[1]),
        )
        self._reserved[pool.flavor] = self._reserved.get(pool.flavor, 0.0) + gpus
        return self._out(pool, gpus, request.vram_gb, f"{pool.name}×{gpus}, зарезервировано с учётом пачки")


POLICIES = {
    p.name: p
    for p in (
        QueueAwarePrioPolicy,
        QuotaAwarePolicy,
        RiskAwarePolicy,
        DurationAwarePolicy,
        WaitCostPolicy,
        PackingAwarePolicy,
        ReservingPolicy,
    )
}
