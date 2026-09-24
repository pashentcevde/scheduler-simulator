#!/usr/bin/env python3
"""
Проверка согласованности конфигурации до применения в кластер.

    python3 python/validate_quotas.py

Проверяет:
  1. сумма номинальных квот по каждому флейвору == физической ёмкости пула
     (когорта не переподписана и не недоиспользована);
  2. lendingLimit <= nominalQuota (иначе kueue отвергнет объект);
  3. cpu/memory заданы пропорционально GPU (8 CPU и 32Gi на карту),
     иначе они начнут ограничивать раньше GPU и исказят эксперимент;
  4. каждый ResourceFlavor ссылается на существующий пул нод.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import ROOT, PoolConfig, parse_quantity  # noqa: E402

GPU = "nvidia.com/gpu"


def load_all(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def main() -> int:
    cfg = PoolConfig.load(ROOT / "config" / "pool.yaml")
    flavors = load_all(ROOT / "manifests" / "02-resource-flavors.yaml")
    queues = [
        d
        for d in load_all(ROOT / "manifests" / "03-cohort-and-queues.yaml")
        if d.get("kind") == "ClusterQueue"
    ]
    errors: list[str] = []
    warns: list[str] = []

    pool_by_flavor = {p.flavor: p for p in cfg.pools}
    products = {p.product for p in cfg.pools}

    # 4. флейворы ссылаются на реальные пулы
    for f in flavors:
        if f.get("kind") != "ResourceFlavor":
            continue
        name = f["metadata"]["name"]
        product = (f["spec"].get("nodeLabels") or {}).get("nvidia.com/gpu.product")
        if name not in pool_by_flavor:
            errors.append(f"ResourceFlavor {name}: нет пула с таким flavor в pool.yaml")
        if product not in products:
            errors.append(f"ResourceFlavor {name}: продукт {product!r} не встречается на нодах")

    nominal: dict[str, float] = defaultdict(float)
    per_queue: dict[str, dict[str, float]] = defaultdict(dict)

    for cq in queues:
        qname = cq["metadata"]["name"]
        for rg in cq["spec"]["resourceGroups"]:
            for fl in rg["flavors"]:
                fname = fl["name"]
                res = {r["name"]: r for r in fl["resources"]}
                gpu = res.get(GPU)
                if gpu is None:
                    errors.append(f"{qname}/{fname}: не задан {GPU}")
                    continue
                gq = parse_quantity(gpu.get("nominalQuota", 0))
                nominal[fname] += gq
                per_queue[qname][fname] = gq

                # 2. lendingLimit <= nominalQuota
                for rname, r in res.items():
                    ll = r.get("lendingLimit")
                    if ll is not None and parse_quantity(ll) > parse_quantity(
                        r.get("nominalQuota", 0)
                    ):
                        errors.append(
                            f"{qname}/{fname}/{rname}: lendingLimit > nominalQuota"
                        )

                # 3. пропорциональность cpu/memory
                cpu = parse_quantity(res.get("cpu", {}).get("nominalQuota", 0))
                mem = parse_quantity(res.get("memory", {}).get("nominalQuota", 0))
                want_cpu = gq * cfg.defaults["cpu_per_gpu"]
                want_mem = gq * cfg.defaults["memory_gi_per_gpu"] * 1024**3
                if abs(cpu - want_cpu) > 1e-6:
                    warns.append(
                        f"{qname}/{fname}: cpu={cpu:g}, ожидалось {want_cpu:g} "
                        f"({cfg.defaults['cpu_per_gpu']} на карту)"
                    )
                if abs(mem - want_mem) > 1:
                    warns.append(
                        f"{qname}/{fname}: memory={mem / 1024**3:g}Gi, ожидалось "
                        f"{want_mem / 1024**3:g}Gi"
                    )

    # 1. сумма квот == ёмкость
    print(f"{'flavor':<14} {'ёмкость':>8} {'квоты':>8}  по очередям")
    for pool in cfg.pools:
        cap = pool.total_gpus
        got = nominal.get(pool.flavor, 0)
        detail = " ".join(
            f"{q}={per_queue[q].get(pool.flavor, 0):g}" for q in sorted(per_queue)
        )
        mark = "ok" if abs(cap - got) < 1e-9 else "РАСХОЖДЕНИЕ"
        print(f"{pool.flavor:<14} {cap:>8} {got:>8g}  {detail}  [{mark}]")
        if abs(cap - got) > 1e-9:
            errors.append(
                f"флейвор {pool.flavor}: сумма номинальных квот {got:g} != "
                f"ёмкости пула {cap}"
            )

    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[ERROR] {e}")

    if errors:
        print(f"\nнайдено ошибок: {len(errors)}")
        return 1
    print("\nконфигурация согласована")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
