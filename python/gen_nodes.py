#!/usr/bin/env python3
"""
Генерация манифеста фейковых GPU-нод для kwok из config/pool.yaml.

    python3 gen_nodes.py --pool config/pool.yaml --out manifests/10-nodes.yaml
    python3 gen_nodes.py --pool config/pool.yaml --print-status-patches

Ноды получают:
  * аннотацию kwok.x-k8s.io/node=fake — только такие ноды берёт под контроль kwok;
  * taint kwok.x-k8s.io/node=fake:NoSchedule — чтобы настоящие поды кластера
    (kueue, kwok, coredns) на них не уехали;
  * метку nvidia.com/gpu.product — по ней матчатся ResourceFlavor'ы kueue
    и nodeSelector пользовательских задач;
  * extended resource nvidia.com/gpu в capacity/allocatable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import (  # noqa: E402
    GPU_RES,
    L_POOL,
    PRODUCT_LABEL,
    PoolConfig,
    dump_yaml,
)


def node_resources(pool, defaults) -> dict:
    gpus = pool.gpus_per_node
    cpu = defaults["cpu_per_gpu"] * gpus + defaults["cpu_reserve"]
    mem = defaults["memory_gi_per_gpu"] * gpus + defaults["memory_gi_reserve"]
    return {
        "cpu": str(cpu),
        "memory": f"{mem}Gi",
        "pods": str(defaults["pods_per_node"]),
        GPU_RES: str(gpus),
        "ephemeral-storage": "1Ti",
    }


def build_nodes(cfg: PoolConfig) -> list[dict]:
    nodes: list[dict] = []
    for pool in cfg.pools:
        for i in range(pool.nodes):
            name = f"gpu-{pool.name}-{i}"
            res = node_resources(pool, cfg.defaults)
            nodes.append(
                {
                    "apiVersion": "v1",
                    "kind": "Node",
                    "metadata": {
                        "name": name,
                        "annotations": {
                            "node.alpha.kubernetes.io/ttl": "0",
                            "kwok.x-k8s.io/node": "fake",
                        },
                        "labels": {
                            "beta.kubernetes.io/arch": "amd64",
                            "beta.kubernetes.io/os": "linux",
                            "kubernetes.io/arch": "amd64",
                            "kubernetes.io/os": "linux",
                            "kubernetes.io/hostname": name,
                            "kubernetes.io/role": "agent",
                            "node-role.kubernetes.io/agent": "",
                            "type": "kwok",
                            PRODUCT_LABEL: pool.product,
                            "nvidia.com/gpu.count": str(pool.gpus_per_node),
                            "nvidia.com/gpu.memory": str(int(pool.vram_gb * 1024)),
                            L_POOL: pool.name,
                        },
                    },
                    "spec": {
                        "taints": [
                            {
                                "key": "kwok.x-k8s.io/node",
                                "value": "fake",
                                "effect": "NoSchedule",
                            }
                        ]
                    },
                    "status": {
                        "allocatable": res,
                        "capacity": res,
                        "nodeInfo": {
                            "architecture": "amd64",
                            "bootID": "",
                            "containerRuntimeVersion": "",
                            "kernelVersion": "",
                            "kubeProxyVersion": "fake",
                            "kubeletVersion": "fake",
                            "machineID": "",
                            "operatingSystem": "linux",
                            "osImage": "",
                            "systemUUID": "",
                        },
                        "phase": "Running",
                    },
                }
            )
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="config/pool.yaml")
    ap.add_argument("--out")
    ap.add_argument(
        "--print-status-patches",
        action="store_true",
        help="вывести 'имя<TAB>json-патч status' для kubectl patch --subresource=status",
    )
    args = ap.parse_args()

    cfg = PoolConfig.load(args.pool)
    nodes = build_nodes(cfg)

    if args.print_status_patches:
        for n in nodes:
            patch = {
                "status": {
                    "allocatable": n["status"]["allocatable"],
                    "capacity": n["status"]["capacity"],
                }
            }
            print(f"{n['metadata']['name']}\t{json.dumps(patch)}")
        return

    doc = "\n---\n".join(dump_yaml(n) for n in nodes)
    header = (
        "# СГЕНЕРИРОВАНО python/gen_nodes.py из config/pool.yaml — не править руками\n"
        f"# нод: {len(nodes)}, GPU всего: {cfg.total_gpus}\n"
    )
    if args.out:
        Path(args.out).write_text(header + doc, encoding="utf-8")
        summary = ", ".join(f"{p.name}={p.total_gpus}" for p in cfg.pools)
        print(f"{args.out}: {len(nodes)} нод, {cfg.total_gpus} GPU ({summary})")
    else:
        sys.stdout.write(header + doc)


if __name__ == "__main__":
    main()
