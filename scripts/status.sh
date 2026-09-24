#!/usr/bin/env bash
# Быстрый взгляд на текущее состояние кластера.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

echo "── ClusterQueues ──────────────────────────────────────────────"
kc get clusterqueues -o custom-columns=\
'QUEUE:.metadata.name,COHORT:.spec.cohortName,PENDING:.status.pendingWorkloads,ADMITTED:.status.admittedWorkloads,RESERVING:.status.reservingWorkloads'

echo
echo "── Занятость GPU по флейворам ─────────────────────────────────"
kc get clusterqueues -o json | python3 "${ROOT}/python/quota_table.py"

echo
echo "── Поды по нодам ──────────────────────────────────────────────"
kc get pods -A -l lab.skalar.ai/lab=true \
  -o custom-columns='NS:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase,REASON:.status.reason,GPU:.spec.containers[0].resources.requests.nvidia\.com/gpu' \
  2>/dev/null | head -40

echo
echo "── Убитые по OOM ──────────────────────────────────────────────"
kc get pods -A -l lab.skalar.ai/lab=true \
  --field-selector status.phase=Failed \
  -o custom-columns='NS:.metadata.namespace,POD:.metadata.name,REASON:.status.reason,MSG:.status.message' \
  2>/dev/null | head -15

echo
echo "── Неудовлетворённые workload'ы (первые 10) ───────────────────"
kc get workloads.kueue.x-k8s.io -A \
  -o custom-columns='NS:.metadata.namespace,NAME:.metadata.name,QUEUE:.spec.queueName,PRIO:.spec.priority,ADMITTED:.status.conditions[?(@.type=="Admitted")].status' \
  2>/dev/null | awk 'NR==1 || $5!="True"' | head -11
