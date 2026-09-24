#!/usr/bin/env bash
# Очистка кластера между прогонами: удаляем Job'ы, поды и Workload'ы
# в неймспейсах команд. Ноды, очереди и квоты остаются.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

for ns in "${LAB_NAMESPACES[@]}"; do
  kc -n "${ns}" delete jobs --all --wait=false >/dev/null 2>&1 || true
done
for ns in "${LAB_NAMESPACES[@]}"; do
  kc -n "${ns}" delete pods --all --grace-period=0 --force --wait=false >/dev/null 2>&1 || true
  kc -n "${ns}" delete workloads.kueue.x-k8s.io --all --wait=false >/dev/null 2>&1 || true
done

log "жду, пока всё исчезнет"
for _ in $(seq 1 60); do
  left=0
  for ns in "${LAB_NAMESPACES[@]}"; do
    n=$(kc -n "${ns}" get pods --no-headers 2>/dev/null | wc -l)
    left=$((left + n))
  done
  [[ "${left}" -eq 0 ]] && break
  sleep 2
done

log "проверяю, что квоты обнулились"
kc get clusterqueues
