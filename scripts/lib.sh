#!/usr/bin/env bash
# Общие переменные и хелперы для скриптов прогона.
# Установка кластера, kwok и kueue здесь НЕ делается — предполагается,
# что всё это уже поднято, а манифесты стенда применены вручную
# (см. README, раздел "Установка").

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Контекст kubectl. Пусто = текущий контекст (kubectl config current-context).
KUBE_CONTEXT="${KUBE_CONTEXT:-}"

LAB_NAMESPACES=(team-a team-b team-c)

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

kc() {
  if [[ -n "${KUBE_CONTEXT}" ]]; then
    kubectl --context "${KUBE_CONTEXT}" "$@"
  else
    kubectl "$@"
  fi
}
