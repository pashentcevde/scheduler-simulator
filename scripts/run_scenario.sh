#!/usr/bin/env bash
# Полный прогон сценария:
#   сброс -> снапшотер (он же модель GPU) -> подача нагрузки ->
#   ожидание разгрузки -> метрики
#
# Кластер всегда один и тот же (kueue с очередями и квотами). Меняется только
# политика выбора ресурсов под задачу:
#   manual       карту выбирает пользователь — как сейчас в проде
#   cheapest     эвристический рекомендатель из задачи #1
#   queue-aware  он же, но с учётом занятости флейворов
#
# Использование:
#   ./scripts/run_scenario.sh <scenario> [manual|cheapest|queue-aware] [флаги submit.py]
#
# Примеры:
#   ./scripts/run_scenario.sh s01_steady_mixed manual
#   ./scripts/run_scenario.sh s09_oom_storm cheapest
#   ./scripts/run_scenario.sh s06_oversizing queue-aware --seed 7
#
# Переменные окружения:
#   KUBE_CONTEXT       контекст kubectl (по умолчанию текущий)
#   SNAPSHOT_INTERVAL  шаг снапшотов, сек (по умолчанию 1)
#   DRAIN_TIMEOUT      сколько ждать разгрузки после подачи, сек (по умолчанию 900)
#   KUBE_FAST          0 — не поднимать kubectl proxy, ходить обычным kubectl
#   KUBE_LIST_FROM_CACHE  1 — списки из watch-кэша apiserver'а. На замерах
#                      выигрыш оказался в пределах шума (узкое место —
#                      сериализация, не etcd), а данные могут отставать;
#                      держи выключенным.
#   SLOW_EVERY         опрашивать Job/ClusterQueue/Workload раз в N тактов
#                      (по умолчанию так, чтобы N*interval было около 1 с)
#   UNTIL_DRAINED      сколько реальных секунд простоя ждать перед остановкой
#                      (по умолчанию 30 сим-минут, пересчитанные в реальные)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SCENARIO_NAME="${1:?укажи имя сценария, например s01_steady_mixed}"
PLACEMENT="${2:-manual}"
shift || true
shift || true
EXTRA_ARGS=("$@")

SCENARIO_FILE="${ROOT}/config/scenarios/${SCENARIO_NAME}.yaml"
[[ -f "${SCENARIO_FILE}" ]] || die "нет файла сценария: ${SCENARIO_FILE}"

SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1}"
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-900}"

TIME_SCALE="$(python3 -c "
import yaml
print(yaml.safe_load(open('${SCENARIO_FILE}'))['time_scale'])
" 2>/dev/null || echo 120)"

# Поды снапшотер опрашивает каждый такт (на них живёт модель GPU), остальное —
# раз в SLOW_EVERY тактов. Держим произведение около секунды: чаще не нужно,
# в статусе Job'а и Workload'а всё равно секундное разрешение, а реже —
# начнёт теряться учёт вытеснений.
SLOW_EVERY="${SLOW_EVERY:-$(python3 -c "
print(max(1, round(1.0 / ${SNAPSHOT_INTERVAL})))
")}"

# Окно тишины перед остановкой задаём в симулированном времени. Фиксированные
# 30 реальных секунд при time_scale 480 — это 4 сим-часа пустого ожидания,
# то есть заметная доля короткого прогона.
UNTIL_DRAINED="${UNTIL_DRAINED:-$(python3 -c "
print(round(max(3.0, 1800.0 / ${TIME_SCALE}), 2))
")}"

# Один kubectl proxy на весь прогон: и снапшотер, и генератор нагрузки ходят
# в него по HTTP вместо форка kubectl на каждый запрос. Это то, что позволяет
# опустить SNAPSHOT_INTERVAL ниже секунды.
PROXY_PID=""
if [[ "${KUBE_FAST:-1}" != "0" && -z "${KUBE_PROXY_URL:-}" ]]; then
  PROXY_LOG="$(mktemp)"
  kubectl ${KUBE_CONTEXT:+--context "${KUBE_CONTEXT}"} proxy --port=0 \
    >"${PROXY_LOG}" 2>&1 &
  PROXY_PID=$!
  for _ in $(seq 1 50); do
    ADDR="$(sed -n 's/.*Starting to serve on[[:space:]]*\(\S*\).*/\1/p' "${PROXY_LOG}")"
    [[ -n "${ADDR}" ]] && break
    sleep 0.2
  done
  if [[ -n "${ADDR:-}" ]]; then
    export KUBE_PROXY_URL="http://${ADDR}"
    log "kubectl proxy на ${ADDR} (быстрый путь к API включён)"
  else
    warn "kubectl proxy не поднялся, работаем через обычный kubectl"
    kill "${PROXY_PID}" 2>/dev/null || true
    PROXY_PID=""
  fi
fi
stop_proxy() {
  [[ -n "${PROXY_PID}" ]] && kill "${PROXY_PID}" 2>/dev/null || true
  PROXY_PID=""
}
trap stop_proxy EXIT INT TERM

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/runs/${SCENARIO_NAME}-${PLACEMENT}-${STAMP}"
mkdir -p "${RUN_DIR}"

log "прогон: ${SCENARIO_NAME} / политика ${PLACEMENT} -> ${RUN_DIR}"

log "чищу кластер от предыдущего прогона"
"${ROOT}/scripts/reset.sh" >/dev/null

log "запускаю снапшотер и модель GPU (интервал ${SNAPSHOT_INTERVAL}s, \
медленный опрос раз в ${SLOW_EVERY} тактов, окно разгрузки ${UNTIL_DRAINED}s)"
python3 "${ROOT}/python/snapshot.py" \
  --run-dir "${RUN_DIR}" \
  --interval "${SNAPSHOT_INTERVAL}" \
  --pool "${ROOT}/config/pool.yaml" \
  --slow-every "${SLOW_EVERY}" \
  --until-drained "${UNTIL_DRAINED}" \
  --duration "$((DRAIN_TIMEOUT + 7200))" &
SNAP_PID=$!

cleanup() {
  if kill -0 "${SNAP_PID}" 2>/dev/null; then
    kill -TERM "${SNAP_PID}" 2>/dev/null || true
    wait "${SNAP_PID}" 2>/dev/null || true
  fi
  stop_proxy
}
trap cleanup EXIT INT TERM

sleep 2

log "подаю нагрузку"
export SNAPSHOT_INTERVAL   # submit.py проверяет по нему разрешение по времени
python3 "${ROOT}/python/submit.py" \
  --scenario "${SCENARIO_FILE}" \
  --placement "${PLACEMENT}" \
  --run-dir "${RUN_DIR}" \
  --pool "${ROOT}/config/pool.yaml" \
  --models "${ROOT}/config/models.yaml" \
  --tenants "${ROOT}/config/tenants.yaml" \
  "${EXTRA_ARGS[@]}"

log "подача закончена, жду разгрузки кластера (максимум ${DRAIN_TIMEOUT}s)"
deadline=$(( $(date +%s) + DRAIN_TIMEOUT ))
while kill -0 "${SNAP_PID}" 2>/dev/null; do
  if [[ $(date +%s) -ge ${deadline} ]]; then
    warn "таймаут ожидания разгрузки — часть задач осталась в очереди"
    warn "(смотри starved_jobs в отчёте: это тоже результат)"
    break
  fi
  sleep 5
done

cleanup
trap - EXIT INT TERM
stop_proxy

log "считаю метрики"
python3 "${ROOT}/python/analyze.py" "${RUN_DIR}" --pool "${ROOT}/config/pool.yaml"

echo
log "готово: ${RUN_DIR}"
log "сравнить с другим прогоном:"
echo "  python3 python/analyze.py --compare runs/<другой> ${RUN_DIR#${ROOT}/}"
