#!/usr/bin/env bash
# Сравнение двух политик размещения на одном и том же потоке задач.
#
#   ./scripts/run_compare.sh s01_steady_mixed                    # manual vs cheapest
#   ./scripts/run_compare.sh s06_oversizing manual queue-aware
#   ./scripts/run_compare.sh s09_oom_storm  manual cheapest
#
# Второй прогон переиспользует plan.json первого, поэтому поток задач
# идентичен вплоть до того, что попросил бы пользователь и сколько памяти
# задача съест на самом деле. Отличается только то, кто выбирает карту.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SCENARIO_NAME="${1:?укажи имя сценария}"
POLICY_A="${2:-manual}"
POLICY_B="${3:-cheapest}"
shift || true; shift || true; shift || true

log "=== A: политика ${POLICY_A} (контрольная группа) ==="
"${ROOT}/scripts/run_scenario.sh" "${SCENARIO_NAME}" "${POLICY_A}" "$@"
RUN_A="$(ls -1dt "${ROOT}"/runs/${SCENARIO_NAME}-${POLICY_A}-* | head -1)"

log "=== B: политика ${POLICY_B} ==="
"${ROOT}/scripts/run_scenario.sh" "${SCENARIO_NAME}" "${POLICY_B}" --plan "${RUN_A}/plan.json" "$@"
RUN_B="$(ls -1dt "${ROOT}"/runs/${SCENARIO_NAME}-${POLICY_B}-* | head -1)"

echo
log "=== сравнение ==="
python3 "${ROOT}/python/analyze.py" --compare "${RUN_A}" "${RUN_B}" --pool "${ROOT}/config/pool.yaml"
