#!/usr/bin/env bash
# Прогон одного сценария на нескольких сидах для двух политик.
#
# Внутри каждого сида план строится один раз (политикой A) и переиспользуется
# политикой B через --plan, поэтому пара A/B строго сравнима. Между сидами
# план разный — это и есть источник статистики.
#
#   ./scripts/run_seeds.sh s05_random_fuzz cheapest queue-aware 1 2 3 4 5
#
# Переменные окружения — те же, что у run_scenario.sh (SNAPSHOT_INTERVAL,
# DRAIN_TIMEOUT, KUBE_CONTEXT).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SCENARIO_NAME="${1:?укажи имя сценария}"
POLICY_A="${2:?укажи первую политику}"
POLICY_B="${3:?укажи вторую политику}"
shift 3
SEEDS=("$@")
[[ ${#SEEDS[@]} -gt 0 ]] || SEEDS=(1 2 3 4 5)

PAIRS=()

for seed in "${SEEDS[@]}"; do
  log "################ seed=${seed} ################"

  log "=== A: ${POLICY_A}, seed ${seed} ==="
  SEED_TAG="seed${seed}" "${ROOT}/scripts/run_scenario.sh" \
    "${SCENARIO_NAME}" "${POLICY_A}" --seed "${seed}"
  RUN_A="$(ls -1dt "${ROOT}"/runs/${SCENARIO_NAME}-seed${seed}-${POLICY_A}-* | head -1)"

  log "=== B: ${POLICY_B}, seed ${seed} (план из A) ==="
  SEED_TAG="seed${seed}" "${ROOT}/scripts/run_scenario.sh" \
    "${SCENARIO_NAME}" "${POLICY_B}" --seed "${seed}" --plan "${RUN_A}/plan.json"
  RUN_B="$(ls -1dt "${ROOT}"/runs/${SCENARIO_NAME}-seed${seed}-${POLICY_B}-* | head -1)"

  python3 "${ROOT}/python/analyze.py" --compare "${RUN_A}" "${RUN_B}" \
    --pool "${ROOT}/config/pool.yaml"
  PAIRS+=("${RUN_A}:${RUN_B}")
done

echo
log "=== сводка по сидам ==="
python3 "${ROOT}/python/aggregate_seeds.py" "${PAIRS[@]}"
