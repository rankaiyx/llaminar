#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/test_mlp_tp_parity"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] Parity test binary not found or not executable: ${BIN}" >&2
  echo "Build first: cmake --build build --target test_mlp_tp_parity" >&2
  exit 2
fi

PASS=0
FAIL=0
SKIP=0
FAILED_CASES=()

log() { echo -e "[$(date +%H:%M:%S)] $*"; }

have_ranks() {
  local np="$1"
  if mpirun -np "${np}" /bin/true &>/dev/null; then
    return 0
  fi
  return 1
}

run_case() {
  local np="$1"; shift
  local filter="$1"; shift
  local tag="$1"; shift
  local extra_env="$*"
  log "RUN np=${np} filter=${filter} tag=${tag} ${extra_env}" 
  if ! have_ranks "${np}"; then
    log "SKIP np=${np} (insufficient slots) tag=${tag}"
    ((SKIP++))
    return 0
  fi
  set +e
  # Unset size to let fixture infer world size dynamically
  unset LLAMINAR_TP_MLP_SIZE || true
  # shellcheck disable=SC2086
  ${extra_env} mpirun -np "${np}" "${BIN}" --gtest_filter="${filter}"
  rc=$?
  set +e
  if [[ $rc -eq 0 ]]; then
    log "PASS tag=${tag}"
    ((PASS++))
  else
    log "FAIL rc=${rc} tag=${tag}"
    ((FAIL++))
    FAILED_CASES+=("${tag}")
  fi
}

log "=== MLP TP Parity Matrix Driver ==="
log "Binary: ${BIN}"

# 2-rank baseline set
run_case 2 MLP_TP_Parity.SmallOddDff70    small_odd70
run_case 2 MLP_TP_Parity.EvenDff64        even64
run_case 2 MLP_TP_Parity.SmallEdgeDff33   edge33
run_case 2 MLP_TP_Parity.RaggedDff65      ragged65
run_case 2 MLP_TP_Parity.LargerEvenDff128 larger128
run_case 2 MLP_TP_Parity.StressLongPrefill stress_prefill
run_case 2 MLP_TP_Parity.PrefillCosmaMode prefill_cosma
run_case 2 MLP_TP_Parity.NegativeMismatch negative_mismatch

# 3-rank variants (only if available)
run_case 3 MLP_TP_Parity.TP3_RaggedDff97  tp3_ragged97
run_case 3 MLP_TP_Parity.TP3_EvenDff90    tp3_even90

# 4-rank variants (only if available)
run_case 4 MLP_TP_Parity.TP4_EvenDff128   tp4_even128
run_case 4 MLP_TP_Parity.TP4_RaggedDff130 tp4_ragged130

log "=== SUMMARY ==="
log "PASS=${PASS} FAIL=${FAIL} SKIP=${SKIP}"
if (( FAIL > 0 )); then
  printf 'Failed cases: %s\n' "${FAILED_CASES[*]}" >&2
  exit 1
fi
exit 0
