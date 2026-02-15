#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BIN="${BIN:-$ROOT_DIR/build_v2_release/llaminar2}"
MODEL="${MODEL:-$ROOT_DIR/models/qwen2.5-0.5b-instruct-q8_0.gguf}"
PROMPT="${PROMPT:-The capital of}"
TOKENS="${TOKENS:-128}"
NP="${NP:-2}"
BIND_TO="${BIND_TO:-socket}"
MAP_BY="${MAP_BY:-socket}"
MPI_LEAVE_PINNED="${MPI_LEAVE_PINNED:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_CMA_CASE="${RUN_CMA_CASE:-0}"

if [[ "${OMP_NUM_THREADS:-}" == "" ]]; then
  cores_per_socket="$(lscpu | awk -F: '/Core\(s\) per socket/{gsub(/ /, "", $2); print $2; exit}')"
  if [[ -n "$cores_per_socket" ]]; then
    export OMP_NUM_THREADS="$cores_per_socket"
  else
    export OMP_NUM_THREADS=28
  fi
fi

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not executable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [--quick]

Runs a global CPU TP tuning matrix and prints decode/collective timing summary.

Environment overrides:
  BIN, MODEL, PROMPT, TOKENS, NP, OUT_DIR
  OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND
  OPENBLAS_NUM_THREADS, MKL_NUM_THREADS
  RUN_CMA_CASE=1       Include CMA transport test case
  TIMEOUT_SEC=<sec>    Wrap each run in timeout if > 0
  EXTRA_ARGS="..."     Extra args passed to llaminar2

Examples:
  $(basename "$0")
  TOKENS=256 NP=2 $(basename "$0")
  RUN_CMA_CASE=1 $(basename "$0")
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--quick" ]]; then
  TOKENS=64
fi

run_case() {
  local label="$1"
  local mca_args="$2"
  local app_args="$3"
  local log_file="$OUT_DIR/${label}.log"

  local mpi_cmd=(mpirun -np "$NP" --bind-to "$BIND_TO" --map-by "$MAP_BY" --mca mpi_leave_pinned "$MPI_LEAVE_PINNED")

  if [[ -n "$mca_args" ]]; then
    # shellcheck disable=SC2206
    local mca_split=( $mca_args )
    mpi_cmd+=("${mca_split[@]}")
  fi

  local app_cmd=("$BIN" --no-mpi-bootstrap --benchmark -m "$MODEL" -d cpu -p "$PROMPT" -n "$TOKENS")
  if [[ -n "$app_args" ]]; then
    # shellcheck disable=SC2206
    local app_split=( $app_args )
    app_cmd+=("${app_split[@]}")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_split=( $EXTRA_ARGS )
    app_cmd+=("${extra_split[@]}")
  fi

  echo "[RUN] $label"
  if [[ "$TIMEOUT_SEC" -gt 0 ]]; then
    if timeout "$TIMEOUT_SEC" env LLAMINAR_PROFILING=1 "${mpi_cmd[@]}" "${app_cmd[@]}" >"$log_file" 2>&1; then
      echo "[OK]  $label"
    else
      echo "[FAIL] $label (see $log_file)"
      return 1
    fi
  else
    if env LLAMINAR_PROFILING=1 "${mpi_cmd[@]}" "${app_cmd[@]}" >"$log_file" 2>&1; then
      echo "[OK]  $label"
    else
      echo "[FAIL] $label (see $log_file)"
      return 1
    fi
  fi

  return 0
}

extract_decode_toks() {
  local log_file="$1"
  awk '
    /║ DECODE/{in_decode=1; next}
    in_decode && /Throughput/{
      if (match($0, /[0-9]+\.[0-9]+/)) {
        print substr($0, RSTART, RLENGTH)
        exit
      }
    }
  ' "$log_file"
}

extract_kernel_total_ms() {
  local log_file="$1"
  local kernel_name="$2"
  awk -F'│' -v k="$kernel_name" '
    index($1, k) > 0 {
      val=$3
      gsub(/ /, "", val)
      print val
      exit
    }
  ' "$log_file"
}

print_summary() {
  local labels=("$@")

  echo
  echo "Collective Tuning Summary"
  printf '%-22s %-12s %-14s %-14s %-s\n' "case" "decode tok/s" "allreduce ms" "allgather ms" "log"
  printf '%-22s %-12s %-14s %-14s %-s\n' "----------------------" "------------" "--------------" "--------------" "----"

  for label in "${labels[@]}"; do
    local log_file="$OUT_DIR/${label}.log"
    local decode="$(extract_decode_toks "$log_file" || true)"
    local allreduce="$(extract_kernel_total_ms "$log_file" "ALLREDUCE" || true)"
    local allgather="$(extract_kernel_total_ms "$log_file" "ALLGATHER" || true)"

    decode="${decode:-n/a}"
    allreduce="${allreduce:-n/a}"
    allgather="${allgather:-n/a}"

    printf '%-22s %-12s %-14s %-14s %-s\n' "$label" "$decode" "$allreduce" "$allgather" "$log_file"
  done

  echo
  echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
  echo "Logs written to: $OUT_DIR"
}

labels=(
  "base_none"
  "vader_default"
  "backend_mpi"
  "backend_upi"
)

run_case "base_none" "--mca btl_vader_single_copy_mechanism none" ""
run_case "vader_default" "" ""
run_case "backend_mpi" "--mca btl_vader_single_copy_mechanism none" "--backend mpi"
run_case "backend_upi" "--mca btl_vader_single_copy_mechanism none" "--backend upi"

if [[ "$RUN_CMA_CASE" == "1" ]]; then
  labels+=("vader_cma")
  run_case "vader_cma" "--mca btl_vader_single_copy_mechanism cma" ""
fi

print_summary "${labels[@]}"
