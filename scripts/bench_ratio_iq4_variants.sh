#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${WORKSPACE_ROOT}/build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel"
TEST_FILTER="ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI"
RUNS="${1:-3}"
OUT_DIR="${2:-/tmp/ratio_iq4_variant_sweep_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] Benchmark binary not found: ${BIN}"
  echo "        Build with: cmake --build build_v2_release --target v2_perf_rocm_ratio_vnni_kernel --parallel"
  exit 1
fi

mkdir -p "${OUT_DIR}"

NAMES=(
  "baseline"
  "decode_mode_1_fused_pair"
  "decode_mode_2_fused_group"
  "decode_mode_3_perm_pair"
  "decode_mode_4_wave_pair_bpermute"
  "prefetch_only"
  "kb_mode_1_min_atomic"
  "kb_mode_2_high_occupancy"
  "cpt2_only"
  "cpt2_lut64"
  "combo_mode1_prefetch_kb2"
)

ENVS=(
  ""
  "LLAMINAR_RATIO_IQ4_DECODE_MODE=1"
  "LLAMINAR_RATIO_IQ4_DECODE_MODE=2"
  "LLAMINAR_RATIO_IQ4_DECODE_MODE=3"
  "LLAMINAR_RATIO_IQ4_DECODE_MODE=4"
  "LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=1"
  "LLAMINAR_RATIO_IQ4_KB_MODE=1"
  "LLAMINAR_RATIO_IQ4_KB_MODE=2"
  "LLAMINAR_RATIO_IQ4_CPT=2"
  "LLAMINAR_RATIO_IQ4_CPT=2 LLAMINAR_RATIO_IQ4_CPT2_LUT64=1"
  "LLAMINAR_RATIO_IQ4_DECODE_MODE=1 LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=1 LLAMINAR_RATIO_IQ4_KB_MODE=2"
)

CSV="${OUT_DIR}/results.csv"
echo "name,run,avg_speedup,iq4_qwo,iq4_ffn_down,iq4_ffn_gate,iq4_avg" > "${CSV}"

echo "[INFO] Output directory: ${OUT_DIR}"
echo "[INFO] Runs per variant: ${RUNS}"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  envs="${ENVS[$i]}"
  echo ""
  echo "===== ${name} ====="

  for run in $(seq 1 "${RUNS}"); do
    log_file="${OUT_DIR}/${name}_run${run}.log"

    if [[ -n "${envs}" ]]; then
      eval "${envs} \"${BIN}\" --gtest_filter=\"${TEST_FILTER}\" --gtest_color=no" > "${log_file}" 2>&1 || true
    else
      "${BIN}" --gtest_filter="${TEST_FILTER}" --gtest_color=no > "${log_file}" 2>&1 || true
    fi

    avg="$(grep -E 'Average ratio/int8 speedup:' "${log_file}" | tail -1 | awk '{print $4}' | tr -d 'x' || true)"
    iq4_qwo="$(grep -F 'IQ4_NL │ Q/Wo' "${log_file}" | tail -1 | awk '{print $(NF-1)}' || true)"
    iq4_ffn_down="$(grep -F 'IQ4_NL │ FFN Down' "${log_file}" | tail -1 | awk '{print $(NF-1)}' || true)"
    iq4_ffn_gate="$(grep -F 'IQ4_NL │ FFN Gate' "${log_file}" | tail -1 | awk '{print $(NF-1)}' || true)"

    iq4_avg="nan"
    if [[ -n "${iq4_qwo}" && -n "${iq4_ffn_down}" && -n "${iq4_ffn_gate}" ]]; then
      iq4_avg="$(python3 - << 'PY' "${iq4_qwo}" "${iq4_ffn_down}" "${iq4_ffn_gate}"
import sys
vals = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
print(sum(vals) / len(vals))
PY
)"
    fi

    if [[ -z "${avg}" ]]; then
      avg="nan"
      echo "run ${run}: avg=nan (see ${log_file})"
    else
      echo "run ${run}: avg=${avg} iq4_avg=${iq4_avg} [qwo=${iq4_qwo:-na}, down=${iq4_ffn_down:-na}, gate=${iq4_ffn_gate:-na}]"
    fi

    echo "${name},${run},${avg},${iq4_qwo:-nan},${iq4_ffn_down:-nan},${iq4_ffn_gate:-nan},${iq4_avg}" >> "${CSV}"
  done
done

SUMMARY="${OUT_DIR}/summary.txt"
python3 - << 'PY' "${CSV}" "${SUMMARY}"
import csv
import math
import statistics
import sys
from collections import defaultdict

csv_path = sys.argv[1]
summary_path = sys.argv[2]

values_all = defaultdict(list)
values_iq4_qwo = defaultdict(list)
values_iq4_down = defaultdict(list)
values_iq4_gate = defaultdict(list)
values_iq4_avg = defaultdict(list)
with open(csv_path, newline='') as f:
    r = csv.DictReader(f)
    def add_value(bucket, key, row, name):
        try:
            v = float(row[key])
            if not math.isnan(v):
                bucket[name].append(v)
        except ValueError:
            pass

    for row in r:
        name = row['name']
        add_value(values_all, 'avg_speedup', row, name)
        add_value(values_iq4_qwo, 'iq4_qwo', row, name)
        add_value(values_iq4_down, 'iq4_ffn_down', row, name)
        add_value(values_iq4_gate, 'iq4_ffn_gate', row, name)
        add_value(values_iq4_avg, 'iq4_avg', row, name)

def summarize(values):
  rows = []
  for name, vals in values.items():
    if not vals:
      continue
    vals_sorted = sorted(vals)
    rows.append((
      name,
      len(vals_sorted),
      statistics.median(vals_sorted),
      statistics.mean(vals_sorted),
      min(vals_sorted),
      max(vals_sorted),
    ))
  rows.sort(key=lambda x: x[2], reverse=True)
  return rows

rows_all = summarize(values_all)
rows_iq4_avg = summarize(values_iq4_avg)
rows_iq4_qwo = summarize(values_iq4_qwo)
rows_iq4_down = summarize(values_iq4_down)
rows_iq4_gate = summarize(values_iq4_gate)

lines = []
def add_table(title, rows):
    lines.append(title)
    lines.append('')
    lines.append(f"{'Variant':40} {'n':>3} {'median':>10} {'mean':>10} {'min':>10} {'max':>10}")
    lines.append('-' * 90)
    for name, n, med, mean, mn, mx in rows:
        lines.append(f"{name:40} {n:>3d} {med:>10.6f} {mean:>10.6f} {mn:>10.6f} {mx:>10.6f}")
    lines.append('')

add_table('Variant ranking by median avg_speedup (global)', rows_all)
add_table('Variant ranking by median IQ4-only average speedup', rows_iq4_avg)
add_table('Variant ranking by median IQ4 Q/Wo speedup', rows_iq4_qwo)
add_table('Variant ranking by median IQ4 FFN Down speedup', rows_iq4_down)
add_table('Variant ranking by median IQ4 FFN Gate speedup', rows_iq4_gate)

text = '\n'.join(lines)
print(text)
with open(summary_path, 'w') as f:
    f.write(text + '\n')
PY

echo ""
echo "[DONE]"
echo "  CSV: ${CSV}"
echo "  Summary: ${SUMMARY}"
