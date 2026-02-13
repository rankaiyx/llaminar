#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${WORKSPACE_ROOT}/build_v2_release"
BENCH_BIN="${BUILD_DIR}/tests/v2/v2_perf_rocm_ratio_vnni_kernel"
GTEST_FILTER="ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI"
GPU_INDEX="${GPU_INDEX:-0}"
OUT_ROOT="${1:-/tmp/rocprof_ratio_vnni_run_$(date +%Y%m%d_%H%M%S)}"

if ! command -v rocprof >/dev/null 2>&1; then
  echo "[ERROR] rocprof not found in PATH"
  exit 1
fi

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "[ERROR] Benchmark binary not found: ${BENCH_BIN}"
  echo "        Build with: cmake --build build_v2_release --target v2_perf_rocm_ratio_vnni_kernel --parallel"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "[INFO] Output root: ${OUT_ROOT}"

action_run_rocprof() {
  local out_dir="$1"
  local input_file="$2"

  mkdir -p "${out_dir}"

  if [[ -n "${input_file}" ]]; then
    HSA_VISIBLE_DEVICES="${GPU_INDEX}" rocprof -i "${input_file}" --stats -o "${out_dir}/rocprof.csv" \
      "${BENCH_BIN}" --gtest_filter="${GTEST_FILTER}" --gtest_color=no > "${out_dir}/test_stdout.log" 2>&1 || true
  else
    HSA_VISIBLE_DEVICES="${GPU_INDEX}" rocprof --stats -o "${out_dir}/rocprof.csv" \
      "${BENCH_BIN}" --gtest_filter="${GTEST_FILTER}" --gtest_color=no > "${out_dir}/test_stdout.log" 2>&1 || true
  fi
}

action_write_metrics_file() {
  local dst="$1"
  local metrics="$2"
  cat > "${dst}" <<EOF
gpu: ${GPU_INDEX}
kernel: gemv_ratio_vnni_grid_kpar_kernel_t gemv_int8_int8_grid_kpar_vnni_kernel_t
pmc: ${metrics}
EOF
}

echo "[1/4] Running full timing pass"
action_run_rocprof "${OUT_ROOT}/full" ""

echo "[2/4] Running split counter passes"
run_counter_pass() {
  local name="$1"
  local metrics="$2"
  local out_dir="${OUT_ROOT}/${name}"
  local metrics_file="${out_dir}/metrics.txt"
  mkdir -p "${out_dir}"
  action_write_metrics_file "${metrics_file}" "${metrics}"
  action_run_rocprof "${out_dir}" "${metrics_file}"
  echo "       - ${name} done"
}

run_counter_pass "g_instr_main" "VFetchInsts FlatVMemInsts VALUBusy SALUBusy VALUInsts SALUInsts Wavefronts"
run_counter_pass "g_sfetch" "SFetchInsts"
run_counter_pass "g_mem_main" "FetchSize MemUnitBusy MemUnitStalled ALUStalledByLDS"
run_counter_pass "g_writes" "WriteSize"
run_counter_pass "g_l2hit" "L2CacheHit"
run_counter_pass "g_misc" "GPUBusy WriteUnitStalled VALUUtilization"

echo "[3/4] Merging counters into per-kernel summary"
python3 - <<'PY' "${OUT_ROOT}"
import csv, glob, os, statistics, sys

out_root = sys.argv[1]
csvs = glob.glob(os.path.join(out_root, 'g_*', 'rocprof.csv'))
metrics = [
    'GPUBusy','MemUnitBusy','MemUnitStalled','WriteUnitStalled','L2CacheHit',
    'VALUUtilization','VALUBusy','SALUBusy','Wavefronts','VALUInsts','SALUInsts',
    'VFetchInsts','SFetchInsts','FlatVMemInsts','FetchSize','WriteSize','ALUStalledByLDS'
]

per_kernel = {}
for p in csvs:
    with open(p, newline='') as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        present = [m for m in metrics if m in cols]
        for row in r:
            k = row.get('KernelName', '')
            if 'gemv_ratio_vnni_grid_kpar_kernel_t' not in k and 'gemv_int8_int8_grid_kpar_vnni_kernel_t' not in k:
                continue
            d = per_kernel.setdefault(k, {
                'count': 0,
                'dur': [],
                'arch_vgpr': row.get('arch_vgpr', ''),
                'sgpr': row.get('sgpr', ''),
                'wgr': row.get('wgr', ''),
            })
            d['count'] += 1
            if row.get('DurationNs'):
                try:
                    d['dur'].append(float(row['DurationNs']))
                except ValueError:
                    pass
            for m in present:
                v = row.get(m, '')
                if v != '':
                    try:
                        d.setdefault(m, []).append(float(v))
                    except ValueError:
                        pass

summary_txt = os.path.join(out_root, 'kernel_bottleneck_summary.txt')
summary_md = os.path.join(out_root, 'kernel_bottleneck_summary.md')

with open(summary_txt, 'w') as out:
    out.write(f'OUTPUT_ROOT {out_root}\n')
    for k, v in sorted(per_kernel.items(), key=lambda kv: -sum(kv[1].get('dur', []))):
        total_ms = sum(v.get('dur', [])) / 1e6
        avg_us = statistics.mean(v.get('dur', [])) / 1e3 if v.get('dur') else float('nan')
        out.write(f'\nKERNEL: {k}\n')
        out.write(f"  calls={v['count']} total_ms={total_ms:.3f} avg_us={avg_us:.3f} arch_vgpr={v['arch_vgpr']} sgpr={v['sgpr']} wgr={v['wgr']}\n")
        for m in metrics:
            vals = v.get(m)
            if vals:
                out.write(f"  {m}={statistics.mean(vals):.3f}\n")

with open(summary_md, 'w') as out:
    out.write('# Ratio-VNNI rocprof bottleneck summary\n\n')
    out.write(f'- Output root: `{out_root}`\n\n')
    out.write('| Kernel | Calls | Total ms | Avg us | arch_vgpr | sgpr | wgr |\n')
    out.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for k, v in sorted(per_kernel.items(), key=lambda kv: -sum(kv[1].get('dur', []))):
        total_ms = sum(v.get('dur', [])) / 1e6
        avg_us = statistics.mean(v.get('dur', [])) / 1e3 if v.get('dur') else float('nan')
        out.write(f"| {k} | {v['count']} | {total_ms:.3f} | {avg_us:.3f} | {v['arch_vgpr']} | {v['sgpr']} | {v['wgr']} |\n")

print(summary_txt)
print(summary_md)
PY

echo "[4/4] Final artifact index"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort > "${OUT_ROOT}/artifacts.txt"
cat "${OUT_ROOT}/artifacts.txt"

echo ""
echo "[DONE]"
echo "  Output root: ${OUT_ROOT}"
echo "  Summary txt: ${OUT_ROOT}/kernel_bottleneck_summary.txt"
echo "  Summary md : ${OUT_ROOT}/kernel_bottleneck_summary.md"
