#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RESULT_DIR_RE = re.compile(r"RPL:\s+result dir\s+'(?P<dir>/tmp/rpl_data_[^']+)'")


@dataclass
class DispatchRecord:
    kernel: str
    duration_ns: float
    wgr: int
    arch_vgpr: int
    sgpr: int
    counters: Dict[str, float] = field(default_factory=dict)


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _most_common_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    try:
        return int(statistics.mode(values))
    except statistics.StatisticsError:
        counts: Dict[int, int] = {}
        for value in values:
            counts[int(value)] = counts.get(int(value), 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def parse_rocprof_stats_csv(path: Path, kernel_filter: Optional[str] = None) -> List[DispatchRecord]:
    records: List[DispatchRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        metadata_fields = {
            "Index",
            "KernelName",
            "gpu-id",
            "queue-id",
            "queue-index",
            "pid",
            "tid",
            "grd",
            "wgr",
            "lds",
            "scr",
            "arch_vgpr",
            "accum_vgpr",
            "sgpr",
            "wave_size",
            "sig",
            "obj",
            "DispatchNs",
            "BeginNs",
            "EndNs",
            "CompleteNs",
            "DurationNs",
        }
        counter_fields = [name for name in fieldnames if name not in metadata_fields]

        for row in reader:
            kernel_name = (row.get("KernelName") or "").strip().strip('"')
            if not kernel_name:
                continue
            if kernel_filter is not None and kernel_filter not in kernel_name:
                continue

            duration_ns = float("nan")
            duration_field = row.get("DurationNs", "")
            begin_field = row.get("BeginNs", "")
            end_field = row.get("EndNs", "")
            if duration_field:
                duration_ns = float(duration_field)
            elif begin_field and end_field:
                duration_ns = float(end_field) - float(begin_field)

            counters: Dict[str, float] = {}
            for counter in counter_fields:
                value = (row.get(counter) or "").strip()
                if not value:
                    continue
                try:
                    counters[counter] = float(value)
                except ValueError:
                    continue

            records.append(
                DispatchRecord(
                    kernel=kernel_name,
                    duration_ns=duration_ns,
                    wgr=int(row.get("wgr", "0") or 0),
                    arch_vgpr=int(row.get("arch_vgpr", "0") or 0),
                    sgpr=int(row.get("sgpr", "0") or 0),
                    counters=counters,
                )
            )

    return records


def summarize_dispatch_records(records: List[DispatchRecord]) -> Dict[str, object]:
    if not records:
        return {"calls": 0}

    durations = [r.duration_ns for r in records if not (r.duration_ns != r.duration_ns)]
    counter_names = sorted({name for r in records for name in r.counters.keys()})
    counter_means = {name: _mean([r.counters[name] for r in records if name in r.counters]) for name in counter_names}

    return {
        "calls": len(records),
        "kernel": records[0].kernel,
        "duration_avg_us": _mean(durations) / 1e3,
        "duration_min_us": min(durations) / 1e3 if durations else float("nan"),
        "duration_max_us": max(durations) / 1e3 if durations else float("nan"),
        "wgr": _most_common_int([r.wgr for r in records]),
        "arch_vgpr": _most_common_int([r.arch_vgpr for r in records]),
        "sgpr": _most_common_int([r.sgpr for r in records]),
        "counters": counter_means,
    }


def derive_cross_pass_metrics(pass1: Dict[str, object], pass2: Dict[str, object]) -> Dict[str, float]:
    p1 = pass1.get("counters", {}) if pass1 else {}
    p2 = pass2.get("counters", {}) if pass2 else {}
    hit = float(p2.get("TCC_HIT_sum", float("nan")))
    miss = float(p2.get("TCC_MISS_sum", float("nan")))
    dur_us = float(pass2.get("duration_avg_us", float("nan")))

    out: Dict[str, float] = {}
    if not (hit != hit or miss != miss) and (hit + miss) > 0:
        out["l2_hit_pct"] = 100.0 * hit / (hit + miss)
    if not (miss != miss or dur_us != dur_us) and dur_us > 0:
        bytes_from_hbm = miss * 64.0
        out["est_hbm_bw_gbps"] = bytes_from_hbm / (dur_us * 1e-6) / 1e9

    lds_insts = float(p1.get("SQ_INSTS_LDS", float("nan")))
    flat_lds_insts = float(p1.get("SQ_INSTS_FLAT_LDS_ONLY", float("nan")))
    lds_wait = float(p1.get("SQ_WAIT_INST_LDS", float("nan")))
    lds_bank_conflict = float(p1.get("SQ_LDS_BANK_CONFLICT", float("nan")))
    pass1_dur_us = float(pass1.get("duration_avg_us", float("nan")))
    sq_waves = float(p1.get("SQ_WAVES", float("nan")))
    valu_insts = float(p1.get("SQ_INSTS_VALU", float("nan")))
    vmem_rd_insts = float(p1.get("SQ_INSTS_VMEM_RD", float("nan")))
    vmem_wr_insts = float(p1.get("SQ_INSTS_VMEM_WR", float("nan")))
    valu_active = float(p1.get("SQ_ACTIVE_INST_VALU", float("nan")))
    valu_thread_cycles = float(p1.get("SQ_THREAD_CYCLES_VALU", float("nan")))
    gui_active = float(p1.get("GRBM_GUI_ACTIVE", float("nan")))

    if not (lds_insts != lds_insts) and lds_insts > 0:
        if not (lds_wait != lds_wait):
            out["lds_wait_per_inst"] = lds_wait / lds_insts
        if not (lds_bank_conflict != lds_bank_conflict):
            out["lds_bank_conflict_per_inst"] = lds_bank_conflict / lds_insts
        if not (flat_lds_insts != flat_lds_insts):
            out["flat_lds_inst_ratio"] = flat_lds_insts / lds_insts
    if not (lds_bank_conflict != lds_bank_conflict) and not (pass1_dur_us != pass1_dur_us) and pass1_dur_us > 0:
        out["lds_bank_conflict_per_us"] = lds_bank_conflict / pass1_dur_us
    if not (lds_wait != lds_wait) and not (pass1_dur_us != pass1_dur_us) and pass1_dur_us > 0:
        out["lds_wait_per_us"] = lds_wait / pass1_dur_us

    if not (sq_waves != sq_waves) and sq_waves > 0:
        if not (valu_insts != valu_insts):
            out["valu_insts_per_wave"] = valu_insts / sq_waves
        if not (vmem_rd_insts != vmem_rd_insts):
            out["vmem_rd_insts_per_wave"] = vmem_rd_insts / sq_waves
        if not (vmem_wr_insts != vmem_wr_insts):
            out["vmem_wr_insts_per_wave"] = vmem_wr_insts / sq_waves
        if not (lds_insts != lds_insts):
            out["lds_insts_per_wave"] = lds_insts / sq_waves

    if not (valu_active != valu_active) and not (valu_insts != valu_insts) and valu_insts > 0:
        out["valu_active_per_inst"] = valu_active / valu_insts
    if not (valu_thread_cycles != valu_thread_cycles) and not (valu_insts != valu_insts) and valu_insts > 0:
        out["valu_thread_cycles_per_inst"] = valu_thread_cycles / valu_insts
    if not (gui_active != gui_active):
        out["gui_active"] = gui_active

    likely_bottleneck = "unknown"
    lds_wait_per_inst = out.get("lds_wait_per_inst", float("nan"))
    lds_bank_conflict_per_inst = out.get("lds_bank_conflict_per_inst", float("nan"))
    l2_hit_pct = out.get("l2_hit_pct", float("nan"))
    hbm_bw = out.get("est_hbm_bw_gbps", float("nan"))
    valu_active_per_inst = out.get("valu_active_per_inst", float("nan"))

    if not (lds_bank_conflict_per_inst != lds_bank_conflict_per_inst) and lds_bank_conflict_per_inst > 0.02:
        likely_bottleneck = "lds-bank-conflict-bound"
    elif not (lds_wait_per_inst != lds_wait_per_inst) and lds_wait_per_inst > 0.25:
        likely_bottleneck = "lds-issue-wait-bound"
    elif not (l2_hit_pct != l2_hit_pct) and not (hbm_bw != hbm_bw) and l2_hit_pct < 45.0 and hbm_bw > 350.0:
        likely_bottleneck = "global-memory-bound"
    elif not (valu_active_per_inst != valu_active_per_inst) and valu_active_per_inst > 1.0:
        likely_bottleneck = "valu-pipeline-pressure"
    else:
        likely_bottleneck = "mixed-or-occupancy"

    out["likely_bottleneck"] = likely_bottleneck
    return out


def _fmt_float(v: object, digits: int = 2) -> str:
    if isinstance(v, (int, float)):
        vf = float(v)
        if vf != vf:
            return "nan"
        return f"{vf:.{digits}f}"
    return str(v)


def write_summary_markdown(
    output_md: Path,
    strategy: str,
    shape: Tuple[int, int, int],
    pass1: Dict[str, object],
    pass2: Dict[str, object],
    derived: Dict[str, float],
    pass1_file: Optional[Path] = None,
    pass2_file: Optional[Path] = None,
) -> None:
    m, n, k = shape
    lines: List[str] = []
    lines.append("# ROCm Strategy Lab Profiler Report")
    lines.append("")
    lines.append(f"- Strategy: `{strategy}`")
    lines.append(f"- Shape: `M={m}, N={n}, K={k}`")
    if pass1_file:
        lines.append(f"- Pass 1 file: `{pass1_file}`")
    if pass2_file:
        lines.append(f"- Pass 2 file: `{pass2_file}`")
    lines.append("")

    lines.append("## Core")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Calls (pass1) | {pass1.get('calls', 0)} |")
    lines.append(f"| Calls (pass2) | {pass2.get('calls', 0)} |")
    lines.append(f"| Grid block size | {pass1.get('wgr', 'n/a')} |")
    lines.append(f"| SGPR | {pass1.get('sgpr', 'n/a')} |")
    lines.append(f"| VGPR | {pass1.get('arch_vgpr', 'n/a')} |")
    lines.append(f"| Pass1 avg us | {_fmt_float(pass1.get('duration_avg_us'))} |")
    lines.append(f"| Pass2 avg us | {_fmt_float(pass2.get('duration_avg_us'))} |")
    lines.append(f"| L2 hit % | {_fmt_float(derived.get('l2_hit_pct', float('nan')), 1)} |")
    lines.append(f"| Est HBM BW (GB/s) | {_fmt_float(derived.get('est_hbm_bw_gbps', float('nan')), 1)} |")
    lines.append(f"| LDS wait / inst | {_fmt_float(derived.get('lds_wait_per_inst', float('nan')), 6)} |")
    lines.append(f"| LDS bank conflict / inst | {_fmt_float(derived.get('lds_bank_conflict_per_inst', float('nan')), 6)} |")
    lines.append(f"| LDS wait / us | {_fmt_float(derived.get('lds_wait_per_us', float('nan')), 3)} |")
    lines.append(f"| LDS bank conflict / us | {_fmt_float(derived.get('lds_bank_conflict_per_us', float('nan')), 3)} |")
    lines.append(f"| VALU inst / wave | {_fmt_float(derived.get('valu_insts_per_wave', float('nan')), 3)} |")
    lines.append(f"| VMEM rd inst / wave | {_fmt_float(derived.get('vmem_rd_insts_per_wave', float('nan')), 3)} |")
    lines.append(f"| LDS inst / wave | {_fmt_float(derived.get('lds_insts_per_wave', float('nan')), 3)} |")
    lines.append(f"| VALU active / inst | {_fmt_float(derived.get('valu_active_per_inst', float('nan')), 6)} |")
    lines.append(f"| VALU thread-cycles / inst | {_fmt_float(derived.get('valu_thread_cycles_per_inst', float('nan')), 6)} |")
    lines.append(f"| FLAT_LDS / LDS inst | {_fmt_float(derived.get('flat_lds_inst_ratio', float('nan')), 6)} |")
    lines.append(f"| GRBM GUI active | {_fmt_float(derived.get('gui_active', float('nan')), 3)} |")
    lines.append("")

    lines.append("## Health Summary")
    lines.append("")
    lines.append("| Signal | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Likely bottleneck | {derived.get('likely_bottleneck', 'unknown')} |")
    lines.append("")

    def write_counter_section(title: str, counters: Dict[str, float]) -> None:
        if not counters:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Counter | Mean |")
        lines.append("|---|---:|")
        for key in sorted(counters.keys()):
            lines.append(f"| {key} | {_fmt_float(counters[key], 3)} |")
        lines.append("")

    write_counter_section("Pass1 Counters", pass1.get("counters", {}))
    write_counter_section("Pass2 Counters", pass2.get("counters", {}))
    output_md.write_text("\n".join(lines) + "\n")


def run_rocprof_pass(
    bench_bin: Path,
    pass_input_file: Path,
    bench_args: Sequence[str],
    cwd: Path,
) -> Tuple[Path, Path, Path, Path]:
    pass_name = pass_input_file.stem
    run_log = cwd / f"{pass_name}.run.log"
    out_csv = cwd / f"{pass_name}.csv"
    cmd = [
        "/opt/rocm/bin/rocprof",
        "--stats",
        "--timestamp",
        "on",
        "-i",
        str(pass_input_file),
        "-o",
        str(out_csv),
        str(bench_bin),
        *bench_args,
    ]
    completed = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    run_log.write_text(
        "\n".join(
            [
                f"[rocprof] exit_code={completed.returncode}",
                "",
                "[rocprof stdout]",
                completed.stdout or "",
                "",
                "[rocprof stderr]",
                completed.stderr or "",
            ]
        )
    )

    if completed.returncode != 0:
        if not out_csv.exists() or out_csv.stat().st_size == 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                cmd,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        print(
            f"[warn] rocprof exited {completed.returncode} but produced CSV; continuing with collected data: {out_csv}",
            file=sys.stderr,
        )

    result_match = RESULT_DIR_RE.search(run_log.read_text(errors="replace"))
    run_dir = Path(result_match.group("dir")).resolve() if result_match else Path("/tmp")
    if not out_csv.exists():
        raise FileNotFoundError(f"rocprof CSV output not found: {out_csv}")
    out_db = out_csv.with_suffix(".db")
    return run_dir, out_csv, out_db, run_log


def _looks_like_counter_rejection(err: subprocess.CalledProcessError) -> bool:
    text = ((err.output or "") + "\n" + (err.stderr or "")).lower()
    hints = (
        "counter",
        "pmc",
        "metric",
        "input file",
        "invalid",
        "unknown",
        "cannot add",
    )
    return any(hint in text for hint in hints)


def _select_target_kernel(
    records: List[DispatchRecord],
    strategy: str,
    kernel_contains: Optional[str],
) -> str:
    if not records:
        raise RuntimeError("No dispatch records found in results file")

    if kernel_contains:
        filtered = [r for r in records if kernel_contains in r.kernel]
        if not filtered:
            raise RuntimeError(f"No kernels matched --kernel-contains={kernel_contains}")
        return filtered[0].kernel

    strategy_hints: List[str] = []
    if strategy:
        tail = strategy
        if strategy.startswith("wave64_"):
            marker = "wave64_cpt8_aligned_mr4_"
            if marker in strategy:
                tail = strategy.split(marker, 1)[1]
                strategy_hints.append(f"gemm_wave64_cpt8_gridstride_aligned_mr4_{tail}")
            strategy_hints.append("gemm_wave64")
        strategy_hints.append(tail)

    if strategy_hints:
        matched_by_hint = [
            r for r in records if any(hint and hint in r.kernel for hint in strategy_hints)
        ]
        if matched_by_hint:
            per_kernel_total: Dict[str, float] = {}
            for rec in matched_by_hint:
                per_kernel_total[rec.kernel] = per_kernel_total.get(rec.kernel, 0.0) + (
                    0.0 if rec.duration_ns != rec.duration_ns else rec.duration_ns
                )
            ranked = sorted(per_kernel_total.items(), key=lambda item: item[1], reverse=True)
            if ranked:
                return ranked[0][0]

    def is_runtime_kernel(name: str) -> bool:
        return (
            name.startswith("__amd_rocclr_")
            or "kernel_gemm_dl_multiple_d" in name
            or "fillBuffer" in name
            or "copyBuffer" in name
        )

    custom = [r for r in records if not is_runtime_kernel(r.kernel)]
    if not custom:
        custom = records

    per_kernel_total: Dict[str, float] = {}
    for rec in custom:
        per_kernel_total[rec.kernel] = per_kernel_total.get(rec.kernel, 0.0) + (
            0.0 if rec.duration_ns != rec.duration_ns else rec.duration_ns
        )

    ranked = sorted(per_kernel_total.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        raise RuntimeError("No kernel candidates available for target selection")
    return ranked[0][0]


def cmd_pmc_collect(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_bin = Path(args.bench_bin).resolve()
    if not bench_bin.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {bench_bin}")

    strategy = args.strategy
    bench_args = [
        f"--device={args.device}",
        f"--m={args.m}",
        f"--n={args.n}",
        f"--k={args.k}",
        f"--warmup={args.warmup}",
        f"--iters={args.iters}",
        "--no-check",
        "--no-prefer-ck-wide",
        "--no-disk-leaderboard",
        f"--strategies={strategy}",
    ]
    if strategy.startswith("lmhead_"):
        bench_args.extend([
            "--full-lm-head",
            "--no-lm-head-chunked",
        ])

    pass1_input = out_dir / "rocprof_pass1.txt"
    pass2_input = out_dir / "rocprof_pass2.txt"

    required_pass1_counters = [
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_VMEM_RD",
        "SQ_INSTS_VMEM_WR",
        "LDSInsts",
    ]
    optional_pass1_counters = [
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_ACTIVE_INST_VALU",
        "SQ_THREAD_CYCLES_VALU",
        "SQ_INSTS_FLAT_LDS_ONLY",
        "SQ_INSTS_LDS",
        "SQ_WAIT_INST_LDS",
        "SQ_LDS_BANK_CONFLICT",
    ]

    def write_pass1_input(counters: Sequence[str]) -> None:
        pass1_input.write_text(
            "\n".join(
                [
                    f"pmc: {' '.join(counters)}",
                    "range:",
                    "",
                ]
            )
        )

    enabled_optional = list(optional_pass1_counters)
    while True:
        active_pass1_counters = [*required_pass1_counters, *enabled_optional]
        write_pass1_input(active_pass1_counters)
        try:
            pass1_dir, pass1_csv, pass1_db, pass1_log = run_rocprof_pass(bench_bin, pass1_input, bench_args, out_dir)
            break
        except subprocess.CalledProcessError as err:
            if not _looks_like_counter_rejection(err):
                raise
            if not enabled_optional:
                raise
            dropped = enabled_optional.pop()
            print(
                f"[warn] pass1 counters rejected by rocprof (exit={err.returncode}); retrying without optional counter '{dropped}'",
                file=sys.stderr,
            )

    pass2_input.write_text(
        "\n".join(
            [
                "pmc: TCC_HIT_sum TCC_MISS_sum",
                "range:",
                "",
            ]
        )
    )

    pass2_dir, pass2_csv, pass2_db, pass2_log = run_rocprof_pass(bench_bin, pass2_input, bench_args, out_dir)

    pass1_all = parse_rocprof_stats_csv(pass1_csv, None)
    target_kernel = _select_target_kernel(pass1_all, strategy, args.kernel_contains)
    pass1_records = [r for r in pass1_all if r.kernel == target_kernel]

    pass2_all = parse_rocprof_stats_csv(pass2_csv, None)
    pass2_records = [r for r in pass2_all if r.kernel == target_kernel]
    pass1_summary = summarize_dispatch_records(pass1_records)
    pass2_summary = summarize_dispatch_records(pass2_records)
    derived = derive_cross_pass_metrics(pass1_summary, pass2_summary)

    json_out = {
        "strategy": strategy,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "device": args.device,
        "pass1_run_dir": str(pass1_dir),
        "pass2_run_dir": str(pass2_dir),
        "pass1_run_log": str(pass1_log),
        "pass2_run_log": str(pass2_log),
        "pass1_csv": str(pass1_csv),
        "pass2_csv": str(pass2_csv),
        "pass1_db": str(pass1_db),
        "pass2_db": str(pass2_db),
        "target_kernel": target_kernel,
        "pass1": pass1_summary,
        "pass2": pass2_summary,
        "derived": derived,
    }

    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(json_out, indent=2) + "\n")
    write_summary_markdown(
        md_path,
        strategy,
        (args.m, args.n, args.k),
        pass1_summary,
        pass2_summary,
        derived,
        pass1_file=pass1_csv,
        pass2_file=pass2_csv,
    )

    print(f"[ok] pass1 csv:      {pass1_csv}")
    print(f"[ok] pass2 csv:      {pass2_csv}")
    print(f"[ok] target kernel:  {target_kernel}")
    print(f"[ok] report json:    {json_path}")
    print(f"[ok] report md:      {md_path}")
    return 0


def _table_name(con: sqlite3.Connection, like: str) -> str:
    row = con.execute(
        "select name from sqlite_master where type='table' and name like ? order by name limit 1",
        (like,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Table not found with pattern: {like}")
    return str(row[0])


def cmd_trace_db(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    kd = _table_name(con, "rocpd_kernel_dispatch_%")
    ks = _table_name(con, "rocpd_info_kernel_symbol_%")

    query = f"""
        SELECT
            s.kernel_name,
            d.workgroup_size_x,
            d.grid_size_x,
            s.sgpr_count,
            s.arch_vgpr_count,
            (d.end - d.start) AS duration_ns
        FROM {kd} d
        JOIN {ks} s ON d.kernel_id = s.id
        WHERE s.kernel_name LIKE ?
    """
    pattern = f"%{args.kernel_contains}%"
    rows = con.execute(query, (pattern,)).fetchall()

    if not rows:
        print(f"No kernel dispatch rows matched kernel filter: {args.kernel_contains}")
        return 1

    durations = [float(r[5]) for r in rows]
    report = {
        "db": str(db_path),
        "kernel_filter": args.kernel_contains,
        "calls": len(rows),
        "duration_avg_us": _mean(durations) / 1e3,
        "duration_min_us": min(durations) / 1e3,
        "duration_max_us": max(durations) / 1e3,
        "grid_x": rows[0][2],
        "block_x": rows[0][1],
        "sgpr": rows[0][3],
        "vgpr": rows[0][4],
        "kernel": rows[0][0],
    }

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trace_report.json"
    md_path = out_dir / "trace_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(
        "\n".join(
            [
                "# ROCm Strategy Lab Trace Report",
                "",
                f"- DB: `{db_path}`",
                f"- Kernel filter: `{args.kernel_contains}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Calls | {report['calls']} |",
                f"| Grid x | {report['grid_x']} |",
                f"| Block x | {report['block_x']} |",
                f"| SGPR | {report['sgpr']} |",
                f"| VGPR | {report['vgpr']} |",
                f"| Avg us | {_fmt_float(report['duration_avg_us'])} |",
                f"| Min us | {_fmt_float(report['duration_min_us'])} |",
                f"| Max us | {_fmt_float(report['duration_max_us'])} |",
                "",
            ]
        )
    )

    print(f"[ok] trace report json: {json_path}")
    print(f"[ok] trace report md:   {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROCm strategy-lab profiling report helper (trace DB + 2-pass PMC)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pmc = sub.add_parser("pmc-collect", help="Run two rocprof counter passes and emit report.")
    p_pmc.add_argument("--strategy", required=True)
    p_pmc.add_argument("--m", type=int, required=True)
    p_pmc.add_argument("--n", type=int, required=True)
    p_pmc.add_argument("--k", type=int, required=True)
    p_pmc.add_argument("--device", type=int, default=0)
    p_pmc.add_argument("--warmup", type=int, default=1)
    p_pmc.add_argument("--iters", type=int, default=1)
    p_pmc.add_argument(
        "--bench-bin",
        default="/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab",
    )
    p_pmc.add_argument(
        "--kernel-contains",
        default=None,
        help="Optional kernel-name substring to force target kernel selection.",
    )
    p_pmc.add_argument("--output-dir", required=True)
    p_pmc.set_defaults(func=cmd_pmc_collect)

    p_trace = sub.add_parser("trace-db", help="Summarize rocprofv3 trace SQLite DB for one kernel filter.")
    p_trace.add_argument("--db", required=True)
    p_trace.add_argument("--kernel-contains", required=True)
    p_trace.add_argument("--output-dir", required=True)
    p_trace.set_defaults(func=cmd_trace_db)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
