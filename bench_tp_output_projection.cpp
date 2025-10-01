// Benchmark: Column-partitioned attention output projection manual vs splitter path
// Author: David Sanftenberg
// Build target: bench_tp_output_projection
// Usage (examples):
//   ./build/bench_tp_output_projection --seq 128,256 --dmodel 2048 --heads 16 --parts 2 --iters 50
//   LLAMINAR_ATTN_TP_FORCE_SPLITTER=1 ./build/bench_tp_output_projection --seq 512 --dmodel 4096 --heads 32 --parts 4
// If LLAMINAR_ATTN_TP_FORCE_SPLITTER is set, the attention kernel code path will use the splitter.
// Here we isolate just the WO projection math to compare packing & loop overheads.

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "tensors/tensor_factory.h"
#include "utils/debug_env.h"
#include "tensors/tp_partition.h" // TPPartitionSpec & splitters
#include <cblas.h>
extern "C" void openblas_set_num_threads(int);

using Clock = std::chrono::high_resolution_clock;

struct Args
{
    std::vector<int> seqs{256};
    int d_model = 4096;
    int heads = 32;
    int iters = 30;
    int warmup = 5;
    std::vector<int> parts_list{2}; // now supports multiple TP partition counts
    bool validate = true;
    bool outer_parallel = true;  // control OpenMP over partitions
    int threads = 1;             // OpenBLAS threads (1 = default serial for stable microbench)
    bool compare_outer = false;  // run both outer_parallel on/off for comparison
    std::string csv_path;        // optional CSV output path
    std::string json_path;       // optional JSON output path (overwrites per invocation)
    int repeat = 1;              // repeat measure to form median
    bool summary_table = false;  // print aggregated summary at end
    bool emit_heuristic = false; // emit heuristic snippet (now generalized across parts)
};

static std::vector<int> parse_list(const std::string &s)
{
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size())
    {
        size_t c = s.find(',', pos);
        auto tok = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        if (!tok.empty())
            out.push_back(std::atoi(tok.c_str()));
        if (c == std::string::npos)
            break;
        pos = c + 1;
    }
    return out;
}

static void parse_cli(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](int idx)
        { return (idx + 1 < argc) ? argv[idx + 1] : (char *)nullptr; };
        if (arg == "--seq" && next(i))
        {
            a.seqs = parse_list(next(i));
            ++i;
        }
        else if (arg == "--dmodel" && next(i))
        {
            a.d_model = std::atoi(next(i));
            ++i;
        }
        else if (arg == "--heads" && next(i))
        {
            a.heads = std::atoi(next(i));
            ++i;
        }
        else if (arg == "--iters" && next(i))
        {
            a.iters = std::atoi(next(i));
            ++i;
        }
        else if (arg == "--warmup" && next(i))
        {
            a.warmup = std::atoi(next(i));
            ++i;
        }
        else if (arg == "--parts" && next(i))
        {
            a.parts_list = parse_list(next(i));
            ++i;
        }
        else if (arg == "--no-validate")
        {
            a.validate = false;
        }
        else if (arg == "--outer-parallel" && next(i))
        {
            int v = std::atoi(next(i));
            a.outer_parallel = (v != 0);
            ++i;
        }
        else if (arg == "--no-outer-parallel")
        {
            a.outer_parallel = false;
        }
        else if (arg == "--threads" && next(i))
        {
            a.threads = std::max(1, std::atoi(next(i)));
            ++i;
        }
        else if (arg == "--compare-outer")
        {
            a.compare_outer = true;
        }
        else if (arg == "--csv" && next(i))
        {
            a.csv_path = next(i);
            ++i;
        }
        else if (arg == "--json" && next(i))
        {
            a.json_path = next(i);
            ++i;
        }
        else if (arg == "--repeat" && next(i))
        {
            a.repeat = std::max(1, std::atoi(next(i)));
            ++i;
        }
        else if (arg == "--summary-table")
        {
            a.summary_table = true;
        }
        else if (arg == "--emit-heuristic")
        {
            a.emit_heuristic = true;
        }
    }
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num) / std::sqrt(den) : 0;
}

int main(int argc, char **argv)
{
    Args args;
    parse_cli(argc, argv, args);
    const auto &env = llaminar::debugEnv();
    if (args.d_model % args.heads != 0)
    {
        std::cerr << "[bench] d_model must be divisible by heads" << std::endl;
        return 1;
    }
    if (args.parts_list.empty())
    {
        std::cerr << "[bench] parts list empty" << std::endl;
        return 1;
    }
    for (int p : args.parts_list)
    {
        if (p <= 0)
        {
            std::cerr << "[bench] invalid parts value " << p << " (must be >0)" << std::endl;
            return 1;
        }
    }
    openblas_set_num_threads(args.threads); // user-controlled threads
    std::cout << "[bench] parts_list=";
    for (size_t i = 0; i < args.parts_list.size(); ++i)
    {
        std::cout << (i ? "," : "") << args.parts_list[i];
    };
    std::cout << " d_model=" << args.d_model << " heads=" << args.heads
              << " outer_parallel=" << (args.outer_parallel ? "1" : "0")
              << " splitter_env=" << (env.attention.tp_force_splitter ? "1" : "0") << "\n";

    int local_head_dim = args.d_model / args.heads; // assume divisible
    struct ResultRow
    {
        int parts;
        int seq;
        bool outer;
        double ms_manual;
        double ms_split;
        double speedup;
        double rel_l2;
        std::string recommend;
    };
    std::vector<ResultRow> all_results;
    all_results.reserve(args.parts_list.size() * args.seqs.size() * (args.compare_outer ? 2 : 1));
    for (int parts_value : args.parts_list)
    {
        int parts = parts_value;
        int local_head_dim = args.d_model / args.heads; // recompute in case heads changes (still constant)
        for (int seq_len : args.seqs)
        {
            size_t A_sz = (size_t)seq_len * local_head_dim;
            size_t B_sz = (size_t)local_head_dim * args.d_model;
            size_t C_sz = (size_t)seq_len * args.d_model;
            std::vector<float> A(A_sz), B(B_sz), C_manual(C_sz), C_split(C_sz);
            // init with deterministic values
            for (size_t i = 0; i < A_sz; ++i)
                A[i] = (float)((i * 1315423911u) % 1000) / 1000.f;
            for (size_t i = 0; i < B_sz; ++i)
                B[i] = (float)((i * 2654435761u) % 1000) / 1000.f;

            struct RunStats
            {
                double total_ms = 0;
                std::vector<long long> part_us;
            };
            auto summarize_parts = [&](const std::vector<long long> &us)
            {
                double min_ms = 1e99, max_ms = 0, sum_ms = 0;
                for (long long v : us)
                {
                    double ms = v / 1000.0;
                    if (ms < min_ms)
                        min_ms = ms;
                    if (ms > max_ms)
                        max_ms = ms;
                    sum_ms += ms;
                }
                double avg_ms = us.empty() ? 0 : sum_ms / us.size();
                return std::tuple<double, double, double>(min_ms, max_ms, avg_ms);
            };

            // Manual path: per-part GEMM directly over original B (no packing)
            auto run_manual = [&](std::vector<float> &C, bool outer) -> RunStats
            {
                RunStats rs;
                rs.part_us.assign(parts, 0);
                auto start = Clock::now();
                for (int it = -args.warmup; it < args.iters; ++it)
                {
                    if (it == 0)
                        start = Clock::now();
                    auto do_part = [&](int part)
                    {
                        auto spec = llaminar::compute_tp_partition(args.d_model, parts, part, llaminar::TPPartitionSpec::Axis::Col);
                        int n_sub = (int)spec.local_dim;
                        int col_off = (int)spec.local_offset;
                        const float *B_sub = B.data() + col_off; // still column offset inside full-width row-major B
                        float *C_sub = C.data() + col_off;
                        auto p0 = Clock::now();
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    seq_len, n_sub, local_head_dim,
                                    1.0f, A.data(), local_head_dim,
                                    B_sub, args.d_model,
                                    0.0f, C_sub, args.d_model);
                        if (it >= 0)
                            rs.part_us[part] += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - p0).count();
                    };
#ifdef _OPENMP
                    if (outer && parts > 1)
                    {
#pragma omp parallel for schedule(static)
                        for (int p = 0; p < parts; ++p)
                            do_part(p);
                    }
                    else
#endif
                    {
                        for (int p = 0; p < parts; ++p)
                            do_part(p);
                    }
                }
                auto end = Clock::now();
                rs.total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                return rs;
            };

            // Reused packed panels path: pre-pack B slice for each partition once, then reuse across iterations
            auto run_splitter = [&](std::vector<float> &C, bool outer) -> RunStats
            {
                RunStats rs;
                rs.part_us.assign(parts, 0);
                // Pre-pack panels for each part: packedB[part] shape [K, n_sub] row-major contiguous
                struct Panel
                {
                    std::vector<float> data;
                    int n_sub = 0;
                    int col_off = 0;
                };
                std::vector<Panel> panels(parts);
                for (int part = 0; part < parts; ++part)
                {
                    auto spec = llaminar::compute_tp_partition(args.d_model, parts, part, llaminar::TPPartitionSpec::Axis::Col);
                    Panel &pan = panels[part];
                    pan.n_sub = (int)spec.local_dim;
                    pan.col_off = (int)spec.local_offset;
                    pan.data.resize((size_t)local_head_dim * pan.n_sub);
                    for (int k = 0; k < local_head_dim; ++k)
                    {
                        const float *row_src = B.data() + k * args.d_model + pan.col_off;
                        float *row_dst = pan.data.data() + k * pan.n_sub;
                        std::memcpy(row_dst, row_src, sizeof(float) * pan.n_sub);
                    }
                }
                auto start = Clock::now();
                for (int it = -args.warmup; it < args.iters; ++it)
                {
                    if (it == 0)
                        start = Clock::now();
                    auto do_part = [&](int part)
                    {
                        const Panel &pan = panels[part];
                        float *C_sub = C.data() + pan.col_off;
                        auto p0 = Clock::now();
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    seq_len, pan.n_sub, local_head_dim,
                                    1.0f, A.data(), local_head_dim,
                                    pan.data.data(), pan.n_sub,
                                    0.0f, C_sub, args.d_model);
                        if (it >= 0)
                            rs.part_us[part] += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - p0).count();
                    };
#ifdef _OPENMP
                    if (outer && parts > 1)
                    {
#pragma omp parallel for schedule(static)
                        for (int p = 0; p < parts; ++p)
                            do_part(p);
                    }
                    else
#endif
                    {
                        for (int p = 0; p < parts; ++p)
                            do_part(p);
                    }
                }
                auto end = Clock::now();
                rs.total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                return rs;
            };
            auto run_mode_once = [&](bool outer_flag)
            {
                // Fresh output buffers for fairness
                std::fill(C_manual.begin(), C_manual.end(), 0.f);
                std::fill(C_split.begin(), C_split.end(), 0.f);
                auto manual_rs = run_manual(C_manual, outer_flag);
                auto split_rs = run_splitter(C_split, outer_flag);
                // Stats
                auto [m_min, m_max, m_avg] = summarize_parts(manual_rs.part_us);
                auto [s_min, s_max, s_avg] = summarize_parts(split_rs.part_us);
                double rel = args.validate ? rel_l2(C_split, C_manual) : 0.0;
                double flops = 2.0 * (double)seq_len * (double)local_head_dim * (double)args.d_model;
                double gflops_manual = (flops * args.iters) / (manual_rs.total_ms * 1e6);
                double gflops_split = (flops * args.iters) / (split_rs.total_ms * 1e6);
                double speedup = manual_rs.total_ms / split_rs.total_ms;
                std::string recommend = (speedup > 1.03 ? "use_split" : (speedup < 0.97 ? "use_manual" : "either"));
                std::cout << "seq=" << seq_len
                          << " parts=" << parts
                          << " outer=" << (outer_flag ? 1 : 0)
                          << " ms_manual=" << manual_rs.total_ms
                          << " ms_split=" << split_rs.total_ms
                          << " speedup=" << speedup
                          << " rel_l2=" << rel
                          << " GF_manual=" << gflops_manual
                          << " GF_split=" << gflops_split
                          << " part_manual(min/max/avg)_ms=" << m_min << "/" << m_max << "/" << m_avg
                          << " part_split(min/max/avg)_ms=" << s_min << "/" << s_max << "/" << s_avg
                          << " recommend=" << recommend
                          << "\n";
                return std::tuple{manual_rs, split_rs, rel, gflops_manual, gflops_split, speedup, m_min, m_max, m_avg, s_min, s_max, s_avg, recommend};
            };
            auto run_mode = [&](bool outer_flag)
            {
                // Repeat logic: gather times & pick median (for ms_manual/ms_split); recompute speedup from medians
                std::vector<double> manual_times, split_times;
                manual_times.reserve(args.repeat);
                split_times.reserve(args.repeat);
                std::vector<std::tuple<RunStats, RunStats, double, double, double, double, double, double, double, double, double, double, std::string>> samples;
                samples.reserve(args.repeat);
                for (int r = 0; r < args.repeat; ++r)
                {
                    samples.push_back(run_mode_once(outer_flag));
                    manual_times.push_back(std::get<0>(samples.back()).total_ms);
                    split_times.push_back(std::get<1>(samples.back()).total_ms);
                }
                auto median = [](std::vector<double> v)
                { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
                double med_manual = median(manual_times);
                double med_split = median(split_times);
                // Use last sample's ancillary stats (approx) or recompute averages across part stats
                const auto &last = samples.back();
                double rel = std::get<2>(last);
                double flops = 2.0 * (double)seq_len * (double)local_head_dim * (double)args.d_model;
                double gflops_manual = (flops * args.iters) / (med_manual * 1e6);
                double gflops_split = (flops * args.iters) / (med_split * 1e6);
                double speedup = med_manual / med_split;
                std::string recommend = (speedup > 1.03 ? "use_split" : (speedup < 0.97 ? "use_manual" : "either"));
                std::cout << "MEDIAN seq=" << seq_len
                          << " parts=" << parts
                          << " outer=" << (outer_flag ? 1 : 0)
                          << " ms_manual=" << med_manual
                          << " ms_split=" << med_split
                          << " speedup=" << speedup
                          << " rel_l2=" << rel
                          << " GF_manual=" << gflops_manual
                          << " GF_split=" << gflops_split
                          << " recommend=" << recommend
                          << (args.repeat > 1 ? " (median)" : "")
                          << "\n";
                return std::tuple{med_manual, med_split, speedup, rel, gflops_manual, gflops_split, recommend};
            };
            std::vector<std::tuple<int, bool, double, double, double, double, double, double, double, double, double, double, std::string>> csv_rows; // seq,outer,...
            auto primary = run_mode(args.outer_parallel);
            csv_rows.push_back({seq_len, args.outer_parallel,
                                std::get<0>(primary), std::get<1>(primary),
                                std::get<2>(primary), std::get<3>(primary), // speedup, rel_l2
                                0, 0, 0, 0, 0, 0,                           // part stats omitted for median mode
                                std::get<6>(primary)});
            all_results.push_back({parts, seq_len, args.outer_parallel, std::get<0>(primary), std::get<1>(primary), std::get<2>(primary), std::get<3>(primary), std::get<6>(primary)});
            if (args.compare_outer && parts > 1)
            {
                auto secondary = run_mode(!args.outer_parallel);
                csv_rows.push_back({seq_len, !args.outer_parallel,
                                    std::get<0>(secondary), std::get<1>(secondary),
                                    std::get<2>(secondary), std::get<3>(secondary),
                                    0, 0, 0, 0, 0, 0,
                                    std::get<6>(secondary)});
                all_results.push_back({parts, seq_len, !args.outer_parallel, std::get<0>(secondary), std::get<1>(secondary), std::get<2>(secondary), std::get<3>(secondary), std::get<6>(secondary)});
            }

            // CSV output
            if (!args.csv_path.empty())
            {
                bool write_header = false;
                struct stat st{};
                if (stat(args.csv_path.c_str(), &st) != 0)
                    write_header = true;
                else if (st.st_size == 0)
                    write_header = true;
                std::ofstream ofs(args.csv_path, std::ios::app);
                if (write_header)
                {
                    ofs << "parts,seq,outer,d_model,heads,threads,ms_manual,ms_split,speedup,rel_l2,manual_part_min_ms,manual_part_max_ms,manual_part_avg_ms,split_part_min_ms,split_part_max_ms,split_part_avg_ms,recommend,repeat\n";
                }
                for (auto &row : csv_rows)
                {
                    ofs << parts << "," << std::get<0>(row) << "," << (std::get<1>(row) ? 1 : 0) << "," << args.d_model << "," << args.heads << "," << args.threads << ","
                        << std::get<2>(row) << "," << std::get<3>(row) << "," << std::get<4>(row) << "," << std::get<5>(row) << ","
                        << std::get<6>(row) << "," << std::get<7>(row) << "," << std::get<8>(row) << ","
                        << std::get<9>(row) << "," << std::get<10>(row) << "," << std::get<11>(row) << ","
                        << std::get<12>(row) << "," << args.repeat << "\n";
                }
            }

            // JSON output (overwrite each sequence append semantics: collect and write at end) -> accumulate in memory
            static std::vector<std::string> json_entries; // one entry per seq * outer-mode(s)
            for (auto &row : csv_rows)
            {
                std::ostringstream j;
                j << "{\"parts\":" << parts
                  << ",\"seq\":" << std::get<0>(row)
                  << ",\"outer\":" << (std::get<1>(row) ? 1 : 0)
                  << ",\"d_model\":" << args.d_model
                  << ",\"heads\":" << args.heads
                  << ",\"threads\":" << args.threads
                  << ",\"ms_manual\":" << std::get<2>(row)
                  << ",\"ms_split\":" << std::get<3>(row)
                  << ",\"speedup\":" << std::get<4>(row)
                  << ",\"rel_l2\":" << std::get<5>(row)
                  << ",\"manual_part_min_ms\":" << std::get<6>(row)
                  << ",\"manual_part_max_ms\":" << std::get<7>(row)
                  << ",\"manual_part_avg_ms\":" << std::get<8>(row)
                  << ",\"split_part_min_ms\":" << std::get<9>(row)
                  << ",\"split_part_max_ms\":" << std::get<10>(row)
                  << ",\"split_part_avg_ms\":" << std::get<11>(row)
                  << ",\"recommend\":\"" << std::get<12>(row) << "\",\"repeat\":" << args.repeat << "}";
                json_entries.push_back(j.str());
            }
            if (&seq_len == &args.seqs.back() && !args.json_path.empty())
            {
                std::ofstream jf(args.json_path);
                jf << "[\n";
                for (size_t i = 0; i < json_entries.size(); ++i)
                {
                    jf << "  " << json_entries[i];
                    if (i + 1 < json_entries.size())
                        jf << ",";
                    jf << "\n";
                }
                jf << "]\n";
            }
        } // end seq loop
    } // end parts loop

    // Summary table
    if (args.summary_table)
    {
        std::cout << "\n=== Summary Table (median times) ===\n";
        std::cout << "parts seq outer ms_manual ms_split speedup recommend\n";
        for (const auto &r : all_results)
        {
            std::cout << r.parts << " " << r.seq << " " << (r.outer ? 1 : 0)
                      << " " << r.ms_manual << " " << r.ms_split << " " << r.speedup << " " << r.recommend << "\n";
        }
    }

    // Heuristic emitter (very lightweight threshold extraction)
    if (args.emit_heuristic)
    {
        // Build per-parts threshold (minimum seq where recommendation == use_split) for outer=selected (or both if compare_outer)
        struct Threshold
        {
            int parts;
            int seq;
            bool outer;
        };
        std::vector<Threshold> thresholds;
        auto capture_thresholds = [&](bool outer_flag)
        {
            for (int p : args.parts_list)
            {
                int best = -1;
                for (const auto &r : all_results)
                {
                    if (r.parts == p && r.outer == outer_flag && r.recommend == "use_split")
                    {
                        if (best < 0 || r.seq < best)
                            best = r.seq;
                    }
                }
                if (best > 0)
                    thresholds.push_back({p, best, outer_flag});
            }
        };
        if (args.compare_outer)
        {
            capture_thresholds(true);
            capture_thresholds(false);
        }
        else
            capture_thresholds(args.outer_parallel);

        std::cout << "\n=== Heuristic Snippet Proposal (generalized) ===\n";
        if (thresholds.empty())
        {
            std::cout << "// No splitter advantage observed across provided parts; keep manual path.\n";
        }
        else
        {
            // Power-law fit: seq_threshold(parts) = C * parts^B
            // Use only one outer flag cohort (first threshold outer group) for model to avoid mixing regimes.
            int chosen_outer = thresholds.front().outer;
            std::vector<std::pair<int, int>> pts;
            pts.reserve(thresholds.size());
            for (auto &t : thresholds)
            {
                if (t.outer == chosen_outer)
                    pts.emplace_back(t.parts, t.seq);
            }
            if (pts.size() >= 2)
            {
                double sum_lp = 0, sum_lt = 0, sum_lp2 = 0, sum_lp_lt = 0;
                int n = 0;
                for (auto &pr : pts)
                {
                    double lp = std::log((double)pr.first);
                    double lt = std::log((double)pr.second);
                    sum_lp += lp;
                    sum_lt += lt;
                    sum_lp2 += lp * lp;
                    sum_lp_lt += lp * lt;
                    ++n;
                }
                double denom = n * sum_lp2 - sum_lp * sum_lp;
                double B = (denom == 0) ? 0 : (n * sum_lp_lt - sum_lp * sum_lt) / denom;
                double A = (sum_lt - B * sum_lp) / n; // log C
                double C = std::exp(A);
                // Compute average relative error
                double err = 0;
                for (auto &pr : pts)
                {
                    double pred = C * std::pow((double)pr.first, B);
                    err += std::abs(pred - pr.second) / pr.second;
                }
                err /= pts.size();
                if (err < 0.35)
                {
                    std::cout << "// Model: threshold_seq(parts) ≈ C * parts^B  (outer_parallel=" << chosen_outer << ")\n";
                    std::cout << "// Fitted C=" << C << " B=" << B << " avg_rel_err=" << err << " n_pts=" << pts.size() << "\n";
                    std::cout << "auto tp_proj_split_threshold = [](int tp_parts)->int { const double C=" << C << "; const double B=" << B << "; return (int)std::ceil(C*std::pow((double)tp_parts, B)); };\n";
                    std::cout << "if(seq_len >= (size_t)tp_proj_split_threshold(tp_parts)" << (args.compare_outer ? " /* consider outer flag if divergent */" : "") << ") use_splitter=true;\n";
                }
                else
                {
                    std::cout << "// Power-law fit error too high (avg_rel_err=" << err << "); enumerating explicit thresholds:\n";
                    std::cout << "bool use_splitter=false;\n";
                    for (auto &pr : pts)
                    {
                        std::cout << "if(tp_parts==" << pr.first << " && seq_len>=" << pr.second << ") use_splitter=true;\n";
                    }
                }
            }
            else
            {
                // Only one point: emit simple condition
                auto pr = pts.front();
                std::cout << "bool use_splitter=false; if(tp_parts==" << pr.first << " && seq_len>=" << pr.second << ") use_splitter=true;\n";
            }
        }
    }
    return 0;
}
