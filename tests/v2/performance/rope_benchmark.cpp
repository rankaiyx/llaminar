#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <omp.h>

#include "v2/kernels/cpu/primitives/RoPEPrimitives.h"
#include "v2/utils/CPUFeatures.h"

using namespace llaminar::v2::kernels::cpu::primitives;

// Simple timer
class Timer
{
    std::chrono::high_resolution_clock::time_point start;

public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

void print_header(const std::string &title)
{
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << std::setw(10) << "SeqLen"
              << std::setw(15) << "Time(ms)"
              << std::setw(15) << "GB/s"
              << std::setw(15) << "MTok/s" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
}

void run_benchmark_q8_1(int head_dim, int num_heads)
{
    print_header("Q8_1 RoPE Benchmark (HeadDim=" + std::to_string(head_dim) + ")");

    std::vector<int> seq_lens = {128, 512, 2048, 4096, 8192};

    for (int seq_len : seq_lens)
    {
        // Setup data
        std::vector<float> q_data(seq_len * num_heads * head_dim);
        std::vector<float> k_data(seq_len * num_heads * head_dim);

        // Initialize with random data
        for (auto &x : q_data)
            x = (float)rand() / RAND_MAX;
        for (auto &x : k_data)
            x = (float)rand() / RAND_MAX;

        // Quantize to Q8_1 (mock)
        // In reality we need Q8_1 blocks.
        // For this benchmark, we'll just allocate the right size and cast.
        // Q8_1 block is 32 bytes for 32 values (block_q8_1 struct).
        // So size is same as bytes.
        size_t num_elements = seq_len * num_heads * head_dim;
        size_t buffer_size = num_elements * sizeof(float); // Just use float buffer size for simplicity of allocation

        // Actually RoPE Q8_1 operates on block_q8_1.
        // We need to allocate block_q8_1 array.
        // 1 block = 32 values.
        size_t num_blocks = num_elements / 32;
        std::vector<block_q8_1> q_q8(num_blocks);
        std::vector<block_q8_1> k_q8(num_blocks);

        // Warmup
        apply_rope_q8_1(q_q8.data(), k_q8.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);

        Timer t;
        int iterations = 10;
        for (int i = 0; i < iterations; ++i)
        {
            apply_rope_q8_1(q_q8.data(), k_q8.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);
        }
        double ms = t.elapsed_ms() / iterations;

        // Calculate metrics
        // Read + Write for Q and K.
        // Q8_1 size: 1 byte per value + scales. Approx 1.125 bytes per value?
        // block_q8_1 is 32 bytes + 2 floats (8 bytes) = 40 bytes for 32 values.
        // So 1.25 bytes per value.
        double bytes_per_val = 40.0 / 32.0;
        double total_bytes = 2 * num_elements * bytes_per_val * 2; // Read+Write for Q and K
        double gb_s = (total_bytes / 1e9) / (ms / 1000.0);
        double mtok_s = (seq_len / 1e6) / (ms / 1000.0);

        std::cout << std::setw(10) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gb_s
                  << std::setw(15) << std::fixed << std::setprecision(2) << mtok_s << std::endl;
    }
}

void run_benchmark_fp16(int head_dim, int num_heads)
{
    print_header("FP16 RoPE Benchmark (HeadDim=" + std::to_string(head_dim) + ")");

    std::vector<int> seq_lens = {128, 512, 2048, 4096, 8192};

    for (int seq_len : seq_lens)
    {
        size_t num_elements = seq_len * num_heads * head_dim;
        std::vector<uint16_t> q_fp16(num_elements);
        std::vector<uint16_t> k_fp16(num_elements);

        // Warmup
        apply_rope_fp16(q_fp16.data(), k_fp16.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);

        Timer t;
        int iterations = 10;
        for (int i = 0; i < iterations; ++i)
        {
            apply_rope_fp16(q_fp16.data(), k_fp16.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);
        }
        double ms = t.elapsed_ms() / iterations;

        double bytes_per_val = 2.0;
        double total_bytes = 2 * num_elements * bytes_per_val * 2;
        double gb_s = (total_bytes / 1e9) / (ms / 1000.0);
        double mtok_s = (seq_len / 1e6) / (ms / 1000.0);

        std::cout << std::setw(10) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gb_s
                  << std::setw(15) << std::fixed << std::setprecision(2) << mtok_s << std::endl;
    }
}

void run_benchmark_bf16(int head_dim, int num_heads)
{
    print_header("BF16 RoPE Benchmark (HeadDim=" + std::to_string(head_dim) + ")");

    std::vector<int> seq_lens = {128, 512, 2048, 4096, 8192};

    for (int seq_len : seq_lens)
    {
        size_t num_elements = seq_len * num_heads * head_dim;
        std::vector<uint16_t> q_bf16(num_elements);
        std::vector<uint16_t> k_bf16(num_elements);

        // Warmup
        apply_rope_bf16(q_bf16.data(), k_bf16.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);

        Timer t;
        int iterations = 10;
        for (int i = 0; i < iterations; ++i)
        {
            apply_rope_bf16(q_bf16.data(), k_bf16.data(), seq_len, head_dim, num_heads, num_heads, 0, 10000.0f, nullptr);
        }
        double ms = t.elapsed_ms() / iterations;

        double bytes_per_val = 2.0;
        double total_bytes = 2 * num_elements * bytes_per_val * 2;
        double gb_s = (total_bytes / 1e9) / (ms / 1000.0);
        double mtok_s = (seq_len / 1e6) / (ms / 1000.0);

        std::cout << std::setw(10) << seq_len
                  << std::setw(15) << std::fixed << std::setprecision(3) << ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gb_s
                  << std::setw(15) << std::fixed << std::setprecision(2) << mtok_s << std::endl;
    }
}

int main()
{
    std::cout << "Running RoPE Benchmarks..." << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    int head_dim = 128;
    int num_heads = 32;

    run_benchmark_q8_1(head_dim, num_heads);
    run_benchmark_fp16(head_dim, num_heads);
    run_benchmark_bf16(head_dim, num_heads);

    return 0;
}
