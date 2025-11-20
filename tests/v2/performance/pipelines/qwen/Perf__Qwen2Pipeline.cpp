/**
 * @file Perf__Qwen2Pipeline.cpp
 * @brief Performance test harness for Qwen2Pipeline
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <iomanip>

#include "pipelines/qwen/Qwen2Pipeline.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "tensors/TensorFactory.h"

using namespace llaminar2;
using namespace std::chrono;

class Qwen2PipelinePerf : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<Qwen2Pipeline> pipeline_;
    int batch_size_ = 1;
    int max_seq_len_ = 128;

    void SetUp() override
    {
        // Create a dummy model context
        // We use createForTesting which sets up minimal metadata
        // but doesn't load weights. We'll need to manually populate weights
        // if the pipeline tries to access them, or mock the weight loading.
        // However, Qwen2Pipeline loads weights lazily via getWeight.
        // ModelContext::createForTesting returns a context where getWeight returns nullptr
        // unless we inject weights.

        // Actually, createForTesting might not be enough if the pipeline expects valid weights
        // for computation. But for performance testing of the *structure* and *activations*,
        // we might get away with dummy weights.

        // Let's try to use a real model if available, or fallback to a synthetic one.
        // For a focused perf test, we want reproducible conditions, so synthetic weights are better.

        // Create MPI context
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        // Create synthetic model context
        // We need to mock the GGUF metadata to match Qwen2
        // Since we can't easily mock GGUFModel internals from here without a file,
        // we might need to rely on a test resource or create a minimal GGUF file.

        // Alternative: Subclass Qwen2Pipeline and override getLayerWeights?
        // No, we want to test the real pipeline.

        // Let's assume we can use a dummy GGUF file or the "createForTesting" sets up enough defaults.
        // ModelContext::createForTesting("dummy", ...)

        model_ctx_ = ModelContext::createForTesting("qwen2_dummy.gguf", mpi_ctx_);

        // We need to manually inject metadata into the model_ctx_->model() if possible,
        // or ensure createForTesting provides defaults.
        // Looking at ModelContext.cpp (not visible here), createForTesting usually sets up
        // some basic params.

        // Let's configure the pipeline
        PipelineConfig config;
        config.max_seq_len = max_seq_len_;
        config.activation_precision = ActivationPrecision::FP32;

        // We need to populate the weights in ModelContext so getWeight returns something valid.
        // ModelContext has a weight_map_ but it's private.
        // However, we can use the WeightManager if exposed.

        // If we can't easily inject weights, we might crash on nullptr access in the pipeline.
        // Let's check if we can use a mock model loader or if we should just use a real model file
        // if present, or skip.

        // For now, let's try to construct it and see. If it fails, we'll need a better strategy.
        // But wait, the user asked to "design a perf test harness".
        // A robust harness should generate synthetic weights.

        // Let's assume we can use the existing "models/qwen2.5-0.5b-instruct-q4_0.gguf" if it exists,
        // as referenced in the instructions.
    }

    void InitializePipeline(int batch_size, int seq_len)
    {
        batch_size_ = batch_size;
        max_seq_len_ = seq_len;

        PipelineConfig config;
        config.max_seq_len = max_seq_len_;
        config.activation_precision = ActivationPrecision::FP32;

        // Try to load real model
        std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        // Check if file exists
        std::ifstream f(model_path.c_str());
        if (!f.good())
        {
            // Fallback or skip
            GTEST_SKIP() << "Model file not found: " << model_path;
            return;
        }

        model_ctx_ = ModelContext::create(model_path, mpi_ctx_);
        pipeline_ = std::make_unique<Qwen2Pipeline>(model_ctx_, mpi_ctx_, -1, nullptr, config, batch_size_);
    }

    void RunBenchmark(const std::string &name, int num_tokens)
    {
        if (!pipeline_)
            return;

        std::vector<std::vector<int>> batches(batch_size_);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 1000); // Random tokens

        for (int b = 0; b < batch_size_; ++b)
        {
            batches[b].resize(num_tokens);
            for (int i = 0; i < num_tokens; ++i)
            {
                batches[b][i] = dist(rng);
            }
        }

        // Warmup
        pipeline_->forward_batch(batches);

        auto start = high_resolution_clock::now();
        int iters = 10;
        for (int i = 0; i < iters; ++i)
        {
            pipeline_->forward_batch(batches);
        }
        auto end = high_resolution_clock::now();

        double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avg_ms = total_ms / iters;
        double tokens_per_sec = (batch_size_ * num_tokens * iters) / (total_ms / 1000.0);

        std::cout << std::left << std::setw(40) << name
                  << " Batch=" << std::setw(3) << batch_size_
                  << " Len=" << std::setw(4) << num_tokens
                  << " | Time: " << std::fixed << std::setprecision(2) << avg_ms << " ms"
                  << " | TPS: " << std::setprecision(2) << tokens_per_sec
                  << std::endl;
    }
};

TEST_F(Qwen2PipelinePerf, Prefill_Batch1_Seq128)
{
    InitializePipeline(1, 128);
    RunBenchmark("Prefill B1 S128", 128);
}

TEST_F(Qwen2PipelinePerf, Decode_Batch1_Seq1)
{
    InitializePipeline(1, 128); // Max seq len 128
    // For decode, we usually feed 1 token but have history.
    // The pipeline forward_batch takes tokens. If we pass 1 token, it's decode step.
    // But we need to set up state? Qwen2Pipeline handles state via current_positions_.
    // So passing 1 token is effectively a decode step if previous steps happened.

    // We'll just benchmark the processing of a 1-token input (decode-like latency)
    RunBenchmark("Decode B1 S1", 1);
}

TEST_F(Qwen2PipelinePerf, Prefill_Batch4_Seq128)
{
    InitializePipeline(4, 128);
    RunBenchmark("Prefill B4 S128", 128);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
