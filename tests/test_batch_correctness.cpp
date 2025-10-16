/**
 * @file test_batch_correctness.cpp
 * @brief Batch vs sequential correctness validation with real models
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include <gtest/gtest.h>
#include "QwenPipelineAdapter.h"
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "MpiContext.h"
#include <memory>
#include <vector>
#include <cmath>
#include <fstream>

using namespace llaminar;

/**
 * @brief Fixture for batch correctness tests
 */
class BatchCorrectnessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use qwen2.5-0.5b-instruct-q4_0.gguf for testing
        model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        
        auto rank = MPIContext::capture().rank;
        
        // Check if model exists
        if (rank == 0)
        {
            std::ifstream model_file(model_path);
            if (!model_file.good())
            {
                GTEST_SKIP() << "Model file not found: " << model_path;
            }
        }
        
        // Register Qwen pipeline
        registerQwenPipeline();
        
        // Disable COSMA for simpler testing
        setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);
        
        // Load model configuration
        ModelLoader loader;
        if (!loader.loadModel(model_path))
        {
            GTEST_SKIP() << "Failed to load model: " << model_path;
        }
        
        TransformerLayerConfig base_config = loader.createLayerConfig();
        config = ModelConfig(base_config, "qwen");
    }

    std::string model_path;
    ModelConfig config;
};

/**
 * @brief Helper to compare two tensors element-by-element
 */
void compareTensors(
    const std::shared_ptr<TensorBase>& a,
    const std::shared_ptr<TensorBase>& b,
    const std::string& name,
    float tolerance = 1e-4)
{
    ASSERT_NE(a, nullptr) << name << " tensor A is null";
    ASSERT_NE(b, nullptr) << name << " tensor B is null";
    
    const auto& shape_a = a->shape();
    const auto& shape_b = b->shape();
    
    ASSERT_EQ(shape_a.size(), shape_b.size()) 
        << name << " shape rank mismatch";
    
    for (size_t i = 0; i < shape_a.size(); ++i)
    {
        ASSERT_EQ(shape_a[i], shape_b[i]) 
            << name << " shape[" << i << "] mismatch";
    }
    
    size_t numel = 1;
    for (auto dim : shape_a) numel *= dim;
    
    const float* data_a = a->data();
    const float* data_b = b->data();
    
    size_t mismatches = 0;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    
    for (size_t i = 0; i < numel; ++i)
    {
        float diff = std::abs(data_a[i] - data_b[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        if (diff > tolerance)
        {
            mismatches++;
            if (mismatches <= 5)  // Only print first 5 mismatches
            {
                ADD_FAILURE() << name << " mismatch at index " << i 
                             << ": " << data_a[i] << " vs " << data_b[i]
                             << " (diff: " << diff << ")";
            }
        }
    }
    
    EXPECT_EQ(mismatches, 0) 
        << name << " had " << mismatches << " mismatches out of " << numel
        << " elements (max diff: " << max_diff << " at index " << max_diff_idx << ")";
}

/**
 * @brief Test: Batch prefill produces same results as sequential prefill
 */
TEST_F(BatchCorrectnessTest, PrefillBatchVsSequential)
{
    auto rank = MPIContext::capture().rank;
    
    const int batch_size = 2;
    const std::vector<int> seq1 = {1, 2, 3, 4};
    const std::vector<int> seq2 = {5, 6, 7, 8, 9};
    
    if (rank == 0)
    {
        std::cout << "\n=== Testing Prefill: Batch vs Sequential ===\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Sequence 1: " << seq1.size() << " tokens\n";
        std::cout << "Sequence 2: " << seq2.size() << " tokens\n";
    }
    
    // ============================================
    // Run batch execution
    // ============================================
    auto batch_pipeline = PipelineFactory::instance().create(config);
    ASSERT_NE(batch_pipeline, nullptr);
    
    auto batch_weights = batch_pipeline->loadWeights(model_path);
    ASSERT_NE(batch_weights, nullptr);
    
    std::vector<std::vector<int>> batch_input = {seq1, seq2};
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;
    
    if (rank == 0) std::cout << "Running batch prefill...\n";
    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));
    ASSERT_NE(batch_logits, nullptr);
    
    // Extract logits for each sequence
    const auto& batch_shape = batch_logits->shape();
    ASSERT_EQ(batch_shape.size(), 2);
    ASSERT_EQ(batch_shape[0], batch_size);
    
    int vocab_size = batch_shape[1];
    const float* batch_data = batch_logits->data();
    
    std::vector<std::vector<float>> batch_results(batch_size);
    for (int seq = 0; seq < batch_size; ++seq)
    {
        batch_results[seq].assign(
            batch_data + seq * vocab_size,
            batch_data + (seq + 1) * vocab_size
        );
    }
    
    // ============================================
    // Run sequential execution
    // ============================================
    std::vector<std::vector<float>> sequential_results(batch_size);
    std::vector<std::vector<int>> sequences = {seq1, seq2};
    
    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0) std::cout << "Running sequential prefill for sequence " << seq << "...\n";
        
        auto seq_pipeline = PipelineFactory::instance().create(config);
        ASSERT_NE(seq_pipeline, nullptr);
        
        auto seq_weights = seq_pipeline->loadWeights(model_path);
        ASSERT_NE(seq_weights, nullptr);
        
        StageContext seq_ctx;
        
        ASSERT_TRUE(seq_pipeline->prefill(sequences[seq], *seq_weights, seq_ctx));
        
        std::shared_ptr<TensorBase> seq_logits;
        ASSERT_TRUE(seq_pipeline->logits(seq_logits));
        ASSERT_NE(seq_logits, nullptr);
        
        const auto& seq_shape = seq_logits->shape();
        ASSERT_EQ(seq_shape.size(), 2);
        ASSERT_EQ(seq_shape[1], vocab_size);
        
        // Get last row (final token's logits)
        int rows = seq_shape[0];
        const float* seq_data = seq_logits->data();
        sequential_results[seq].assign(
            seq_data + (rows - 1) * vocab_size,
            seq_data + rows * vocab_size
        );
    }
    
    // ============================================
    // Compare results
    // ============================================
    if (rank == 0) std::cout << "Comparing results...\n";
    
    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0) std::cout << "Comparing sequence " << seq << "...\n";
        
        ASSERT_EQ(batch_results[seq].size(), sequential_results[seq].size());
        
        size_t mismatches = 0;
        float max_diff = 0.0f;
        
        for (size_t i = 0; i < batch_results[seq].size(); ++i)
        {
            float diff = std::abs(batch_results[seq][i] - sequential_results[seq][i]);
            max_diff = std::max(max_diff, diff);
            
            if (diff > 1e-4f)
            {
                mismatches++;
                if (mismatches <= 3)
                {
                    ADD_FAILURE() << "Sequence " << seq << " token " << i 
                                 << " mismatch: batch=" << batch_results[seq][i]
                                 << " sequential=" << sequential_results[seq][i]
                                 << " diff=" << diff;
                }
            }
        }
        
        EXPECT_EQ(mismatches, 0) 
            << "Sequence " << seq << " had " << mismatches 
            << " mismatches (max diff: " << max_diff << ")";
    }
    
    if (rank == 0)
    {
        std::cout << "✓ Batch prefill matches sequential prefill\n";
    }
}

/**
 * @brief Test: Batch decode produces same results as sequential decode
 */
TEST_F(BatchCorrectnessTest, DecodeBatchVsSequential)
{
    auto rank = MPIContext::capture().rank;
    
    const int batch_size = 2;
    const std::vector<int> seq1 = {1, 2, 3};
    const std::vector<int> seq2 = {4, 5, 6, 7};
    const int decode_steps = 3;
    
    if (rank == 0)
    {
        std::cout << "\n=== Testing Decode: Batch vs Sequential ===\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Decode steps: " << decode_steps << "\n";
    }
    
    // ============================================
    // Run batch execution
    // ============================================
    auto batch_pipeline = PipelineFactory::instance().create(config);
    ASSERT_NE(batch_pipeline, nullptr);
    
    auto batch_weights = batch_pipeline->loadWeights(model_path);
    ASSERT_NE(batch_weights, nullptr);
    
    std::vector<std::vector<int>> batch_input = {seq1, seq2};
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;
    
    // Prefill
    if (rank == 0) std::cout << "Batch prefill...\n";
    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));
    
    // Decode steps
    std::vector<std::vector<std::vector<float>>> batch_decode_results(decode_steps);
    
    for (int step = 0; step < decode_steps; ++step)
    {
        if (rank == 0) std::cout << "Batch decode step " << (step + 1) << "...\n";
        
        // Use token ID 42 for all sequences (deterministic)
        std::vector<int> next_tokens(batch_size, 42);
        
        std::shared_ptr<TensorBase> decode_logits;
        ASSERT_TRUE(batch_pipeline->decodeBatch(next_tokens, *batch_weights, batch_ctx, decode_logits));
        ASSERT_NE(decode_logits, nullptr);
        
        const auto& shape = decode_logits->shape();
        ASSERT_EQ(shape.size(), 2);
        ASSERT_EQ(shape[0], batch_size);
        
        int vocab_size = shape[1];
        const float* data = decode_logits->data();
        
        batch_decode_results[step].resize(batch_size);
        for (int seq = 0; seq < batch_size; ++seq)
        {
            batch_decode_results[step][seq].assign(
                data + seq * vocab_size,
                data + (seq + 1) * vocab_size
            );
        }
    }
    
    // ============================================
    // Run sequential execution
    // ============================================
    std::vector<std::vector<std::vector<float>>> sequential_decode_results(decode_steps);
    for (auto& step_results : sequential_decode_results)
    {
        step_results.resize(batch_size);
    }
    
    std::vector<std::vector<int>> sequences = {seq1, seq2};
    
    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0) std::cout << "Sequential prefill+decode for sequence " << seq << "...\n";
        
        auto seq_pipeline = PipelineFactory::instance().create(config);
        ASSERT_NE(seq_pipeline, nullptr);
        
        auto seq_weights = seq_pipeline->loadWeights(model_path);
        ASSERT_NE(seq_weights, nullptr);
        
        StageContext seq_ctx;
        
        // Prefill
        ASSERT_TRUE(seq_pipeline->prefill(sequences[seq], *seq_weights, seq_ctx));
        
        // Decode steps
        for (int step = 0; step < decode_steps; ++step)
        {
            ASSERT_TRUE(seq_pipeline->decode(42, *seq_weights, seq_ctx));
            
            std::shared_ptr<TensorBase> seq_logits;
            ASSERT_TRUE(seq_pipeline->logits(seq_logits));
            ASSERT_NE(seq_logits, nullptr);
            
            const auto& shape = seq_logits->shape();
            ASSERT_EQ(shape.size(), 2);
            
            int vocab_size = shape[1];
            int rows = shape[0];
            const float* data = seq_logits->data();
            
            sequential_decode_results[step][seq].assign(
                data + (rows - 1) * vocab_size,
                data + rows * vocab_size
            );
        }
    }
    
    // ============================================
    // Compare results
    // ============================================
    if (rank == 0) std::cout << "Comparing decode results...\n";
    
    for (int step = 0; step < decode_steps; ++step)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            if (rank == 0) std::cout << "Comparing step " << (step + 1) 
                                    << " sequence " << seq << "...\n";
            
            const auto& batch_result = batch_decode_results[step][seq];
            const auto& seq_result = sequential_decode_results[step][seq];
            
            ASSERT_EQ(batch_result.size(), seq_result.size());
            
            size_t mismatches = 0;
            float max_diff = 0.0f;
            
            for (size_t i = 0; i < batch_result.size(); ++i)
            {
                float diff = std::abs(batch_result[i] - seq_result[i]);
                max_diff = std::max(max_diff, diff);
                
                if (diff > 1e-4f)
                {
                    mismatches++;
                    if (mismatches <= 3)
                    {
                        ADD_FAILURE() << "Step " << step << " seq " << seq 
                                     << " token " << i << " mismatch: batch=" 
                                     << batch_result[i] << " sequential=" 
                                     << seq_result[i] << " diff=" << diff;
                    }
                }
            }
            
            EXPECT_EQ(mismatches, 0) 
                << "Step " << step << " sequence " << seq << " had " 
                << mismatches << " mismatches (max diff: " << max_diff << ")";
        }
    }
    
    if (rank == 0)
    {
        std::cout << "✓ Batch decode matches sequential decode\n";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
