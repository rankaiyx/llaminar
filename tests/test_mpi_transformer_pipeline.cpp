#include "mpi_transformer_pipeline.h"
#include "logger.h"
#include "tensors/tensor_factory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace llaminar;

class MPITransformerPipelineTest : public ::testing::Test

    for (size_t i = 0;
i < output1_size; ++i)
{
    EXPECT_NEAR(output1_data[i], output2_data[i], 1e-5f);
rmerPipelineTest:
    public
    ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Verify MPI is initialized
            int flag;
            MPI_Initialized(&flag);
            if (!flag)
            {
                throw std::runtime_error("MPI should be initialized before running tests");
            }

            // Set up test configuration for a small transformer model
            config_.n_head = 8;
            config_.n_head_kv = 8;
            config_.head_dim = 64;
            config_.d_model = 512; // 8 * 64
            config_.d_ff = 2048;   // 4 * d_model
            config_.vocab_size = 1000;
            config_.max_seq_len = 64;
            config_.n_layers = 2; // Small for testing
            config_.eps = 1e-6f;

            // Create transformer pipeline
            pipeline_ = createMPITransformerPipeline(config_);

            // Initialize random generator with fixed seed for reproducibility
            generator_.seed(42);
        }

        void TearDown() override
        {
            pipeline_.reset();
        }

        void fillRandomData(std::shared_ptr<TensorBase> & tensor, float min_val = -0.01f, float max_val = 0.01f)
        {
            std::uniform_real_distribution<float> dist(min_val, max_val);
            auto *data = const_cast<float *>(tensor->data());
            size_t total_size = 1;
            for (int dim : tensor->shape())
            {
                total_size *= dim;
            }
            for (size_t i = 0; i < total_size; ++i)
            {
                data[i] = dist(generator_);
            }
        }

        MPITransformerPipeline::ModelWeights createTestWeights()
        {
            MPITransformerPipeline::ModelWeights weights;

            // Token embedding
            weights.token_embedding = llaminar::TensorFactory::create_simple(std::vector<int>{config_.vocab_size, config_.d_model});
            fillRandomData(weights.token_embedding);

            // Output layer weights
            weights.output_norm_weight = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model});
            fillRandomData(weights.output_norm_weight, 0.8f, 1.2f); // Norm weights should be near 1

            weights.lm_head = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.vocab_size});
            fillRandomData(weights.lm_head);

            // Per-layer weights
            for (int i = 0; i < config_.n_layers; ++i)
            {
                // Attention weights
                weights.attn_norm_weight.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model}));
                fillRandomData(weights.attn_norm_weight[i], 0.8f, 1.2f);

                weights.wq.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head * config_.head_dim}));
                fillRandomData(weights.wq[i]);

                weights.wk.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head_kv * config_.head_dim}));
                fillRandomData(weights.wk[i]);

                weights.wv.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head_kv * config_.head_dim}));
                fillRandomData(weights.wv[i]);

                weights.wo.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.n_head * config_.head_dim, config_.d_model}));
                fillRandomData(weights.wo[i]);

                // FFN weights
                weights.ffn_norm_weight.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model}));
                fillRandomData(weights.ffn_norm_weight[i], 0.8f, 1.2f);

                weights.w_gate.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.d_ff}));
                fillRandomData(weights.w_gate[i]);

                weights.w_up.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.d_ff}));
                fillRandomData(weights.w_up[i]);

                weights.w_down.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_ff, config_.d_model}));
                fillRandomData(weights.w_down[i]);
            }

            return weights;
        }

        std::vector<int> createTestTokens(int seq_len)
        {
            std::vector<int> tokens;
            std::uniform_int_distribution<int> dist(0, config_.vocab_size - 1);

            for (int i = 0; i < seq_len; ++i)
            {
                tokens.push_back(dist(generator_));
            }

            return tokens;
        }

        void validateOutput(const std::shared_ptr<TensorBase> &output, int seq_len)
        {
            ASSERT_TRUE(output);
            ASSERT_EQ(output->shape().size(), 2);
            ASSERT_EQ(output->shape()[0], seq_len);
            ASSERT_EQ(output->shape()[1], config_.vocab_size);

            // Check output is not zero and has reasonable values
            bool has_nonzero = false;
            bool has_reasonable_values = true;
            float max_val = -std::numeric_limits<float>::infinity();
            float min_val = std::numeric_limits<float>::infinity();

            const auto *output_data = output->data();
            size_t output_size = output->shape()[0] * output->shape()[1];
            for (size_t i = 0; i < output_size; ++i)
            {
                const auto &val = output_data[i];
                if (std::abs(val) > 1e-6f)
                {
                    has_nonzero = true;
                }
                if (std::abs(val) > 100.0f)
                { // Check for exploding values
                    has_reasonable_values = false;
                }
                max_val = std::max(max_val, val);
                min_val = std::min(min_val, val);
            }

            EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";
            EXPECT_TRUE(has_reasonable_values) << "Output values should be reasonable (not exploding)";

            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                LOG_INFO("Output validation: min=" << std::fixed << std::setprecision(6) << min_val
                                                   << ", max=" << max_val << ", shape=[" << seq_len << ", " << config_.vocab_size << "]");
            }
        }

    protected:
        std::unique_ptr<MPITransformerPipeline> pipeline_;
        MPITransformerPipeline::LayerConfig config_;
        std::mt19937 generator_;
    };

    TEST_F(MPITransformerPipelineTest, BasicFunctionality)
    {
        // Test basic pipeline execution with simple input
        const int seq_len = 4;
        auto tokens = createTestTokens(seq_len);
        auto weights = createTestWeights();
        auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

        ASSERT_TRUE(pipeline_->execute(tokens, weights, output));
        validateOutput(output, seq_len);
    }

    TEST_F(MPITransformerPipelineTest, ValidationTests)
    {
        // Test weight validation
        auto valid_weights = createTestWeights();
        EXPECT_TRUE(pipeline_->validate(valid_weights));

        // Test invalid embedding shape
        auto invalid_weights = valid_weights;
        invalid_weights.token_embedding = llaminar::TensorFactory::create_simple(std::vector<int>{config_.vocab_size + 1, config_.d_model});
        EXPECT_FALSE(pipeline_->validate(invalid_weights));

        // Test missing layer weights
        invalid_weights = valid_weights;
        invalid_weights.wq.pop_back();
        EXPECT_FALSE(pipeline_->validate(invalid_weights));

        // Test invalid attention weight shape
        invalid_weights = valid_weights;
        invalid_weights.wq[0] = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head * config_.head_dim + 1});
        EXPECT_FALSE(pipeline_->validate(invalid_weights));
    }

    TEST_F(MPITransformerPipelineTest, DifferentSequenceLengths)
    {
        // Test with various sequence lengths
        auto weights = createTestWeights();

        std::vector<int> test_lengths = {1, 2, 8, 16};

        for (int seq_len : test_lengths)
        {
            auto tokens = createTestTokens(seq_len);
            auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

            ASSERT_TRUE(pipeline_->execute(tokens, weights, output))
                << "Failed for sequence length " << seq_len;
            validateOutput(output, seq_len);
        }
    }

    TEST_F(MPITransformerPipelineTest, ConsistencyAcrossRuns)
    {
        // Test that multiple runs with same input produce same output
        const int seq_len = 4;
        auto tokens = createTestTokens(seq_len);
        auto weights = createTestWeights();

        // First run
        auto output1 = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});
        ASSERT_TRUE(pipeline_->execute(tokens, weights, output1));

        // Second run with same inputs
        auto output2 = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});
        ASSERT_TRUE(pipeline_->execute(tokens, weights, output2));

        // Compare outputs
        ASSERT_EQ(output1->data.size(), output2->data.size());

        for (size_t i = 0; i < output1->data.size(); ++i)
        {
            EXPECT_NEAR(output1->data[i], output2->data[i], 1e-5f)
                << "Mismatch at index " << i;
        }
    }

    TEST_F(MPITransformerPipelineTest, PerformanceBenchmark)
    {
        // Benchmark pipeline performance
        const int seq_len = 16;
        const int num_runs = 3;

        auto tokens = createTestTokens(seq_len);
        auto weights = createTestWeights();
        auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

        std::vector<double> run_times;

        for (int run = 0; run < num_runs; ++run)
        {
            auto start = std::chrono::high_resolution_clock::now();

            ASSERT_TRUE(pipeline_->execute(tokens, weights, output));

            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            run_times.push_back(time_ms);
        }

        // Calculate statistics
        double total_time = std::accumulate(run_times.begin(), run_times.end(), 0.0);
        double avg_time = total_time / num_runs;
        double min_time = *std::min_element(run_times.begin(), run_times.end());
        double max_time = *std::max_element(run_times.begin(), run_times.end());

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            LOG_INFO("Performance benchmark (" << num_runs << " runs, seq_len=" << seq_len << "):");
            LOG_INFO("  Average: " << std::fixed << std::setprecision(2) << avg_time << "ms");
            LOG_INFO("  Min: " << min_time << "ms, Max: " << max_time << "ms");
            LOG_INFO("  Throughput: " << (seq_len * config_.n_layers * 1000.0 / avg_time)
                                      << " tokens*layers/second");
        }

        validateOutput(output, seq_len);
    }

    TEST_F(MPITransformerPipelineTest, LoadBalancingAnalysis)
    {
        // Test load balancing across MPI processes
        const int seq_len = 8;
        auto tokens = createTestTokens(seq_len);
        auto weights = createTestWeights();
        auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

        auto start = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(pipeline_->execute(tokens, weights, output));
        auto end = std::chrono::high_resolution_clock::now();

        double local_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Gather timing from all processes
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<double> all_times(size);
        MPI_Gather(&local_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double total_time = std::accumulate(all_times.begin(), all_times.end(), 0.0);
            double avg_time = total_time / size;
            double min_time = *std::min_element(all_times.begin(), all_times.end());
            double max_time = *std::max_element(all_times.begin(), all_times.end());
            double load_imbalance = (max_time - min_time) / avg_time * 100.0;

            LOG_INFO("Load balancing analysis:");
            LOG_INFO("  Process times: " << std::fixed << std::setprecision(2));
            for (int i = 0; i < size; ++i)
            {
                LOG_INFO("    Rank " << i << ": " << all_times[i] << "ms");
            }
            LOG_INFO("  Load imbalance: " << load_imbalance << "%");

            // Good load balancing should have less than 20% imbalance
            EXPECT_LT(load_imbalance, 20.0) << "Load imbalance too high";
        }

        validateOutput(output, seq_len);
    }

    // Main function for running MPI tests
    int main(int argc, char **argv)
    {
        // Initialize MPI
        MPI_Init(&argc, &argv);

        // Initialize Google Test
        ::testing::InitGoogleTest(&argc, argv);

        // Get MPI rank and size
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0)
        {
            std::cout << "Running MPITransformerPipeline tests with " << size << " MPI processes" << std::endl;
        }

        // Run the tests
        int result = RUN_ALL_TESTS();

        // Synchronize before finalizing
        MPI_Barrier(MPI_COMM_WORLD);

        // Finalize MPI
        MPI_Finalize();

        return result;
    }