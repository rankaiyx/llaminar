#include "mpi_transformer_pipeline.h"
#include "cosma_prefill_manager.h"
#include "tensors/tensor_factory.h"
#include "model_loader.h"
#include "logger.h"
#include "test_timeout_guard.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <functional>

extern "C"
{
#include "llama.h"
}

using namespace llaminar;

namespace
{
    struct MPIFinalizer
    {
        ~MPIFinalizer()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized)
            {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized)
                {
                    MPI_Finalize();
                }
            }
        }
    } mpi_finalizer;

    std::string find_test_model()
    {
        namespace fs = std::filesystem;
        fs::path models_dir{"models"};
        if (!fs::exists(models_dir))
        {
            return {};
        }

        // Preferred candidates in order of specificity (smaller quantized model first)
        const std::vector<std::string> preferred = {
            "qwen2.5-0.5b-instruct-q4_0.gguf",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-0.5b-instruct-q5_0.gguf",
            "qwen2.5-0.5b-instruct-fp16.gguf"};

        for (const auto &candidate : preferred)
        {
            fs::path path = models_dir / candidate;
            if (fs::exists(path))
            {
                return path.string();
            }
        }

        for (const auto &entry : fs::directory_iterator(models_dir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            auto name = entry.path().filename().string();
            if (name.size() >= 5 && name.substr(name.size() - 5) == ".gguf")
            {
                return entry.path().string();
            }
        }

        return {};
    }

    void broadcast_string(std::string &value, int root, MPI_Comm comm)
    {
        int length = static_cast<int>(value.size());
        MPI_Bcast(&length, 1, MPI_INT, root, comm);
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank != root)
        {
            value.assign(length, '\0');
        }
        if (length > 0)
        {
            MPI_Bcast(value.data(), length, MPI_CHAR, root, comm);
        }
    }

    struct LlamaContextGuard
    {
        llama_model *model{nullptr};
        llama_context *ctx{nullptr};

        ~LlamaContextGuard()
        {
            if (ctx)
            {
                llama_free(ctx);
            }
            if (model)
            {
                llama_model_free(model);
            }
        }
    };

    struct GoldenScenario
    {
        int seq_len;
        int max_layers;
        const char *name;
    };

    struct TokenPattern
    {
        const char *name;
        std::function<void(std::vector<int> &, int)> populate;
    };

    struct ComparisonMetrics
    {
        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        double rel_l2 = 0.0;
    };

    ComparisonMetrics compute_metrics(const std::vector<float> &candidate, const std::vector<float> &baseline)
    {
        ComparisonMetrics metrics;
        if (candidate.size() != baseline.size())
        {
            return metrics;
        }

        double sum_abs = 0.0;
        double diff_sq = 0.0;
        double ref_sq = 0.0;

        for (size_t i = 0; i < candidate.size(); ++i)
        {
            double diff = static_cast<double>(candidate[i]) - static_cast<double>(baseline[i]);
            double ref = static_cast<double>(baseline[i]);
            TransformerLayerConfig base_config = loader.createLayerConfig();
            const std::vector<GoldenScenario> scenarios = {
                {32, 2, "seq32_layers2"},
                {128, 4, "seq128_layers4"}};
            const std::vector<std::string> cosma_env_modes = {"direct", "replicated"};
            const std::vector<TokenPattern> token_patterns = {
                {"arithmetic_mod", [](std::vector<int> &tokens, int vocab)
                 {
                     const int safe_vocab = std::max(1, vocab);
                     for (size_t i = 0; i < tokens.size(); ++i)
                     {
                         tokens[i] = (static_cast<int>(i) * 7) % safe_vocab;
                     }
                 }},
                {"repeated_mid", [](std::vector<int> &tokens, int vocab)
                 {
                     const int safe_vocab = std::max(1, vocab);
                     const int value = safe_vocab > 1 ? safe_vocab / 2 : 0;
                     std::fill(tokens.begin(), tokens.end(), value);
                 }},
                {"alternating_pair", [](std::vector<int> &tokens, int vocab)
                 {
                     const int safe_vocab = std::max(1, vocab);
                     const int a = 0;
                     const int b = safe_vocab > 1 ? 1 : 0;
                     for (size_t i = 0; i < tokens.size(); ++i)
                     {
                         tokens[i] = (i % 2 == 0) ? a : b;
                     }
                 }}};

            CosmaPrefillManager &manager = CosmaPrefillManager::instance();
            constexpr float kPointwiseTolerance = 2e-3f;
            constexpr double kRelL2Tolerance = 5e-4;

            for (const auto &scenario : scenarios)
            {
                SCOPED_TRACE(::testing::Message() << "scenario=" << scenario.name);

                TransformerLayerConfig config = base_config;
                config.n_layers = std::min(config.n_layers, scenario.max_layers);
                config.max_seq_len = scenario.seq_len;

                auto weights = loadModelWeights(loader, config);
                MPITransformerPipeline pipeline(config);
                const int vocab = config.vocab_size;

                for (const auto &pattern : token_patterns)
                {
                    SCOPED_TRACE(::testing::Message() << "token_pattern=" << pattern.name);

                    std::vector<int> token_ids(scenario.seq_len, 0);
                    pattern.populate(token_ids, vocab);

                    const int64_t total_logits = static_cast<int64_t>(scenario.seq_len) * static_cast<int64_t>(vocab);
                    ASSERT_LE(total_logits, static_cast<int64_t>(std::numeric_limits<int>::max()))
                        << "Total logits exceed MPI broadcast capacity";
                    const int broadcast_count = static_cast<int>(total_logits);
                    std::vector<float> llama_logits(total_logits, 0.0f);

                    if (rank == 0)
                    {
                        llama_backend_init();

                        llama_model_params mparams = llama_model_default_params();
                        mparams.n_gpu_layers = 0;
                        mparams.use_mmap = false;
                        mparams.use_mlock = false;
                        mparams.check_tensors = false;

                        LlamaContextGuard guard;
                        guard.model = llama_model_load_from_file(model_path.c_str(), mparams);
                        ASSERT_NE(guard.model, nullptr) << "Failed to load llama.cpp model";

                        llama_context_params cparams = llama_context_default_params();
                        cparams.n_ctx = std::max<uint32_t>(scenario.seq_len, cparams.n_ctx);
                        cparams.n_batch = std::max<uint32_t>(scenario.seq_len, cparams.n_batch);
                        cparams.n_ubatch = std::max<uint32_t>(scenario.seq_len, cparams.n_ubatch);
                        cparams.n_threads = 1;
                        cparams.n_threads_batch = 1;
                        cparams.offload_kqv = false;
                        cparams.no_perf = true;
                        cparams.embeddings = false;

                        guard.ctx = llama_init_from_model(guard.model, cparams);
                        ASSERT_NE(guard.ctx, nullptr) << "Failed to initialize llama.cpp context";
                        llama_set_n_threads(guard.ctx, 1, 1);

                        llama_batch batch = llama_batch_init(scenario.seq_len, 0, 1);
                        for (int i = 0; i < scenario.seq_len; ++i)
                        {
                            batch.token[i] = token_ids[i];
                            batch.pos[i] = i;
                            batch.n_seq_id[i] = 1;
                            batch.seq_id[i][0] = 0;
                            batch.logits[i] = 1;
                        }
                        batch.n_tokens = scenario.seq_len;

                        int32_t rc = llama_decode(guard.ctx, batch);
                        ASSERT_EQ(rc, 0) << "llama_decode failed";
                        llama_synchronize(guard.ctx);

                        for (int i = 0; i < scenario.seq_len; ++i)
                        {
                            float *row = llama_get_logits_ith(guard.ctx, i);
                            ASSERT_NE(row, nullptr) << "Missing logits for token index " << i;
                            std::memcpy(llama_logits.data() + static_cast<int64_t>(i) * vocab, row, sizeof(float) * static_cast<size_t>(vocab));
                        }

                        llama_batch_free(batch);
                        llama_backend_free();
                    }

                    MPI_Bcast(llama_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

                    manager.reset_stats();
                    manager.set_force_cosma(false);
                    manager.set_threshold(std::numeric_limits<int>::max());

                    std::shared_ptr<TensorBase> openblas_output;
                    ASSERT_TRUE(pipeline.execute(token_ids, weights, openblas_output));

                    std::vector<float> openblas_logits(total_logits, 0.0f);
                    if (openblas_output && openblas_output->data())
                    {
                        std::memcpy(openblas_logits.data(), openblas_output->data(), sizeof(float) * static_cast<size_t>(total_logits));
                    }
                    MPI_Bcast(openblas_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

                    const auto openblas_metrics = compute_metrics(openblas_logits, llama_logits);
                    EXPECT_LT(openblas_metrics.max_abs, kPointwiseTolerance)
                        << "OpenBLAS max abs diff exceeds tolerance";
                    EXPECT_LT(openblas_metrics.rel_l2, kRelL2Tolerance)
                        << "OpenBLAS relative L2 drift too large";

                    EXPECT_EQ(manager.stats().cosma_path_calls.load(), 0) << "COSMA path should not have been used";

                    for (const auto &mode : cosma_env_modes)
                    {
                        SCOPED_TRACE(::testing::Message() << "cosma_mode=" << mode);

                        bool restore_direct = false;
                        bool restore_replicated = false;
                        std::string prev_force_direct;
                        std::string prev_force_replicated;

                        if (const char *existing = std::getenv("LLAMINAR_COSMA_FORCE_DIRECT"))
                        {
                            restore_direct = true;
                            prev_force_direct = existing;
                        }
                        if (const char *existing = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED"))
                        {
                            restore_replicated = true;
                            prev_force_replicated = existing;
                        }

                        if (mode == "direct")
                        {
                            setenv("LLAMINAR_COSMA_FORCE_DIRECT", "1", /*overwrite*/ 1);
                            unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
                        }
                        else if (mode == "replicated")
                        {
                            setenv("LLAMINAR_COSMA_FORCE_REPLICATED", "1", 1);
                            unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
                        }

                        manager.reset_stats();
                        manager.set_threshold(1);
                        manager.set_force_cosma(true);

                        std::shared_ptr<TensorBase> cosma_output;
                        ASSERT_TRUE(pipeline.execute(token_ids, weights, cosma_output));

                        std::vector<float> cosma_logits(total_logits, 0.0f);
                        if (cosma_output && cosma_output->data())
                        {
                            std::memcpy(cosma_logits.data(), cosma_output->data(), sizeof(float) * static_cast<size_t>(total_logits));
                        }
                        MPI_Bcast(cosma_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

                        const auto cosma_metrics = compute_metrics(cosma_logits, llama_logits);
                        EXPECT_LT(cosma_metrics.max_abs, kPointwiseTolerance)
                            << "COSMA max abs diff exceeds tolerance";
                        EXPECT_LT(cosma_metrics.rel_l2, kRelL2Tolerance)
                            << "COSMA relative L2 drift too large";

                        EXPECT_GT(manager.stats().cosma_path_calls.load(), 0)
                            << "COSMA path should have been invoked";
                        EXPECT_EQ(manager.stats().fast_path_calls.load(), 0)
                            << "Fast path should not trigger when COSMA forced";

                        if (restore_direct)
                        {
                            setenv("LLAMINAR_COSMA_FORCE_DIRECT", prev_force_direct.c_str(), 1);
                        }
                        else
                        {
                            unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
                        }
                        if (restore_replicated)
                        {
                            setenv("LLAMINAR_COSMA_FORCE_REPLICATED", prev_force_replicated.c_str(), 1);
                        }
                        else
                        {
                            unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
                        }
                    }
                }
            }
        }

        manager.set_force_cosma(false);
        manager.set_threshold(4096);
        manager.reset_stats();

        watchdog.disarm();
    }
