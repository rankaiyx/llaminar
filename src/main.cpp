#include <iostream>
#include <mpi.h>
#include <string>
#include <memory>
#include <iomanip>
#include <chrono>

// Llaminar modular components
#include "argument_parser.h"
#include "logger.h"
#include "topology_manager.h"
#include "graph_compute.h"
#include "model_loader.h"
#include "mpi_transformer_pipeline.h"

// Chat interface components
#include "chat/tokenizer_interface.h"
#include "chat/gguf_tokenizer.h"
#include "chat/chat_session.h"
#include "chat/chat_interface.h"
#include "chat/response_generator.h"
// Note: inference_engine.h removed until llama.cpp dependency is resolved

using namespace llaminar;

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try
    {
        // 1. Parse command line arguments
        ArgumentParser parser(argc, argv);
        LlaminarParams params;

        if (!parser.parse(params))
        {
            MPI_Finalize();
            return 1;
        }

        // Handle help and version
        if (params.show_help)
        {
            if (rank == 0)
                parser.printUsage();
            MPI_Finalize();
            return 0;
        }

        if (params.show_version)
        {
            if (rank == 0)
                parser.printVersion();
            MPI_Finalize();
            return 0;
        }

        // 2. Initialize logging
        initializeLogging();
        Logger::getInstance().setLogLevel(params.log_level);

        if (rank == 0)
        {
            LOG_INFO("Llaminar LLM Inference Engine starting...");
            LOG_INFO("MPI initialized with " << size << " processes");
            LOG_DEBUG("Log level set to: " << Logger::getInstance().logLevelToString(params.log_level));
        }

        // 3. Detect system topology
        TopologyManager topology_manager;
        SystemTopology topology = topology_manager.detectSystemTopology(
            params.use_hyperthreading,
            params.detect_gpus);

        if (rank == 0 && params.print_topology)
        {
            topology_manager.printSystemTopology(topology);
        }

        LOG_DEBUG("System initialization complete");

        // 4. Handle different execution modes
        if (params.interactive && !params.model_file.empty())
        {
            // Interactive chat mode - this will be implemented after model loading
            LOG_INFO("Interactive chat mode requested");
            LOG_INFO("Model file: " << params.model_file);
        }
        else if (params.inference_mode || !params.model_file.empty())
        {
            // Standard inference mode (non-interactive)
            LOG_INFO("Running inference mode");
            if (!params.model_file.empty())
            {
                LOG_INFO("Model file: " << params.model_file);
            }
        }
        else
        {
            // COSMA benchmark mode
            LOG_INFO("Running COSMA benchmark mode");

            // Create compute graph for benchmarking
            ComputeGraph graph;
            auto node = std::make_shared<MatMulNode>(
                "cosma_benchmark",
                params.m, params.n, params.k);
            graph.addNode(node);

            // Execute benchmark
            LOG_INFO("Executing compute graph...");
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < params.num_repeat; ++i)
            {
                bool success = graph.execute();
                if (!success)
                {
                    LOG_ERROR("Compute graph execution failed on iteration " << i);
                    MPI_Finalize();
                    return 1;
                }

                LOG_DEBUG("Completed iteration " << (i + 1) << "/" << params.num_repeat);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);

            // Report performance results
            LOG_INFO("=== Performance Results ===");
            LOG_INFO("Total execution time: " << duration.count() << " ms");
            LOG_INFO("Average time per iteration: " << duration.count() / params.num_repeat << " ms");

            // Calculate GFLOPS (2*m*n*k operations per matrix multiply)
            double total_ops = 2.0 * params.m * params.n * params.k * params.num_repeat;
            double gflops = total_ops / (duration.count() / 1000.0) / 1e9;

            if (rank == 0)
            {
                LOG_INFO("Matrix dimensions: " << params.m << "x" << params.n << "x" << params.k);
                LOG_INFO("Total operations: " << total_ops / 1e9 << " GFLOP");
                LOG_INFO("Achieved performance: " << gflops << " GFLOPS");
                LOG_INFO("Performance per MPI rank: " << gflops / size << " GFLOPS");
            }
        }

        // 5. Load model if specified
        std::unique_ptr<ModelLoader> model_loader;
        std::unique_ptr<MPITransformerPipeline> transformer_pipeline;
        std::unique_ptr<MPITransformerPipeline::ModelWeights> model_weights;

        if (!params.model_file.empty())
        {
            LOG_INFO("Loading model from: " << params.model_file);
            model_loader = std::make_unique<ModelLoader>();

            try
            {
                if (model_loader->loadModel(params.model_file))
                {
                    LOG_INFO("Model loaded successfully");

                    // Extract transformer configuration from GGUF metadata
                    auto config = model_loader->createLayerConfig();
                    LOG_INFO("Extracted model config: " << config.n_layers << " layers, "
                                                        << config.n_head << " heads, " << config.d_model << " dimensions");
                    LOG_INFO("Model config details: vocab_size=" << config.vocab_size
                                                                 << ", max_seq_len=" << config.max_seq_len << ", d_ff=" << config.d_ff);

                    // Model is fully loaded (including weights) after successful loadModel() call
                    LOG_INFO("Model weights are available for tensor loading");

                    // Initialize MPI transformer pipeline with model configuration
                    transformer_pipeline = std::make_unique<MPITransformerPipeline>(config);
                    LOG_INFO("Transformer pipeline initialized successfully");

                    // Load model weights for transformer pipeline
                    try
                    {
                        model_weights = std::make_unique<MPITransformerPipeline::ModelWeights>(
                            loadModelWeights(params.model_file, config));
                        LOG_INFO("Model weights loaded for transformer pipeline");
                    }
                    catch (const std::exception &e)
                    {
                        LOG_ERROR("Failed to load model weights: " << e.what());
                        MPI_Finalize();
                        return 1;
                    }
                }
                else
                {
                    LOG_ERROR("Failed to load model: " << params.model_file);
                    MPI_Finalize();
                    return 1;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Exception loading model: " << e.what());
                MPI_Finalize();
                return 1;
            }
        }

        // 6. Create compute graph and execute workload
        ComputeGraph graph;

        // Interactive chat mode
        if (params.interactive && transformer_pipeline)
        {
            if (rank == 0)
            {
                // Create tokenizer
                auto tokenizer = chat::createTokenizer(*model_loader);
                if (!tokenizer || !tokenizer->isReady())
                {
                    LOG_ERROR("Failed to initialize tokenizer for chat interface");
                    MPI_Finalize();
                    return 1;
                }
                // Create shared tokenizer for multiple usage
                auto tokenizer_shared = std::shared_ptr<chat::TokenizerInterface>(tokenizer.release());

                // Create chat session - needs a copy of tokenizer
                auto tokenizer_copy = chat::createTokenizer(*model_loader);
                auto session = std::make_unique<chat::ChatSession>(std::move(tokenizer_copy), params);

                // Create response generator
                auto shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
                auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params);

                // Launch chat interface
                chat::ChatInterface chat_ui(std::move(session), shared_pipeline, std::move(generator), params);
                chat_ui.run();
            }
            else
            {
                // Non-root ranks: wait for inference requests (to be implemented)
                // For now, just idle
                while (true)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            MPI_Finalize();
            return 0;
        }

        // Non-interactive prompt processing (--prompt or --eval)
        if (transformer_pipeline && (!params.prompt.empty() || params.eval_only))
        {
            if (rank == 0)
            {
                LOG_INFO("Running non-interactive prompt processing");

                // Create tokenizer
                auto tokenizer = chat::createTokenizer(*model_loader);
                if (!tokenizer || !tokenizer->isReady())
                {
                    LOG_ERROR("Failed to initialize tokenizer for prompt processing");
                    MPI_Finalize();
                    return 1;
                }
                auto tokenizer_shared = std::shared_ptr<chat::TokenizerInterface>(tokenizer.release());

                // Create response generator with pre-loaded weights
                auto shared_pipeline = std::shared_ptr<MPITransformerPipeline>(std::move(transformer_pipeline));
                auto generator = std::make_unique<chat::ResponseGenerator>(tokenizer_shared, shared_pipeline, params, *model_weights);

                // Process prompt
                std::string prompt_text = params.prompt.empty() ? "hello world" : params.prompt;
                LOG_INFO("Processing prompt: \"" << prompt_text << "\"");

                // Tokenize prompt
                std::vector<int32_t> prompt_tokens = tokenizer_shared->tokenize(prompt_text);
                LOG_INFO("Tokenized to " << prompt_tokens.size() << " tokens");

                if (params.eval_only)
                {
                    // Just print tokenization and exit
                    std::cout << "Prompt: \"" << prompt_text << "\"" << std::endl;
                    std::cout << "Tokens: [";
                    for (size_t i = 0; i < prompt_tokens.size(); ++i)
                    {
                        if (i > 0)
                            std::cout << ", ";
                        std::cout << prompt_tokens[i];
                    }
                    std::cout << "]" << std::endl;
                    std::cout << "Token count: " << prompt_tokens.size() << std::endl;

                    // Optionally print token strings
                    std::cout << "Token strings: [";
                    for (size_t i = 0; i < prompt_tokens.size(); ++i)
                    {
                        if (i > 0)
                            std::cout << ", ";
                        std::string token_str = tokenizer_shared->detokenize({prompt_tokens[i]});
                        std::cout << "\"" << token_str << "\"";
                    }
                    std::cout << "]" << std::endl;
                }
                else
                {
                    // Generate response
                    try
                    {
                        std::string response = generator->generateResponse(prompt_tokens);
                        std::cout << "Response: " << response << std::endl;
                    }
                    catch (const std::exception &e)
                    {
                        LOG_ERROR("Exception during response generation: " << e.what());
                        MPI_Finalize();
                        return 1;
                    }
                }

                // Signal completion to other ranks
                int completion_signal = 1;
                MPI_Bcast(&completion_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            else
            {
                // Non-root ranks: wait for completion signal
                int completion_signal = 0;
                MPI_Bcast(&completion_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                LOG_DEBUG("Non-root rank " << rank << " received completion signal");
            }
            MPI_Finalize();
            return 0;
        }

        if (transformer_pipeline && model_weights)
        {
            // Model inference workflow
            LOG_INFO("Setting up transformer inference pipeline...");

            // Test inference with simple token sequence
            std::vector<int> input_tokens = {1, 2, 3, 4, 5}; // Simple test sequence
            LOG_INFO("Running inference on " << input_tokens.size() << " tokens");

            try
            {
                std::shared_ptr<TensorBase> output;
                bool success = transformer_pipeline->execute(input_tokens, *model_weights, output);

                if (success && rank == 0)
                {
                    LOG_INFO("Inference completed successfully!");
                    LOG_INFO("Output shape: [" << output->shape()[0] << ", " << output->shape()[1] << "]");

                    // Print first few output values
                    const float *output_data = output->data();
                    LOG_INFO("First 10 output values:");
                    for (int i = 0; i < std::min(10, (int)output->shape()[1]); ++i)
                    {
                        LOG_INFO("  [" << i << "] = " << output_data[i]);
                    }
                }
                else if (!success)
                {
                    LOG_ERROR("Transformer pipeline execution failed");
                    MPI_Finalize();
                    return 1;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Exception during inference: " << e.what());
                MPI_Finalize();
                return 1;
            }
        }
        else if (model_loader)
        {
            // Model loaded but no transformer pipeline (fallback)
            LOG_INFO("Model loaded but transformer pipeline not initialized");
            LOG_WARN("Running matrix multiplication demo instead");

            int matrix_size = params.m;
            LOG_INFO("Running matrix multiplication benchmark: " << matrix_size << "x" << matrix_size);

            auto node = std::make_shared<MatMulNode>(
                "model_matmul_demo",
                matrix_size, matrix_size, matrix_size);

            graph.addNode(node);
        }
        else
        {
            // Legacy COSMA benchmark mode
            LOG_INFO("Running legacy COSMA benchmark mode");

            auto node = std::make_shared<MatMulNode>(
                "cosma_benchmark",
                params.m, params.n, params.k);

            graph.addNode(node);
        }

        // 7. Execute compute graph (if not already executed by transformer pipeline)
        if (!transformer_pipeline || !model_weights)
        {
            LOG_INFO("Executing compute graph...");

            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < params.num_repeat; ++i)
            {
                bool success = graph.execute();

                if (!success)
                {
                    LOG_ERROR("Compute graph execution failed on iteration " << i);
                    MPI_Finalize();
                    return 1;
                }

                if (rank == 0 && params.log_level >= LogLevel::VERBOSITY_DEBUG)
                {
                    LOG_DEBUG("Completed iteration " << (i + 1) << "/" << params.num_repeat);
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // 8. Report performance results
            if (rank == 0)
            {
                LOG_INFO("=== Performance Results ===");
                LOG_INFO("Total execution time: " << duration.count() << " ms");
                LOG_INFO("Average time per iteration: " << (duration.count() / params.num_repeat) << " ms");

                if (params.profile_kernels)
                {
                    LOG_INFO("=== Kernel Performance ===");
                    LOG_INFO("Kernel profiling not implemented yet");
                }

                // Calculate GFLOPS for matrix multiplication
                if (!model_loader)
                {
                    long long ops = 2LL * params.m * params.n * params.k * params.num_repeat;
                    double gflops = (double)ops / (duration.count() * 1e6);
                    LOG_INFO("Achieved performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS");
                }
            }
        }
        else
        {
            LOG_INFO("Transformer inference completed - skipping compute graph execution");
        }

        // 9. Validation if requested
        if (params.validate_results && rank == 0)
        {
            LOG_INFO("Running result validation...");
            // TODO: Add result validation logic
            LOG_INFO("Validation completed successfully");
        }

        LOG_INFO("Llaminar execution completed successfully");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in main: " << e.what());
        MPI_Finalize();
        return 1;
    }
    catch (...)
    {
        LOG_ERROR("Unknown exception in main");
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}