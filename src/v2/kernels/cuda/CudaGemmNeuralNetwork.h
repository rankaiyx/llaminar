/**
 * @file CudaGemmNeuralNetwork.h
 * @brief ONNX Runtime neural network ranking model for CUDA GEMM config selection
 *
 * This module wraps an ONNX neural network trained to RANK CUDA GEMM configurations.
 * The NN provides perfect ranking accuracy on unseen shapes without hardware profiling.
 *
 * Architecture (November 3, 2025): 101 features → 256 → 128 → 64 → 1 (ranking score)
 * Training:  R² = 0.9981 with profiling features
 * Validation: 100% top-30 hit rate on 26 unseen test cases (WITHOUT profiling!)
 * Model Size: ~265 KB (.onnx file)
 * Inference: ~10μs per prediction
 *
 * RANKING PERFORMANCE (What Matters):
 * - Perfect generalization: 100% top-1, top-5, top-10, top-30 hit rates
 * - Tested on: Qwen 1.5B-72B, DeepSeek 671B, OddBatch, OddDim
 * - 97,344 validation configurations across 26 diverse test cases
 * - Rank correlation: Kendall's tau = 0.35, Spearman's rho = 0.50
 *
 * ⚠️  CRITICAL: This is a RANKING model, NOT a performance predictor!
 * - Absolute values are meaningless (trained with profiling, inference without)
 * - Relative ordering is perfect (use for ranking only)
 * - DO NOT interpret output as GFLOPS (it's just a ranking score)
 *
 * Feature Categories (101 total):
 * - 73 base features: m, n, k, tile sizes, thread config, ratios, work metrics, etc.
 * - 28 zero-padded features: Profiling features used during training, zero at inference
 *
 * Zero-Padding Strategy:
 * - Training: 84 base + 17 profiling features (101 total)
 * - Inference: 73 base + 28 zeros (profiling features unavailable)
 * - Result: Perfect ranking despite wrong absolute predictions
 *
 * Usage:
 * ```cpp
 * auto &nn = CudaGemmNeuralNetwork::instance();
 * double ranking_score = nn.rankConfig(config, m, n, k);  // Higher = better
 * // DO NOT use as GFLOPS! Only for relative ranking!
 * ```
 *
 * @author David Sanftenberg
 * @date November 3, 2025 (Refactored as pure ranking model)
 */

#pragma once

#ifdef HAVE_ONNX_RUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include "CudaGemmConfig.h"

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief ONNX-based neural network for CUDA GEMM performance prediction
         *
         * Singleton class that loads an ONNX model and provides predictions.
         */
        class CudaGemmNeuralNetwork
        {
        public:
            /**
             * @brief Get singleton instance
             */
            static CudaGemmNeuralNetwork &instance();

            /**
             * @brief Rank a GEMM configuration (higher score = better performance)
             *
             * ⚠️  CRITICAL: This returns a RANKING SCORE, not GFLOPS!
             * - Use only for relative comparison (config A vs config B)
             * - Absolute value is meaningless
             * - Higher score means better predicted performance
             *
             * @param config CUDA GEMM configuration
             * @param m Number of rows (batch size)
             * @param n Number of columns
             * @param k Inner dimension
             * @return Ranking score (higher = better, absolute value meaningless)
             */
            double rankConfig(const CudaGemmConfig &config, int m, int n, int k);

            /**
             * @brief DEPRECATED: Use rankConfig() instead
             * @deprecated This function name suggests absolute prediction (wrong!)
             */
            [[deprecated("Use rankConfig() - this is a ranking model, not a predictor")]]
            double predict(const CudaGemmConfig &config, int m, int n, int k)
            {
                return rankConfig(config, m, n, k);
            }

            /**
             * @brief Check if neural network is initialized
             */
            bool isInitialized() const { return initialized_; }

            /**
             * @brief Get model path
             */
            std::string getModelPath() const { return model_path_; }

        private:
            CudaGemmNeuralNetwork();
            ~CudaGemmNeuralNetwork() = default;

            // Non-copyable, non-movable (singleton)
            CudaGemmNeuralNetwork(const CudaGemmNeuralNetwork &) = delete;
            CudaGemmNeuralNetwork &operator=(const CudaGemmNeuralNetwork &) = delete;

            /**
             * @brief Initialize ONNX Runtime and load model
             */
            void initialize();

            /**
             * @brief Extract features from config and problem size
             *
             * CRITICAL: Feature extraction MUST match Python training exactly!
             * See: python/validate_heuristic.py (engineer_features function)
             *
             * @return 101-element feature vector (84 base + 17 zero-padded profiling)
             */
            std::array<float, 101> extractFeatures(const CudaGemmConfig &config, int m, int n, int k);

            /**
             * @brief Load scaler parameters (mean, scale) from .txt file
             * @return true if successful, false otherwise
             */
            bool loadScalerParameters();

            bool initialized_ = false;
            std::string model_path_;
            std::string scaler_path_;

            // ONNX Runtime components
            Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "CudaGemmNN"};
            Ort::SessionOptions session_options_;
            std::unique_ptr<Ort::Session> session_;

            // Feature scaler parameters (from StandardScaler in Python)
            std::array<float, 101> feature_mean_;  // Mean of each feature
            std::array<float, 101> feature_scale_; // Std dev of each feature

            // Input/output tensor info (store as strings to keep memory alive)
            std::string input_name_storage_;
            std::string output_name_storage_;
            std::vector<const char *> input_names_;
            std::vector<const char *> output_names_;
        };

    } // namespace cuda
} // namespace llaminar2

#endif // HAVE_ONNX_RUNTIME
