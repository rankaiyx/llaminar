/**
 * @file Test__Sampler.cpp
 * @brief Unit tests for Sampler class
 * @author David Sanftenberg
 * @date 2025-11-07
 *
 * Tests for token sampling strategies including:
 * - Greedy sampling (argmax)
 * - Temperature scaling
 * - Top-k sampling
 * - Top-p (nucleus) sampling
 * - Seed reproducibility
 * - Edge cases (empty logits, single token, uniform distribution)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "../../../src/v2/utils/Sampler.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Test fixture for Sampler tests
     */
    class SamplerTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create sampler with fixed seed for deterministic tests
            sampler_ = std::make_unique<Sampler>(12345);

            // Standard logits for testing (5 tokens)
            // Token 2 has highest logit (3.0)
            standard_logits_ = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f};

            // Uniform logits (all same value)
            uniform_logits_ = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

            // Single peak logits (one clearly dominant token)
            peaked_logits_ = {0.1f, 0.2f, 10.0f, 0.1f, 0.2f};
        }

        void TearDown() override
        {
            sampler_.reset();
        }

        std::unique_ptr<Sampler> sampler_;
        std::vector<float> standard_logits_;
        std::vector<float> uniform_logits_;
        std::vector<float> peaked_logits_;
    };

    // =============================================================================
    // Basic Functionality Tests
    // =============================================================================

    TEST_F(SamplerTest, SamplerCreation)
    {
        // Should construct without errors
        EXPECT_NE(sampler_, nullptr);
    }

    TEST_F(SamplerTest, SamplerWithSeed)
    {
        // Create two samplers with same seed
        Sampler sampler1(42);
        Sampler sampler2(42);

        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Should produce identical sequences
        for (int i = 0; i < 10; ++i)
        {
            SamplingParams params;
            params.temperature = 0.8f;

            int token1 = sampler1.sample(logits, params);
            int token2 = sampler2.sample(logits, params);

            EXPECT_EQ(token1, token2) << "Iteration " << i;
        }
    }

    // =============================================================================
    // Greedy Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, GreedySampling_StandardLogits)
    {
        int token = sampler_->sample_greedy(standard_logits_);

        // Should select token with highest logit (index 2, value 3.0)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, GreedySampling_UniformLogits)
    {
        int token = sampler_->sample_greedy(uniform_logits_);

        // With uniform logits, should select first occurrence of max
        EXPECT_EQ(token, 0);
    }

    TEST_F(SamplerTest, GreedySampling_SingleToken)
    {
        std::vector<float> single = {5.0f};
        int token = sampler_->sample_greedy(single);

        EXPECT_EQ(token, 0);
    }

    TEST_F(SamplerTest, GreedySampling_Deterministic)
    {
        // Greedy sampling should always return same result
        int first = sampler_->sample_greedy(standard_logits_);

        for (int i = 0; i < 10; ++i)
        {
            int token = sampler_->sample_greedy(standard_logits_);
            EXPECT_EQ(token, first) << "Iteration " << i;
        }
    }

    // =============================================================================
    // Temperature Scaling Tests
    // =============================================================================

    TEST_F(SamplerTest, TemperatureZero_IsGreedy)
    {
        // Temperature 0 should behave like greedy sampling
        int token = sampler_->sample_temperature(standard_logits_, 0.0f);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, TemperatureOne_Standard)
    {
        // Temperature 1.0 is standard softmax (no scaling)
        // With fixed seed, should produce consistent result
        int token = sampler_->sample_temperature(standard_logits_, 1.0f);

        // Should be valid token index
        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TemperatureHigh_MoreRandom)
    {
        // High temperature should allow more diversity
        // Run multiple samples and check we don't always get the peak
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_temperature(peaked_logits_, 2.0f);
        }

        // Count unique tokens sampled
        std::sort(samples.begin(), samples.end());
        auto unique_end = std::unique(samples.begin(), samples.end());
        int num_unique = std::distance(samples.begin(), unique_end);

        // With high temperature on peaked logits, should sample multiple tokens
        EXPECT_GT(num_unique, 1) << "High temperature should allow diversity";
    }

    TEST_F(SamplerTest, TemperatureLow_LessRandom)
    {
        // Low temperature should strongly favor the peak
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_temperature(peaked_logits_, 0.1f);
        }

        // Count how often we sample the peak (token 2)
        int peak_count = std::count(samples.begin(), samples.end(), 2);

        // With very low temperature, should almost always sample the peak
        EXPECT_GT(peak_count, 90) << "Low temperature should favor peak token";
    }

    // =============================================================================
    // Top-k Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, TopK_K1_IsGreedy)
    {
        // Top-k with k=1 should be equivalent to greedy
        int token = sampler_->sample_top_k(standard_logits_, 1, 1.0f);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, TopK_K2_OnlyTopTokens)
    {
        // Top-k with k=2 should only sample from top 2 tokens
        // For standard_logits: top 2 are indices 2 (3.0) and 1 (2.0)
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(standard_logits_, 2, 1.0f);
        }

        // All samples should be either token 1 or 2
        for (int token : samples)
        {
            EXPECT_TRUE(token == 1 || token == 2)
                << "Top-k=2 sampled token " << token << ", expected 1 or 2";
        }
    }

    TEST_F(SamplerTest, TopK_KLargerThanVocab)
    {
        // If k > vocab_size, should consider all tokens (standard sampling)
        int token = sampler_->sample_top_k(standard_logits_, 100, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TopK_WithTemperature)
    {
        // Top-k with temperature should apply temp scaling before filtering
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(peaked_logits_, 3, 0.5f);
        }

        // Should only sample from top 3 tokens
        for (int token : samples)
        {
            EXPECT_LT(token, 5) << "Invalid token ID";
        }
    }

    // =============================================================================
    // Top-p (Nucleus) Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, TopP_P1_AllTokens)
    {
        // Top-p with p=1.0 should consider all tokens
        int token = sampler_->sample_top_p(standard_logits_, 1.0f, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TopP_SmallP_FewTokens)
    {
        // Small p should only consider a few high-probability tokens
        // For peaked_logits, token 2 has very high probability
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_p(peaked_logits_, 0.1f, 1.0f);
        }

        // Count unique tokens
        std::sort(samples.begin(), samples.end());
        auto unique_end = std::unique(samples.begin(), samples.end());
        int num_unique = std::distance(samples.begin(), unique_end);

        // With very small p on peaked distribution, should sample very few tokens
        EXPECT_LE(num_unique, 3) << "Small top-p should limit diversity";
    }

    TEST_F(SamplerTest, TopP_WithTemperature)
    {
        // Top-p with temperature should apply temp scaling before filtering
        int token = sampler_->sample_top_p(standard_logits_, 0.9f, 0.8f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    // =============================================================================
    // Unified sample() Interface Tests
    // =============================================================================

    TEST_F(SamplerTest, Sample_GreedyMode)
    {
        SamplingParams params;
        params.temperature = 0.0f; // Greedy

        int token = sampler_->sample(standard_logits_, params);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, Sample_TopKMode)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 2;

        // Sample multiple times
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample(standard_logits_, params);
        }

        // All samples should be from top 2 tokens (indices 1 or 2)
        for (int token : samples)
        {
            EXPECT_TRUE(token == 1 || token == 2);
        }
    }

    TEST_F(SamplerTest, Sample_TopPMode)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_p = 0.5f;

        int token = sampler_->sample(standard_logits_, params);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, Sample_CombinedTopKTopP)
    {
        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 3;
        params.top_p = 0.9f;

        int token = sampler_->sample(standard_logits_, params);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    // =============================================================================
    // Seed Reproducibility Tests
    // =============================================================================

    TEST_F(SamplerTest, SeedReproducibility_SameSeed)
    {
        // Two samplers with same seed should produce identical sequences
        Sampler sampler1(12345);
        Sampler sampler2(12345);

        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 40;

        for (int i = 0; i < 20; ++i)
        {
            int token1 = sampler1.sample(standard_logits_, params);
            int token2 = sampler2.sample(standard_logits_, params);

            EXPECT_EQ(token1, token2) << "Iteration " << i;
        }
    }

    TEST_F(SamplerTest, SeedReproducibility_SetSeed)
    {
        Sampler sampler(0); // Start with random seed

        // Set fixed seed
        sampler.set_seed(42);

        SamplingParams params;
        params.temperature = 1.0f;

        // Record first sequence
        std::vector<int> sequence1;
        for (int i = 0; i < 10; ++i)
        {
            sequence1.push_back(sampler.sample(standard_logits_, params));
        }

        // Reset seed and generate again
        sampler.set_seed(42);
        std::vector<int> sequence2;
        for (int i = 0; i < 10; ++i)
        {
            sequence2.push_back(sampler.sample(standard_logits_, params));
        }

        // Sequences should match
        EXPECT_EQ(sequence1, sequence2);
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    TEST_F(SamplerTest, EdgeCase_SingleToken)
    {
        std::vector<float> single = {5.0f};

        // Greedy
        EXPECT_EQ(sampler_->sample_greedy(single), 0);

        // Temperature
        EXPECT_EQ(sampler_->sample_temperature(single, 1.0f), 0);

        // Top-k
        EXPECT_EQ(sampler_->sample_top_k(single, 1, 1.0f), 0);

        // Top-p
        EXPECT_EQ(sampler_->sample_top_p(single, 0.5f, 1.0f), 0);
    }

    TEST_F(SamplerTest, EdgeCase_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};

        // All logits equal, should select first token
        int token = sampler_->sample_greedy(zeros);
        EXPECT_EQ(token, 0);

        // Temperature sampling should still work (uniform distribution)
        token = sampler_->sample_temperature(zeros, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
    }

    TEST_F(SamplerTest, EdgeCase_AllSameValue)
    {
        // Uniform distribution should work correctly
        int token = sampler_->sample_temperature(uniform_logits_, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(uniform_logits_.size()));
    }

    TEST_F(SamplerTest, EdgeCase_NegativeLogits)
    {
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};

        // Greedy should still select max (index 2, value -1.0)
        int token = sampler_->sample_greedy(negative);
        EXPECT_EQ(token, 2);

        // Temperature sampling should work
        token = sampler_->sample_temperature(negative, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
    }

    TEST_F(SamplerTest, EdgeCase_ExtremeLogits)
    {
        std::vector<float> extreme = {-1000.0f, -1000.0f, 100.0f, -1000.0f};

        // With extreme difference, greedy should select the peak
        int token = sampler_->sample_greedy(extreme);
        EXPECT_EQ(token, 2);

        // Even with temperature, should almost always sample the peak
        Sampler sampler(42);
        for (int i = 0; i < 10; ++i)
        {
            token = sampler.sample_temperature(extreme, 1.0f);
            EXPECT_EQ(token, 2) << "Extreme logit difference should dominate";
        }
    }

    // =============================================================================
    // SamplingParams Tests
    // =============================================================================

    TEST_F(SamplerTest, SamplingParams_IsGreedy_TemperatureZero)
    {
        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_TRUE(params.is_greedy());
    }

    TEST_F(SamplerTest, SamplingParams_IsGreedy_TopK1)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 1;
        params.top_p = 1.0f;

        EXPECT_TRUE(params.is_greedy());
    }

    TEST_F(SamplerTest, SamplingParams_NotGreedy)
    {
        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 40;
        params.top_p = 0.95f;

        EXPECT_FALSE(params.is_greedy());
    }

    // =============================================================================
    // Large Vocabulary Tests
    // =============================================================================

    TEST_F(SamplerTest, LargeVocabulary_Greedy)
    {
        // Simulate large vocabulary (e.g., 50k tokens)
        std::vector<float> large_logits(50000, 0.0f);
        large_logits[12345] = 10.0f; // Peak at specific token

        int token = sampler_->sample_greedy(large_logits);
        EXPECT_EQ(token, 12345);
    }

    TEST_F(SamplerTest, LargeVocabulary_TopK)
    {
        // Large vocabulary with top-k
        std::vector<float> large_logits(50000, 0.0f);
        large_logits[100] = 5.0f;
        large_logits[200] = 4.0f;
        large_logits[300] = 3.0f;

        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(large_logits, 3, 1.0f);
        }

        // Should only sample from tokens 100, 200, 300
        for (int token : samples)
        {
            EXPECT_TRUE(token == 100 || token == 200 || token == 300)
                << "Large vocab top-k sampled unexpected token " << token;
        }
    }

} // anonymous namespace
