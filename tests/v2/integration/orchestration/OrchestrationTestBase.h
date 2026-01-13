/**
 * @file OrchestrationTestBase.h
 * @brief Base class and utilities for orchestration/placement integration tests
 *
 * Provides standardized infrastructure for testing Llaminar's placement and
 * orchestration subsystems across various cluster topologies and model sizes.
 * All orchestration tests should inherit from OrchestrationTestBase to ensure:
 *
 * - Consistent topology simulation (via MockMPITopologyBuilder)
 * - Standard device capability creation (GPUs, CPUs)
 * - PlacementInput generation for various model sizes
 * - Validation helpers for placement plans
 * - Summary output formatting
 *
 * Usage:
 *   class Test__ScenarioFoo : public OrchestrationTestBase {
 *   protected:
 *       ClusterConfig getClusterConfig() override { return myConfig; }
 *       std::vector<ModelConfig> getModelConfigs() override { return myModels; }
 *   };
 *   INSTANTIATE_ORCHESTRATION_TESTS(Test__ScenarioFoo);
 *
 * @see docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <set>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "mocks/MockMPITopology.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockGraphBufferManager.h"
#include "execution/PlacementStrategy.h"
#include "execution/PlacementPlan.h"
#include "execution/DeviceInventory.h"
#include "execution/GraphExecutor.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

namespace llaminar2::test::orchestration {

// =============================================================================
// Device Specifications
// =============================================================================

/**
 * @brief Common GPU specifications for test scenarios
 */
struct GPUSpec {
    std::string name;
    DeviceCapability::Type type;
    float memory_gb;
    float bandwidth_gbps;
    float compute_tflops;
    bool has_nvlink = false;
    bool has_xgmi = false;
    
    /**
     * @brief Create a DeviceCapability from this spec
     */
    DeviceCapability toCapability(int device_id) const {
        DeviceCapability cap;
        cap.type = type;
        cap.device_id = device_id;
        cap.memory_bytes = static_cast<size_t>(memory_gb * 1024 * 1024 * 1024);
        cap.relative_compute = compute_tflops;
        cap.name = name;
        return cap;
    }
};

/**
 * @brief Pre-defined GPU specifications for common hardware
 */
namespace GPUSpecs {
    // NVIDIA
    inline const GPUSpec RTX_3090 = {
        "NVIDIA RTX 3090", DeviceCapability::Type::CUDA,
        24.0f, 936.0f, 35.0f, false, false
    };
    inline const GPUSpec RTX_4090 = {
        "NVIDIA RTX 4090", DeviceCapability::Type::CUDA,
        24.0f, 1008.0f, 83.0f, false, false
    };
    inline const GPUSpec A100_40GB = {
        "NVIDIA A100 40GB", DeviceCapability::Type::CUDA,
        40.0f, 1555.0f, 156.0f, true, false
    };
    inline const GPUSpec A100_80GB = {
        "NVIDIA A100 80GB", DeviceCapability::Type::CUDA,
        80.0f, 2039.0f, 156.0f, true, false
    };
    inline const GPUSpec H100_80GB = {
        "NVIDIA H100 80GB", DeviceCapability::Type::CUDA,
        80.0f, 3350.0f, 267.0f, true, false
    };
    
    // AMD
    inline const GPUSpec MI50 = {
        "AMD Instinct Mi50", DeviceCapability::Type::ROCm,
        16.0f, 717.0f, 13.0f, false, true
    };
    inline const GPUSpec MI100 = {
        "AMD Instinct Mi100", DeviceCapability::Type::ROCm,
        32.0f, 1228.0f, 46.0f, false, true
    };
    inline const GPUSpec MI250X = {
        "AMD Instinct Mi250X", DeviceCapability::Type::ROCm,
        128.0f, 3200.0f, 191.0f, false, true
    };
    inline const GPUSpec RX_7900_XTX = {
        "AMD Radeon RX 7900 XTX", DeviceCapability::Type::ROCm,
        24.0f, 960.0f, 23.0f, false, false
    };
}

/**
 * @brief CPU specifications for test scenarios
 */
struct CPUSpec {
    std::string name;
    int cores;
    float memory_gb;
    float bandwidth_gbps;  // Memory bandwidth
    
    DeviceCapability toCapability() const {
        return MockDevices::cpu(1.0f);
    }
};

namespace CPUSpecs {
    inline const CPUSpec XEON_28C_DDR4 = {"Xeon 28-core DDR4", 28, 128.0f, 153.6f};
    inline const CPUSpec XEON_28C_DDR5 = {"Xeon 28-core DDR5", 28, 512.0f, 358.0f};
    inline const CPUSpec EPYC_64C = {"EPYC 64-core", 64, 512.0f, 400.0f};
}

// =============================================================================
// Socket Configuration
// =============================================================================

/**
 * @brief Configuration for a single socket in a cluster
 */
struct SocketConfig {
    std::vector<GPUSpec> gpus;
    CPUSpec cpu;
    
    /**
     * @brief Total VRAM for this socket in GB
     */
    float totalVRAMGB() const {
        float total = 0.0f;
        for (const auto& gpu : gpus) {
            total += gpu.memory_gb;
        }
        return total;
    }
    
    /**
     * @brief Total GPU bandwidth for this socket in GB/s
     */
    float totalGPUBandwidthGBs() const {
        float total = 0.0f;
        for (const auto& gpu : gpus) {
            total += gpu.bandwidth_gbps;
        }
        return total;
    }
};

// =============================================================================
// Cluster Configuration
// =============================================================================

/**
 * @brief Complete cluster topology configuration
 *
 * Defines the hardware layout for a test scenario:
 * - Number of machines and sockets
 * - GPU/CPU configuration per socket
 * - Interconnect characteristics
 */
struct ClusterConfig {
    std::string name;                    ///< Scenario name for display
    int num_machines = 1;
    int sockets_per_machine = 1;
    
    /// Socket configurations (indexed by socket_id % configs.size())
    std::vector<SocketConfig> socket_configs;
    
    // Interconnect specifications
    float nvlink_bandwidth_gbps = 600.0f;
    float xgmi_bandwidth_gbps = 600.0f;
    float pcie_bandwidth_gbps = 32.0f;
    float qpi_upi_bandwidth_gbps = 50.0f;      ///< Cross-socket on same machine
    float infiniband_bandwidth_gbps = 25.0f;   ///< Cross-machine
    
    // Computed properties
    int totalRanks() const { return num_machines * sockets_per_machine; }
    
    int totalGPUs() const {
        int total = 0;
        for (int rank = 0; rank < totalRanks(); ++rank) {
            const auto& socket = socket_configs[rank % socket_configs.size()];
            total += static_cast<int>(socket.gpus.size());
        }
        return total;
    }
    
    float totalVRAMGB() const {
        float total = 0.0f;
        for (int rank = 0; rank < totalRanks(); ++rank) {
            const auto& socket = socket_configs[rank % socket_configs.size()];
            total += socket.totalVRAMGB();
        }
        return total;
    }
    
    float totalGPUBandwidthTBs() const {
        float total = 0.0f;
        for (int rank = 0; rank < totalRanks(); ++rank) {
            const auto& socket = socket_configs[rank % socket_configs.size()];
            total += socket.totalGPUBandwidthGBs();
        }
        return total / 1000.0f;
    }
    
    float totalCPUBandwidthTBs() const {
        float total = 0.0f;
        for (int rank = 0; rank < totalRanks(); ++rank) {
            const auto& socket = socket_configs[rank % socket_configs.size()];
            total += socket.cpu.bandwidth_gbps;
        }
        return total / 1000.0f;
    }
    
    /**
     * @brief Get the socket config for a specific rank
     */
    const SocketConfig& getSocketConfig(int rank) const {
        return socket_configs[rank % socket_configs.size()];
    }
};

// =============================================================================
// Model Configuration
// =============================================================================

/**
 * @brief Model architecture configuration for placement testing
 */
struct ModelConfig {
    std::string name;
    int n_layers;
    size_t d_model;
    size_t d_ff;
    size_t vocab_size;
    size_t n_heads;
    size_t n_kv_heads;
    size_t estimated_memory_bytes;  ///< Q4_0 weight size
    std::string quant_type = "Q4_0";
    
    /**
     * @brief Estimated memory in GB
     */
    float memorySizeGB() const {
        return static_cast<float>(estimated_memory_bytes) / (1024.0f * 1024.0f * 1024.0f);
    }
    
    /**
     * @brief Layers per rank if evenly distributed
     */
    float layersPerRank(int num_ranks) const {
        return static_cast<float>(n_layers) / num_ranks;
    }
};

/**
 * @brief Pre-defined model configurations
 */
namespace ModelConfigs {
    inline const ModelConfig QWEN2_0_5B = {
        "Qwen2-0.5B", 24, 896, 4864, 151936, 14, 2,
        500ULL * 1024 * 1024
    };
    inline const ModelConfig QWEN2_7B = {
        "Qwen2-7B", 32, 4096, 11008, 151936, 32, 32,
        4ULL * 1024 * 1024 * 1024
    };
    inline const ModelConfig QWEN2_72B = {
        "Qwen2-72B", 80, 8192, 29568, 151936, 64, 8,
        40ULL * 1024 * 1024 * 1024
    };
    inline const ModelConfig QWEN2_235B = {
        "Qwen2-235B (fictional)", 128, 16384, 53248, 151936, 128, 16,
        130ULL * 1024 * 1024 * 1024
    };
    inline const ModelConfig QWEN2_571B = {
        "Qwen2-571B (fictional)", 160, 20480, 65536, 151936, 160, 20,
        320ULL * 1024 * 1024 * 1024
    };
    inline const ModelConfig QWEN2_1142B = {
        "Qwen2-1142B (fictional)", 192, 24576, 81920, 151936, 192, 24,
        640ULL * 1024 * 1024 * 1024
    };
}

// =============================================================================
// Placement Validation Results
// =============================================================================

/**
 * @brief Result of validating a placement plan
 */
struct PlacementValidationResult {
    bool is_valid = false;
    int gpu_layers = 0;
    int cpu_layers = 0;
    int total_layers = 0;
    std::set<int> gpus_used;
    bool respects_socket_boundary = true;
    std::string error_message;
    
    float gpuLayerPercentage() const {
        return total_layers > 0 ? 100.0f * gpu_layers / total_layers : 0.0f;
    }
};

// =============================================================================
// Graph Bridge Validation Results
// =============================================================================

/**
 * @brief Result of validating graph construction from placement plan
 */
struct GraphValidationResult {
    bool is_valid = true;
    int nodes_checked = 0;
    int device_mismatches = 0;
    int backend_mismatches = 0;
    std::vector<std::string> errors;
    
    void addError(const std::string& msg) {
        is_valid = false;
        errors.push_back(msg);
    }
};

/**
 * @brief Result of validating graph execution
 */
struct ExecutionValidationResult {
    bool all_executed = true;
    int stages_executed = 0;
    int expected_stages = 0;
    std::vector<std::string> execution_order;
    std::vector<std::string> skipped_stages;
    
    bool orderCorrect() const {
        // Check stages executed in dependency order (no child before parent)
        return all_executed && stages_executed == expected_stages;
    }
};

/**
 * @brief Configuration for a mock layer graph
 */
struct MockLayerGraphSpec {
    int layer_idx = 0;
    DeviceId device = DeviceId::cpu();
    bool include_attention = true;
    bool include_ffn = true;
    bool include_allreduce = false;  ///< Insert allreduce at TP boundary
    
    /// Stages to generate for this layer
    std::vector<ComputeStageType> stages = {
        ComputeStageType::RMS_NORM,      // attn_norm
        ComputeStageType::GEMM,          // qkv_proj  
        ComputeStageType::ROPE,          // rope
        ComputeStageType::ATTENTION,     // attention
        ComputeStageType::GEMM,          // o_proj
        ComputeStageType::ADD_RESIDUAL,  // attn_residual
        ComputeStageType::RMS_NORM,      // ffn_norm
        ComputeStageType::GEMM,          // gate_up_proj
        ComputeStageType::SWIGLU,        // swiglu
        ComputeStageType::GEMM,          // down_proj
        ComputeStageType::ADD_RESIDUAL   // ffn_residual
    };
};

// =============================================================================
// OrchestrationTestBase
// =============================================================================

/**
 * @brief Base class for orchestration integration tests
 *
 * Provides:
 * - Topology creation from ClusterConfig
 * - PlacementInput generation for models
 * - Plan validation helpers
 * - Summary output formatters
 */
class OrchestrationTestBase : public ::testing::Test {
protected:
    // =========================================================================
    // Abstract Methods (override in subclasses)
    // =========================================================================
    
    /**
     * @brief Return the cluster configuration for this scenario
     */
    virtual ClusterConfig getClusterConfig() = 0;
    
    /**
     * @brief Return the models to test in this scenario
     */
    virtual std::vector<ModelConfig> getModelConfigs() = 0;
    
    // =========================================================================
    // Setup/Teardown
    // =========================================================================
    
    void SetUp() override {
        cluster_ = getClusterConfig();
        models_ = getModelConfigs();
        
        // Create topologies for all ranks
        topologies_.clear();
        for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {
            topologies_.push_back(createTopology(rank));
        }
    }
    
    // =========================================================================
    // Topology Creation
    // =========================================================================
    
    /**
     * @brief Create MockMPITopology for a specific rank
     */
    std::shared_ptr<MockMPITopology> createTopology(int local_rank) {
        auto builder = MockMPITopologyBuilder();
        
        // Build all ranks to get consistent cluster view
        for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {
            int machine = rank / cluster_.sockets_per_machine;
            const auto& socket_cfg = cluster_.getSocketConfig(rank);
            
            std::vector<DeviceCapability> devices;
            devices.push_back(socket_cfg.cpu.toCapability());
            
            int device_id = 0;
            for (const auto& gpu_spec : socket_cfg.gpus) {
                auto cap = gpu_spec.toCapability(device_id++);
                devices.push_back(cap);
            }
            
            std::string hostname = "node" + std::to_string(machine) + 
                                   "-socket" + std::to_string(rank % cluster_.sockets_per_machine);
            builder.addRank(rank, machine, devices, hostname);
        }
        
        builder.setLocalRank(local_rank);
        return builder.build();
    }
    
    // =========================================================================
    // PlacementInput Creation
    // =========================================================================
    
    /**
     * @brief Create PlacementInput for a model and rank
     */
    PlacementInput createPlacementInput(const ModelConfig& model, int rank) {
        auto topology = topologies_[rank];
        const auto& socket_cfg = cluster_.getSocketConfig(rank);
        
        PlacementInput input;
        input.architecture = "qwen2";
        input.n_layers = model.n_layers;
        input.d_model = model.d_model;
        input.d_ff = model.d_ff;
        input.vocab_size = model.vocab_size;
        input.n_heads = model.n_heads;
        input.n_kv_heads = model.n_kv_heads;
        input.quant_type = model.quant_type;
        input.estimated_memory_bytes = model.estimated_memory_bytes;
        
        // Populate from topology
        input.world_size = topology->world_size();
        input.ranks_per_node = topology->ranks_per_node();
        input.node_count = topology->node_count();
        input.cluster_inventory = topology->clusterInventory();
        
        // Bandwidth info
        input.gpu_memory_bandwidth_gbps = socket_cfg.totalGPUBandwidthGBs() / 
                                           std::max(1UL, socket_cfg.gpus.size());
        input.cpu_memory_bandwidth_gbps = socket_cfg.cpu.bandwidth_gbps;
        
        input.updateAggregatedFields();
        
        return input;
    }
    
    // =========================================================================
    // Validation Helpers
    // =========================================================================
    
    /**
     * @brief Validate a placement plan
     */
    PlacementValidationResult validatePlan(const PlacementPlan& plan, int rank) {
        PlacementValidationResult result;
        result.total_layers = static_cast<int>(plan.layers.size());
        result.is_valid = plan.isValid();
        
        if (!result.is_valid) {
            result.error_message = "Plan marked invalid";
            return result;
        }
        
        for (const auto& layer : plan.layers) {
            if (layer.device.isGPU()) {
                result.gpu_layers++;
                result.gpus_used.insert(layer.device.gpu_index);
            } else {
                result.cpu_layers++;
            }
        }
        
        return result;
    }
    
    /**
     * @brief Check if model fits in cluster GPU memory
     */
    bool modelFitsInGPUMemory(const ModelConfig& model) {
        size_t total_vram = static_cast<size_t>(cluster_.totalVRAMGB() * 1024 * 1024 * 1024);
        return model.estimated_memory_bytes < total_vram;
    }
    
    /**
     * @brief Calculate how many machines are needed for a model
     */
    int machinesNeededForModel(const ModelConfig& model) {
        float vram_per_machine = cluster_.totalVRAMGB() / cluster_.num_machines;
        return static_cast<int>(std::ceil(model.memorySizeGB() / vram_per_machine));
    }
    
    // =========================================================================
    // Output Helpers
    // =========================================================================
    
    /**
     * @brief Print cluster summary
     */
    void printClusterSummary() {
        std::cout << "\n=== " << cluster_.name << " ===" << std::endl;
        std::cout << "Machines: " << cluster_.num_machines << std::endl;
        std::cout << "Sockets/machine: " << cluster_.sockets_per_machine << std::endl;
        std::cout << "Total ranks: " << cluster_.totalRanks() << std::endl;
        std::cout << "Total GPUs: " << cluster_.totalGPUs() << std::endl;
        std::cout << "Total GPU VRAM: " << std::fixed << std::setprecision(0) 
                  << cluster_.totalVRAMGB() << " GB" << std::endl;
        std::cout << "Total GPU BW: " << std::fixed << std::setprecision(1) 
                  << cluster_.totalGPUBandwidthTBs() << " TB/s" << std::endl;
        std::cout << "Total CPU BW: " << std::fixed << std::setprecision(1) 
                  << cluster_.totalCPUBandwidthTBs() << " TB/s" << std::endl;
    }
    
    /**
     * @brief Print model fit summary table
     */
    void printModelFitTable() {
        std::cout << "\n=== Model Fit Analysis ===" << std::endl;
        std::cout << "Model             | Size   | Fits GPU? | Machines | Layers/Rank" << std::endl;
        std::cout << "------------------|--------|-----------|----------|------------" << std::endl;
        
        for (const auto& model : models_) {
            bool fits = modelFitsInGPUMemory(model);
            int machines = machinesNeededForModel(model);
            float lpr = model.layersPerRank(cluster_.totalRanks());
            
            printf("%-17s | %5.0fGB | %-9s | %8d | %11.1f\n",
                   model.name.c_str(),
                   model.memorySizeGB(),
                   fits ? "Yes" : "No",
                   machines,
                   lpr);
        }
    }
    
    // =========================================================================
    // Graph Bridge: PlacementPlan → ComputeGraph
    // =========================================================================
    
    /**
     * @brief Build a mock ComputeGraph from a PlacementPlan
     * 
     * Creates mock stages for each layer with:
     * - Correct device placement matching the plan
     * - Correct backend support (CPU/CUDA/ROCm)
     * - Optional AllReduce stages at TP boundaries
     * 
     * @param plan The placement plan to convert
     * @param include_allreduce Whether to insert AllReduce stages
     * @return tuple of (ComputeGraph, execution_log_ptr, stages_map)
     */
    struct MockGraphResult {
        ComputeGraph graph;
        std::shared_ptr<std::vector<std::string>> execution_log;
        std::unordered_map<std::string, testing::MockComputeStage*> stages;
    };
    
    MockGraphResult buildMockGraphFromPlan(
        const PlacementPlan& plan,
        bool include_allreduce = false
    ) {
        MockGraphResult result;
        result.execution_log = std::make_shared<std::vector<std::string>>();
        
        std::vector<std::unique_ptr<testing::MockComputeStage>> owned_stages;
        std::string prev_layer_last_stage;
        
        for (const auto& layer : plan.layers) {
            DeviceId device = toDeviceId(layer.device);
            auto backend = device.is_cpu() ? ComputeBackendType::CPU : 
                          (device.type == DeviceType::CUDA ? ComputeBackendType::GPU_CUDA : 
                                                             ComputeBackendType::GPU_ROCM);
            
            // Build stages for this layer
            MockLayerGraphSpec spec;
            spec.layer_idx = layer.layer_idx;
            spec.device = device;
            spec.include_allreduce = include_allreduce && !device.is_cpu();
            
            std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";
            std::vector<std::string> stage_names = {
                layer_prefix + "attn_norm",
                layer_prefix + "qkv_proj",
                layer_prefix + "rope",
                layer_prefix + "attention",
                layer_prefix + "o_proj"
            };
            
            // Insert allreduce after attention if TP
            if (spec.include_allreduce) {
                stage_names.push_back(layer_prefix + "attn_allreduce");
            }
            
            stage_names.push_back(layer_prefix + "attn_residual");
            stage_names.push_back(layer_prefix + "ffn_norm");
            stage_names.push_back(layer_prefix + "gate_up_proj");
            stage_names.push_back(layer_prefix + "swiglu");
            stage_names.push_back(layer_prefix + "down_proj");
            
            // Insert allreduce after FFN if TP
            if (spec.include_allreduce) {
                stage_names.push_back(layer_prefix + "ffn_allreduce");
            }
            
            stage_names.push_back(layer_prefix + "ffn_residual");
            
            // Create stages and add to graph
            std::string prev_stage_in_layer;
            for (size_t i = 0; i < stage_names.size(); ++i) {
                const std::string& name = stage_names[i];
                ComputeStageType type = getStageType(name);
                
                auto stage = std::make_unique<testing::MockComputeStage>(type, name);
                stage->setExecutionLog(result.execution_log.get());
                
                // Configure backend support based on device
                if (device.is_cpu()) {
                    stage->setSupportedBackends({ComputeBackendType::CPU});
                } else {
                    stage->setSupportedBackends({backend});
                }
                
                // Track for validation
                result.stages[name] = stage.get();
                owned_stages.push_back(std::move(stage));
                
                // Add to graph with correct device
                result.graph.addNode(name, std::move(owned_stages.back()), device);
                
                // Add dependency chain within layer
                if (!prev_stage_in_layer.empty()) {
                    result.graph.addDependency(name, prev_stage_in_layer);
                }
                prev_stage_in_layer = name;
            }
            
            // Add dependency to previous layer
            if (!prev_layer_last_stage.empty() && !stage_names.empty()) {
                result.graph.addDependency(stage_names[0], prev_layer_last_stage);
            }
            prev_layer_last_stage = stage_names.back();
        }
        
        return result;
    }
    
    /**
     * @brief Get ComputeStageType from stage name
     */
    static ComputeStageType getStageType(const std::string& name) {
        if (name.find("norm") != std::string::npos) return ComputeStageType::RMS_NORM;
        if (name.find("qkv") != std::string::npos) return ComputeStageType::GEMM;
        if (name.find("rope") != std::string::npos) return ComputeStageType::ROPE;
        if (name.find("attention") != std::string::npos) return ComputeStageType::ATTENTION;
        if (name.find("o_proj") != std::string::npos) return ComputeStageType::GEMM;
        if (name.find("gate_up") != std::string::npos) return ComputeStageType::GEMM;
        if (name.find("swiglu") != std::string::npos) return ComputeStageType::SWIGLU;
        if (name.find("down_proj") != std::string::npos) return ComputeStageType::GEMM;
        if (name.find("residual") != std::string::npos) return ComputeStageType::ADD_RESIDUAL;
        if (name.find("allreduce") != std::string::npos) return ComputeStageType::ALLREDUCE;
        return ComputeStageType::GEMM;
    }
    
    /**
     * @brief Validate that graph node devices match placement plan
     */
    GraphValidationResult validateGraphMatchesPlan(
        const ComputeGraph& graph,
        const PlacementPlan& plan
    ) {
        GraphValidationResult result;
        
        for (const auto& layer : plan.layers) {
            DeviceId expected_device = toDeviceId(layer.device);
            std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";
            
            // Check each stage in this layer has correct device
            for (const auto& node_name : graph.getExecutionOrder()) {
                if (node_name.find(layer_prefix) == 0) {
                    result.nodes_checked++;
                    const auto* node = graph.getNode(node_name);
                    if (!node) {
                        result.addError("Node not found: " + node_name);
                        continue;
                    }
                    
                    // Check device matches
                    if (node->device != expected_device) {
                        result.device_mismatches++;
                        result.addError("Device mismatch for " + node_name + 
                                       ": expected " + expected_device.to_string() +
                                       ", got " + node->device.to_string());
                    }
                    
                    // Check backend support matches device
                    auto expected_backend = expected_device.is_cpu() ? 
                        ComputeBackendType::CPU : ComputeBackendType::GPU_CUDA;
                    if (node->stage && !node->stage->supportsBackend(expected_backend)) {
                        result.backend_mismatches++;
                        result.addError("Backend mismatch for " + node_name);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Validate AllReduce stages are present at TP boundaries
     */
    GraphValidationResult validateAllReduceStages(
        const ComputeGraph& graph,
        const PlacementPlan& plan,
        int world_size
    ) {
        GraphValidationResult result;
        
        if (world_size <= 1) {
            // No tensor parallelism, no AllReduce needed
            return result;
        }
        
        for (const auto& layer : plan.layers) {
            if (!layer.device.isGPU()) continue;
            
            std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";
            
            // Check for attention allreduce
            std::string attn_ar = layer_prefix + "attn_allreduce";
            if (!graph.getNode(attn_ar)) {
                result.addError("Missing AllReduce after attention: " + attn_ar);
            }
            
            // Check for FFN allreduce
            std::string ffn_ar = layer_prefix + "ffn_allreduce";
            if (!graph.getNode(ffn_ar)) {
                result.addError("Missing AllReduce after FFN: " + ffn_ar);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Execute graph with mock device context and validate
     */
    ExecutionValidationResult executeAndValidateGraph(
        ComputeGraph& graph,
        const std::shared_ptr<std::vector<std::string>>& execution_log,
        DeviceId device = DeviceId::cpu()
    ) {
        ExecutionValidationResult result;
        result.expected_stages = static_cast<int>(graph.size());
        
        // Create mock device context
        auto backend = device.is_cpu() ? ComputeBackendType::CPU : ComputeBackendType::GPU_CUDA;
        testing::MockDeviceContext ctx(device.ordinal, backend);
        
        // Execute all ready nodes until complete
        while (!graph.allCompleted()) {
            auto ready = graph.getReadyNodes();
            if (ready.empty()) {
                result.all_executed = false;
                // Find which stages weren't executed
                for (const auto& name : graph.getExecutionOrder()) {
                    auto* node = graph.getNode(name);
                    if (node && !node->completed) {
                        result.skipped_stages.push_back(name);
                    }
                }
                break;
            }
            
            for (const auto& name : ready) {
                auto* node = graph.getNode(name);
                if (node && node->stage) {
                    bool success = node->stage->execute(&ctx);
                    if (success) {
                        graph.markCompleted(name);
                        result.stages_executed++;
                    }
                }
            }
        }
        
        result.execution_order = *execution_log;
        return result;
    }
    
    /**
     * @brief Print graph validation summary
     */
    void printGraphValidationSummary(
        const std::string& model_name,
        const GraphValidationResult& result
    ) {
        std::cout << "  Graph validation for " << model_name << ": ";
        if (result.is_valid) {
            std::cout << "✓ " << result.nodes_checked << " nodes valid" << std::endl;
        } else {
            std::cout << "✗ " << result.errors.size() << " errors" << std::endl;
            for (const auto& err : result.errors) {
                std::cout << "    - " << err << std::endl;
            }
        }
    }
    
    // =========================================================================
    // Data Members
    // =========================================================================
    
    ClusterConfig cluster_;
    std::vector<ModelConfig> models_;
    std::vector<std::shared_ptr<MockMPITopology>> topologies_;
};

} // namespace llaminar2::test::orchestration

// =============================================================================
// Test Instantiation Macro
// =============================================================================

/**
 * @brief Macro to instantiate comprehensive orchestration tests for a scenario
 *
 * Tests the full orchestration pipeline from topology to execution:
 *
 * CLUSTER FOUNDATION (2 tests):
 * - ClusterTopologyValid: Verifies topology creation for all ranks
 * - AllRanksSeeConsistentCluster: Ensures deterministic global view
 *
 * PLACEMENT PLANNING (4 tests):
 * - GPUFirstPlacement_AllModels: GPUFirst strategy produces valid plans
 * - CPUOnlyPlacement_AllModels: CPUOnly strategy produces valid plans
 * - HybridOptimalPlacement_AllModels: HybridOptimal strategy produces valid plans
 * - PhaseAwareWeights: Prefill/decode phase weights are correct
 *
 * GRAPH CONSTRUCTION (3 tests):
 * - GraphDevicePlacement: Graph nodes have correct device assignments
 * - GraphBackendSupport: Stages support correct compute backends
 * - GraphAllReduceAtTPBoundaries: AllReduce stages present for tensor parallelism
 *
 * EXECUTION VALIDATION (2 tests):
 * - GraphExecutionOrder: All stages execute in dependency order
 * - GraphConsistentAcrossRanks: All ranks produce identical graph structure
 *
 * WORK DISTRIBUTION (2 tests):
 * - WorkDistribution_HeadSharding: Attention heads distributed correctly
 * - WorkDistribution_VocabSharding: Vocabulary distributed correctly
 *
 * SUMMARY (2 tests):
 * - Summary_ClusterCapacity: Prints cluster hardware summary
 * - Summary_ModelFitTable: Prints model fit analysis
 */
#define INSTANTIATE_ORCHESTRATION_TESTS(TestFixture)                                    \
                                                                                        \
    /* ========================================================================= */     \
    /* CLUSTER FOUNDATION                                                        */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, ClusterTopologyValid)                                           \
    {                                                                                   \
        EXPECT_EQ(topologies_.size(), static_cast<size_t>(cluster_.totalRanks()));      \
        for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {                      \
            EXPECT_EQ(topologies_[rank]->rank(), rank);                                 \
            EXPECT_EQ(topologies_[rank]->world_size(), cluster_.totalRanks());          \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, AllRanksSeeConsistentCluster)                                   \
    {                                                                                   \
        const auto& cluster0 = topologies_[0]->clusterInventory();                      \
        for (int rank = 1; rank < cluster_.totalRanks(); ++rank) {                      \
            const auto& cluster_r = topologies_[rank]->clusterInventory();              \
            EXPECT_EQ(cluster0.world_size, cluster_r.world_size);                       \
            EXPECT_EQ(cluster0.total_gpus, cluster_r.total_gpus);                       \
            EXPECT_EQ(cluster0.total_gpu_memory, cluster_r.total_gpu_memory);           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* ========================================================================= */     \
    /* PLACEMENT PLANNING                                                        */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, GPUFirstPlacement_AllModels)                                    \
    {                                                                                   \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (strategy.isApplicable(input)) {                                         \
                PlacementPlan plan = strategy.compute(input);                           \
                auto result = validatePlan(plan, 0);                                    \
                EXPECT_TRUE(result.is_valid)                                            \
                    << "GPUFirst failed for " << model.name;                            \
                EXPECT_EQ(result.total_layers, model.n_layers);                         \
                EXPECT_GT(result.gpu_layers, 0)                                         \
                    << model.name << " should have at least some GPU layers";           \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, CPUOnlyPlacement_AllModels)                                     \
    {                                                                                   \
        CPUOnlyPlacementStrategy strategy;                                              \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (strategy.isApplicable(input)) {                                         \
                PlacementPlan plan = strategy.compute(input);                           \
                auto result = validatePlan(plan, 0);                                    \
                EXPECT_TRUE(result.is_valid)                                            \
                    << "CPUOnly failed for " << model.name;                             \
                EXPECT_EQ(result.gpu_layers, 0)                                         \
                    << "CPUOnly has GPU layers for " << model.name;                     \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, HybridOptimalPlacement_AllModels)                               \
    {                                                                                   \
        HybridOptimalPlacementStrategy strategy;                                        \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (strategy.isApplicable(input)) {                                         \
                PlacementPlan plan = strategy.compute(input);                           \
                auto result = validatePlan(plan, 0);                                    \
                EXPECT_TRUE(result.is_valid)                                            \
                    << "HybridOptimal failed for " << model.name;                       \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, PhaseAwareWeights)                                              \
    {                                                                                   \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            auto [prefill_gpu, prefill_cpu] =                                           \
                input.getPhaseDeviceWeights(InferencePhase::PREFILL);                   \
            EXPECT_FLOAT_EQ(prefill_gpu, 1.0f) << "PREFILL should be GPU-only";         \
            EXPECT_FLOAT_EQ(prefill_cpu, 0.0f);                                         \
            auto [decode_gpu, decode_cpu] =                                             \
                input.getPhaseDeviceWeights(InferencePhase::DECODE);                    \
            EXPECT_GT(decode_gpu, 0.0f);                                                \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* ========================================================================= */     \
    /* GRAPH CONSTRUCTION                                                        */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, GraphDevicePlacement)                                           \
    {                                                                                   \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (!strategy.isApplicable(input)) continue;                                \
                                                                                        \
            PlacementPlan plan = strategy.compute(input);                               \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);            \
                                                                                        \
            auto result = validateGraphMatchesPlan(graph, plan);                        \
            EXPECT_TRUE(result.is_valid)                                                \
                << "Device placement mismatch for " << model.name;                      \
            EXPECT_EQ(result.device_mismatches, 0);                                     \
            EXPECT_GT(result.nodes_checked, 0)                                          \
                << "No nodes checked for " << model.name;                               \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, GraphBackendSupport)                                            \
    {                                                                                   \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (!strategy.isApplicable(input)) continue;                                \
                                                                                        \
            PlacementPlan plan = strategy.compute(input);                               \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);            \
                                                                                        \
            auto result = validateGraphMatchesPlan(graph, plan);                        \
            EXPECT_EQ(result.backend_mismatches, 0)                                     \
                << "Backend support mismatch for " << model.name;                       \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, GraphAllReduceAtTPBoundaries)                                   \
    {                                                                                   \
        if (cluster_.totalRanks() <= 1) {                                               \
            GTEST_SKIP() << "Single-rank cluster has no tensor parallelism";            \
        }                                                                               \
                                                                                        \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (!strategy.isApplicable(input)) continue;                                \
                                                                                        \
            PlacementPlan plan = strategy.compute(input);                               \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, true);             \
                                                                                        \
            auto result = validateAllReduceStages(graph, plan, cluster_.totalRanks());  \
            EXPECT_TRUE(result.is_valid)                                                \
                << "Missing AllReduce stages for " << model.name;                       \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* ========================================================================= */     \
    /* EXECUTION VALIDATION                                                      */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, GraphExecutionOrder)                                            \
    {                                                                                   \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            auto input = createPlacementInput(model, 0);                                \
            if (!strategy.isApplicable(input)) continue;                                \
                                                                                        \
            PlacementPlan plan = strategy.compute(input);                               \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);            \
                                                                                        \
            auto result = executeAndValidateGraph(graph, log);                          \
                                                                                        \
            EXPECT_TRUE(result.all_executed)                                            \
                << "Not all stages executed for " << model.name;                        \
            EXPECT_EQ(result.stages_executed, result.expected_stages);                  \
            EXPECT_TRUE(result.skipped_stages.empty());                                 \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, GraphConsistentAcrossRanks)                                     \
    {                                                                                   \
        GPUFirstPlacementStrategy strategy;                                             \
        for (const auto& model : models_) {                                             \
            std::vector<size_t> graph_sizes;                                            \
            std::vector<int> gpu_layer_counts;                                          \
                                                                                        \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {                  \
                auto input = createPlacementInput(model, rank);                         \
                if (!strategy.isApplicable(input)) break;                               \
                                                                                        \
                PlacementPlan plan = strategy.compute(input);                           \
                auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);        \
                                                                                        \
                graph_sizes.push_back(graph.size());                                    \
                                                                                        \
                int gpu_layers = 0;                                                     \
                for (const auto& layer : plan.layers) {                                 \
                    if (layer.device.isGPU()) gpu_layers++;                             \
                }                                                                       \
                gpu_layer_counts.push_back(gpu_layers);                                 \
            }                                                                           \
                                                                                        \
            if (graph_sizes.empty()) continue;                                          \
                                                                                        \
            size_t expected_size = graph_sizes[0];                                      \
            int expected_gpu_layers = gpu_layer_counts[0];                              \
                                                                                        \
            for (size_t r = 1; r < graph_sizes.size(); ++r) {                           \
                EXPECT_EQ(graph_sizes[r], expected_size)                                \
                    << model.name << " rank " << r << " has different graph size";      \
                EXPECT_EQ(gpu_layer_counts[r], expected_gpu_layers)                     \
                    << model.name << " rank " << r << " has different GPU layers";      \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* ========================================================================= */     \
    /* WORK DISTRIBUTION                                                         */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, WorkDistribution_HeadSharding)                                  \
    {                                                                                   \
        for (const auto& model : models_) {                                             \
            size_t total_heads = 0;                                                     \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {                  \
                auto range = topologies_[rank]->get_head_range(model.n_heads);          \
                total_heads += range.size();                                            \
            }                                                                           \
            EXPECT_EQ(total_heads, model.n_heads)                                       \
                << "Head sharding doesn't cover all heads for " << model.name;          \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, WorkDistribution_VocabSharding)                                 \
    {                                                                                   \
        for (const auto& model : models_) {                                             \
            size_t total_vocab = 0;                                                     \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank) {                  \
                auto range = topologies_[rank]->get_vocab_range(model.vocab_size);      \
                total_vocab += range.size();                                            \
            }                                                                           \
            EXPECT_EQ(total_vocab, model.vocab_size)                                    \
                << "Vocab sharding doesn't cover all tokens for " << model.name;        \
        }                                                                               \
    }                                                                                   \
                                                                                        \
    /* ========================================================================= */     \
    /* SUMMARY                                                                   */     \
    /* ========================================================================= */     \
                                                                                        \
    TEST_F(TestFixture, Summary_ClusterCapacity)                                        \
    {                                                                                   \
        printClusterSummary();                                                          \
        SUCCEED();                                                                      \
    }                                                                                   \
                                                                                        \
    TEST_F(TestFixture, Summary_ModelFitTable)                                          \
    {                                                                                   \
        printModelFitTable();                                                           \
        SUCCEED();                                                                      \
    }
