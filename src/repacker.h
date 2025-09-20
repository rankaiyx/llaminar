#pragma once

#include "common.h"
#include "model_loader.h"
#include "tensors/tensor_base.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

// Forward declarations
struct GGUFModel;
struct GGUFTensorInfo;

// COSMA tensor format wrapper
class COSMATensor
{
public:
    COSMATensor(const std::vector<int> &shape, const std::vector<double> &data,
                const std::string &name = "");
    ~COSMATensor() = default;

    // Data access
    const std::vector<double> &getData() const { return data_; }
    std::vector<double> &getData() { return data_; }

    // Shape information
    const std::vector<int> &getShape() const { return shape_; }
    const std::string &getName() const { return name_; }

    // COSMA-specific operations
    bool isDistributed() const { return distributed_; }
    void setDistributed(bool distributed) { distributed_ = distributed; }

    // Conversion to/from modern TensorBase
    std::shared_ptr<llaminar::TensorBase> toTensorBase() const;
    static std::shared_ptr<COSMATensor> fromTensorBase(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                       const std::string &name = "");

    // Memory layout optimization for COSMA
    void optimizeForCOSMA();
    bool isOptimized() const { return optimized_; }

private:
    std::vector<int> shape_;
    std::vector<double> data_;
    std::string name_;
    bool distributed_;
    bool optimized_;
};

// Tensor repacker class for GGUF to COSMA conversion
class TensorRepacker
{
public:
    TensorRepacker(); // Constructor is defined (not defaulted)
    ~TensorRepacker() = default;

    // Main repacking interface
    std::shared_ptr<COSMATensor> repackFromGGUF(const GGUFTensorInfo &tensor_info,
                                                const std::vector<uint8_t> &raw_data);

    std::shared_ptr<COSMATensor> repackFromTensorBase(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                                      const std::string &name = "");

    // Batch operations
    std::vector<std::shared_ptr<COSMATensor>> repackModel(const GGUFModel &model,
                                                          ModelLoader &loader);

    // Performance optimization
    void setOptimizationLevel(int level) { optimization_level_ = level; }
    int getOptimizationLevel() const { return optimization_level_; }

    // Memory management
    void setMemoryLimit(size_t limit_bytes) { memory_limit_ = limit_bytes; }
    size_t getMemoryLimit() const { return memory_limit_; }

    // Debugging and profiling - note: function name matches implementation
    void printReppackingInfo() const;
    void resetStats();

private:
    // Configuration
    int optimization_level_ = 1;
    size_t memory_limit_ = 0;            // 0 means no limit
    bool optimize_memory_layout_ = true; // Member variable used in implementation

    // Statistics
    mutable size_t total_tensors_repacked_ = 0;
    mutable size_t total_bytes_processed_ = 0;
    mutable double total_time_spent_ = 0.0;
    mutable size_t processed_tensor_count_ = 0;
    mutable size_t total_memory_usage_ = 0;

    // Configuration for distribution and data types
    std::string preferred_data_type_ = "float32";
    std::string distribution_strategy_ = "auto";

    // Helper methods that exist in the implementation
    bool shouldDistribute(const std::vector<int> &shape) const;
    void optimizeMemoryLayout(COSMATensor &tensor) const;
    std::vector<double> convertToDouble(const std::vector<uint8_t> &raw_data,
                                        uint32_t gguf_type) const;

    // Quantization methods
    std::vector<double> dequantizeQ8_0(const uint8_t *data, size_t n_elements);
    std::vector<double> dequantizeQ4_0(const uint8_t *data, size_t n_elements);
    std::vector<double> dequantizeF16(const uint8_t *data, size_t n_elements);
    std::vector<double> convertF32ToDouble(const uint8_t *data, size_t n_elements);

    // Layout optimization methods
    void optimizeRowMajorLayout(std::vector<double> &data, const std::vector<int> &shape);
    void optimizeColumnMajorLayout(std::vector<double> &data, const std::vector<int> &shape);

    // Distribution strategy methods
    void applyDistributionStrategy(COSMATensor &tensor);
    void distributeByRows(COSMATensor &tensor);
    void distributeByColumns(COSMATensor &tensor);
    void distributeByBlocks(COSMATensor &tensor);

    // Utility methods
    size_t calculateMemoryUsage(const std::vector<int> &shape) const;
    void updateStatistics(const COSMATensor &tensor);
};

namespace TensorConversion
{
    // Utility functions for tensor conversion
    std::shared_ptr<COSMATensor> convert(const std::shared_ptr<llaminar::TensorBase> &tensor,
                                         const std::string &name = "");
    std::shared_ptr<llaminar::TensorBase> convert(const std::shared_ptr<COSMATensor> &cosma_tensor);

    // Type conversion utilities
    std::vector<double> floatToDouble(const std::vector<float> &float_data);
    std::vector<float> doubleToFloat(const std::vector<double> &double_data);
}
