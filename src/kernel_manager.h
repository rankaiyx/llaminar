#pragma once

#include "common.h"
#include <functional>
#include <unordered_map>
#include <memory>
#include <vector>

// Forward declarations
class Tensor;

// Base kernel interface
class Kernel
{
public:
    Kernel(const std::string &name, const std::string &operation_type);
    virtual ~Kernel() = default;

    // Core kernel functionality
    virtual bool execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                         std::vector<std::shared_ptr<Tensor>> &outputs) = 0;

    virtual bool validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                          const std::vector<std::shared_ptr<Tensor>> &outputs) const = 0;

    // Kernel information
    const std::string &getName() const { return name_; }
    const std::string &getOperationType() const { return operation_type_; }

    // Performance profiling
    double getLastExecutionTime() const { return last_execution_time_ms_; }
    size_t getExecutionCount() const { return execution_count_; }

    void recordExecutionTime(double time_ms)
    {
        last_execution_time_ms_ = time_ms;
        execution_count_++;
    }

protected:
    std::string name_;
    std::string operation_type_;
    double last_execution_time_ms_;
    size_t execution_count_;
};

// Kernel factory function type
using KernelFactory = std::function<std::shared_ptr<Kernel>()>;

// Kernel manager for registration and dispatch
class KernelManager
{
public:
    static KernelManager &getInstance();

    // Kernel registration
    void registerKernel(const std::string &operation_type, KernelFactory factory);

    template <typename KernelType>
    void registerKernel(const std::string &operation_type)
    {
        registerKernel(operation_type, []() -> std::shared_ptr<Kernel>
                       { return std::make_shared<KernelType>(); });
    }

    // Kernel creation and execution
    std::shared_ptr<Kernel> createKernel(const std::string &operation_type) const;

    bool executeKernel(const std::string &operation_type,
                       const std::vector<std::shared_ptr<Tensor>> &inputs,
                       std::vector<std::shared_ptr<Tensor>> &outputs);

    // Kernel information
    std::vector<std::string> getRegisteredOperations() const;
    bool isRegistered(const std::string &operation_type) const;

    // Performance profiling
    void printPerformanceReport() const;
    void resetPerformanceCounters();

private:
    KernelManager() = default;

    std::unordered_map<std::string, KernelFactory> kernel_factories_;
    std::unordered_map<std::string, std::shared_ptr<Kernel>> kernel_instances_;
};

// Macro for easy kernel registration
#define REGISTER_KERNEL(operation_type, kernel_class) \
    KernelManager::getInstance().registerKernel<kernel_class>(operation_type)

// Simple tensor class for kernel interface (will be expanded)
class Tensor
{
public:
    Tensor(const std::vector<int> &shape, const std::vector<double> &data);
    Tensor(int rows, int cols, const std::vector<double> &data);
    ~Tensor() = default;

    // Data access
    const std::vector<double> &getData() const { return data_; }
    std::vector<double> &getData() { return data_; }
    void setData(const std::vector<double> &data) { data_ = data; }

    // Shape information
    const std::vector<int> &getShape() const { return shape_; }
    int getRank() const { return shape_.size(); }
    size_t getSize() const { return data_.size(); }

    // Matrix-specific accessors (for 2D tensors)
    int getRows() const { return shape_.size() >= 1 ? shape_[0] : 0; }
    int getCols() const { return shape_.size() >= 2 ? shape_[1] : 1; }

    // Utility
    void reshape(const std::vector<int> &new_shape);
    bool isCompatibleWith(const Tensor &other) const;
    void print() const;

private:
    std::vector<int> shape_;
    std::vector<double> data_;
};