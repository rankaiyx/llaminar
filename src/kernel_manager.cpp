#include "kernel_manager.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>

// Kernel implementation
Kernel::Kernel(const std::string &name, const std::string &operation_type)
    : name_(name), operation_type_(operation_type),
      last_execution_time_ms_(0.0), execution_count_(0) {}

// KernelManager implementation
KernelManager &KernelManager::getInstance()
{
    static KernelManager instance;
    return instance;
}

void KernelManager::registerKernel(const std::string &operation_type, KernelFactory factory)
{
    kernel_factories_[operation_type] = factory;
    std::cout << "Registered kernel: " << operation_type << std::endl;
}

std::shared_ptr<Kernel> KernelManager::createKernel(const std::string &operation_type) const
{
    auto it = kernel_factories_.find(operation_type);
    if (it != kernel_factories_.end())
    {
        return it->second();
    }

    std::cerr << "Error: Unknown kernel operation type: " << operation_type << std::endl;
    return nullptr;
}

bool KernelManager::executeKernel(const std::string &operation_type,
                                  const std::vector<std::shared_ptr<Tensor>> &inputs,
                                  std::vector<std::shared_ptr<Tensor>> &outputs)
{
    // Get or create kernel instance
    auto it = kernel_instances_.find(operation_type);
    if (it == kernel_instances_.end())
    {
        auto kernel = createKernel(operation_type);
        if (!kernel)
        {
            return false;
        }
        kernel_instances_[operation_type] = kernel;
        it = kernel_instances_.find(operation_type);
    }

    auto kernel = it->second;

    // Validate inputs and outputs
    if (!kernel->validate(inputs, outputs))
    {
        std::cerr << "Error: Kernel validation failed for " << operation_type << std::endl;
        return false;
    }

    // Execute kernel with timing
    auto start = std::chrono::high_resolution_clock::now();
    bool success = kernel->execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();

    if (success)
    {
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();
        kernel->recordExecutionTime(execution_time);
    }

    return success;
}

std::vector<std::string> KernelManager::getRegisteredOperations() const
{
    std::vector<std::string> operations;
    for (const auto &pair : kernel_factories_)
    {
        operations.push_back(pair.first);
    }
    return operations;
}

bool KernelManager::isRegistered(const std::string &operation_type) const
{
    return kernel_factories_.find(operation_type) != kernel_factories_.end();
}

void KernelManager::printPerformanceReport() const
{
    std::cout << "\n=== Kernel Performance Report ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Operation"
              << std::setw(15) << "Executions"
              << std::setw(15) << "Last Time (ms)"
              << std::setw(15) << "Total Time (ms)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto &pair : kernel_instances_)
    {
        const auto &kernel = pair.second;
        double total_time = kernel->getLastExecutionTime() * kernel->getExecutionCount();

        std::cout << std::left << std::setw(20) << pair.first
                  << std::setw(15) << kernel->getExecutionCount()
                  << std::setw(15) << std::fixed << std::setprecision(2) << kernel->getLastExecutionTime()
                  << std::setw(15) << std::fixed << std::setprecision(2) << total_time
                  << std::endl;
    }
    std::cout << "=================================" << std::endl;
}

void KernelManager::resetPerformanceCounters()
{
    kernel_instances_.clear();
}

// Tensor implementation
Tensor::Tensor(const std::vector<int> &shape, const std::vector<double> &data)
    : shape_(shape), data_(data)
{
    // Validate that data size matches shape
    size_t expected_size = 1;
    for (int dim : shape)
    {
        expected_size *= dim;
    }

    if (data.size() != expected_size)
    {
        std::cerr << "Warning: Data size (" << data.size()
                  << ") doesn't match shape size (" << expected_size << ")" << std::endl;
    }
}

Tensor::Tensor(int rows, int cols, const std::vector<double> &data)
    : shape_({rows, cols}), data_(data)
{
    size_t expected_size = rows * cols;

    if (data.size() != expected_size)
    {
        std::cerr << "Warning: Data size (" << data.size()
                  << ") doesn't match matrix size (" << expected_size << ")" << std::endl;
    }
}

void Tensor::reshape(const std::vector<int> &new_shape)
{
    size_t new_size = 1;
    for (int dim : new_shape)
    {
        new_size *= dim;
    }

    if (new_size != data_.size())
    {
        std::cerr << "Error: Cannot reshape tensor. Size mismatch." << std::endl;
        return;
    }

    shape_ = new_shape;
}

bool Tensor::isCompatibleWith(const Tensor &other) const
{
    return shape_ == other.shape_;
}

void Tensor::print() const
{
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        std::cout << shape_[i];
        if (i < shape_.size() - 1)
            std::cout << ", ";
    }
    std::cout << "], size: " << data_.size() << std::endl;

    // Print first few elements if not too large
    if (data_.size() <= 20)
    {
        std::cout << "Data: [";
        for (size_t i = 0; i < data_.size(); ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << data_[i];
            if (i < data_.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    else
    {
        std::cout << "Data: [" << data_[0] << ", " << data_[1] << ", ..., "
                  << data_[data_.size() - 2] << ", " << data_[data_.size() - 1] << "]" << std::endl;
    }
}