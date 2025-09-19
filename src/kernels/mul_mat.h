#pragma once

#include "../kernel_manager.h"
#include <memory>

// COSMA matrix multiplication kernel
class MatMulKernel : public Kernel
{
public:
    MatMulKernel();
    ~MatMulKernel() override = default;

    bool execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                 std::vector<std::shared_ptr<Tensor>> &outputs) override;

    bool validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                  const std::vector<std::shared_ptr<Tensor>> &outputs) const override;

    // COSMA-specific configuration
    void setStrategy(const std::string &strategy) { strategy_ = strategy; }
    void setBlockSizes(int block_m, int block_n, int block_k)
    {
        block_m_ = block_m;
        block_n_ = block_n;
        block_k_ = block_k;
    }

private:
    std::string strategy_;
    int block_m_, block_n_, block_k_;

    // Helper functions
    bool executeCOSMA(const Tensor &A, const Tensor &B, Tensor &C);
    void initializeCOSMAContext();
};