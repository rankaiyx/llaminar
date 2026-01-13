/**
 * @file Test__ROCmIntraVendorP2P.cpp
 * @brief Test ROCm-to-ROCm P2P transfers between two MI50 GPUs
 *
 * Tests the CrossVendorP2PEngine's allow_same_vendor mode for
 * intra-vendor transfers across PCIe between two AMD GPUs.
 */

#include <gtest/gtest.h>

#include "backends/benchmarks/CrossVendorP2P.h"
#include "utils/Logger.h"

#include <iomanip>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2 {
namespace test {

class Test__ROCmIntraVendorP2P : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if we have at least 2 ROCm devices
#ifdef HAVE_ROCM
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count < 2) {
            GTEST_SKIP() << "Need at least 2 ROCm devices for intra-vendor P2P test";
        }
        
        // Get device info
        for (int i = 0; i < device_count; ++i) {
            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, i);
            LOG_INFO("ROCm device " << i << ": " << props.name 
                     << " (" << (props.totalGlobalMem / (1024*1024)) << " MB)");
        }
#else
        GTEST_SKIP() << "ROCm not available";
#endif
    }
};

TEST_F(Test__ROCmIntraVendorP2P, CanTransferWithSameVendorAllowed)
{
    DeviceId src{DeviceType::ROCm, 0};
    DeviceId dst{DeviceType::ROCm, 1};
    
    // Without allow_same_vendor: should return false
    EXPECT_FALSE(CrossVendorP2PHelper::canTransfer(src, dst));
    
    // With allow_same_vendor: should return true
    EXPECT_TRUE(CrossVendorP2PHelper::canTransfer(src, dst, true));
    
    // Same device should always return false
    EXPECT_FALSE(CrossVendorP2PHelper::canTransfer(src, src, true));
}

TEST_F(Test__ROCmIntraVendorP2P, InitializeSameVendorEngine)
{
    DeviceId src{DeviceType::ROCm, 0};
    DeviceId dst{DeviceType::ROCm, 1};
    
    CrossVendorP2PConfig config;
    config.buffer_size = 16 * 1024 * 1024;  // 16 MB staging
    config.chunk_size = 4 * 1024 * 1024;    // 4 MB chunks
    config.num_buffers = 2;                  // Double-buffer
    config.enable_pipelining = true;
    config.allow_same_vendor = true;       // Enable ROCm↔ROCm
    
    CrossVendorP2PEngine engine(config);
    bool ok = engine.initialize(src, dst);
    ASSERT_TRUE(ok) << "Failed to initialize ROCm↔ROCm P2P engine";
    
    LOG_INFO("Theoretical max throughput: " << engine.theoreticalMaxGbps() << " GB/s");
}

TEST_F(Test__ROCmIntraVendorP2P, TransferSmallBuffer)
{
    DeviceId src{DeviceType::ROCm, 0};
    DeviceId dst{DeviceType::ROCm, 1};
    
    CrossVendorP2PConfig config;
    config.buffer_size = 16 * 1024 * 1024;
    config.chunk_size = 4 * 1024 * 1024;
    config.num_buffers = 2;
    config.enable_pipelining = true;
    config.allow_same_vendor = true;
    
    CrossVendorP2PEngine engine(config);
    ASSERT_TRUE(engine.initialize(src, dst));
    
#ifdef HAVE_ROCM
    // Allocate test buffers
    const size_t test_size = 1 * 1024 * 1024;  // 1 MB
    void* d_src = nullptr;
    void* d_dst = nullptr;
    
    hipSetDevice(src.ordinal);
    ASSERT_EQ(hipMalloc(&d_src, test_size), hipSuccess);
    ASSERT_EQ(hipMemset(d_src, 0xAB, test_size), hipSuccess);
    
    hipSetDevice(dst.ordinal);
    ASSERT_EQ(hipMalloc(&d_dst, test_size), hipSuccess);
    ASSERT_EQ(hipMemset(d_dst, 0x00, test_size), hipSuccess);
    
    // Transfer
    auto result = engine.transfer(d_src, d_dst, test_size);
    EXPECT_TRUE(result.success) << "Transfer failed";
    EXPECT_EQ(result.bytes_transferred, test_size);
    EXPECT_GT(result.throughput_gbps, 0.0);
    
    LOG_INFO("Transfer throughput: " << result.throughput_gbps << " GB/s");
    
    // Verify data
    std::vector<uint8_t> host_data(test_size);
    hipSetDevice(dst.ordinal);
    ASSERT_EQ(hipMemcpy(host_data.data(), d_dst, test_size, hipMemcpyDeviceToHost), hipSuccess);
    
    // Check first few bytes
    bool data_ok = true;
    for (size_t i = 0; i < 1024 && data_ok; ++i) {
        if (host_data[i] != 0xAB) {
            data_ok = false;
            LOG_ERROR("Data mismatch at byte " << i << ": expected 0xAB, got 0x" 
                      << std::hex << (int)host_data[i]);
        }
    }
    EXPECT_TRUE(data_ok) << "Data verification failed";
    
    // Cleanup
    hipSetDevice(src.ordinal);
    hipFree(d_src);
    hipSetDevice(dst.ordinal);
    hipFree(d_dst);
#endif
}

TEST_F(Test__ROCmIntraVendorP2P, BenchmarkThroughput)
{
    DeviceId src{DeviceType::ROCm, 0};
    DeviceId dst{DeviceType::ROCm, 1};
    
    CrossVendorP2PConfig config;
    config.buffer_size = 32 * 1024 * 1024;   // 32 MB staging
    config.chunk_size = 8 * 1024 * 1024;     // 8 MB chunks
    config.num_buffers = 2;
    config.enable_pipelining = true;
    config.allow_same_vendor = true;
    config.auto_tune = false;                // Skip auto-tune for faster test
    
    CrossVendorP2PEngine engine(config);
    ASSERT_TRUE(engine.initialize(src, dst));
    
    // Test different sizes
    const size_t TEST_SIZES[] = {
        1 * 1024 * 1024,    // 1 MB
        4 * 1024 * 1024,    // 4 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        128 * 1024 * 1024   // 128 MB
    };
    
    LOG_INFO("=== ROCm↔ROCm P2P Benchmark (via host staging) ===");
    LOG_INFO("  Size (MB)    Throughput (GB/s)");
    LOG_INFO("  ---------    -----------------");
    
    for (size_t size : TEST_SIZES) {
        auto result = engine.benchmark(size, 5);
        if (result.success) {
            LOG_INFO("  " << std::setw(8) << (size / (1024*1024)) 
                     << "     " << std::fixed << std::setprecision(2) 
                     << result.throughput_gbps);
        } else {
            LOG_WARN("  " << (size / (1024*1024)) << " MB: FAILED");
        }
        EXPECT_TRUE(result.success);
        
        // We expect at least ~1 GB/s even with host staging
        if (result.success) {
            EXPECT_GT(result.throughput_gbps, 0.5) << "Throughput too low for " << (size/(1024*1024)) << " MB";
        }
    }
}

TEST_F(Test__ROCmIntraVendorP2P, BidirectionalTransfer)
{
    // Test transfer in both directions
    DeviceId gpu0{DeviceType::ROCm, 0};
    DeviceId gpu1{DeviceType::ROCm, 1};
    
    CrossVendorP2PConfig config;
    config.buffer_size = 16 * 1024 * 1024;
    config.chunk_size = 4 * 1024 * 1024;
    config.num_buffers = 2;
    config.enable_pipelining = true;
    config.allow_same_vendor = true;
    config.auto_tune = false;
    
    // GPU0 → GPU1
    CrossVendorP2PEngine engine_0to1(config);
    ASSERT_TRUE(engine_0to1.initialize(gpu0, gpu1));
    
    // GPU1 → GPU0
    CrossVendorP2PEngine engine_1to0(config);
    ASSERT_TRUE(engine_1to0.initialize(gpu1, gpu0));
    
    const size_t test_size = 32 * 1024 * 1024;  // 32 MB
    
    auto result_0to1 = engine_0to1.benchmark(test_size, 3);
    auto result_1to0 = engine_1to0.benchmark(test_size, 3);
    
    EXPECT_TRUE(result_0to1.success);
    EXPECT_TRUE(result_1to0.success);
    
    LOG_INFO("GPU0→GPU1: " << result_0to1.throughput_gbps << " GB/s");
    LOG_INFO("GPU1→GPU0: " << result_1to0.throughput_gbps << " GB/s");
    
    // Both directions should have similar throughput (within 20%)
    if (result_0to1.success && result_1to0.success) {
        double ratio = result_0to1.throughput_gbps / result_1to0.throughput_gbps;
        EXPECT_GT(ratio, 0.8) << "Asymmetric throughput detected";
        EXPECT_LT(ratio, 1.2) << "Asymmetric throughput detected";
    }
}

}  // namespace test
}  // namespace llaminar2
