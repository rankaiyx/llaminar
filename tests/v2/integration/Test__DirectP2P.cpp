/**
 * @file Test__DirectP2P.cpp
 * @brief Test EXPERIMENTAL direct cross-vendor P2P via DMA-BUF
 *
 * Tests whether we can share memory directly between CUDA and ROCm
 * using Linux DMA-BUF / POSIX file descriptors.
 */

#include <gtest/gtest.h>

#include "backends/benchmarks/DirectP2P.h"
#include "utils/Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace llaminar2
{
    namespace test
    {

        class Test__DirectP2P : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Ensure we have both CUDA and ROCm
#ifndef HAVE_CUDA
                GTEST_SKIP() << "CUDA not available";
#endif
#ifndef HAVE_ROCM
                GTEST_SKIP() << "ROCm not available";
#endif
            }
        };

        TEST_F(Test__DirectP2P, ProbeCapabilities)
        {
            auto caps = DirectP2PEngine::probeCapabilities();

            LOG_INFO("\n" << caps.describe());

            // At minimum, we should get kernel version
            EXPECT_FALSE(caps.kernel_version.empty());

            // Log what we found
            LOG_INFO("CUDA DMA-BUF export: " << (caps.dmabuf_export_cuda ? "YES" : "NO"));
            LOG_INFO("CUDA DMA-BUF import: " << (caps.dmabuf_import_cuda ? "YES" : "NO"));
            LOG_INFO("ROCm DMA-BUF export: " << (caps.dmabuf_export_rocm ? "YES" : "NO"));
            LOG_INFO("ROCm DMA-BUF import: " << (caps.dmabuf_import_rocm ? "YES" : "NO"));
            LOG_INFO("Direct P2P possible: " << (caps.canDoDirectP2P() ? "YES" : "NO"));
        }

        TEST_F(Test__DirectP2P, InitializeEngine)
        {
            DirectP2PEngine engine;

            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            bool init = engine.initialize(cuda_dev, rocm_dev);
            EXPECT_TRUE(init) << "Engine initialization should succeed (even if direct P2P not available)";
        }

        TEST_F(Test__DirectP2P, AllocateExportableCUDA)
        {
            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initialize(cuda_dev, rocm_dev));

            size_t size = 1024 * 1024; // 1 MB
            auto buffer = engine.allocateExportable(cuda_dev, size);

            if (buffer)
            {
                LOG_INFO("CUDA exportable buffer allocated:");
                LOG_INFO("  device_ptr: " << buffer->device_ptr);
                LOG_INFO("  dmabuf_fd: " << buffer->dmabuf_fd);
                LOG_INFO("  is_exported: " << buffer->is_exported);

                // Try to import into ROCm
                if (buffer->is_exported)
                {
                    void *imported = engine.importBuffer(*buffer, rocm_dev);
                    if (imported)
                    {
                        LOG_INFO("  Imported into ROCm: " << imported);
                        LOG_INFO("  *** DIRECT P2P MAY BE POSSIBLE! ***");
                    }
                    else
                    {
                        LOG_INFO("  Import into ROCm FAILED");
                    }
                }
            }
            else
            {
                LOG_WARN("CUDA exportable buffer allocation failed - this is expected on many systems");
            }
        }

        TEST_F(Test__DirectP2P, AllocateExportableROCm)
        {
            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initialize(cuda_dev, rocm_dev));

            size_t size = 1024 * 1024; // 1 MB
            auto buffer = engine.allocateExportable(rocm_dev, size);

            if (buffer)
            {
                LOG_INFO("ROCm exportable buffer allocated:");
                LOG_INFO("  device_ptr: " << buffer->device_ptr);
                LOG_INFO("  dmabuf_fd: " << buffer->dmabuf_fd);
                LOG_INFO("  is_exported: " << buffer->is_exported);

                // Try to import into CUDA
                if (buffer->is_exported)
                {
                    void *imported = engine.importBuffer(*buffer, cuda_dev);
                    if (imported)
                    {
                        LOG_INFO("  Imported into CUDA: " << imported);
                        LOG_INFO("  *** DIRECT P2P MAY BE POSSIBLE! ***");
                    }
                    else
                    {
                        LOG_INFO("  Import into CUDA FAILED");
                    }
                }
            }
            else
            {
                LOG_WARN("ROCm exportable buffer allocation failed - ROCm DMA-BUF export is very limited");
            }
        }

        TEST_F(Test__DirectP2P, BenchmarkWithFallback)
        {
            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initialize(cuda_dev, rocm_dev));

            size_t size = 64 * 1024 * 1024; // 64 MB
            int iterations = 3;

            LOG_INFO("\n=== DirectP2P Benchmark (64 MB, 3 iterations) ===");
            auto result = engine.benchmark(size, iterations);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Used DMA-BUF: " << result.used_dmabuf);
            LOG_INFO("  Fell back to staged: " << result.fell_back_to_staged);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");

            // The benchmark should at least succeed via fallback
            EXPECT_TRUE(result.success);
            EXPECT_GT(result.throughput_gbps, 0.5); // At least 0.5 GB/s via fallback
        }

        //----------------------------------------------------------------------
        // PCIe BAR Direct P2P Tests (the working path!)
        //----------------------------------------------------------------------

        TEST_F(Test__DirectP2P, PCIeBar_ProbeCapabilities)
        {
            auto caps = DirectP2PEngine::probeCapabilities();

            LOG_INFO("\n=== PCIe BAR Capability Probe ===");
            LOG_INFO("AMD BARs found: " << caps.discovered_bars.size());
            for (const auto& bar : caps.discovered_bars)
            {
                LOG_INFO("  - " << bar.pci_address << ": " 
                         << (bar.bar_size / (1024*1024*1024)) << " GB");
            }
            LOG_INFO("BAR access: " << (caps.pcie_bar_accessible ? "YES" : "NO (need root?)"));
            LOG_INFO("CUDA IOMEMORY: " << (caps.pcie_bar_iomemory_supported ? "YES" : "NO"));
            LOG_INFO("PCIe BAR P2P possible: " << (caps.canDoPCIeBarP2P() ? "YES" : "NO"));

            // We should at least detect AMD GPUs
            if (caps.discovered_bars.empty())
            {
                LOG_WARN("No AMD GPU BARs found - test environment may not have AMD GPU");
            }
        }

        TEST_F(Test__DirectP2P, PCIeBar_Initialize)
        {
            auto caps = DirectP2PEngine::probeCapabilities();
            
            if (!caps.canDoPCIeBarP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available (need root + AMD GPU)";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            LOG_INFO("\n=== PCIe BAR Initialization ===");
            bool init = engine.initializePCIeBar(cuda_dev, rocm_dev);
            
            EXPECT_TRUE(init) << "PCIe BAR initialization should succeed with root access";
            EXPECT_TRUE(engine.isPCIeBarActive());
            
            if (init)
            {
                LOG_INFO("CUDA BAR pointer: " << engine.getCudaBarPointer());
                LOG_INFO("Mapped size: " << (engine.getBarMappedSize() / (1024*1024)) << " MB");
            }
        }

        TEST_F(Test__DirectP2P, PCIeBar_TransferToNVIDIA)
        {
            auto caps = DirectP2PEngine::probeCapabilities();
            
            if (!caps.canDoPCIeBarP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            size_t size = 16 * 1024 * 1024; // 16 MB
            
            // Allocate CUDA buffer
            void* cuda_buf = nullptr;
#ifdef HAVE_CUDA
            cudaMalloc(&cuda_buf, size);
            ASSERT_NE(cuda_buf, nullptr);
#endif

            LOG_INFO("\n=== PCIe BAR Transfer: AMD → NVIDIA ===");
            auto result = engine.transferViaPCIeBar(cuda_buf, 0, size, 
                                                     DirectP2PEngine::Direction::ToNVIDIA);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Used PCIe BAR: " << result.used_pcie_bar);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_TRUE(result.used_pcie_bar);
            EXPECT_GT(result.throughput_gbps, 0.5); // At least 0.5 GB/s

#ifdef HAVE_CUDA
            cudaFree(cuda_buf);
#endif
        }

        TEST_F(Test__DirectP2P, PCIeBar_TransferToAMD)
        {
            auto caps = DirectP2PEngine::probeCapabilities();
            
            if (!caps.canDoPCIeBarP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            size_t size = 16 * 1024 * 1024; // 16 MB
            
            // Allocate CUDA buffer
            void* cuda_buf = nullptr;
#ifdef HAVE_CUDA
            cudaMalloc(&cuda_buf, size);
            ASSERT_NE(cuda_buf, nullptr);
            cudaMemset(cuda_buf, 0xAB, size); // Initialize
#endif

            LOG_INFO("\n=== PCIe BAR Transfer: NVIDIA → AMD ===");
            auto result = engine.transferViaPCIeBar(cuda_buf, 0, size, 
                                                     DirectP2PEngine::Direction::ToAMD);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Used PCIe BAR: " << result.used_pcie_bar);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_TRUE(result.used_pcie_bar);
            EXPECT_GT(result.throughput_gbps, 0.5); // At least 0.5 GB/s

#ifdef HAVE_CUDA
            cudaFree(cuda_buf);
#endif
        }

        TEST_F(Test__DirectP2P, PCIeBar_Benchmark)
        {
            auto caps = DirectP2PEngine::probeCapabilities();
            
            if (!caps.canDoPCIeBarP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            LOG_INFO("\n=== PCIe BAR Direct P2P Benchmark ===");
            size_t size = 64 * 1024 * 1024; // 64 MB
            auto result = engine.benchmarkPCIeBar(size, 5);

            LOG_INFO("\nFinal Results:");
            LOG_INFO("  AMD → NVIDIA (read): " << result.read_gbps << " GB/s");
            LOG_INFO("  NVIDIA → AMD (write): " << result.write_gbps << " GB/s");
            LOG_INFO("  Average: " << result.throughput_gbps << " GB/s");

            EXPECT_TRUE(result.success);
            EXPECT_TRUE(result.used_pcie_bar);
            EXPECT_GT(result.read_gbps, 0.5);   // Should get at least 0.5 GB/s read
            EXPECT_GT(result.write_gbps, 0.5);  // Should get at least 0.5 GB/s write
            
            // Write is typically faster than read for PCIe BAR
            LOG_INFO("  Write/Read ratio: " << (result.write_gbps / result.read_gbps));
        }

    } // namespace test
} // namespace llaminar2
