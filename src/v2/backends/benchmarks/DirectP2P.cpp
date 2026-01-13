/**
 * @file DirectP2P.cpp
 * @brief EXPERIMENTAL: Direct cross-vendor P2P implementation
 *
 * Implements PCIe BAR-based direct P2P between CUDA and ROCm devices.
 * Also includes experimental DMA-BUF support (often fails).
 * Falls back to pipelined host-staged transfer if direct P2P fails.
 *
 * ## PCIe BAR Implementation
 *
 * The key insight is that AMD GPU's VRAM is exposed via PCIe BAR0, and
 * CUDA's cuMemHostRegister with IOMEMORY flag can register this "device I/O
 * memory" for DMA access. This bypasses the normal P2P restrictions.
 *
 * Steps:
 * 1. Find AMD GPU's sysfs path (e.g., /sys/bus/pci/devices/0000:b1:00.0)
 * 2. Open and mmap() resource0 (BAR0)
 * 3. cuMemHostRegister() with CU_MEMHOSTREGISTER_IOMEMORY flag
 * 4. cuMemHostGetDevicePointer() to get CUDA-usable address
 * 5. cuMemcpyDtoD() between CUDA memory and BAR-mapped AMD memory
 */

#include "DirectP2P.h"
#include "CrossVendorP2P.h" // Fallback
#include "utils/Logger.h"

#include <chrono>
#include <cstring>
#include <map>
#include <sstream>
#include <iomanip>
#include <sys/utsname.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>

#ifdef HAVE_CUDA
#include "DirectP2P_CUDA.h"
#include <cuda.h>
#endif

#ifdef HAVE_ROCM
#include "DirectP2P_ROCm.h"
#endif

namespace llaminar2
{

    //--------------------------------------------------------------------------
    // DirectP2PBuffer Implementation
    //--------------------------------------------------------------------------

    DirectP2PBuffer::~DirectP2PBuffer()
    {
        if (device_ptr && dmabuf_fd >= 0)
        {
#ifdef HAVE_CUDA
            if (owner_device.type == DeviceType::CUDA)
            {
                cuda_direct_p2p::freeExportable(owner_device.ordinal, device_ptr, dmabuf_fd);
            }
#endif
#ifdef HAVE_ROCM
            if (owner_device.type == DeviceType::ROCm)
            {
                rocm_direct_p2p::freeExportable(owner_device.ordinal, device_ptr, dmabuf_fd);
            }
#endif
        }
    }

    DirectP2PBuffer::DirectP2PBuffer(DirectP2PBuffer &&other) noexcept
        : device_ptr(other.device_ptr), dmabuf_fd(other.dmabuf_fd),
          size(other.size), owner_device(other.owner_device),
          is_exported(other.is_exported)
    {
        other.device_ptr = nullptr;
        other.dmabuf_fd = -1;
    }

    DirectP2PBuffer &DirectP2PBuffer::operator=(DirectP2PBuffer &&other) noexcept
    {
        if (this != &other)
        {
            device_ptr = other.device_ptr;
            dmabuf_fd = other.dmabuf_fd;
            size = other.size;
            owner_device = other.owner_device;
            is_exported = other.is_exported;

            other.device_ptr = nullptr;
            other.dmabuf_fd = -1;
        }
        return *this;
    }

    //--------------------------------------------------------------------------
    // PCIe BAR Discovery Functions
    //--------------------------------------------------------------------------

    namespace
    {
        /**
         * @brief Check if a PCI device is an AMD GPU
         */
        bool isAmdGpu(const std::string &pci_path)
        {
            // Read vendor ID
            std::string vendor_path = pci_path + "/vendor";
            std::ifstream vendor_file(vendor_path);
            if (!vendor_file.is_open())
                return false;

            std::string vendor_id;
            vendor_file >> vendor_id;

            // AMD vendor ID is 0x1002
            return vendor_id == "0x1002";
        }

        /**
         * @brief Get BAR0 size from sysfs resource file
         */
        size_t getBar0Size(const std::string &pci_path)
        {
            // Parse /sys/bus/pci/devices/XXXX:XX:XX.X/resource
            std::string resource_path = pci_path + "/resource";
            std::ifstream resource_file(resource_path);
            if (!resource_file.is_open())
                return 0;

            // First line is BAR0
            std::string line;
            if (!std::getline(resource_file, line))
                return 0;

            // Format: "0xSTART 0xEND 0xFLAGS"
            unsigned long long start, end, flags;
            if (sscanf(line.c_str(), "%llx %llx %llx", &start, &end, &flags) != 3)
                return 0;

            if (end <= start)
                return 0;
            return static_cast<size_t>(end - start + 1);
        }

        /**
         * @brief Discover all AMD GPU PCIe BARs
         */
        std::vector<PCIeBarInfo> discoverAmdBars()
        {
            std::vector<PCIeBarInfo> bars;

            const char *pci_path = "/sys/bus/pci/devices";
            DIR *dir = opendir(pci_path);
            if (!dir)
            {
                LOG_DEBUG("Cannot open " << pci_path);
                return bars;
            }

            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr)
            {
                if (entry->d_name[0] == '.')
                    continue;

                std::string device_path = std::string(pci_path) + "/" + entry->d_name;

                if (isAmdGpu(device_path))
                {
                    PCIeBarInfo info;
                    info.pci_address = entry->d_name;
                    info.sysfs_path = device_path + "/resource0";
                    info.bar_size = getBar0Size(device_path);
                    info.is_amd = true;

                    if (info.bar_size > 0)
                    {
                        LOG_DEBUG("Found AMD GPU BAR: " << info.pci_address
                                                        << " size=" << (info.bar_size / (1024 * 1024 * 1024)) << " GB");
                        bars.push_back(std::move(info));
                    }
                }
            }

            closedir(dir);
            return bars;
        }

        /**
         * @brief Check if we have permission to access PCIe BAR
         */
        bool canAccessBar(const std::string &bar_path)
        {
            int fd = open(bar_path.c_str(), O_RDWR);
            if (fd >= 0)
            {
                close(fd);
                return true;
            }
            return false;
        }

#ifdef HAVE_CUDA
        /**
         * @brief Check if CUDA supports IOMEMORY registration
         */
        bool checkCudaIomemorySupport()
        {
            // Check CUDA driver version - IOMEMORY support is in most modern drivers
            int driverVersion = 0;
            if (cuDriverGetVersion(&driverVersion) == CUDA_SUCCESS)
            {
                // IOMEMORY flag exists in CUDA 11.0+ (version 11000+)
                return driverVersion >= 11000;
            }
            return false;
        }
#endif
    } // anonymous namespace

    //--------------------------------------------------------------------------
    // DirectP2PCapability Implementation
    //--------------------------------------------------------------------------

    std::string DirectP2PCapability::describe() const
    {
        std::ostringstream ss;
        ss << "Direct P2P Capabilities:\n";
        ss << "  Kernel: " << kernel_version << "\n";
        ss << "  CUDA Driver: " << cuda_driver_version << "\n";
        ss << "  ROCm Driver: " << rocm_driver_version << "\n";
        ss << "\n  === PCIe BAR (Recommended Path) ===\n";
        ss << "  PCIe BAR Access: " << (pcie_bar_accessible ? "YES" : "NO (need root?)") << "\n";
        ss << "  CUDA IOMEMORY Support: " << (pcie_bar_iomemory_supported ? "YES" : "NO") << "\n";
        ss << "  AMD BARs Found: " << discovered_bars.size() << "\n";
        for (const auto &bar : discovered_bars)
        {
            ss << "    - " << bar.pci_address << ": "
               << (bar.bar_size / (1024 * 1024 * 1024)) << " GB\n";
        }
        ss << "\n  === DMA-BUF (Experimental) ===\n";
        ss << "  CUDA DMA-BUF Export: " << (dmabuf_export_cuda ? "YES" : "NO") << "\n";
        ss << "  CUDA DMA-BUF Import: " << (dmabuf_import_cuda ? "YES" : "NO") << "\n";
        ss << "  ROCm DMA-BUF Export: " << (dmabuf_export_rocm ? "YES" : "NO") << "\n";
        ss << "  ROCm DMA-BUF Import: " << (dmabuf_import_rocm ? "YES" : "NO") << "\n";
        ss << "\n  IOMMU: " << (iommu_enabled ? "ENABLED (may affect DMA-BUF)" : "disabled") << "\n";
        ss << "  Direct P2P Possible: " << (canDoDirectP2P() ? "YES" : "NO") << "\n";
        if (canDoPCIeBarP2P())
        {
            ss << "  >>> PCIe BAR P2P is AVAILABLE <<<\n";
        }
        return ss.str();
    }

    //--------------------------------------------------------------------------
    // DirectP2PEngine Implementation
    //--------------------------------------------------------------------------

    struct DirectP2PEngine::Impl
    {
        // Streams for async operations (multiple for concurrent transfers)
        void *cuda_stream = nullptr;
        void *cuda_stream_read = nullptr;  // For overlapped reads
        void *cuda_stream_write = nullptr; // For overlapped writes
        void *rocm_stream = nullptr;
        int cuda_device = -1;
        int rocm_device = -1;

        // Single BAR (legacy API)
        PCIeBarInfo bar_info;
        CUdeviceptr cuda_bar_ptr = 0; // CUDA device pointer to BAR
        bool bar_mapped = false;

        // Multi-GPU BAR resources (new API)
        std::vector<PCIeBarInfo> mapped_bars;          // All mapped AMD GPU BARs
        std::map<int, size_t> rocm_ordinal_to_bar_idx; // Map ordinal to bar index

        // CUDA context (retained)
        CUcontext cuda_ctx = nullptr;
        CUdevice cu_device = 0;

        // Fallback engine
        std::unique_ptr<CrossVendorP2PEngine> fallback_engine;

        ~Impl()
        {
            cleanup();
        }

        void cleanup()
        {
            // Cleanup PCIe BAR
            if (bar_mapped && bar_info.mapped_ptr)
            {
#ifdef HAVE_CUDA
                cuMemHostUnregister(bar_info.mapped_ptr);
#endif
                munmap(bar_info.mapped_ptr, bar_info.mapped_size);
                bar_info.mapped_ptr = nullptr;
                bar_info.mapped_size = 0;
                cuda_bar_ptr = 0;
                bar_mapped = false;
            }

            if (bar_info.bar_fd >= 0)
            {
                close(bar_info.bar_fd);
                bar_info.bar_fd = -1;
            }

#ifdef HAVE_CUDA
            if (cuda_stream)
            {
                cuda_direct_p2p::destroyStream(cuda_stream);
                cuda_stream = nullptr;
            }
#endif
#ifdef HAVE_ROCM
            if (rocm_stream)
            {
                rocm_direct_p2p::destroyStream(rocm_stream);
                rocm_stream = nullptr;
            }
#endif
            fallback_engine.reset();
        }
    };

    DirectP2PEngine::DirectP2PEngine()
        : impl_(std::make_unique<Impl>())
    {
    }

    DirectP2PEngine::~DirectP2PEngine() = default;

    DirectP2PCapability DirectP2PEngine::probeCapabilities()
    {
        DirectP2PCapability caps;

        // Get kernel version
        struct utsname uname_data;
        if (uname(&uname_data) == 0)
        {
            caps.kernel_version = uname_data.release;
        }

        // Check for IOMMU
        FILE *f = fopen("/sys/kernel/iommu_groups/0/type", "r");
        if (f)
        {
            caps.iommu_enabled = true;
            fclose(f);
        }

        // Discover AMD GPU BARs
        caps.discovered_bars = discoverAmdBars();

        // Check if we can access any BAR with usable size (need root or udev rule)
        for (const auto &bar : caps.discovered_bars)
        {
            if (bar.bar_size > 0 && canAccessBar(bar.sysfs_path))
            {
                caps.pcie_bar_accessible = true;
                break;
            }
        }

#ifdef HAVE_CUDA
        caps.cuda_driver_version = cuda_direct_p2p::getDriverVersion();
        caps.dmabuf_export_cuda = cuda_direct_p2p::supportsDmaBufExport();
        caps.dmabuf_import_cuda = cuda_direct_p2p::supportsDmaBufImport();
        caps.pcie_bar_iomemory_supported = checkCudaIomemorySupport();
#endif

#ifdef HAVE_ROCM
        caps.rocm_driver_version = rocm_direct_p2p::getDriverVersion();
        caps.dmabuf_export_rocm = rocm_direct_p2p::supportsDmaBufExport();
        caps.dmabuf_import_rocm = rocm_direct_p2p::supportsDmaBufImport();
#endif

        return caps;
    }

    bool DirectP2PEngine::initialize(DeviceId device_a, DeviceId device_b)
    {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
        LOG_ERROR("DirectP2PEngine requires both HAVE_CUDA and HAVE_ROCM");
        return false;
#else
        if (device_a.type == device_b.type)
        {
            LOG_ERROR("DirectP2PEngine: devices must be different vendors");
            return false;
        }

        impl_->cleanup();

        device_a_ = device_a;
        device_b_ = device_b;

        // Identify CUDA and ROCm devices
        if (device_a.type == DeviceType::CUDA)
        {
            impl_->cuda_device = device_a.ordinal;
            impl_->rocm_device = device_b.ordinal;
        }
        else
        {
            impl_->cuda_device = device_b.ordinal;
            impl_->rocm_device = device_a.ordinal;
        }

        LOG_INFO("Creating streams - CUDA device: " << impl_->cuda_device << ", ROCm device: " << impl_->rocm_device);

        // Create streams
        impl_->cuda_stream = cuda_direct_p2p::createStream(impl_->cuda_device);
        LOG_INFO("CUDA stream: " << (impl_->cuda_stream ? "OK" : "FAILED"));

        impl_->rocm_stream = rocm_direct_p2p::createStream(impl_->rocm_device);
        LOG_INFO("ROCm stream: " << (impl_->rocm_stream ? "OK" : "FAILED"));

        if (!impl_->cuda_stream || !impl_->rocm_stream)
        {
            LOG_ERROR("Failed to create GPU streams - CUDA: " << (impl_->cuda_stream ? "OK" : "NULL")
                                                              << ", ROCm: " << (impl_->rocm_stream ? "OK" : "NULL"));
            impl_->cleanup();
            return false;
        }

        // Check if direct P2P is possible
        auto caps = probeCapabilities();
        LOG_INFO(caps.describe());

        if (caps.canDoDirectP2P())
        {
            LOG_INFO("Direct P2P via DMA-BUF appears supported - will try direct path");
            direct_p2p_active_ = true;
        }
        else
        {
            LOG_WARN("Direct P2P not fully supported - will use fallback with direct attempt");
            direct_p2p_active_ = false;
        }

        // Always set up fallback engine
        CrossVendorP2PConfig fallback_config;
        fallback_config.buffer_size = 64 * 1024 * 1024;
        fallback_config.chunk_size = 16 * 1024 * 1024;
        fallback_config.num_buffers = 2;
        fallback_config.enable_pipelining = true;

        impl_->fallback_engine = std::make_unique<CrossVendorP2PEngine>(fallback_config);
        if (!impl_->fallback_engine->initialize(device_a, device_b))
        {
            LOG_WARN("Fallback engine initialization failed");
        }

        return true;
#endif
    }

    //--------------------------------------------------------------------------
    // PCIe BAR Direct P2P Implementation
    //--------------------------------------------------------------------------

    bool DirectP2PEngine::initializePCIeBar(DeviceId cuda_device, DeviceId rocm_device,
                                            size_t bar_offset, size_t map_size)
    {
#ifndef HAVE_CUDA
        LOG_ERROR("PCIe BAR P2P requires CUDA support");
        return false;
#else
        if (cuda_device.type != DeviceType::CUDA)
        {
            LOG_ERROR("First device must be CUDA for PCIe BAR P2P");
            return false;
        }
        if (rocm_device.type != DeviceType::ROCm)
        {
            LOG_ERROR("Second device must be ROCm for PCIe BAR P2P");
            return false;
        }

        impl_->cleanup();
        impl_->cuda_device = cuda_device.ordinal;
        impl_->rocm_device = rocm_device.ordinal;

        // Discover AMD GPU BARs
        auto bars = discoverAmdBars();
        if (bars.empty())
        {
            LOG_ERROR("No AMD GPU BARs found");
            return false;
        }

        // Find the largest BAR (MI50s have 32GB BARs)
        PCIeBarInfo *best_bar = nullptr;
        for (auto &bar : bars)
        {
            if (bar.bar_size > 0 && (!best_bar || bar.bar_size > best_bar->bar_size))
            {
                best_bar = &bar;
            }
        }

        if (!best_bar)
        {
            LOG_ERROR("No AMD GPU BAR with usable size found");
            return false;
        }

        impl_->bar_info = *best_bar;
        LOG_INFO("Using AMD BAR: " << impl_->bar_info.pci_address
                                   << " (" << (impl_->bar_info.bar_size / (1024 * 1024 * 1024)) << " GB)");

        // Open BAR file
        impl_->bar_info.bar_fd = open(impl_->bar_info.sysfs_path.c_str(), O_RDWR | O_SYNC);
        if (impl_->bar_info.bar_fd < 0)
        {
            LOG_ERROR("Cannot open BAR at " << impl_->bar_info.sysfs_path
                                            << " (errno=" << errno << ")");
            LOG_ERROR("PCIe BAR access requires elevated privileges. Options:");
            LOG_ERROR("  1. Run with sudo");
            LOG_ERROR("  2. Create udev rule: sudo tee /etc/udev/rules.d/99-amd-gpu-bar.rules << 'EOF'");
            LOG_ERROR("     SUBSYSTEM==\"pci\", ATTR{vendor}==\"0x1002\", ATTR{class}==\"0x030000\", \\");
            LOG_ERROR("         RUN+=\"/bin/chmod 0666 /sys/bus/pci/devices/%k/resource0\"");
            LOG_ERROR("     EOF");
            LOG_ERROR("     sudo udevadm control --reload-rules && sudo udevadm trigger");
            LOG_ERROR("  3. One-time: sudo chmod 666 " << impl_->bar_info.sysfs_path);
            return false;
        }

        // Determine map size
        size_t actual_map_size = map_size;
        if (actual_map_size == 0)
        {
            // Default to 64MB (reasonable for transfers, not the whole BAR)
            actual_map_size = 64 * 1024 * 1024;
        }
        actual_map_size = std::min(actual_map_size, impl_->bar_info.bar_size - bar_offset);

        // mmap the BAR
        void *mapped = mmap(nullptr, actual_map_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, impl_->bar_info.bar_fd, bar_offset);
        if (mapped == MAP_FAILED)
        {
            LOG_ERROR("Failed to mmap BAR (errno=" << errno << ")");
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        impl_->bar_info.mapped_ptr = mapped;
        impl_->bar_info.mapped_size = actual_map_size;
        LOG_INFO("Mapped " << (actual_map_size / (1024 * 1024)) << " MB of BAR at offset " << bar_offset);

        // Initialize CUDA context
        CUresult err;
        err = cuInit(0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuInit failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        CUdevice cu_device;
        err = cuDeviceGet(&cu_device, cuda_device.ordinal);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDeviceGet failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        // Use primary context instead of creating a new one
        CUcontext cu_ctx;
        err = cuDevicePrimaryCtxRetain(&cu_ctx, cu_device);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDevicePrimaryCtxRetain failed: " << err);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        err = cuCtxSetCurrent(cu_ctx);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuCtxSetCurrent failed: " << err);
            cuDevicePrimaryCtxRelease(cu_device);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        // Register BAR memory with CUDA using IOMEMORY flag
        // This is the key - IOMEMORY tells CUDA this is device I/O memory
        // NOTE: This requires root privileges - CUDA's IOMEMORY registration
        // checks for CAP_SYS_ADMIN, not just file permissions
        unsigned int flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY;
        err = cuMemHostRegister(mapped, actual_map_size, flags);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuMemHostRegister with IOMEMORY failed: " << err);
            if (err == 800) // CUDA_ERROR_INVALID_VALUE
            {
                LOG_ERROR("CUDA IOMEMORY registration requires root privileges.");
                LOG_ERROR("The NVIDIA driver checks CAP_SYS_ADMIN for DMA to device I/O memory.");
                LOG_ERROR("Options:");
                LOG_ERROR("  1. Run with sudo");
                LOG_ERROR("  2. Run as root user");
                LOG_ERROR("  3. Use host-staged transfer (CrossVendorP2PEngine) instead");
            }
            cuCtxDestroy(cu_ctx);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        // Get device pointer for the registered memory
        err = cuMemHostGetDevicePointer(&impl_->cuda_bar_ptr, mapped, 0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuMemHostGetDevicePointer failed: " << err);
            cuMemHostUnregister(mapped);
            cuCtxDestroy(cu_ctx);
            munmap(mapped, actual_map_size);
            close(impl_->bar_info.bar_fd);
            impl_->bar_info.bar_fd = -1;
            return false;
        }

        impl_->bar_mapped = true;
        pcie_bar_active_ = true;

        LOG_INFO("PCIe BAR P2P initialized successfully!");
        LOG_INFO("  CUDA device pointer to AMD BAR: " << std::hex << impl_->cuda_bar_ptr << std::dec);
        LOG_INFO("  Mapped size: " << (actual_map_size / (1024 * 1024)) << " MB");

        return true;
#endif
    }

    void *DirectP2PEngine::getCudaBarPointer() const
    {
#ifdef HAVE_CUDA
        return reinterpret_cast<void *>(impl_->cuda_bar_ptr);
#else
        return nullptr;
#endif
    }

    size_t DirectP2PEngine::getBarMappedSize() const
    {
        return impl_->bar_info.mapped_size;
    }

    void *DirectP2PEngine::getBarHostPtr() const
    {
        return impl_->bar_info.mapped_ptr;
    }

    size_t DirectP2PEngine::getBarOffset() const
    {
        // The bar_offset was always 0 in our implementation since we map from the start
        // If we need to track this explicitly, we'd need to add a member variable
        return 0;
    }

    DirectP2PResult DirectP2PEngine::transferViaPCIeBar(void *cuda_ptr, size_t bar_offset,
                                                        size_t num_bytes, Direction direction)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_)
        {
            result.error_message = "PCIe BAR not initialized - call initializePCIeBar() first";
            return result;
        }

        if (bar_offset + num_bytes > impl_->bar_info.mapped_size)
        {
            result.error_message = "Transfer exceeds mapped BAR region";
            return result;
        }

        CUdeviceptr bar_addr = impl_->cuda_bar_ptr + bar_offset;
        CUdeviceptr cuda_addr = reinterpret_cast<CUdeviceptr>(cuda_ptr);

        auto start = std::chrono::high_resolution_clock::now();

        CUresult err;
        if (direction == Direction::ToNVIDIA)
        {
            // Read from AMD BAR → CUDA memory
            err = cuMemcpyDtoD(cuda_addr, bar_addr, num_bytes);
            result.transfer_path = "AMD BAR → NVIDIA (read from BAR)";
        }
        else
        {
            // Write CUDA memory → AMD BAR
            err = cuMemcpyDtoD(bar_addr, cuda_addr, num_bytes);
            result.transfer_path = "NVIDIA → AMD BAR (write to BAR)";
        }

        if (err != CUDA_SUCCESS)
        {
            result.error_message = "cuMemcpyDtoD failed: " + std::to_string(err);
            return result;
        }

        // Sync
        cuCtxSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.throughput_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (result.transfer_time_ms / 1000.0);
        result.success = true;
        result.used_pcie_bar = true;

        return result;
#endif
    }

    DirectP2PResult DirectP2PEngine::benchmarkPCIeBar(size_t num_bytes, int iterations)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_)
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        if (num_bytes > impl_->bar_info.mapped_size)
        {
            result.error_message = "Transfer size exceeds mapped BAR region";
            return result;
        }

        LOG_INFO("=== PCIe BAR Direct P2P Benchmark ===");
        LOG_INFO("Transfer size: " << (num_bytes / (1024 * 1024)) << " MB");
        LOG_INFO("Iterations: " << iterations);

        // Allocate CUDA test buffer
        CUdeviceptr cuda_buf;
        CUresult err = cuMemAlloc(&cuda_buf, num_bytes);
        if (err != CUDA_SUCCESS)
        {
            result.error_message = "Failed to allocate CUDA buffer";
            return result;
        }

        // Initialize CUDA buffer
        cuMemsetD8(cuda_buf, 0xAB, num_bytes);
        cuCtxSynchronize();

        // Warmup
        LOG_INFO("Warming up...");
        cuMemcpyDtoD(cuda_buf, impl_->cuda_bar_ptr, num_bytes);
        cuMemcpyDtoD(impl_->cuda_bar_ptr, cuda_buf, num_bytes);
        cuCtxSynchronize();

        // Benchmark AMD → NVIDIA (read from BAR)
        double read_total_ms = 0.0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(cuda_buf, impl_->cuda_bar_ptr, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            read_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_read_ms = read_total_ms / iterations;
        result.read_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_read_ms / 1000.0);

        // Benchmark NVIDIA → AMD (write to BAR)
        double write_total_ms = 0.0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(impl_->cuda_bar_ptr, cuda_buf, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            write_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_write_ms = write_total_ms / iterations;
        result.write_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_write_ms / 1000.0);

        // Overall average
        result.throughput_gbps = (result.read_gbps + result.write_gbps) / 2.0;
        result.transfer_time_ms = (avg_read_ms + avg_write_ms) / 2.0;
        result.success = true;
        result.used_pcie_bar = true;
        result.transfer_path = "PCIe BAR Direct P2P";

        LOG_INFO("\n╔══════════════════════════════════════════════════════════════╗");
        LOG_INFO("║           PCIe BAR DIRECT P2P BENCHMARK RESULTS              ║");
        LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
        LOG_INFO("║ AMD → NVIDIA (read from BAR):  " << std::fixed << std::setprecision(2)
                                                     << std::setw(6) << result.read_gbps << " GB/s                    ║");
        LOG_INFO("║ NVIDIA → AMD (write to BAR):   " << std::fixed << std::setprecision(2)
                                                     << std::setw(6) << result.write_gbps << " GB/s                    ║");
        LOG_INFO("╚══════════════════════════════════════════════════════════════╝");

        cuMemFree(cuda_buf);
        return result;
#endif
    }

    std::unique_ptr<DirectP2PBuffer> DirectP2PEngine::allocateExportable(DeviceId device, size_t size)
    {
        auto buffer = std::make_unique<DirectP2PBuffer>();
        buffer->size = size;
        buffer->owner_device = device;

#ifdef HAVE_CUDA
        if (device.type == DeviceType::CUDA)
        {
            int fd = -1;
            buffer->device_ptr = cuda_direct_p2p::allocateExportable(device.ordinal, size, &fd);
            buffer->dmabuf_fd = fd;
            buffer->is_exported = (fd >= 0);

            if (buffer->device_ptr)
            {
                LOG_INFO("Allocated exportable CUDA buffer: " << size << " bytes, fd=" << fd);
            }
        }
#endif

#ifdef HAVE_ROCM
        if (device.type == DeviceType::ROCm)
        {
            int fd = -1;
            buffer->device_ptr = rocm_direct_p2p::allocateExportable(device.ordinal, size, &fd);
            buffer->dmabuf_fd = fd;
            buffer->is_exported = (fd >= 0);

            if (buffer->device_ptr)
            {
                LOG_INFO("Allocated exportable ROCm buffer: " << size << " bytes, fd=" << fd);
            }
        }
#endif

        if (!buffer->device_ptr)
        {
            return nullptr;
        }

        return buffer;
    }

    void *DirectP2PEngine::importBuffer(const DirectP2PBuffer &buffer, DeviceId target_device)
    {
        if (buffer.dmabuf_fd < 0)
        {
            LOG_ERROR("Buffer not exported (no DMA-BUF fd)");
            return nullptr;
        }

        if (buffer.owner_device.type == target_device.type)
        {
            LOG_ERROR("Cannot import to same vendor");
            return nullptr;
        }

#ifdef HAVE_CUDA
        if (target_device.type == DeviceType::CUDA)
        {
            return cuda_direct_p2p::importDmaBuf(target_device.ordinal,
                                                 buffer.dmabuf_fd, buffer.size);
        }
#endif

#ifdef HAVE_ROCM
        if (target_device.type == DeviceType::ROCm)
        {
            return rocm_direct_p2p::importDmaBuf(target_device.ordinal,
                                                 buffer.dmabuf_fd, buffer.size);
        }
#endif

        return nullptr;
    }

    DirectP2PResult DirectP2PEngine::transfer(DeviceId src_device, const void *src_ptr,
                                              DeviceId dst_device, void *dst_ptr,
                                              size_t num_bytes)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
        result.error_message = "Requires both CUDA and ROCm";
        return result;
#else
        auto start = std::chrono::high_resolution_clock::now();

        // Try direct copy first (this would work if DMA-BUF import succeeded
        // and gives us a valid cross-vendor pointer)
        bool direct_success = false;

        // For now, we can't do truly direct copy without explicit DMA-BUF setup
        // The memory regions aren't mapped across vendors by default

        // Fall back to pipelined transfer
        if (!direct_success && impl_->fallback_engine)
        {
            auto fallback_result = impl_->fallback_engine->transfer(src_ptr, dst_ptr, num_bytes);
            if (fallback_result.success)
            {
                result.throughput_gbps = fallback_result.throughput_gbps;
                result.success = true;
                result.fell_back_to_staged = true;
            }
            else
            {
                result.error_message = "Fallback transfer also failed";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (result.success)
        {
            result.throughput_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                     (result.transfer_time_ms / 1000.0);
        }

        return result;
#endif
    }

    DirectP2PResult DirectP2PEngine::benchmark(size_t num_bytes, int iterations)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
        result.error_message = "Requires both CUDA and ROCm";
        return result;
#else
        LOG_INFO("=== DirectP2PEngine Benchmark ===");
        LOG_INFO("Testing " << (num_bytes / (1024 * 1024)) << " MB transfer, "
                            << iterations << " iterations");

        // Probe capabilities
        auto caps = probeCapabilities();
        LOG_INFO(caps.describe());

        // Try allocating exportable buffer on CUDA side
        DeviceId cuda_dev = DeviceId::cuda(impl_->cuda_device);
        DeviceId rocm_dev = DeviceId::rocm(impl_->rocm_device);

        LOG_INFO("\n--- Testing DMA-BUF Export/Import ---");

        auto cuda_buffer = allocateExportable(cuda_dev, num_bytes);
        if (cuda_buffer && cuda_buffer->is_exported)
        {
            LOG_INFO("CUDA buffer exported with fd=" << cuda_buffer->dmabuf_fd);

            // Try to import into ROCm
            void *rocm_imported = importBuffer(*cuda_buffer, rocm_dev);
            if (rocm_imported)
            {
                LOG_INFO("Successfully imported into ROCm as ptr=" << rocm_imported);
                result.used_dmabuf = true;

                // Try direct copy!
                LOG_INFO("\n--- Attempting Direct Cross-Vendor Copy ---");

                // Initialize source buffer
                // (would need to do this via CUDA, but for benchmark we'll skip)

                // The actual copy would be:
                // rocm_direct_p2p::asyncCopy(impl_->rocm_device,
                //                            rocm_imported,      // dst (ROCm view of CUDA memory)
                //                            cuda_buffer->device_ptr, // src (native CUDA ptr)
                //                            num_bytes,
                //                            impl_->rocm_stream);

                LOG_INFO("Direct copy would go here - but needs same physical memory!");
                rocm_direct_p2p::releaseImported(impl_->rocm_device, rocm_imported);
            }
            else
            {
                LOG_WARN("Failed to import CUDA buffer into ROCm");
            }
        }
        else
        {
            LOG_WARN("CUDA exportable buffer allocation failed or not exported");
        }

        // Fall back to pipelined benchmark
        LOG_INFO("\n--- Fallback: Pipelined Host-Staged Transfer ---");

        if (impl_->fallback_engine)
        {
            auto fallback_result = impl_->fallback_engine->benchmark(num_bytes, iterations);
            result.throughput_gbps = fallback_result.throughput_gbps;
            result.success = fallback_result.success;
            result.fell_back_to_staged = true;

            LOG_INFO("Pipelined transfer: " << result.throughput_gbps << " GB/s");
        }

        return result;
#endif
    }

    //--------------------------------------------------------------------------
    // Multi-GPU and Concurrent Transfer Implementation
    //--------------------------------------------------------------------------

    bool DirectP2PEngine::initializePCIeBarMultiGPU(DeviceId cuda_device,
                                                    const std::vector<DeviceId> &rocm_devices,
                                                    size_t map_size)
    {
#ifndef HAVE_CUDA
        LOG_ERROR("PCIe BAR P2P requires CUDA support");
        return false;
#else
        if (cuda_device.type != DeviceType::CUDA)
        {
            LOG_ERROR("First device must be CUDA for PCIe BAR P2P");
            return false;
        }

        for (const auto &dev : rocm_devices)
        {
            if (dev.type != DeviceType::ROCm)
            {
                LOG_ERROR("All target devices must be ROCm for PCIe BAR P2P");
                return false;
            }
        }

        impl_->cleanup();
        impl_->cuda_device = cuda_device.ordinal;
        impl_->mapped_bars.clear();
        impl_->rocm_ordinal_to_bar_idx.clear();

        // Discover all AMD GPU BARs
        auto discovered_bars = discoverAmdBars();
        if (discovered_bars.empty())
        {
            LOG_ERROR("No AMD GPU BARs found");
            return false;
        }

        LOG_INFO("Found " << discovered_bars.size() << " AMD GPU BARs");

        // Initialize CUDA context
        CUresult err = cuInit(0);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuInit failed: " << err);
            return false;
        }

        CUdevice cu_device;
        err = cuDeviceGet(&cu_device, cuda_device.ordinal);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDeviceGet failed: " << err);
            return false;
        }
        impl_->cu_device = cu_device;

        CUcontext cu_ctx;
        err = cuDevicePrimaryCtxRetain(&cu_ctx, cu_device);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuDevicePrimaryCtxRetain failed: " << err);
            return false;
        }
        impl_->cuda_ctx = cu_ctx;

        err = cuCtxSetCurrent(cu_ctx);
        if (err != CUDA_SUCCESS)
        {
            LOG_ERROR("cuCtxSetCurrent failed: " << err);
            cuDevicePrimaryCtxRelease(cu_device);
            return false;
        }

        // Create streams for overlapped operations
        CUstream read_stream, write_stream;
        err = cuStreamCreate(&read_stream, CU_STREAM_NON_BLOCKING);
        if (err == CUDA_SUCCESS)
        {
            impl_->cuda_stream_read = reinterpret_cast<void *>(read_stream);
        }
        err = cuStreamCreate(&write_stream, CU_STREAM_NON_BLOCKING);
        if (err == CUDA_SUCCESS)
        {
            impl_->cuda_stream_write = reinterpret_cast<void *>(write_stream);
        }

        // Determine map size per BAR
        size_t actual_map_size = map_size > 0 ? map_size : 64 * 1024 * 1024; // Default 64 MB

        // Map BAR for each requested ROCm device
        size_t bar_idx = 0;
        for (const auto &rocm_dev : rocm_devices)
        {
            // Find matching BAR by device ordinal
            // Note: This is heuristic - BAR order may not match device ordinal
            // For now, use ordinal as index into discovered BARs
            if (bar_idx >= discovered_bars.size())
            {
                LOG_WARN("Not enough AMD GPU BARs for all requested devices");
                break;
            }

            PCIeBarInfo &bar = discovered_bars[bar_idx];
            LOG_INFO("Mapping BAR for ROCm device " << rocm_dev.ordinal
                                                    << " from " << bar.pci_address);

            // Open BAR file
            bar.bar_fd = open(bar.sysfs_path.c_str(), O_RDWR | O_SYNC);
            if (bar.bar_fd < 0)
            {
                LOG_ERROR("Cannot open BAR at " << bar.sysfs_path << " (errno=" << errno << ")");
                LOG_ERROR("Run: sudo chmod 666 " << bar.sysfs_path);
                continue;
            }

            // Map the BAR
            size_t to_map = std::min(actual_map_size, bar.bar_size);
            void *mapped = mmap(nullptr, to_map, PROT_READ | PROT_WRITE,
                                MAP_SHARED, bar.bar_fd, 0);
            if (mapped == MAP_FAILED)
            {
                LOG_ERROR("Failed to mmap BAR for " << bar.pci_address);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                continue;
            }

            bar.mapped_ptr = mapped;
            bar.mapped_size = to_map;
            bar.device_ordinal = rocm_dev.ordinal;

            // Register with CUDA
            unsigned int flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY;
            err = cuMemHostRegister(mapped, to_map, flags);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemHostRegister failed for " << bar.pci_address << ": " << err);
                munmap(mapped, to_map);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                bar.mapped_ptr = nullptr;
                continue;
            }

            // Get CUDA device pointer
            CUdeviceptr cuda_ptr;
            err = cuMemHostGetDevicePointer(&cuda_ptr, mapped, 0);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemHostGetDevicePointer failed: " << err);
                cuMemHostUnregister(mapped);
                munmap(mapped, to_map);
                close(bar.bar_fd);
                bar.bar_fd = -1;
                bar.mapped_ptr = nullptr;
                continue;
            }

            bar.cuda_device_ptr = reinterpret_cast<void *>(cuda_ptr);

            // Store the mapping
            impl_->rocm_ordinal_to_bar_idx[rocm_dev.ordinal] = impl_->mapped_bars.size();
            impl_->mapped_bars.push_back(std::move(bar));

            LOG_INFO("  Mapped " << (to_map / (1024 * 1024)) << " MB, CUDA ptr: "
                                 << std::hex << cuda_ptr << std::dec);

            bar_idx++;
        }

        if (impl_->mapped_bars.empty())
        {
            LOG_ERROR("Failed to map any AMD GPU BARs");
            return false;
        }

        pcie_bar_active_ = true;
        LOG_INFO("PCIe BAR Multi-GPU initialized: " << impl_->mapped_bars.size() << " BARs mapped");

        return true;
#endif
    }

    size_t DirectP2PEngine::getNumMappedBars() const
    {
        return impl_->mapped_bars.size();
    }

    void *DirectP2PEngine::getCudaBarPointerForDevice(int rocm_ordinal) const
    {
        auto it = impl_->rocm_ordinal_to_bar_idx.find(rocm_ordinal);
        if (it == impl_->rocm_ordinal_to_bar_idx.end())
        {
            return nullptr;
        }
        return impl_->mapped_bars[it->second].cuda_device_ptr;
    }

    DirectP2PResult DirectP2PEngine::broadcastToAMD(const void *cuda_src, size_t num_bytes,
                                                    void *stream)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes * impl_->mapped_bars.size();

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        CUstream cu_stream = stream ? reinterpret_cast<CUstream>(stream)
                                    : reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        auto start = std::chrono::high_resolution_clock::now();

        // Write to all AMD BARs (uses fast posted writes ~2.65 GB/s per target)
        // All writes share PCIe bandwidth but simplify programming
        for (auto &bar : impl_->mapped_bars)
        {
            if (num_bytes > bar.mapped_size)
            {
                LOG_WARN("Transfer size exceeds BAR mapped size for device " << bar.device_ordinal);
                continue;
            }

            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(cuda_src);

            CUresult err = cuMemcpyDtoDAsync(bar_ptr, src_ptr, num_bytes, cu_stream);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemcpyDtoDAsync to BAR " << bar.device_ordinal << " failed: " << err);
            }
        }

        // Wait for all writes to complete
        cuStreamSynchronize(cu_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Total throughput = total bytes / time
        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.write_gbps = result.throughput_gbps;
        result.dual_write_gbps = result.throughput_gbps;
        result.success = true;
        result.used_pcie_bar = true;
        result.transfer_path = "NVIDIA -> " + std::to_string(impl_->mapped_bars.size()) + "x AMD (broadcast)";

        return result;
#endif
    }

    DirectP2PResult DirectP2PEngine::gatherFromAMDOverlapped(void *cuda_dst, size_t num_bytes,
                                                             void *stream)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes * impl_->mapped_bars.size();

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        // Use separate streams for overlapped reads
        CUstream read_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_read);
        CUstream write_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        auto start = std::chrono::high_resolution_clock::now();

        // Launch reads from all AMD BARs using different streams for overlap
        // This achieves ~1.22 GB/s total from 2 BARs (vs ~0.79 GB/s sequential)
        size_t offset = 0;
        for (size_t i = 0; i < impl_->mapped_bars.size(); ++i)
        {
            auto &bar = impl_->mapped_bars[i];
            if (num_bytes > bar.mapped_size)
            {
                LOG_WARN("Transfer size exceeds BAR mapped size for device " << bar.device_ordinal);
                continue;
            }

            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(cuda_dst) + offset;

            // Alternate streams for overlap
            CUstream s = (i % 2 == 0) ? read_stream : write_stream;
            if (!s)
                s = read_stream;

            CUresult err = cuMemcpyDtoDAsync(dst_ptr, bar_ptr, num_bytes, s);
            if (err != CUDA_SUCCESS)
            {
                LOG_ERROR("cuMemcpyDtoDAsync from BAR " << bar.device_ordinal << " failed: " << err);
            }

            offset += num_bytes;
        }

        // Wait for all reads to complete
        if (read_stream)
            cuStreamSynchronize(read_stream);
        if (write_stream)
            cuStreamSynchronize(write_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.read_gbps = result.throughput_gbps;
        result.dual_read_gbps = result.throughput_gbps;
        result.success = true;
        result.used_pcie_bar = true;
        result.used_overlap = true;
        result.transfer_path = std::to_string(impl_->mapped_bars.size()) + "x AMD -> NVIDIA (overlapped gather)";

        return result;
#endif
    }

    DirectP2PResult DirectP2PEngine::transferOverlapped(void *cuda_read_dst, size_t read_bar_offset,
                                                        size_t read_bytes,
                                                        const void *cuda_write_src, size_t write_bar_offset,
                                                        size_t write_bytes)
    {
        DirectP2PResult result;
        result.bytes_transferred = read_bytes + write_bytes;

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_ || impl_->mapped_bars.size() < 2)
        {
            result.error_message = "Need at least 2 AMD BARs for overlapped transfer";
            return result;
        }

        CUstream read_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_read);
        CUstream write_stream = reinterpret_cast<CUstream>(impl_->cuda_stream_write);

        if (!read_stream || !write_stream)
        {
            result.error_message = "Streams not initialized";
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Simultaneously:
        // - Read from BAR 0 (AMD->NVIDIA) on read_stream
        // - Write to BAR 1 (NVIDIA->AMD) on write_stream
        // This achieves ~1.58 GB/s effective (30% faster than sequential)

        if (cuda_read_dst && read_bytes > 0)
        {
            auto &bar0 = impl_->mapped_bars[0];
            if (read_bar_offset + read_bytes <= bar0.mapped_size)
            {
                CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar0.cuda_device_ptr) + read_bar_offset;
                CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(cuda_read_dst);
                cuMemcpyDtoDAsync(dst_ptr, bar_ptr, read_bytes, read_stream);
            }
        }

        if (cuda_write_src && write_bytes > 0)
        {
            auto &bar1 = impl_->mapped_bars[1];
            if (write_bar_offset + write_bytes <= bar1.mapped_size)
            {
                CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar1.cuda_device_ptr) + write_bar_offset;
                CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(cuda_write_src);
                cuMemcpyDtoDAsync(bar_ptr, src_ptr, write_bytes, write_stream);
            }
        }

        // Wait for both to complete
        cuStreamSynchronize(read_stream);
        cuStreamSynchronize(write_stream);

        auto end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps = (result.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) /
                                 (result.transfer_time_ms / 1000.0);
        result.concurrent_gbps = result.throughput_gbps;
        result.success = true;
        result.used_pcie_bar = true;
        result.used_overlap = true;
        result.transfer_path = "Overlapped: AMD->NVIDIA + NVIDIA->AMD";

        return result;
#endif
    }

    DirectP2PResult DirectP2PEngine::benchmarkAllModes(size_t num_bytes, int iterations)
    {
        DirectP2PResult result;
        result.bytes_transferred = num_bytes;

#ifndef HAVE_CUDA
        result.error_message = "Requires CUDA";
        return result;
#else
        if (!pcie_bar_active_ || impl_->mapped_bars.empty())
        {
            result.error_message = "PCIe BAR not initialized";
            return result;
        }

        LOG_INFO("");
        LOG_INFO("================================================================");
        LOG_INFO("       COMPREHENSIVE PCIe BAR P2P BENCHMARK                     ");
        LOG_INFO("================================================================");
        LOG_INFO(" Transfer size: " << (num_bytes / (1024 * 1024)) << " MB per operation");
        LOG_INFO(" Iterations: " << iterations);
        LOG_INFO(" Mapped BARs: " << impl_->mapped_bars.size());
        LOG_INFO("----------------------------------------------------------------");

        // Allocate test buffer
        CUdeviceptr cuda_buf;
        CUresult err = cuMemAlloc(&cuda_buf, num_bytes);
        if (err != CUDA_SUCCESS)
        {
            result.error_message = "Failed to allocate CUDA buffer";
            return result;
        }
        cuMemsetD8(cuda_buf, 0xAB, num_bytes);

        // Warmup
        for (auto &bar : impl_->mapped_bars)
        {
            CUdeviceptr bar_ptr = reinterpret_cast<CUdeviceptr>(bar.cuda_device_ptr);
            cuMemcpyDtoD(bar_ptr, cuda_buf, std::min(num_bytes, bar.mapped_size));
            cuMemcpyDtoD(cuda_buf, bar_ptr, std::min(num_bytes, bar.mapped_size));
        }
        cuCtxSynchronize();

        // 1. Single write (NVIDIA -> AMD)
        double write_total = 0;
        auto &bar0 = impl_->mapped_bars[0];
        CUdeviceptr bar0_ptr = reinterpret_cast<CUdeviceptr>(bar0.cuda_device_ptr);
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(bar0_ptr, cuda_buf, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            write_total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        result.write_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / ((write_total / iterations) / 1000.0);
        LOG_INFO(" Single WRITE (NVIDIA->AMD):  " << std::fixed << std::setprecision(2)
                                                  << result.write_gbps << " GB/s");

        // 2. Single read (AMD -> NVIDIA)
        double read_total = 0;
        for (int i = 0; i < iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuMemcpyDtoD(cuda_buf, bar0_ptr, num_bytes);
            cuCtxSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            read_total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        result.read_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / ((read_total / iterations) / 1000.0);
        LOG_INFO(" Single READ (AMD->NVIDIA):   " << std::fixed << std::setprecision(2)
                                                  << result.read_gbps << " GB/s");

        // 3. Dual write (if 2+ BARs)
        if (impl_->mapped_bars.size() >= 2)
        {
            CUstream s1, s2;
            cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING);
            cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING);

            CUdeviceptr bar1_ptr = reinterpret_cast<CUdeviceptr>(impl_->mapped_bars[1].cuda_device_ptr);

            double dual_write_total = 0;
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(bar0_ptr, cuda_buf, num_bytes, s1);
                cuMemcpyDtoDAsync(bar1_ptr, cuda_buf, num_bytes, s2);
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                dual_write_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.dual_write_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                     ((dual_write_total / iterations) / 1000.0);
            LOG_INFO(" Dual WRITE (NVIDIA->2xAMD):  " << std::fixed << std::setprecision(2)
                                                      << result.dual_write_gbps << " GB/s (total)");

            // 4. Dual read
            double dual_read_total = 0;
            CUdeviceptr cuda_buf2;
            cuMemAlloc(&cuda_buf2, num_bytes);
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(cuda_buf, bar0_ptr, num_bytes, s1);
                cuMemcpyDtoDAsync(cuda_buf2, bar1_ptr, num_bytes, s2);
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                dual_read_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.dual_read_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                    ((dual_read_total / iterations) / 1000.0);
            LOG_INFO(" Dual READ (2xAMD->NVIDIA):   " << std::fixed << std::setprecision(2)
                                                      << result.dual_read_gbps << " GB/s (total)");

            // 5. Concurrent read+write
            double concurrent_total = 0;
            for (int i = 0; i < iterations; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                cuMemcpyDtoDAsync(cuda_buf, bar0_ptr, num_bytes, s1);  // Read from BAR0
                cuMemcpyDtoDAsync(bar1_ptr, cuda_buf2, num_bytes, s2); // Write to BAR1
                cuStreamSynchronize(s1);
                cuStreamSynchronize(s2);
                auto end = std::chrono::high_resolution_clock::now();
                concurrent_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            result.concurrent_gbps = (2 * num_bytes / (1024.0 * 1024.0 * 1024.0)) /
                                     ((concurrent_total / iterations) / 1000.0);
            LOG_INFO(" Concurrent READ+WRITE:       " << std::fixed << std::setprecision(2)
                                                      << result.concurrent_gbps << " GB/s (effective)");
            result.used_overlap = true;

            cuMemFree(cuda_buf2);
            cuStreamDestroy(s1);
            cuStreamDestroy(s2);
        }

        LOG_INFO("----------------------------------------------------------------");
        LOG_INFO(" Write/Read asymmetry: " << std::fixed << std::setprecision(1)
                                           << (result.write_gbps / result.read_gbps) << "x");
        if (result.concurrent_gbps > 0)
        {
            double seq_time = (num_bytes / result.read_gbps + num_bytes / result.write_gbps);
            double conc_time = (2 * num_bytes / result.concurrent_gbps);
            double speedup = seq_time / conc_time;
            LOG_INFO(" Overlap speedup vs sequential: " << std::fixed << std::setprecision(1)
                                                        << ((speedup - 1) * 100) << "%");
        }
        LOG_INFO("================================================================");

        result.success = true;
        result.used_pcie_bar = true;
        result.throughput_gbps = (result.read_gbps + result.write_gbps) / 2.0;

        cuMemFree(cuda_buf);
        return result;
#endif
    }

} // namespace llaminar2
