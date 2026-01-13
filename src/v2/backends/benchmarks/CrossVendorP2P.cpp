/**
 * @file CrossVendorP2P.cpp
 * @brief Cross-vendor P2P transfer engine implementation
 *
 * This file coordinates CUDA and ROCm operations via the backend-specific
 * helper functions, without including either GPU runtime header directly.
 *
 * The key optimization is double-buffered pipelined transfers:
 *
 *   Time →
 *   Buffer A: [D2H from src]---->[H2D to dst]
 *   Buffer B:        [D2H from src]---->[H2D to dst]
 *
 * With pipelining, we overlap D2H and H2D operations on different buffers,
 * achieving throughput closer to min(d2h_rate, h2d_rate) instead of
 * harmonic_mean(d2h_rate, h2d_rate).
 */

#include "CrossVendorP2P.h"
#include "utils/Logger.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <algorithm>

#ifdef HAVE_CUDA
#include "CrossVendorP2P_CUDA.h"
#endif

#ifdef HAVE_ROCM
#include "CrossVendorP2P_ROCm.h"
#endif

namespace llaminar2
{

    //--------------------------------------------------------------------------
    // Implementation Details
    //--------------------------------------------------------------------------

    struct CrossVendorP2PEngine::Impl
    {
        // Host staging buffers (dual-pinned)
        std::vector<void *> staging_buffers;
        size_t buffer_size = 0;

        // CUDA resources
        void *cuda_stream = nullptr;
        int cuda_device = -1;

        // ROCm resources  
        void *rocm_stream = nullptr;
        void *rocm_stream_dst = nullptr;  // Second stream for same-vendor ROCm transfers
        int rocm_device = -1;
        int rocm_device_dst = -1;  // Second device for same-vendor ROCm transfers

        // Which runtime handles source vs destination
        bool cuda_is_source = false;
        
        // Same-vendor transfer mode
        bool same_vendor_rocm = false;  // true = ROCm↔ROCm transfer
        bool same_vendor_cuda = false;  // true = CUDA↔CUDA transfer

        ~Impl()
        {
            cleanup();
        }

        void cleanup()
        {
            // Cleanup staging buffers
            for (void *buf : staging_buffers)
            {
                if (buf)
                {
#ifdef HAVE_CUDA
                    if (!same_vendor_rocm)
                        cuda_p2p::unregisterHostMemory(buf);
#endif
#ifdef HAVE_ROCM
                    if (!same_vendor_cuda)
                        rocm_p2p::unregisterHostMemory(buf);
#endif
                    std::free(buf);
                }
            }
            staging_buffers.clear();

            // Cleanup streams
#ifdef HAVE_CUDA
            if (cuda_stream)
            {
                cuda_p2p::destroyStream(cuda_stream);
                cuda_stream = nullptr;
            }
#endif
#ifdef HAVE_ROCM
            if (rocm_stream)
            {
                rocm_p2p::destroyStream(rocm_stream);
                rocm_stream = nullptr;
            }
            if (rocm_stream_dst)
            {
                rocm_p2p::destroyStream(rocm_stream_dst);
                rocm_stream_dst = nullptr;
            }
#endif
        }
    };

    //--------------------------------------------------------------------------
    // CrossVendorP2PEngine
    //--------------------------------------------------------------------------

    CrossVendorP2PEngine::CrossVendorP2PEngine(const CrossVendorP2PConfig &config)
        : impl_(std::make_unique<Impl>()), config_(config)
    {
    }

    CrossVendorP2PEngine::~CrossVendorP2PEngine() = default;

    CrossVendorP2PEngine::CrossVendorP2PEngine(CrossVendorP2PEngine &&) noexcept = default;
    CrossVendorP2PEngine &CrossVendorP2PEngine::operator=(CrossVendorP2PEngine &&) noexcept = default;

    bool CrossVendorP2PEngine::initialize(DeviceId src, DeviceId dst)
    {
        // Check for same-vendor mode
        bool same_vendor = (src.type == dst.type);
        
        if (same_vendor && !config_.allow_same_vendor)
        {
            LOG_ERROR("CrossVendorP2PEngine: src and dst are same vendor, set allow_same_vendor=true");
            return false;
        }
        
        if (same_vendor && src.ordinal == dst.ordinal)
        {
            LOG_ERROR("CrossVendorP2PEngine: src and dst are the same device");
            return false;
        }

        if (src.type != DeviceType::CUDA && src.type != DeviceType::ROCm)
        {
            LOG_ERROR("CrossVendorP2PEngine: src must be CUDA or ROCm");
            return false;
        }
        
        if (dst.type != DeviceType::CUDA && dst.type != DeviceType::ROCm)
        {
            LOG_ERROR("CrossVendorP2PEngine: dst must be CUDA or ROCm");
            return false;
        }

        impl_->cleanup();

        src_device_ = src;
        dst_device_ = dst;
        impl_->same_vendor_rocm = same_vendor && (src.type == DeviceType::ROCm);
        impl_->same_vendor_cuda = same_vendor && (src.type == DeviceType::CUDA);

        // Allocate staging buffers with 4KB alignment for DMA
        impl_->buffer_size = config_.buffer_size;
        impl_->staging_buffers.resize(config_.num_buffers);

        for (int i = 0; i < config_.num_buffers; ++i)
        {
            void *buf = std::aligned_alloc(4096, impl_->buffer_size);
            if (!buf)
            {
                LOG_ERROR("Failed to allocate staging buffer " << i);
                impl_->cleanup();
                return false;
            }

            // Initialize for first-touch NUMA placement
            std::memset(buf, 0, impl_->buffer_size);
            impl_->staging_buffers[i] = buf;

#ifdef HAVE_CUDA
            if (!impl_->same_vendor_rocm)
            {
                // Register with CUDA for fast DMA
                if (!cuda_p2p::registerHostMemory(buf, impl_->buffer_size))
                {
                    LOG_WARN("Failed to register buffer " << i << " with CUDA");
                }
            }
#endif

#ifdef HAVE_ROCM
            if (!impl_->same_vendor_cuda)
            {
                // Register with ROCm for fast DMA
                if (!rocm_p2p::registerHostMemory(buf, impl_->buffer_size))
                {
                    LOG_WARN("Failed to register buffer " << i << " with ROCm");
                }
            }
#endif
        }

        // Setup device ordinals and streams based on mode
        if (impl_->same_vendor_rocm)
        {
#ifdef HAVE_ROCM
            // ROCm↔ROCm: two ROCm devices, no CUDA
            impl_->rocm_device = src.ordinal;      // Source device
            impl_->rocm_device_dst = dst.ordinal;  // Destination device
            impl_->cuda_device = -1;
            
            impl_->rocm_stream = rocm_p2p::createStream(impl_->rocm_device);
            if (!impl_->rocm_stream)
            {
                LOG_ERROR("Failed to create ROCm source stream");
                impl_->cleanup();
                return false;
            }
            
            impl_->rocm_stream_dst = rocm_p2p::createStream(impl_->rocm_device_dst);
            if (!impl_->rocm_stream_dst)
            {
                LOG_ERROR("Failed to create ROCm destination stream");
                impl_->cleanup();
                return false;
            }
#else
            LOG_ERROR("ROCm not available");
            return false;
#endif
        }
        else if (impl_->same_vendor_cuda)
        {
            LOG_ERROR("Same-vendor CUDA transfers not yet implemented");
            return false;
        }
        else
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            LOG_ERROR("Cross-vendor requires both HAVE_CUDA and HAVE_ROCM");
            return false;
#else
            // Cross-vendor: one CUDA, one ROCm
            impl_->cuda_is_source = (src.type == DeviceType::CUDA);
            
            if (impl_->cuda_is_source)
            {
                impl_->cuda_device = src.ordinal;
                impl_->rocm_device = dst.ordinal;
            }
            else
            {
                impl_->rocm_device = src.ordinal;
                impl_->cuda_device = dst.ordinal;
            }

            impl_->cuda_stream = cuda_p2p::createStream(impl_->cuda_device);
            if (!impl_->cuda_stream)
            {
                LOG_ERROR("Failed to create CUDA stream");
                impl_->cleanup();
                return false;
            }

            impl_->rocm_stream = rocm_p2p::createStream(impl_->rocm_device);
            if (!impl_->rocm_stream)
            {
                LOG_ERROR("Failed to create ROCm stream");
                impl_->cleanup();
                return false;
            }
#endif
        }

        initialized_ = true;

        // Auto-tune chunk size if requested
        if (config_.auto_tune)
        {
            LOG_INFO("CrossVendorP2PEngine: Running auto-tune for optimal chunk size...");
            autoTuneChunkSize();
        }

        std::string mode = impl_->same_vendor_rocm ? "ROCm↔ROCm" : 
                          (impl_->same_vendor_cuda ? "CUDA↔CUDA" : "Cross-Vendor");
        LOG_INFO("P2PEngine initialized (" << mode << "): " << src << " -> " << dst
                 << " with " << config_.num_buffers << " x " 
                 << (impl_->buffer_size / (1024 * 1024)) << " MB staging buffers"
                 << ", chunk_size=" << (config_.chunk_size / (1024 * 1024)) << " MB");

        return true;
    }

    CrossVendorP2PResult CrossVendorP2PEngine::transfer(const void *d_src, void *d_dst, size_t num_bytes)
    {
        CrossVendorP2PResult result;
        result.bytes_transferred = num_bytes;

        if (!initialized_)
        {
            LOG_ERROR("CrossVendorP2PEngine not initialized");
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Set up function pointers based on transfer mode
        std::function<bool(void*, int, void*, const void*, size_t)> src_d2h;
        std::function<bool(void*, int, void*, const void*, size_t)> dst_h2d;
        std::function<void(void*)> src_sync;
        std::function<void(void*)> dst_sync;
        void *src_stream = nullptr;
        void *dst_stream = nullptr;
        int src_ord = 0;
        int dst_ord = 0;

#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            // ROCm↔ROCm transfer
            src_d2h = rocm_p2p::asyncD2H;
            dst_h2d = rocm_p2p::asyncH2D;
            src_sync = rocm_p2p::syncStream;
            dst_sync = rocm_p2p::syncStream;
            src_stream = impl_->rocm_stream;
            dst_stream = impl_->rocm_stream_dst;
            src_ord = impl_->rocm_device;
            dst_ord = impl_->rocm_device_dst;
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            // Cross-vendor transfer
            src_d2h = impl_->cuda_is_source ? cuda_p2p::asyncD2H : rocm_p2p::asyncD2H;
            dst_h2d = impl_->cuda_is_source ? rocm_p2p::asyncH2D : cuda_p2p::asyncH2D;
            src_sync = impl_->cuda_is_source ? cuda_p2p::syncStream : rocm_p2p::syncStream;
            dst_sync = impl_->cuda_is_source ? rocm_p2p::syncStream : cuda_p2p::syncStream;
            src_stream = impl_->cuda_is_source ? impl_->cuda_stream : impl_->rocm_stream;
            dst_stream = impl_->cuda_is_source ? impl_->rocm_stream : impl_->cuda_stream;
            src_ord = impl_->cuda_is_source ? impl_->cuda_device : impl_->rocm_device;
            dst_ord = impl_->cuda_is_source ? impl_->rocm_device : impl_->cuda_device;
        }
#else
        {
            LOG_ERROR("Required GPU backend not available");
            return result;
        }
#endif

        size_t chunk_size = std::min(config_.chunk_size, impl_->buffer_size);

        if (config_.enable_pipelining && config_.num_buffers >= 2)
        {
            // True concurrent pipelined transfer with double buffering
            //
            // Timeline (D2H on src GPU, H2D on dst GPU - they can overlap!):
            //   Src GPU: |--D2H[0]--|--D2H[1]--|--D2H[2]--|--D2H[3]--|
            //   Dst GPU:      |--H2D[0]--|--H2D[1]--|--H2D[2]--|--H2D[3]--|
            //   Buffer:   A        B        A        B        A        B
            //
            // The trick: D2H into buffer B can run while H2D from buffer A runs,
            // because they're on DIFFERENT GPUs with DIFFERENT streams!

            size_t num_chunks = (num_bytes + chunk_size - 1) / chunk_size;
            
            // Track which chunks are in flight
            // For double buffer: buffer 0 handles even chunks, buffer 1 handles odd chunks
            size_t chunks_started_d2h = 0;  // How many chunks have started D2H
            size_t chunks_started_h2d = 0;  // How many chunks have started H2D
            size_t chunks_completed = 0;    // How many chunks fully done

            // Start first D2H
            {
                size_t offset = 0;
                size_t this_chunk = std::min(chunk_size, num_bytes - offset);
                const char *src_ptr = static_cast<const char *>(d_src) + offset;
                
                if (!src_d2h(src_stream, src_ord, impl_->staging_buffers[0], src_ptr, this_chunk))
                {
                    LOG_ERROR("D2H failed for chunk 0");
                    return result;
                }
                chunks_started_d2h = 1;
            }

            while (chunks_completed < num_chunks)
            {
                // Wait for the oldest D2H that hasn't been followed by H2D yet
                if (chunks_started_h2d < chunks_started_d2h)
                {
                    src_sync(src_stream);  // Ensure D2H for this chunk is done
                    
                    // Start H2D for this chunk
                    int buf_idx = chunks_started_h2d % 2;
                    size_t offset = chunks_started_h2d * chunk_size;
                    size_t this_chunk = std::min(chunk_size, num_bytes - offset);
                    char *dst_ptr = static_cast<char *>(d_dst) + offset;
                    
                    if (!dst_h2d(dst_stream, dst_ord, dst_ptr, impl_->staging_buffers[buf_idx], this_chunk))
                    {
                        LOG_ERROR("H2D failed for chunk " << chunks_started_h2d);
                        return result;
                    }
                    chunks_started_h2d++;
                }

                // Try to start next D2H if we have a free buffer and more data
                // Buffer is free if its H2D has completed (chunks_completed caught up)
                if (chunks_started_d2h < num_chunks)
                {
                    int next_buf = chunks_started_d2h % 2;
                    // Buffer is free if we've completed H2D for the chunk that used it
                    size_t buf_available_after_chunk = (chunks_started_d2h >= 2) ? (chunks_started_d2h - 2) : 0;
                    
                    // If that chunk's H2D is done (or buffer was never used), we can reuse
                    if (chunks_started_d2h < 2 || chunks_completed > buf_available_after_chunk)
                    {
                        size_t offset = chunks_started_d2h * chunk_size;
                        size_t this_chunk = std::min(chunk_size, num_bytes - offset);
                        const char *src_ptr = static_cast<const char *>(d_src) + offset;
                        
                        if (!src_d2h(src_stream, src_ord, impl_->staging_buffers[next_buf], src_ptr, this_chunk))
                        {
                            LOG_ERROR("D2H failed for chunk " << chunks_started_d2h);
                            return result;
                        }
                        chunks_started_d2h++;
                    }
                }

                // Check if oldest H2D is complete so we can mark chunk as done
                if (chunks_started_h2d > chunks_completed)
                {
                    dst_sync(dst_stream);  // Wait for H2D stream
                    chunks_completed = chunks_started_h2d;  // All queued H2D ops are now done
                }
            }
        }
        else
        {
            // Simple sequential transfer (no pipelining)
            size_t offset = 0;
            while (offset < num_bytes)
            {
                size_t this_chunk = std::min(chunk_size, num_bytes - offset);
                const char *src_ptr = static_cast<const char *>(d_src) + offset;
                char *dst_ptr = static_cast<char *>(d_dst) + offset;
                void *staging = impl_->staging_buffers[0];

                // D2H
                if (!src_d2h(src_stream, src_ord, staging, src_ptr, this_chunk))
                {
                    LOG_ERROR("D2H failed at offset " << offset);
                    return result;
                }
                src_sync(src_stream);

                // H2D
                if (!dst_h2d(dst_stream, dst_ord, dst_ptr, staging, this_chunk))
                {
                    LOG_ERROR("H2D failed at offset " << offset);
                    return result;
                }
                dst_sync(dst_stream);

                offset += this_chunk;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps = (num_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_ms / 1000.0);
        result.success = true;

        return result;
    }

    CrossVendorP2PResult CrossVendorP2PEngine::benchmark(size_t num_bytes, int iterations)
    {
        CrossVendorP2PResult result;
        result.bytes_transferred = num_bytes;

        if (!initialized_)
        {
            LOG_ERROR("CrossVendorP2PEngine not initialized");
            return result;
        }

        // Allocate test buffers
        void *d_src = nullptr;
        void *d_dst = nullptr;

#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            // ROCm↔ROCm: allocate on source and destination devices
            d_src = rocm_p2p::allocDevice(impl_->rocm_device, num_bytes);
            d_dst = rocm_p2p::allocDevice(impl_->rocm_device_dst, num_bytes);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            // Cross-vendor allocation
            if (impl_->cuda_is_source)
            {
                d_src = cuda_p2p::allocDevice(impl_->cuda_device, num_bytes);
                d_dst = rocm_p2p::allocDevice(impl_->rocm_device, num_bytes);
            }
            else
            {
                d_src = rocm_p2p::allocDevice(impl_->rocm_device, num_bytes);
                d_dst = cuda_p2p::allocDevice(impl_->cuda_device, num_bytes);
            }
        }
#else
        {
            LOG_ERROR("Required GPU backend not available for benchmark");
            return result;
        }
#endif

        if (!d_src || !d_dst)
        {
            LOG_ERROR("Failed to allocate test buffers");
            // Cleanup any partial allocation
#ifdef HAVE_ROCM
            if (impl_->same_vendor_rocm)
            {
                if (d_src) rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                if (d_dst) rocm_p2p::freeDevice(impl_->rocm_device_dst, d_dst);
            }
            else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            {
                if (impl_->cuda_is_source)
                {
                    if (d_src) cuda_p2p::freeDevice(impl_->cuda_device, d_src);
                    if (d_dst) rocm_p2p::freeDevice(impl_->rocm_device, d_dst);
                }
                else
                {
                    if (d_src) rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                    if (d_dst) cuda_p2p::freeDevice(impl_->cuda_device, d_dst);
                }
            }
#endif
            return result;
        }

        // Initialize source buffer
#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            rocm_p2p::memsetDevice(impl_->rocm_device, d_src, 0xAB, num_bytes);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            if (impl_->cuda_is_source)
            {
                cuda_p2p::memsetDevice(impl_->cuda_device, d_src, 0xAB, num_bytes);
            }
            else
            {
                rocm_p2p::memsetDevice(impl_->rocm_device, d_src, 0xAB, num_bytes);
            }
        }
#endif

        // Warmup
        transfer(d_src, d_dst, num_bytes);

        // Timed iterations
        double total_gbps = 0.0;
        for (int i = 0; i < iterations; ++i)
        {
            auto r = transfer(d_src, d_dst, num_bytes);
            if (!r.success)
            {
                LOG_ERROR("Benchmark iteration " << i << " failed");
                break;
            }
            total_gbps += r.throughput_gbps;
        }

        // Cleanup
#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            rocm_p2p::freeDevice(impl_->rocm_device, d_src);
            rocm_p2p::freeDevice(impl_->rocm_device_dst, d_dst);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            if (impl_->cuda_is_source)
            {
                cuda_p2p::freeDevice(impl_->cuda_device, d_src);
                rocm_p2p::freeDevice(impl_->rocm_device, d_dst);
            }
            else
            {
                rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                cuda_p2p::freeDevice(impl_->cuda_device, d_dst);
            }
        }
#endif

        result.throughput_gbps = total_gbps / iterations;
        result.success = true;

        return result;
    }

    void CrossVendorP2PEngine::autoTuneChunkSize()
    {
        if (!initialized_)
        {
            LOG_WARN("Auto-tune failed: not initialized");
            return;
        }

        // Test chunk sizes: 2MB, 4MB, 8MB, 16MB, 32MB
        const size_t CHUNK_SIZES[] = {
            2 * 1024 * 1024,
            4 * 1024 * 1024,
            8 * 1024 * 1024,
            16 * 1024 * 1024,
            32 * 1024 * 1024
        };
        const int NUM_CHUNK_SIZES = sizeof(CHUNK_SIZES) / sizeof(CHUNK_SIZES[0]);

        // Use a reasonable test size (64 MB - typical allreduce)
        const size_t TEST_SIZE = 64 * 1024 * 1024;
        const int TEST_ITERATIONS = 3;

        // Allocate test buffers
        void *d_src = nullptr;
        void *d_dst = nullptr;

#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            d_src = rocm_p2p::allocDevice(impl_->rocm_device, TEST_SIZE);
            d_dst = rocm_p2p::allocDevice(impl_->rocm_device_dst, TEST_SIZE);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            if (impl_->cuda_is_source)
            {
                d_src = cuda_p2p::allocDevice(impl_->cuda_device, TEST_SIZE);
                d_dst = rocm_p2p::allocDevice(impl_->rocm_device, TEST_SIZE);
            }
            else
            {
                d_src = rocm_p2p::allocDevice(impl_->rocm_device, TEST_SIZE);
                d_dst = cuda_p2p::allocDevice(impl_->cuda_device, TEST_SIZE);
            }
        }
#else
        {
            LOG_WARN("Auto-tune failed: no GPU backend available");
            return;
        }
#endif

        if (!d_src || !d_dst)
        {
            LOG_WARN("Auto-tune failed: could not allocate test buffers");
#ifdef HAVE_ROCM
            if (impl_->same_vendor_rocm)
            {
                if (d_src) rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                if (d_dst) rocm_p2p::freeDevice(impl_->rocm_device_dst, d_dst);
            }
            else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            {
                if (d_src)
                {
                    if (impl_->cuda_is_source)
                        cuda_p2p::freeDevice(impl_->cuda_device, d_src);
                    else
                        rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                }
                if (d_dst)
                {
                    if (impl_->cuda_is_source)
                        rocm_p2p::freeDevice(impl_->rocm_device, d_dst);
                    else
                        cuda_p2p::freeDevice(impl_->cuda_device, d_dst);
                }
            }
#endif
            return;
        }

        // Initialize source buffer
#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            rocm_p2p::memsetDevice(impl_->rocm_device, d_src, 0xAB, TEST_SIZE);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            if (impl_->cuda_is_source)
            {
                cuda_p2p::memsetDevice(impl_->cuda_device, d_src, 0xAB, TEST_SIZE);
            }
            else
            {
                rocm_p2p::memsetDevice(impl_->rocm_device, d_src, 0xAB, TEST_SIZE);
            }
        }
#endif

        double best_throughput = 0.0;
        size_t best_chunk_size = config_.chunk_size;

        LOG_DEBUG("Auto-tune: testing chunk sizes...");

        for (int i = 0; i < NUM_CHUNK_SIZES; ++i)
        {
            size_t test_chunk = CHUNK_SIZES[i];
            if (test_chunk > impl_->buffer_size)
            {
                continue;  // Skip if chunk exceeds buffer size
            }

            // Temporarily set chunk size
            size_t saved_chunk = config_.chunk_size;
            config_.chunk_size = test_chunk;

            // Warmup
            transfer(d_src, d_dst, TEST_SIZE);

            // Timed test
            double total_gbps = 0.0;
            for (int j = 0; j < TEST_ITERATIONS; ++j)
            {
                auto r = transfer(d_src, d_dst, TEST_SIZE);
                if (r.success)
                {
                    total_gbps += r.throughput_gbps;
                }
            }
            double avg_gbps = total_gbps / TEST_ITERATIONS;

            LOG_DEBUG("  chunk_size=" << (test_chunk / (1024 * 1024)) << "MB: " << avg_gbps << " GB/s");

            if (avg_gbps > best_throughput)
            {
                best_throughput = avg_gbps;
                best_chunk_size = test_chunk;
            }

            config_.chunk_size = saved_chunk;
        }

        // Apply best chunk size
        config_.chunk_size = best_chunk_size;
        theoretical_max_gbps_ = best_throughput;

        LOG_INFO("Auto-tune complete: optimal chunk_size=" << (best_chunk_size / (1024 * 1024)) 
                 << "MB, throughput=" << best_throughput << " GB/s");

        // Cleanup
#ifdef HAVE_ROCM
        if (impl_->same_vendor_rocm)
        {
            rocm_p2p::freeDevice(impl_->rocm_device, d_src);
            rocm_p2p::freeDevice(impl_->rocm_device_dst, d_dst);
        }
        else
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        {
            if (impl_->cuda_is_source)
            {
                cuda_p2p::freeDevice(impl_->cuda_device, d_src);
                rocm_p2p::freeDevice(impl_->rocm_device, d_dst);
            }
            else
            {
                rocm_p2p::freeDevice(impl_->rocm_device, d_src);
                cuda_p2p::freeDevice(impl_->cuda_device, d_dst);
            }
        }
#endif
    }

    //--------------------------------------------------------------------------
    // CrossVendorP2PHelper
    //--------------------------------------------------------------------------

    bool CrossVendorP2PHelper::canTransfer(DeviceId src, DeviceId dst)
    {
        // Cross-vendor transfer is possible between any CUDA and ROCm device
        bool src_is_gpu = (src.type == DeviceType::CUDA || src.type == DeviceType::ROCm);
        bool dst_is_gpu = (dst.type == DeviceType::CUDA || dst.type == DeviceType::ROCm);
        bool different_vendor = (src.type != dst.type);

        // Standard cross-vendor
        if (src_is_gpu && dst_is_gpu && different_vendor)
        {
            return true;
        }

        // Note: Same-vendor transfers are also supported when same_vendor_allowed=true
        // in the config, but canTransfer() can't know the config at this static level.
        // Use the instance method or check config separately.
        return false;
    }

    bool CrossVendorP2PHelper::canTransfer(DeviceId src, DeviceId dst, bool allow_same_vendor)
    {
        bool src_is_gpu = (src.type == DeviceType::CUDA || src.type == DeviceType::ROCm);
        bool dst_is_gpu = (dst.type == DeviceType::CUDA || dst.type == DeviceType::ROCm);
        bool different_vendor = (src.type != dst.type);

        // Standard cross-vendor
        if (src_is_gpu && dst_is_gpu && different_vendor)
        {
            return true;
        }

        // Same-vendor transfer (e.g., ROCm↔ROCm across PCIe)
        if (allow_same_vendor && src_is_gpu && dst_is_gpu && !different_vendor)
        {
            // Must be different devices
            return src.ordinal != dst.ordinal;
        }

        return false;
    }

    double CrossVendorP2PHelper::estimateTransferTimeMs(DeviceId src, DeviceId dst,
                                                         size_t num_bytes, bool pipelined)
    {
        // Rough estimates based on typical PCIe bandwidth
        // These should be calibrated per-system
        double src_d2h_gbps = 4.0;  // Conservative estimate
        double dst_h2d_gbps = 4.0;

        double size_gb = num_bytes / (1024.0 * 1024.0 * 1024.0);

        if (pipelined)
        {
            // With pipelining, limited by slower of the two
            double effective_gbps = std::min(src_d2h_gbps, dst_h2d_gbps);
            return (size_gb / effective_gbps) * 1000.0;
        }
        else
        {
            // Without pipelining, both transfers are serial
            double d2h_time = size_gb / src_d2h_gbps;
            double h2d_time = size_gb / dst_h2d_gbps;
            return (d2h_time + h2d_time) * 1000.0;
        }
    }

    CrossVendorP2PResult CrossVendorP2PHelper::quickBenchmark(DeviceId src, DeviceId dst, size_t num_bytes)
    {
        CrossVendorP2PEngine engine;
        if (!engine.initialize(src, dst))
        {
            return {};
        }
        return engine.benchmark(num_bytes, 3);
    }

} // namespace llaminar2
