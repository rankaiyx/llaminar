/**
 * @file MockMPITopology.h
 * @brief Mock MPI topology for unit testing distributed logic without real MPI
 *
 * This mock enables:
 * - Testing topology-dependent code without MPI runtime
 * - Simulating heterogeneous clusters (CPU-only, GPU, mixed)
 * - Testing work distribution algorithms deterministically
 * - Configuring custom cluster configurations for edge cases
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "interfaces/IMPITopology.h"
#include "utils/MPITopology.h"  // For RankPlacement, WorkRange, DeviceCapability
#include "execution/DeviceInventory.h"  // For ClusterInventory
#include "tensors/TensorSlice.h"  // For SliceMetadata
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

namespace llaminar2::test {

/**
 * @brief Builder for MockMPITopology
 *
 * Provides a fluent interface for constructing mock topologies:
 * @code
 * auto mock = MockMPITopologyBuilder()
 *     .addRank(0, 0, {DeviceCapability::cpu()})         // Rank 0: CPU only
 *     .addRank(1, 0, {DeviceCapability::cpu(), DeviceCapability::cuda(0)})  // Rank 1: CPU + GPU
 *     .setLocalRank(0)
 *     .build();
 * @endcode
 */
class MockMPITopologyBuilder;

/**
 * @brief Mock MPI topology for unit testing
 *
 * Provides configurable behavior for testing topology-dependent logic:
 * - Configurable rank and world_size
 * - Custom device configurations per rank
 * - Deterministic work distribution for verification
 * - Heterogeneous cluster simulation
 *
 * Usage:
 * @code
 * // Simple 2-rank CPU-only topology
 * auto mock = MockMPITopology::createSimple(0, 2);
 *
 * // Custom heterogeneous cluster
 * auto mock = MockMPITopologyBuilder()
 *     .addCPUOnlyRank(0, 0)
 *     .addGPURank(1, 0, 0)  // Rank 1 with CUDA:0
 *     .setLocalRank(0)
 *     .build();
 *
 * // Use in tests
 * WorkRange heads = mock->get_head_range(14);
 * EXPECT_EQ(heads.start, 0);
 * EXPECT_EQ(heads.end, 7);
 * @endcode
 */
class MockMPITopology : public IMPITopology {
public:
    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create a simple uniform topology (all CPU-only ranks)
     * @param rank Local rank (0-indexed)
     * @param world_size Total number of ranks
     * @param ranks_per_node Ranks per physical node (default: all on one node)
     * @return Shared pointer to mock topology
     */
    static std::shared_ptr<MockMPITopology> createSimple(
        int rank,
        int world_size,
        int ranks_per_node = -1);

    /**
     * @brief Create topology from explicit placements
     * @param rank Local rank
     * @param world_size Total ranks
     * @param placements Vector of RankPlacement for all ranks
     * @return Shared pointer to mock topology
     */
    static std::shared_ptr<MockMPITopology> create(
        int rank,
        int world_size,
        const std::vector<RankPlacement>& placements);

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Construct mock with explicit configuration
     * @param rank Local rank
     * @param world_size Total ranks
     * @param placements Vector of RankPlacement for all ranks
     */
    MockMPITopology(int rank, int world_size, const std::vector<RankPlacement>& placements);

    ~MockMPITopology() override = default;

    // =========================================================================
    // IMPITopology Implementation - Identity
    // =========================================================================

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }

    // =========================================================================
    // IMPITopology Implementation - Topology Info
    // =========================================================================

    int node_count() const override { return node_count_; }
    int ranks_per_node() const override { return ranks_per_node_; }
    const RankPlacement& placement() const override { return placements_[rank_]; }
    const RankPlacement& get_placement(int r) const override;
    const std::vector<RankPlacement>& all_placements() const override { return placements_; }

    // =========================================================================
    // IMPITopology Implementation - Role Queries
    // =========================================================================

    bool is_coordinator() const override { return rank_ == 0; }
    bool is_compute_participant() const override { return compute_participant_; }
    bool is_node_leader() const override { return placements_[rank_].local_rank == 0; }

    // =========================================================================
    // IMPITopology Implementation - Work Distribution
    // =========================================================================

    WorkRange get_head_range(int total_heads) const override {
        return WorkRange::for_rank_equal(static_cast<size_t>(total_heads), rank_, world_size_);
    }

    WorkRange get_kv_head_range(int total_kv_heads) const override {
        return WorkRange::for_rank_equal(static_cast<size_t>(total_kv_heads), rank_, world_size_);
    }

    WorkRange get_column_range(size_t total_cols) const override {
        return WorkRange::for_rank_equal(total_cols, rank_, world_size_);
    }

    WorkRange get_row_range(size_t total_rows) const override {
        return WorkRange::for_rank_equal(total_rows, rank_, world_size_);
    }

    WorkRange get_vocab_range(size_t vocab_size) const override {
        return WorkRange::for_rank_equal(vocab_size, rank_, world_size_);
    }

    WorkRange get_ffn_range(size_t ffn_dim) const override {
        return WorkRange::for_rank_equal(ffn_dim, rank_, world_size_);
    }

    // =========================================================================
    // IMPITopology Implementation - Capability Info
    // =========================================================================

    const ClusterInventory& clusterInventory() const override;
    std::vector<float> get_compute_weights() const override;
    bool has_accelerator() const override;
    int get_device() const override;
    const std::vector<DeviceCapability>& get_devices() const override {
        return placements_[rank_].devices;
    }

    // =========================================================================
    // IMPITopology Implementation - SliceMetadata
    // =========================================================================

    SliceMetadata createRowParallelMeta(
        size_t original_rows,
        size_t original_cols,
        bool inner_is_presliced = false) const override;

    SliceMetadata createColumnParallelMeta(
        size_t original_rows,
        size_t original_cols,
        bool inner_is_presliced = false) const override;

    // =========================================================================
    // IMPITopology Implementation - Debugging
    // =========================================================================

    std::string to_string() const override;

    // =========================================================================
    // Mock-Specific Configuration
    // =========================================================================

    /**
     * @brief Set whether this rank participates in compute
     * @param participate If false, this rank is observer-only
     */
    void set_compute_participant(bool participate) { compute_participant_ = participate; }

private:
    int rank_;
    int world_size_;
    int node_count_;
    int ranks_per_node_;
    bool compute_participant_ = true;

    std::vector<RankPlacement> placements_;
    mutable ClusterInventory cluster_inventory_;
    mutable bool cluster_inventory_built_ = false;

    void buildClusterInventory() const;
    void calculateNodeInfo();
};

/**
 * @brief Builder for MockMPITopology with fluent interface
 */
class MockMPITopologyBuilder {
public:
    MockMPITopologyBuilder() = default;

    /**
     * @brief Add a rank with explicit placement
     * @param rank Rank ID
     * @param node_id Physical node ID
     * @param devices Device capabilities for this rank
     * @param hostname Optional hostname
     * @return Builder reference for chaining
     */
    MockMPITopologyBuilder& addRank(
        int rank,
        int node_id,
        const std::vector<DeviceCapability>& devices,
        const std::string& hostname = "");

    /**
     * @brief Add a CPU-only rank
     * @param rank Rank ID
     * @param node_id Physical node ID
     * @return Builder reference for chaining
     */
    MockMPITopologyBuilder& addCPUOnlyRank(int rank, int node_id);

    /**
     * @brief Add a rank with CPU and one CUDA GPU
     * @param rank Rank ID
     * @param node_id Physical node ID
     * @param cuda_device_id CUDA device index
     * @param gpu_memory_gb GPU memory in GB (default: 8)
     * @return Builder reference for chaining
     */
    MockMPITopologyBuilder& addGPURank(
        int rank,
        int node_id,
        int cuda_device_id,
        float gpu_memory_gb = 8.0f);

    /**
     * @brief Add a rank with CPU and ROCm GPU
     * @param rank Rank ID
     * @param node_id Physical node ID
     * @param rocm_device_id ROCm device index
     * @param gpu_memory_gb GPU memory in GB (default: 16)
     * @return Builder reference for chaining
     */
    MockMPITopologyBuilder& addROCmRank(
        int rank,
        int node_id,
        int rocm_device_id,
        float gpu_memory_gb = 16.0f);

    /**
     * @brief Set which rank is the local rank
     * @param rank Local rank ID
     * @return Builder reference for chaining
     */
    MockMPITopologyBuilder& setLocalRank(int rank);

    /**
     * @brief Build the mock topology
     * @return Shared pointer to the mock topology
     * @throws std::invalid_argument if configuration is invalid
     */
    std::shared_ptr<MockMPITopology> build();

private:
    std::vector<RankPlacement> placements_;
    int local_rank_ = 0;
};

// =========================================================================
// Helper Functions for Creating DeviceCapability
// =========================================================================

namespace MockDevices {

/**
 * @brief Create a CPU device capability
 * @param compute_power Relative compute power (default: 1.0)
 * @return DeviceCapability configured as CPU
 */
inline DeviceCapability cpu(float compute_power = 1.0f) {
    DeviceCapability dev;
    dev.type = DeviceCapability::Type::CPU;
    dev.device_id = 0;
    dev.relative_compute = compute_power;
    dev.name = "CPU";
    return dev;
}

/**
 * @brief Create a CUDA GPU device capability
 * @param device_id CUDA device index
 * @param memory_gb VRAM in GB
 * @param compute_power Relative compute power (default: 10.0)
 * @return DeviceCapability configured as CUDA GPU
 */
inline DeviceCapability cuda(int device_id, float memory_gb = 8.0f, float compute_power = 10.0f) {
    DeviceCapability dev;
    dev.type = DeviceCapability::Type::CUDA;
    dev.device_id = device_id;
    dev.memory_bytes = static_cast<size_t>(memory_gb * 1024 * 1024 * 1024);
    dev.relative_compute = compute_power;
    dev.name = "CUDA:" + std::to_string(device_id);
    return dev;
}

/**
 * @brief Create a ROCm GPU device capability
 * @param device_id ROCm device index
 * @param memory_gb VRAM in GB
 * @param compute_power Relative compute power (default: 10.0)
 * @return DeviceCapability configured as ROCm GPU
 */
inline DeviceCapability rocm(int device_id, float memory_gb = 16.0f, float compute_power = 10.0f) {
    DeviceCapability dev;
    dev.type = DeviceCapability::Type::ROCm;
    dev.device_id = device_id;
    dev.memory_bytes = static_cast<size_t>(memory_gb * 1024 * 1024 * 1024);
    dev.relative_compute = compute_power;
    dev.name = "ROCm:" + std::to_string(device_id);
    return dev;
}

} // namespace MockDevices

// =========================================================================
// Implementation
// =========================================================================

inline std::shared_ptr<MockMPITopology> MockMPITopology::createSimple(
    int rank, int world_size, int ranks_per_node)
{
    if (ranks_per_node < 0) {
        ranks_per_node = world_size;  // All on one node by default
    }

    std::vector<RankPlacement> placements;
    placements.reserve(world_size);

    for (int r = 0; r < world_size; ++r) {
        RankPlacement p;
        p.rank = r;
        p.node_id = r / ranks_per_node;
        p.local_rank = r % ranks_per_node;
        p.socket_id = p.local_rank;
        p.numa_node = p.local_rank;
        p.hostname = "node" + std::to_string(p.node_id);
        p.devices.push_back(MockDevices::cpu());
        placements.push_back(std::move(p));
    }

    return std::make_shared<MockMPITopology>(rank, world_size, placements);
}

inline std::shared_ptr<MockMPITopology> MockMPITopology::create(
    int rank, int world_size, const std::vector<RankPlacement>& placements)
{
    return std::make_shared<MockMPITopology>(rank, world_size, placements);
}

inline MockMPITopology::MockMPITopology(
    int rank, int world_size, const std::vector<RankPlacement>& placements)
    : rank_(rank), world_size_(world_size), placements_(placements)
{
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("MockMPITopology: rank must be in [0, world_size)");
    }
    if (world_size < 1) {
        throw std::invalid_argument("MockMPITopology: world_size must be >= 1");
    }
    if (static_cast<int>(placements.size()) != world_size) {
        throw std::invalid_argument("MockMPITopology: placements.size() must equal world_size");
    }

    calculateNodeInfo();
}

inline const RankPlacement& MockMPITopology::get_placement(int r) const
{
    if (r < 0 || r >= world_size_) {
        throw std::out_of_range("MockMPITopology::get_placement: rank out of range");
    }
    return placements_[r];
}

inline const ClusterInventory& MockMPITopology::clusterInventory() const
{
    if (!cluster_inventory_built_) {
        buildClusterInventory();
    }
    return cluster_inventory_;
}

inline std::vector<float> MockMPITopology::get_compute_weights() const
{
    std::vector<float> weights;
    weights.reserve(world_size_);
    for (const auto& p : placements_) {
        weights.push_back(p.total_compute_power());
    }
    return weights;
}

inline bool MockMPITopology::has_accelerator() const
{
    for (const auto& dev : placements_[rank_].devices) {
        if (dev.type == DeviceCapability::Type::CUDA ||
            dev.type == DeviceCapability::Type::ROCm) {
            return true;
        }
    }
    return false;
}

inline int MockMPITopology::get_device() const
{
    // Return first GPU device ID, or 0 for CPU
    for (const auto& dev : placements_[rank_].devices) {
        if (dev.type == DeviceCapability::Type::CUDA ||
            dev.type == DeviceCapability::Type::ROCm) {
            return dev.device_id;
        }
    }
    return 0;
}

inline SliceMetadata MockMPITopology::createRowParallelMeta(
    size_t original_rows, size_t original_cols, bool inner_is_presliced) const
{
    WorkRange row_range = get_row_range(original_rows);
    
    SliceMetadata meta;
    meta.mode = SliceMode::ROW_PARALLEL;
    meta.slice_start = row_range.start;
    meta.slice_end = row_range.end;
    meta.original_rows = original_rows;
    meta.original_cols = original_cols;
    meta.world_size = world_size_;
    meta.rank = rank_;
    meta.inner_is_presliced = inner_is_presliced;
    return meta;
}

inline SliceMetadata MockMPITopology::createColumnParallelMeta(
    size_t original_rows, size_t original_cols, bool inner_is_presliced) const
{
    WorkRange col_range = get_column_range(original_cols);
    
    SliceMetadata meta;
    meta.mode = SliceMode::COLUMN_PARALLEL;
    meta.slice_start = col_range.start;
    meta.slice_end = col_range.end;
    meta.original_rows = original_rows;
    meta.original_cols = original_cols;
    meta.world_size = world_size_;
    meta.rank = rank_;
    meta.inner_is_presliced = inner_is_presliced;
    return meta;
}

inline std::string MockMPITopology::to_string() const
{
    std::ostringstream ss;
    ss << "MockMPITopology[rank=" << rank_ << "/" << world_size_
       << " nodes=" << node_count_
       << " ranks_per_node=" << ranks_per_node_;
    
    const auto& p = placements_[rank_];
    ss << " devices={";
    for (size_t i = 0; i < p.devices.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << p.devices[i].name;
    }
    ss << "}]";
    return ss.str();
}

inline void MockMPITopology::calculateNodeInfo()
{
    // Calculate node_count and ranks_per_node from placements
    int max_node_id = 0;
    std::vector<int> ranks_on_node;
    
    for (const auto& p : placements_) {
        if (p.node_id > max_node_id) {
            max_node_id = p.node_id;
        }
    }
    
    node_count_ = max_node_id + 1;
    ranks_on_node.resize(node_count_, 0);
    
    for (const auto& p : placements_) {
        ranks_on_node[p.node_id]++;
    }
    
    // Use max ranks on any node as ranks_per_node
    ranks_per_node_ = 0;
    for (int count : ranks_on_node) {
        if (count > ranks_per_node_) {
            ranks_per_node_ = count;
        }
    }
}

inline void MockMPITopology::buildClusterInventory() const
{
    cluster_inventory_.world_size = world_size_;
    cluster_inventory_.node_count = node_count_;
    cluster_inventory_.ranks.resize(world_size_);

    for (int r = 0; r < world_size_; ++r) {
        const auto& rp = placements_[r];
        auto& ri = cluster_inventory_.ranks[r];
        
        ri.rank = rp.rank;
        ri.node_id = rp.node_id;
        ri.local_rank = rp.local_rank;
        ri.hostname = rp.hostname;
        ri.numa_nodes = 1;

        for (const auto& dev : rp.devices) {
            if (dev.type == DeviceCapability::Type::CPU) {
                ri.cpu.type = DeviceType::CPU;
                ri.cpu.local_device_id = dev.device_id;
                ri.cpu.memory_bytes = dev.memory_bytes;
                ri.cpu.compute_units = static_cast<int>(dev.compute_units);
                ri.cpu.name = dev.name;
            } else {
                DeviceInfo gpu;
                if (dev.type == DeviceCapability::Type::CUDA) {
                    gpu.type = DeviceType::CUDA;
                } else if (dev.type == DeviceCapability::Type::ROCm) {
                    gpu.type = DeviceType::ROCm;
                }
                gpu.local_device_id = dev.device_id;
                gpu.memory_bytes = dev.memory_bytes;
                gpu.compute_units = static_cast<int>(dev.compute_units);
                gpu.name = dev.name;
                ri.gpus.push_back(gpu);
            }
        }
    }

    cluster_inventory_.buildNodeAggregations();
    cluster_inventory_built_ = true;
}

// =========================================================================
// Builder Implementation
// =========================================================================

inline MockMPITopologyBuilder& MockMPITopologyBuilder::addRank(
    int rank, int node_id, const std::vector<DeviceCapability>& devices,
    const std::string& hostname)
{
    // Ensure placements vector is large enough
    if (static_cast<int>(placements_.size()) <= rank) {
        placements_.resize(rank + 1);
    }

    RankPlacement p;
    p.rank = rank;
    p.node_id = node_id;
    p.devices = devices;
    p.hostname = hostname.empty() ? ("node" + std::to_string(node_id)) : hostname;
    
    // Calculate local_rank based on existing placements on same node
    int local_rank = 0;
    for (size_t i = 0; i < placements_.size(); ++i) {
        if (static_cast<int>(i) != rank && placements_[i].node_id == node_id) {
            local_rank++;
        }
    }
    p.local_rank = local_rank;
    p.socket_id = local_rank;
    p.numa_node = local_rank;

    placements_[rank] = std::move(p);
    return *this;
}

inline MockMPITopologyBuilder& MockMPITopologyBuilder::addCPUOnlyRank(int rank, int node_id)
{
    return addRank(rank, node_id, {MockDevices::cpu()});
}

inline MockMPITopologyBuilder& MockMPITopologyBuilder::addGPURank(
    int rank, int node_id, int cuda_device_id, float gpu_memory_gb)
{
    return addRank(rank, node_id, {
        MockDevices::cpu(),
        MockDevices::cuda(cuda_device_id, gpu_memory_gb)
    });
}

inline MockMPITopologyBuilder& MockMPITopologyBuilder::addROCmRank(
    int rank, int node_id, int rocm_device_id, float gpu_memory_gb)
{
    return addRank(rank, node_id, {
        MockDevices::cpu(),
        MockDevices::rocm(rocm_device_id, gpu_memory_gb)
    });
}

inline MockMPITopologyBuilder& MockMPITopologyBuilder::setLocalRank(int rank)
{
    local_rank_ = rank;
    return *this;
}

inline std::shared_ptr<MockMPITopology> MockMPITopologyBuilder::build()
{
    if (placements_.empty()) {
        throw std::invalid_argument("MockMPITopologyBuilder: no ranks added");
    }

    // Validate all ranks are filled
    for (size_t i = 0; i < placements_.size(); ++i) {
        if (placements_[i].devices.empty()) {
            throw std::invalid_argument(
                "MockMPITopologyBuilder: rank " + std::to_string(i) + " not configured");
        }
    }

    if (local_rank_ < 0 || local_rank_ >= static_cast<int>(placements_.size())) {
        throw std::invalid_argument("MockMPITopologyBuilder: local_rank out of range");
    }

    return std::make_shared<MockMPITopology>(
        local_rank_,
        static_cast<int>(placements_.size()),
        placements_);
}

} // namespace llaminar2::test

// =========================================================================
// IMPITopology::createMock Implementation (in llaminar2 namespace)
// =========================================================================

namespace llaminar2 {

inline std::shared_ptr<IMPITopology> IMPITopology::createMock(
    int rank, int world_size, const std::vector<RankPlacement>& placements)
{
    return test::MockMPITopology::create(rank, world_size, placements);
}

} // namespace llaminar2
