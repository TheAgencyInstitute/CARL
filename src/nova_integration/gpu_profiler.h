#pragma once

#include "nova_context.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>

/**
 * Dynamic GPU Profiler for CARL AI Operations - Nova-CARL Integration
 * 
 * NOW INTEGRATED with Nova-CARL Enhanced Queue System:
 * - Monitors ALL queue families discovered by Nova-CARL extensions
 * - Direct access to Nova's enhanced Queues struct
 * - Leverages Nova's queue family indices and properties
 * - Eliminates abstraction layer - works directly with Nova core
 */

namespace CARL {
namespace GPU {

enum class ProfileEventType {
    COMPUTE_DISPATCH,
    MEMORY_TRANSFER,
    PIPELINE_CREATION,
    SYNCHRONIZATION,
    QUEUE_SUBMIT,
    CUSTOM_MARKER
};

// Nova-CARL Integration: Use Nova's queue family system directly
enum class QueueType {
    GRAPHICS_COMPUTE = 0,    // Family 0: Graphics+Compute+Transfer+Sparse (Nova compatible)
    DEDICATED_COMPUTE = 1,   // Family 1: 4x Dedicated Compute+Transfer+Sparse (Nova-CARL)
    VIDEO_DECODE = 2,        // Family 2: Video decode (Nova-CARL)
    VIDEO_ENCODE = 3,        // Family 3: Video encode (Nova-CARL) 
    SPARSE_BINDING = 4       // Family 4: Dedicated sparse binding (Nova-CARL)
};

struct ProfileEvent {
    std::string name;
    ProfileEventType type;
    QueueType queue_type;
    uint32_t queue_index;
    uint64_t gpu_start_timestamp;
    uint64_t gpu_end_timestamp;
    std::chrono::steady_clock::time_point cpu_start;
    std::chrono::steady_clock::time_point cpu_end;
    size_t memory_used_bytes;
    uint32_t work_groups_x, work_groups_y, work_groups_z;
    std::unordered_map<std::string, float> custom_metrics;
};

struct QueueStats {
    uint32_t total_submissions;
    uint64_t total_gpu_time_ns;
    uint64_t total_cpu_time_ns;
    uint32_t active_commands;
    float utilization_percent;
    uint64_t last_activity_timestamp;
};

struct MemoryProfile {
    size_t device_local_used;      // VRAM usage
    size_t device_local_total;     // Total VRAM (16GB on RX 6800 XT)
    size_t host_visible_used;      // System memory mapped to GPU
    size_t host_visible_total;     // Total system memory accessible
    float fragmentation_ratio;
    uint32_t allocation_count;
    uint32_t pool_hits;
    uint32_t pool_misses;
};

class GPUProfiler {
public:
    // Nova-CARL Integration Constructor
    GPUProfiler(NovaContext* context);
    GPUProfiler(NovaCore* nova_core);  // Direct Nova core access
    ~GPUProfiler();
    
    bool initialize();
    void shutdown();
    
    // Nova Integration
    void attachToNova(NovaCore* nova_core);
    const Queues& getNovaQueues() const { return *_nova_queues; }
    
    // Profiling Control
    void enable(bool enable = true);
    void disable() { enable(false); }
    bool isEnabled() const { return _enabled; }
    
    // Event Tracking
    void beginEvent(const std::string& name, ProfileEventType type, QueueType queue_type = QueueType::GRAPHICS_COMPUTE);
    void endEvent();
    void markEvent(const std::string& name, ProfileEventType type = ProfileEventType::CUSTOM_MARKER);
    
    // GPU Timestamp Queries
    void insertTimestamp(VkCommandBuffer cmd_buffer, const std::string& name);
    void queryTimestamps();
    
    // Queue-Specific Profiling
    void trackQueueSubmission(QueueType queue_type, uint32_t queue_index, VkCommandBuffer cmd_buffer);
    void trackMemoryAllocation(size_t bytes, bool device_local);
    void trackMemoryDeallocation(size_t bytes, bool device_local);
    
    // Performance Analysis
    std::vector<ProfileEvent> getEvents(ProfileEventType type = ProfileEventType::COMPUTE_DISPATCH) const;
    QueueStats getQueueStats(QueueType queue_type, uint32_t queue_index = 0) const;
    MemoryProfile getMemoryProfile() const;
    
    // Real-time Monitoring
    float getCurrentGPUUtilization() const;
    float getQueueUtilization(QueueType queue_type, uint32_t queue_index = 0) const;
    uint64_t getAverageFrameTime() const; // For training iterations
    
    // Dynamic Optimization Hints
    struct OptimizationHint {
        std::string suggestion;
        float impact_estimate;  // 0.0 to 1.0
        bool actionable;
    };
    std::vector<OptimizationHint> getOptimizationHints() const;
    
    // RDNA2-Specific Metrics
    struct RDNA2Metrics {
        float compute_unit_utilization;     // CU usage across 72 CUs
        float memory_bandwidth_utilization; // % of 512 GB/s used
        float l2_cache_hit_rate;            // Estimated cache performance
        uint32_t active_wavefronts;         // RDNA2 wavefront utilization
        float shader_alu_busy_percent;      // ALU utilization
    };
    RDNA2Metrics getRDNA2Metrics() const;
    
    // Export/Reporting
    void exportToJSON(const std::string& filename) const;
    void printSummary() const;
    void resetStats();
    
private:
    NovaContext* _context;
    NovaCore* _core;
    VkDevice _device;
    bool _enabled;
    
    // Nova-CARL Integration
    const Queues* _nova_queues;  // Direct access to Nova's enhanced queue system
    
    // Enhanced timestamp query pools for all 5 queue families
    VkQueryPool _timestamp_pools[5];  // Graphics, Compute(4), Video Decode, Video Encode, Sparse
    uint32_t _timestamp_index[5];
    static const uint32_t MAX_TIMESTAMPS = 1024;
    
    // Event tracking
    std::vector<ProfileEvent> _events;
    std::unordered_map<std::string, size_t> _active_events; // name -> event index
    mutable std::mutex _events_mutex;
    
    // Queue statistics
    std::unordered_map<uint64_t, QueueStats> _queue_stats; // (queue_family << 32) | queue_index
    
    // Memory tracking  
    MemoryProfile _memory_profile;
    mutable std::mutex _memory_mutex;
    
    // Performance counters
    uint64_t _frame_counter;
    std::chrono::steady_clock::time_point _last_frame_time;
    std::vector<uint64_t> _frame_times; // Rolling window
    static const size_t FRAME_HISTORY_SIZE = 100;
    
    // RDNA2-specific monitoring
    RDNA2Metrics _rdna2_metrics;
    std::chrono::steady_clock::time_point _last_metrics_update;
    
    // Internal methods
    bool _createTimestampPools();
    void _destroyTimestampPools();
    uint64_t _makeQueueKey(QueueType queue_type, uint32_t queue_index) const;
    void _updateQueueStats(uint64_t queue_key, uint64_t gpu_time_ns);
    void _updateRDNA2Metrics();
    void _analyzePerformanceBottlenecks(std::vector<OptimizationHint>& hints) const;
    float _estimateMemoryBandwidthUtilization() const;
    uint32_t _estimateActiveWavefronts() const;
};

// Global Profiler Instance
class GlobalProfiler {
public:
    static GPUProfiler* getInstance();
    static bool initialize(NovaContext* context);
    static void shutdown();
    
    // Convenience macros
    static void beginEvent(const std::string& name, ProfileEventType type = ProfileEventType::COMPUTE_DISPATCH);
    static void endEvent();
    static void markEvent(const std::string& name);
    
private:
    static std::unique_ptr<GPUProfiler> _instance;
    static std::mutex _instance_mutex;
};

} // namespace GPU
} // namespace CARL

// Profiling Macros for Easy Integration
#define CARL_PROFILE_GPU_EVENT(name, type) \
    CARL::GPU::GlobalProfiler::beginEvent(name, type); \
    struct ProfileEventGuard { \
        ~ProfileEventGuard() { CARL::GPU::GlobalProfiler::endEvent(); } \
    } _guard;

#define CARL_PROFILE_COMPUTE(name) CARL_PROFILE_GPU_EVENT(name, CARL::GPU::ProfileEventType::COMPUTE_DISPATCH)
#define CARL_PROFILE_TRANSFER(name) CARL_PROFILE_GPU_EVENT(name, CARL::GPU::ProfileEventType::MEMORY_TRANSFER)
#define CARL_PROFILE_MARK(name) CARL::GPU::GlobalProfiler::markEvent(name)

/**
 * Usage Example:
 * 
 * // Initialize profiler
 * CARL::GPU::GlobalProfiler::initialize(context);
 * 
 * // Profile compute operation
 * {
 *     CARL_PROFILE_COMPUTE("Matrix Multiplication 1024x1024");
 *     auto result = compute->matrixMultiply(A, B);
 * }
 * 
 * // Check performance
 * auto profiler = CARL::GPU::GlobalProfiler::getInstance();
 * auto rdna2_metrics = profiler->getRDNA2Metrics();
 * std::cout << "CU Utilization: " << rdna2_metrics.compute_unit_utilization << "%" << std::endl;
 * 
 * // Get optimization hints
 * auto hints = profiler->getOptimizationHints();
 * for (const auto& hint : hints) {
 *     std::cout << "Suggestion: " << hint.suggestion 
 *               << " (Impact: " << hint.impact_estimate << ")" << std::endl;
 * }
 */