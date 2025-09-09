#pragma once

#include "nova_context.h"
#include "gpu_profiler.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

/**
 * Advanced Queue Manager for AMD RX 6800 XT - Nova Integration
 * 
 * EXTENDS Nova's basic queue management to leverage ALL available queues:
 * - Nova Default: Only uses 1 queue per family (missing 3 compute queues!)
 * - Our Extension: Discovers and utilizes ALL queues for maximum throughput
 * 
 * Queue Discovery Results on AMD RX 6800 XT:
 * - Queue Family 0: 1x Graphics+Compute (Nova: ✓ Used | Ours: Enhanced)
 * - Queue Family 1: 4x Dedicated Compute (Nova: ✗ Only 1st | Ours: All 4)  
 * - Queue Family 2: 1x Video Decode (Nova: ✗ Unused | Ours: Supported)
 * 
 * Performance Impact: 4x parallel compute operations vs Nova's single queue
 */

namespace CARL {
namespace GPU {

enum class WorkloadType {
    // Pure compute workloads (use dedicated compute queues)
    MATRIX_OPERATIONS,
    CONVOLUTION_COMPUTE,
    ACTIVATION_FUNCTIONS,
    NEURAL_NETWORK_TRAINING,
    SNN_SPIKE_PROCESSING,
    
    // Graphics-compute hybrid (use graphics+compute queue)
    RENDER_TO_TEXTURE,
    COMPUTE_WITH_GRAPHICS,
    VISUALIZATION,
    
    // Memory operations (can use any queue with transfer)
    MEMORY_UPLOAD,
    MEMORY_DOWNLOAD,
    BUFFER_COPY,
    
    // Video operations (use video decode queue)
    VIDEO_PREPROCESSING,
    VIDEO_FEATURE_EXTRACTION
};

enum class Priority {
    LOW = 0,
    NORMAL = 1, 
    HIGH = 2,
    CRITICAL = 3
};

struct QueueSubmission {
    VkCommandBuffer command_buffer;
    VkSemaphore wait_semaphore;
    VkSemaphore signal_semaphore;
    VkFence fence;
    WorkloadType workload_type;
    Priority priority;
    std::string debug_name;
    std::function<void()> completion_callback;
    std::chrono::steady_clock::time_point submit_time;
};

struct QueueInfo {
    QueueType type;
    uint32_t family_index;
    uint32_t queue_index;
    VkQueue handle;
    VkCommandPool command_pool;
    std::queue<QueueSubmission> pending_submissions;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool is_busy;
    float load_factor; // 0.0 to 1.0
    uint64_t last_submission_time;
    uint32_t completed_operations;
};

// Enhanced Queue Information (extends Nova's basic Queues struct)
struct EnhancedQueues {
    // Nova's original queues (compatibility layer)
    VkQueue nova_graphics;
    VkQueue nova_present; 
    VkQueue nova_transfer;
    VkQueue nova_compute;
    
    // Our multi-queue extensions
    std::vector<VkQueue> all_compute_queues;      // All 4 dedicated compute queues
    std::vector<VkQueue> all_graphics_queues;     // Graphics family queues
    std::vector<VkQueue> video_decode_queues;     // Video decode family
    
    // Queue family information
    uint32_t compute_family_index;
    uint32_t graphics_family_index; 
    uint32_t video_family_index;
    
    // Command pools per queue
    std::vector<VkCommandPool> compute_command_pools;
    std::vector<VkCommandPool> graphics_command_pools;
    std::vector<VkCommandPool> video_command_pools;
};

class QueueManager {
public:
    QueueManager(NovaContext* context);
    ~QueueManager();
    
    bool initialize();
    void shutdown();
    
    // Nova Integration - Extend existing queue discovery  
    bool extendNovaQueues();  // Discover additional queues Nova missed
    void initializeFromNova(NovaCore* nova_core);  // Use Nova's device setup
    
    // Enhanced Queue Discovery and Setup
    std::vector<QueueInfo> getAvailableQueues() const;
    std::vector<QueueInfo> getNovaQueues() const;    // Nova's original 4 queues
    std::vector<QueueInfo> getExtendedQueues() const; // Our additional queues
    bool isQueueFamilyAvailable(QueueType type) const;
    uint32_t getQueueCount(QueueType type) const;
    uint32_t getTotalComputeQueues() const; // Nova: 1, Ours: 4
    
    // Workload Submission
    std::future<void> submit(VkCommandBuffer cmd_buffer, 
                            WorkloadType workload_type,
                            Priority priority = Priority::NORMAL,
                            const std::string& debug_name = "");
    
    std::future<void> submitWithDependency(VkCommandBuffer cmd_buffer,
                                          WorkloadType workload_type, 
                                          VkSemaphore wait_semaphore,
                                          VkSemaphore signal_semaphore = VK_NULL_HANDLE,
                                          Priority priority = Priority::NORMAL);
    
    // Parallel Execution Patterns
    struct ParallelBatch {
        std::vector<VkCommandBuffer> command_buffers;
        std::vector<WorkloadType> workload_types;
        std::vector<std::string> debug_names;
        Priority priority = Priority::NORMAL;
    };
    
    std::vector<std::future<void>> submitParallel(const ParallelBatch& batch);
    
    // AI-Specific Batch Operations
    std::future<void> submitMatrixBatch(const std::vector<VkCommandBuffer>& matrices,
                                       Priority priority = Priority::HIGH);
    
    std::future<void> submitConvolutionBatch(const std::vector<VkCommandBuffer>& conv_ops,
                                            Priority priority = Priority::HIGH);
    
    std::future<void> submitNeuralNetworkLayer(const std::vector<VkCommandBuffer>& layer_ops,
                                              bool wait_for_completion = true);
    
    // Queue Load Balancing
    QueueInfo* selectOptimalQueue(WorkloadType workload_type, Priority priority);
    void redistributeLoad();
    float getSystemUtilization() const;
    
    // Synchronization and Dependencies
    VkSemaphore createSemaphore(const std::string& debug_name = "");
    VkFence createFence(bool signaled = false, const std::string& debug_name = "");
    void waitForFences(const std::vector<VkFence>& fences, uint64_t timeout_ns = UINT64_MAX);
    void waitForQueueIdle(QueueType type, uint32_t queue_index = 0);
    void waitForAllQueues();
    
    // Performance Monitoring
    struct QueuePerformance {
        QueueType type;
        uint32_t queue_index;
        float utilization_percent;
        uint32_t pending_operations;
        uint32_t completed_operations;
        uint64_t average_execution_time_ns;
        float throughput_ops_per_second;
    };
    
    std::vector<QueuePerformance> getPerformanceStats() const;
    void resetPerformanceCounters();
    
    // Advanced Features
    
    // Multi-Queue Neural Network Training
    struct TrainingPipeline {
        VkCommandBuffer forward_pass;
        VkCommandBuffer backward_pass;
        VkCommandBuffer weight_update;
        VkSemaphore forward_complete;
        VkSemaphore backward_complete;
    };
    
    std::future<void> submitTrainingPipeline(const TrainingPipeline& pipeline);
    
    // Asynchronous Memory Operations
    std::future<void> asyncMemoryUpload(VkBuffer dst, const void* src_data, size_t size);
    std::future<void> asyncMemoryDownload(const VkBuffer src, void* dst_data, size_t size);
    
    // Queue Affinity (pin specific operations to specific queues)
    void setQueueAffinity(WorkloadType workload_type, QueueType preferred_queue_type);
    void clearQueueAffinity(WorkloadType workload_type);
    
    // Debug and Diagnostics
    void enableDebugMode(bool enable) { _debug_mode = enable; }
    void printQueueStatus() const;
    void dumpSubmissionHistory() const;
    
private:
    NovaContext* _context;
    NovaCore* _core; 
    VkDevice _device;
    VkPhysicalDevice _physical_device;
    
    // Enhanced queue management (extends Nova)
    EnhancedQueues _enhanced_queues;
    std::vector<QueueInfo> _queues;
    std::unordered_map<WorkloadType, QueueType> _queue_affinity;
    
    // Nova integration state
    bool _nova_initialized;
    
    // Threading and synchronization
    std::vector<std::thread> _worker_threads;
    std::atomic<bool> _shutdown_requested;
    std::mutex _submission_mutex;
    
    // Performance tracking
    mutable std::mutex _stats_mutex;
    std::unordered_map<uint64_t, QueuePerformance> _performance_stats;
    
    // Resource pools
    std::vector<VkSemaphore> _semaphore_pool;
    std::vector<VkFence> _fence_pool;
    std::mutex _resource_mutex;
    
    bool _debug_mode;
    uint64_t _submission_counter;
    
    // Internal methods
    bool _discoverQueues();
    bool _createCommandPools();
    void _startWorkerThreads();
    void _stopWorkerThreads();
    void _queueWorker(size_t queue_index);
    
    uint64_t _makeQueueKey(QueueType type, uint32_t queue_index) const;
    QueueInfo* _findQueue(QueueType type, uint32_t queue_index);
    QueueInfo* _findLeastLoadedQueue(QueueType preferred_type);
    
    void _updateQueueStats(QueueInfo* queue, uint64_t execution_time_ns);
    WorkloadType _detectWorkloadType(VkCommandBuffer cmd_buffer) const;
    
    // Resource management
    VkSemaphore _getSemaphoreFromPool();
    VkFence _getFenceFromPool(bool signaled);
    void _returnSemaphoreToPool(VkSemaphore semaphore);
    void _returnFenceToPool(VkFence fence);
};

// Specialized Queue Managers for AI Components

class NeuralNetworkQueueManager {
public:
    NeuralNetworkQueueManager(QueueManager* base_manager);
    
    // Layer-by-layer execution with optimal queue assignment
    std::future<void> executeLayer(VkCommandBuffer layer_cmd,
                                  const std::string& layer_name,
                                  bool synchronous = false);
    
    // Pipeline multiple layers across dedicated compute queues  
    std::future<void> executePipeline(const std::vector<VkCommandBuffer>& layers,
                                     const std::vector<std::string>& layer_names);
    
    // Batch processing across multiple inputs
    std::future<void> processBatch(const std::vector<VkCommandBuffer>& batch_operations,
                                  uint32_t batch_size);
    
private:
    QueueManager* _base_manager;
    uint32_t _current_compute_queue; // Round-robin across 4 dedicated compute queues
};

class MatrixOperationQueueManager {
public:
    MatrixOperationQueueManager(QueueManager* base_manager);
    
    // Distribute large matrix operations across multiple compute queues
    std::future<void> parallelMatrixMultiply(const std::vector<VkCommandBuffer>& matrix_chunks,
                                            uint32_t matrix_size);
    
    // Batched matrix operations
    std::future<void> batchedMatrixOps(const std::vector<VkCommandBuffer>& operations,
                                      const std::vector<std::string>& op_names);
    
private:
    QueueManager* _base_manager;
};

// Global Queue Manager
namespace Global {
    bool initialize(NovaContext* context);
    void shutdown();
    
    QueueManager* getQueueManager();
    NeuralNetworkQueueManager* getNeuralNetworkQueues();
    MatrixOperationQueueManager* getMatrixQueues();
}

} // namespace GPU  
} // namespace CARL

/**
 * RDNA2 RX 6800 XT Optimization Notes:
 * 
 * Queue Family 0 (Graphics+Compute): 1 queue
 * - Best for: Operations that need both graphics and compute
 * - Use cases: Visualization, render-to-texture, hybrid operations
 * 
 * Queue Family 1 (Dedicated Compute): 4 queues ⭐ OPTIMAL FOR AI
 * - Best for: Pure compute workloads 
 * - Use cases: Matrix multiplication, convolution, neural network training
 * - Strategy: Distribute AI operations across all 4 queues for maximum throughput
 * 
 * Queue Family 2 (Video Decode): 1 queue  
 * - Best for: Video preprocessing for vision models
 * - Use cases: Video frame extraction, preprocessing pipelines
 * 
 * Expected Performance:
 * - 4x parallel compute operations vs single queue
 * - ~3-4x throughput improvement for matrix-heavy workloads
 * - Better GPU utilization (targeting >90% across all CUs)
 */