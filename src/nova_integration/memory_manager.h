#pragma once

#include "nova_context.h"
#include "compute_wrapper.h"
#include <unordered_map>
#include <memory>
#include <mutex>

/**
 * CARL Memory Manager - GPU Memory Pool Management
 * 
 * Efficient GPU memory allocation and management for AI operations.
 * Reduces allocation overhead and fragmentation for training and inference.
 */

namespace CARL {
namespace GPU {

enum class PoolType {
    TENSOR_STORAGE,     // Large tensors for neural networks
    TEMPORARY_COMPUTE,  // Intermediate computation results
    SPIKE_BUFFERS,      // SNN spike trains and membrane potentials
    WEIGHT_STORAGE,     // Model parameters and gradients
    ACTIVATION_CACHE    // Cached activations for backprop
};

struct MemoryStats {
    size_t total_allocated;
    size_t total_free;
    size_t peak_usage;
    size_t current_usage;
    uint32_t allocation_count;
    uint32_t deallocation_count;
    uint32_t pool_hits;
    uint32_t pool_misses;
};

class MemoryPool {
public:
    MemoryPool(size_t block_size, size_t pool_size, PoolType type);
    ~MemoryPool();
    
    ComputeBuffer* allocate(size_t size, DataType data_type, 
                           const std::vector<uint32_t>& dimensions);
    void deallocate(ComputeBuffer* buffer);
    
    MemoryStats getStats() const;
    void defragment();
    void clear();
    
    PoolType getType() const { return _type; }
    size_t getBlockSize() const { return _block_size; }
    
private:
    struct Block {
        ComputeBuffer buffer;
        bool is_free;
        size_t actual_size;
        uint64_t last_used_frame;
    };
    
    PoolType _type;
    size_t _block_size;
    size_t _pool_size;
    std::vector<std::unique_ptr<Block>> _blocks;
    std::vector<size_t> _free_blocks;
    
    mutable std::mutex _mutex;
    MemoryStats _stats;
    uint64_t _current_frame;
    
    Block* _findFreeBlock(size_t required_size);
    bool _expandPool();
};

class MemoryManager {
public:
    MemoryManager(NovaContext* context);
    ~MemoryManager();
    
    bool initialize();
    void shutdown();
    
    // Buffer Allocation
    ComputeBuffer* allocateBuffer(size_t size_bytes, DataType type,
                                 const std::vector<uint32_t>& dimensions,
                                 PoolType pool_type = PoolType::TENSOR_STORAGE);
    void deallocateBuffer(ComputeBuffer* buffer);
    
    // Specialized Allocators for AI Components
    ComputeBuffer* allocateTensor(const std::vector<uint32_t>& dimensions, 
                                 DataType type = DataType::FLOAT32);
    ComputeBuffer* allocateWeights(uint32_t input_size, uint32_t output_size,
                                  DataType type = DataType::FLOAT32);
    ComputeBuffer* allocateActivations(const std::vector<uint32_t>& dimensions,
                                      DataType type = DataType::FLOAT32);
    ComputeBuffer* allocateSpikeBuffer(uint32_t neuron_count, uint32_t time_steps,
                                      DataType type = DataType::BOOL);
    
    // Memory Pools
    void configurePool(PoolType type, size_t block_size, size_t pool_size);
    MemoryStats getPoolStats(PoolType type) const;
    MemoryStats getTotalStats() const;
    
    // Memory Management
    void defragmentPools();
    void clearUnusedMemory();
    void frameAdvance(); // Call each training iteration
    
    // Memory Transfer Optimization
    void* mapBuffer(ComputeBuffer* buffer, bool read_only = false);
    void unmapBuffer(ComputeBuffer* buffer);
    bool copyBufferAsync(ComputeBuffer* src, ComputeBuffer* dst);
    void waitForCopies();
    
    // Performance Optimization
    void prefetchToGPU(ComputeBuffer* buffer);
    void evictFromGPU(ComputeBuffer* buffer);
    void setMemoryBudget(size_t max_bytes);
    size_t getMemoryBudget() const { return _memory_budget; }
    
    // Debug and Profiling
    void enableDebugMode(bool enable) { _debug_mode = enable; }
    void dumpMemoryMap() const;
    std::vector<ComputeBuffer*> getActiveBuffers() const;
    
private:
    NovaContext* _context;
    NovaCore* _core;
    VkDevice _device;
    
    std::unordered_map<PoolType, std::unique_ptr<MemoryPool>> _pools;
    std::unordered_map<ComputeBuffer*, PoolType> _buffer_pool_map;
    std::unordered_map<void*, ComputeBuffer*> _mapped_buffers;
    
    size_t _memory_budget;
    uint64_t _current_frame;
    bool _debug_mode;
    mutable std::mutex _global_mutex;
    
    // Async copy operations
    struct CopyOperation {
        VkCommandBuffer cmd_buffer;
        VkFence fence;
        bool completed;
    };
    std::vector<CopyOperation> _pending_copies;
    VkCommandPool _copy_command_pool;
    
    // Default pool configurations
    void _setupDefaultPools();
    PoolType _selectPoolType(size_t size, DataType type);
    bool _createCopyCommandPool();
    void _cleanupCopyOperations();
    
    // Memory budget enforcement
    void _enforceMemoryBudget();
    ComputeBuffer* _evictLRUBuffer();
    
    // Utility
    static std::string _poolTypeToString(PoolType type);
};

// Global Memory Manager Instance (Singleton)
class GlobalMemoryManager {
public:
    static MemoryManager* getInstance();
    static bool initialize(NovaContext* context);
    static void shutdown();
    
private:
    static std::unique_ptr<MemoryManager> _instance;
    static std::mutex _instance_mutex;
};

} // namespace GPU
} // namespace CARL