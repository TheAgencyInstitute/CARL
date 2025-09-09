#pragma once

/**
 * CARL GPU - Main Interface for GPU-Accelerated AI Operations
 * 
 * Single header that provides access to all CARL GPU acceleration features.
 * This is the primary interface for AI components (RL, GAN, CNN, SNN).
 */

#include "nova_context.h"
#include "compute_wrapper.h"
#include "memory_manager.h"

namespace CARL {
namespace GPU {

// Forward declarations for AI components
class RLAccelerator;
class GANAccelerator;  
class CNNAccelerator;
class SNNAccelerator;

/**
 * Main GPU System - Coordinates all GPU operations for CARL
 */
class GPUSystem {
public:
    GPUSystem();
    ~GPUSystem();
    
    // System Lifecycle
    bool initialize(const NovaContextConfig& config = {});
    void shutdown();
    bool isInitialized() const;
    
    // Core Components Access
    NovaContext* getContext() { return _context.get(); }
    ComputeWrapper* getCompute() { return _compute.get(); }
    MemoryManager* getMemory() { return _memory.get(); }
    
    // AI Component Accelerators  
    RLAccelerator* getRL() { return _rl.get(); }
    GANAccelerator* getGAN() { return _gan.get(); }
    CNNAccelerator* getCNN() { return _cnn.get(); }
    SNNAccelerator* getSNN() { return _snn.get(); }
    
    // System Information
    GPUInfo getDeviceInfo() const;
    MemoryStats getMemoryStats() const;
    bool hasError() const;
    std::string getLastError() const;
    
    // Performance Management
    void frameAdvance(); // Call each training iteration
    void waitForIdle();
    void enableProfiling(bool enable);
    
    // System Configuration
    void setMemoryBudget(size_t max_bytes);
    void configureMemoryPool(PoolType type, size_t block_size, size_t pool_size);
    
private:
    std::unique_ptr<NovaContext> _context;
    std::unique_ptr<ComputeWrapper> _compute;
    std::unique_ptr<MemoryManager> _memory;
    
    // AI Accelerators
    std::unique_ptr<RLAccelerator> _rl;
    std::unique_ptr<GANAccelerator> _gan;
    std::unique_ptr<CNNAccelerator> _cnn;
    std::unique_ptr<SNNAccelerator> _snn;
    
    bool _initialized;
    std::string _error_message;
    
    bool _initializeAccelerators();
    void _shutdownAccelerators();
    void _setError(const std::string& message);
};

/**
 * Global GPU System Access (Singleton)
 * Provides convenient access to GPU acceleration throughout CARL
 */
namespace Global {
    bool initialize(const NovaContextConfig& config = {});
    void shutdown();
    
    // Quick access functions
    GPUSystem* getGPU();
    NovaContext* getContext();
    ComputeWrapper* getCompute();
    MemoryManager* getMemory();
    
    // AI accelerators
    RLAccelerator* getRL();
    GANAccelerator* getGAN();
    CNNAccelerator* getCNN();
    SNNAccelerator* getSNN();
    
    // Utility
    bool isAvailable();
    GPUInfo getDeviceInfo();
}

} // namespace GPU
} // namespace CARL

/**
 * Convenience macros for common operations
 */
#define CARL_GPU_AVAILABLE() CARL::GPU::Global::isAvailable()
#define CARL_GPU() CARL::GPU::Global::getGPU()
#define CARL_COMPUTE() CARL::GPU::Global::getCompute()  
#define CARL_MEMORY() CARL::GPU::Global::getMemory()
#define CARL_RL() CARL::GPU::Global::getRL()
#define CARL_GAN() CARL::GPU::Global::getGAN()
#define CARL_CNN() CARL::GPU::Global::getCNN()
#define CARL_SNN() CARL::GPU::Global::getSNN()

/**
 * Usage Example:
 * 
 * #include "carl_gpu.h"
 * 
 * // Initialize GPU system
 * if (!CARL::GPU::Global::initialize()) {
 *     // Fall back to CPU
 * }
 * 
 * // Use CNN accelerator
 * auto cnn = CARL_CNN();
 * auto input = CARL_MEMORY()->allocateTensor({1, 224, 224, 3});
 * auto output = cnn->convolution(input, kernel);
 * 
 * // Cleanup
 * CARL::GPU::Global::shutdown();
 */