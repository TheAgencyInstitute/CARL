#pragma once

#include "../../nova/Nova.h"
#include "../core/Types/Types.h"
#include <memory>
#include <string>

/**
 * CARL Nova Context - GPU Acceleration Interface
 * 
 * Provides simplified interface to Nova's Vulkan compute capabilities
 * for CARL AI system components (RL, GAN, CNN, SNN)
 */

namespace CARL {
namespace GPU {

enum class DeviceType {
    ANY,
    DISCRETE_GPU,
    INTEGRATED_GPU,
    CPU_FALLBACK
};

enum class ComputeCapability {
    BASIC_MATH,      // Matrix ops, activation functions
    AI_TRAINING,     // Neural network training loops
    NEUROMORPHIC,    // SNN spike processing
    ALL_FEATURES
};

struct GPUInfo {
    std::string name;
    DeviceType type;
    size_t memory_bytes;
    uint32_t compute_units;
    bool supports_fp16;
    bool supports_int8;
};

struct NovaContextConfig {
    std::string app_name = "CARL AI System";
    DeviceType preferred_device = DeviceType::DISCRETE_GPU;
    ComputeCapability required_features = ComputeCapability::AI_TRAINING;
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB default
    bool enable_profiling = true;
    std::string debug_level = "info";
};

class NovaContext {
public:
    NovaContext();
    ~NovaContext();
    
    // Initialization
    bool initialize(const NovaContextConfig& config);
    void shutdown();
    bool isInitialized() const { return _initialized; }
    
    // Device Information
    GPUInfo getDeviceInfo() const;
    bool isComputeCapable() const;
    
    // Nova Access
    Nova* getNova() { return _nova.get(); }
    NovaCore* getCore() { return _core; }
    
    // Compute Pipeline Management
    VkPipeline createComputePipeline(const std::string& shader_path);
    void destroyComputePipeline(VkPipeline pipeline);
    
    // Error Handling
    bool hasError() const { return !_error_message.empty(); }
    std::string getLastError() const { return _error_message; }
    void clearError() { _error_message.clear(); }

private:
    std::unique_ptr<Nova> _nova;
    NovaCore* _core;
    NovaContextConfig _config;
    bool _initialized;
    std::string _error_message;
    
    bool _validateDevice();
    bool _setupMemoryPools();
    void _setError(const std::string& message);
};

} // namespace GPU
} // namespace CARL