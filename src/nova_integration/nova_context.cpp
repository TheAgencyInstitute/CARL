#include "nova_context.h"
#include "../../nova/Core/components/logger.h"
#include <cassert>

namespace CARL {
namespace GPU {

NovaContext::NovaContext() 
    : _nova(nullptr)
    , _core(nullptr) 
    , _initialized(false)
{
}

NovaContext::~NovaContext() {
    shutdown();
}

bool NovaContext::initialize(const NovaContextConfig& config) {
    if (_initialized) {
        _setError("NovaContext already initialized");
        return false;
    }
    
    _config = config;
    
    try {
        // Create Nova configuration
        NovaConfig nova_config = {
            .name = config.app_name,
            .screen = { 1600, 1200 },  // Headless compute doesn't need display
            .debug_level = config.debug_level,
            .dimensions = "3D",
            .camera_type = "fixed",
            .compute = true  // Enable compute capabilities
        };
        
        report(LOGGER::INFO, "CARL - Initializing Nova GPU context");
        
        // Create Nova instance
        _nova = std::make_unique<Nova>(nova_config);
        if (!_nova) {
            _setError("Failed to create Nova instance");
            return false;
        }
        
        // Get access to Nova core for compute operations
        _core = _nova.get()->_architect; // Access private member (friend class needed)
        if (!_core) {
            _setError("Failed to access Nova core");
            return false;
        }
        
        // Validate device capabilities
        if (!_validateDevice()) {
            return false;
        }
        
        // Setup memory pools for AI operations
        if (!_setupMemoryPools()) {
            return false;
        }
        
        _initialized = true;
        report(LOGGER::INFO, "CARL - Nova GPU context initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        _setError(std::string("Nova initialization failed: ") + e.what());
        return false;
    }
}

void NovaContext::shutdown() {
    if (!_initialized) {
        return;
    }
    
    report(LOGGER::INFO, "CARL - Shutting down Nova GPU context");
    
    // Nova destructor handles Vulkan cleanup
    _nova.reset();
    _core = nullptr;
    _initialized = false;
}

GPUInfo NovaContext::getDeviceInfo() const {
    if (!_initialized || !_core) {
        return {"Unknown", DeviceType::CPU_FALLBACK, 0, 0, false, false};
    }
    
    GPUInfo info;
    
    // Get device properties from Vulkan physical device
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(_core->physical_device, &props);
    
    info.name = std::string(props.deviceName);
    
    // Determine device type
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            info.type = DeviceType::DISCRETE_GPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            info.type = DeviceType::INTEGRATED_GPU;
            break;
        default:
            info.type = DeviceType::ANY;
            break;
    }
    
    // Get memory properties
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(_core->physical_device, &mem_props);
    
    // Find largest device-local heap
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.memory_bytes = std::max(info.memory_bytes, 
                                       static_cast<size_t>(mem_props.memoryHeaps[i].size));
        }
    }
    
    // Get compute capabilities (simplified)
    info.compute_units = props.limits.maxComputeWorkGroupCount[0];
    
    // Check for additional features
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(_core->physical_device, &features);
    
    // Note: Actual FP16/INT8 support requires extension queries
    info.supports_fp16 = features.shaderFloat64;  // Simplified check
    info.supports_int8 = true;  // Most modern GPUs support int8
    
    return info;
}

bool NovaContext::isComputeCapable() const {
    if (!_initialized || !_core) {
        return false;
    }
    
    // Check if device supports compute queues
    return _core->queues.compute_family.has_value();
}

VkPipeline NovaContext::createComputePipeline(const std::string& shader_path) {
    if (!_initialized || !_core) {
        _setError("NovaContext not initialized");
        return VK_NULL_HANDLE;
    }
    
    try {
        // Use Nova's existing compute pipeline creation
        _core->constructComputePipeline();
        return _core->compute_pipeline->instance;
        
    } catch (const std::exception& e) {
        _setError(std::string("Failed to create compute pipeline: ") + e.what());
        return VK_NULL_HANDLE;
    }
}

void NovaContext::destroyComputePipeline(VkPipeline pipeline) {
    if (!_initialized || !_core || pipeline == VK_NULL_HANDLE) {
        return;
    }
    
    // Nova handles pipeline cleanup in destructor
    // For now, just log the destruction
    report(LOGGER::VERBOSE, "CARL - Destroying compute pipeline");
}

bool NovaContext::_validateDevice() {
    if (!_core) {
        _setError("No Nova core available for device validation");
        return false;
    }
    
    // Check for compute queue support
    if (!isComputeCapable()) {
        _setError("Device does not support compute operations");
        return false;
    }
    
    GPUInfo info = getDeviceInfo();
    
    // Minimum memory requirement for AI operations (512MB)
    if (info.memory_bytes < 512 * 1024 * 1024) {
        _setError("Insufficient GPU memory for AI operations");
        return false;
    }
    
    report(LOGGER::INFO, "CARL - GPU Device: %s (%zu MB)", 
           info.name.c_str(), info.memory_bytes / (1024 * 1024));
    
    return true;
}

bool NovaContext::_setupMemoryPools() {
    // Nova handles memory allocation through VMA
    // This could be extended to pre-allocate pools for AI operations
    report(LOGGER::VERBOSE, "CARL - Memory pools configured via Nova VMA");
    return true;
}

void NovaContext::_setError(const std::string& message) {
    _error_message = message;
    report(LOGGER::ERROR, "CARL - %s", message.c_str());
}

} // namespace GPU
} // namespace CARL