#pragma once

#include "nova_context.h"
#include "../core/Types/Types.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <functional>

/**
 * CARL Compute Wrapper - AI Operations on GPU
 * 
 * High-level interface for AI mathematical operations using Nova's
 * Vulkan compute shaders. Designed for CARL's RL, GAN, CNN, and SNN components.
 */

namespace CARL {
namespace GPU {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    BOOL
};

enum class OperationType {
    MATRIX_MULTIPLY,
    CONVOLUTION_2D,
    ACTIVATION,
    POOLING,
    REDUCTION,
    SPIKE_PROCESS,
    CUSTOM
};

struct ComputeBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_data;
    size_t size_bytes;
    DataType data_type;
    std::vector<uint32_t> dimensions;  // Shape: [batch, height, width, channels, ...]
    
    ComputeBuffer() : buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE), 
                     mapped_data(nullptr), size_bytes(0), data_type(DataType::FLOAT32) {}
};

struct ComputeKernel {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_layout;
    OperationType type;
    std::string name;
    uint32_t local_size_x, local_size_y, local_size_z;
    
    ComputeKernel() : pipeline(VK_NULL_HANDLE), layout(VK_NULL_HANDLE),
                     descriptor_set(VK_NULL_HANDLE), descriptor_layout(VK_NULL_HANDLE),
                     type(OperationType::CUSTOM), local_size_x(1), local_size_y(1), local_size_z(1) {}
};

class ComputeWrapper {
public:
    ComputeWrapper(NovaContext* context);
    ~ComputeWrapper();
    
    // Buffer Management
    ComputeBuffer* createBuffer(size_t size_bytes, DataType type, 
                               const std::vector<uint32_t>& dimensions,
                               VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    void destroyBuffer(ComputeBuffer* buffer);
    
    // Data Transfer
    bool uploadData(ComputeBuffer* buffer, const void* data, size_t size_bytes);
    bool downloadData(ComputeBuffer* buffer, void* data, size_t size_bytes);
    void copyBuffer(ComputeBuffer* src, ComputeBuffer* dst);
    
    // Kernel Management
    ComputeKernel* createKernel(const std::string& shader_source, 
                               OperationType type = OperationType::CUSTOM);
    ComputeKernel* createKernel(const std::vector<uint32_t>& spirv_code,
                               OperationType type = OperationType::CUSTOM);
    void destroyKernel(ComputeKernel* kernel);
    
    // Compute Execution
    bool bindBuffers(ComputeKernel* kernel, const std::vector<ComputeBuffer*>& buffers);
    bool dispatch(ComputeKernel* kernel, uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void waitForCompletion();
    
    // High-Level AI Operations
    bool matrixMultiply(ComputeBuffer* a, ComputeBuffer* b, ComputeBuffer* result);
    bool convolution2D(ComputeBuffer* input, ComputeBuffer* kernel, ComputeBuffer* output,
                      uint32_t stride_x = 1, uint32_t stride_y = 1);
    bool activation(ComputeBuffer* input, ComputeBuffer* output, const std::string& function);
    bool pooling2D(ComputeBuffer* input, ComputeBuffer* output, 
                   uint32_t pool_size, const std::string& type = "max");
    
    // SNN Operations
    bool processSpikes(ComputeBuffer* spikes, ComputeBuffer* weights, ComputeBuffer* output);
    bool updateSynapses(ComputeBuffer* pre_spikes, ComputeBuffer* post_spikes, 
                       ComputeBuffer* weights, float learning_rate);
    
    // Error Handling
    bool hasError() const { return !_error_message.empty(); }
    std::string getLastError() const { return _error_message; }
    void clearError() { _error_message.clear(); }
    
    // Performance Profiling
    struct ProfileInfo {
        uint64_t gpu_time_ns;
        uint64_t cpu_time_ns;
        size_t memory_used_bytes;
    };
    ProfileInfo getLastProfileInfo() const { return _last_profile; }
    void enableProfiling(bool enable) { _profiling_enabled = enable; }

private:
    NovaContext* _context;
    NovaCore* _core;
    VkDevice _device;
    VkCommandPool _command_pool;
    VkCommandBuffer _command_buffer;
    VkDescriptorPool _descriptor_pool;
    
    std::vector<ComputeBuffer*> _buffers;
    std::vector<ComputeKernel*> _kernels;
    
    std::string _error_message;
    ProfileInfo _last_profile;
    bool _profiling_enabled;
    
    // Internal Operations
    bool _initializeResources();
    void _cleanupResources();
    VkDescriptorSetLayout _createDescriptorLayout(uint32_t binding_count);
    VkDescriptorSet _allocateDescriptorSet(VkDescriptorSetLayout layout);
    bool _compileShader(const std::string& source, std::vector<uint32_t>& spirv);
    void _setError(const std::string& message);
    
    // Built-in Kernels
    ComputeKernel* _matrix_multiply_kernel;
    ComputeKernel* _conv2d_kernel;
    ComputeKernel* _activation_kernel;
    ComputeKernel* _pooling_kernel;
    ComputeKernel* _spike_kernel;
    
    bool _createBuiltinKernels();
    void _destroyBuiltinKernels();
};

// Utility Functions
size_t getDataTypeSize(DataType type);
std::string getDataTypeName(DataType type);
uint32_t calculateElements(const std::vector<uint32_t>& dimensions);
std::vector<uint32_t> calculateWorkGroups(const std::vector<uint32_t>& dimensions, 
                                         uint32_t local_size_x, 
                                         uint32_t local_size_y = 1, 
                                         uint32_t local_size_z = 1);

} // namespace GPU
} // namespace CARL