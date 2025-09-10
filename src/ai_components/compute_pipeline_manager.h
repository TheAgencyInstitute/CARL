#pragma once

#include "../nova/Core/core.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/**
 * Compute Pipeline Manager for CARL AI System
 * Handles shader loading, pipeline creation, and compute dispatch
 */

namespace CARL {
namespace AI {

enum class ShaderType {
    MATRIX_MULTIPLY,
    CONVOLUTION_2D,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX,
    POOLING_MAX,
    POOLING_AVERAGE,
    BATCH_NORMALIZATION,
    GRADIENT_DESCENT,
    SNN_SPIKE_UPDATE,
    SPARSE_ATTENTION,
    GAN_GENERATOR,
    GAN_DISCRIMINATOR,
    RL_Q_LEARNING,
    RL_POLICY_GRADIENT,
    NEURAL_VISUALIZATION,
    SPARSE_MEMORY_MANAGER,
    GAN_LOSS_COMPUTATION,
    GAN_PROGRESSIVE_TRAINING
};

struct ComputePipelineInfo {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descriptor_layout;
    VkShaderModule shader_module;
    std::string shader_path;
    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
};

struct PushConstantData {
    uint32_t data[32]; // 128 bytes max push constant size
    uint32_t size;
};

class ComputePipelineManager {
public:
    ComputePipelineManager(VkDevice device, VkPhysicalDevice physical_device);
    ~ComputePipelineManager();
    
    bool initialize();
    void shutdown();
    
    // Pipeline Management
    bool createComputePipeline(ShaderType shader_type, const std::string& shader_path);
    bool createAllPipelines();
    
    // Shader Operations
    VkShaderModule loadShaderModule(const std::string& filepath);
    void destroyShaderModule(VkShaderModule module);
    
    // Descriptor Management
    VkDescriptorSetLayout createDescriptorSetLayout(const std::vector<VkDescriptorType>& binding_types);
    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout);
    void updateDescriptorSet(VkDescriptorSet descriptor_set, 
                           const std::vector<VkBuffer>& buffers,
                           const std::vector<VkDescriptorType>& types);
    
    // Compute Dispatch
    void dispatchCompute(ShaderType shader_type,
                        VkCommandBuffer cmd_buffer,
                        VkDescriptorSet descriptor_set,
                        const PushConstantData& push_constants,
                        uint32_t group_count_x,
                        uint32_t group_count_y,
                        uint32_t group_count_z);
    
    // Utility
    ComputePipelineInfo* getPipelineInfo(ShaderType shader_type);
    const std::string& getShaderPath(ShaderType shader_type) const;
    
private:
    VkDevice _device;
    VkPhysicalDevice _physical_device;
    VkDescriptorPool _descriptor_pool;
    
    std::unordered_map<ShaderType, ComputePipelineInfo> _pipelines;
    std::unordered_map<ShaderType, std::string> _shader_paths;
    
    // Helper methods
    void initializeShaderPaths();
    bool createDescriptorPool();
    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout descriptor_layout);
    
    // Cleanup
    void destroyPipeline(ComputePipelineInfo& info);
};

} // namespace AI
} // namespace CARL