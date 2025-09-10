#include "compute_pipeline_manager.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace CARL {
namespace AI {

ComputePipelineManager::ComputePipelineManager(VkDevice device, VkPhysicalDevice physical_device)
    : _device(device), _physical_device(physical_device), _descriptor_pool(VK_NULL_HANDLE) {
    initializeShaderPaths();
}

ComputePipelineManager::~ComputePipelineManager() {
    shutdown();
}

bool ComputePipelineManager::initialize() {
    if (!createDescriptorPool()) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }
    
    if (!createAllPipelines()) {
        std::cerr << "Failed to create compute pipelines" << std::endl;
        return false;
    }
    
    std::cout << "ComputePipelineManager initialized with " << _pipelines.size() << " pipelines" << std::endl;
    return true;
}

void ComputePipelineManager::shutdown() {
    // Destroy all pipelines
    for (auto& pair : _pipelines) {
        destroyPipeline(pair.second);
    }
    _pipelines.clear();
    
    // Destroy descriptor pool
    if (_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
        _descriptor_pool = VK_NULL_HANDLE;
    }
}

void ComputePipelineManager::initializeShaderPaths() {
    const std::string shader_dir = "src/shaders/compiled/";
    
    _shader_paths[ShaderType::MATRIX_MULTIPLY] = shader_dir + "matrix_multiply.comp.spv";
    _shader_paths[ShaderType::CONVOLUTION_2D] = shader_dir + "convolution2d.comp.spv";
    _shader_paths[ShaderType::ACTIVATION_RELU] = shader_dir + "activation_relu.comp.spv";
    _shader_paths[ShaderType::ACTIVATION_SOFTMAX] = shader_dir + "activation_softmax.comp.spv";
    _shader_paths[ShaderType::POOLING_MAX] = shader_dir + "pooling_max.comp.spv";
    _shader_paths[ShaderType::POOLING_AVERAGE] = shader_dir + "pooling_average.comp.spv";
    _shader_paths[ShaderType::BATCH_NORMALIZATION] = shader_dir + "batch_normalization.comp.spv";
    _shader_paths[ShaderType::GRADIENT_DESCENT] = shader_dir + "gradient_descent.comp.spv";
    _shader_paths[ShaderType::SNN_SPIKE_UPDATE] = shader_dir + "snn_spike_update.comp.spv";
    _shader_paths[ShaderType::SPARSE_ATTENTION] = shader_dir + "sparse_attention.comp.spv";
    _shader_paths[ShaderType::GAN_GENERATOR] = shader_dir + "gan_generator.comp.spv";
    _shader_paths[ShaderType::GAN_DISCRIMINATOR] = shader_dir + "gan_discriminator.comp.spv";
    _shader_paths[ShaderType::RL_Q_LEARNING] = shader_dir + "rl_q_learning.comp.spv";
    _shader_paths[ShaderType::RL_POLICY_GRADIENT] = shader_dir + "rl_policy_gradient.comp.spv";
    _shader_paths[ShaderType::NEURAL_VISUALIZATION] = shader_dir + "neural_visualization.comp.spv";
    _shader_paths[ShaderType::SPARSE_MEMORY_MANAGER] = shader_dir + "sparse_memory_manager.comp.spv";
    _shader_paths[ShaderType::GAN_LOSS_COMPUTATION] = shader_dir + "gan_loss_computation.comp.spv";
    _shader_paths[ShaderType::GAN_PROGRESSIVE_TRAINING] = shader_dir + "gan_progressive_training.comp.spv";
}

bool ComputePipelineManager::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 2000;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 500;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = 100;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1000;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    
    return vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptor_pool) == VK_SUCCESS;
}

VkShaderModule ComputePipelineManager::loadShaderModule(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return VK_NULL_HANDLE;
    }
    
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module for: " << filepath << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return shaderModule;
}

VkDescriptorSetLayout ComputePipelineManager::createDescriptorSetLayout(const std::vector<VkDescriptorType>& binding_types) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    
    for (size_t i = 0; i < binding_types.size(); i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = static_cast<uint32_t>(i);
        binding.descriptorCount = 1;
        binding.descriptorType = binding_types[i];
        binding.pImmutableSamplers = nullptr;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return layout;
}

VkPipelineLayout ComputePipelineManager::createPipelineLayout(VkDescriptorSetLayout descriptor_layout) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_layout;
    
    // Push constants for shader parameters
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128; // 32 uint32_t values
    
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    VkPipelineLayout layout;
    if (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return layout;
}

bool ComputePipelineManager::createComputePipeline(ShaderType shader_type, const std::string& shader_path) {
    ComputePipelineInfo info = {};
    
    // Load shader module
    info.shader_module = loadShaderModule(shader_path);
    if (info.shader_module == VK_NULL_HANDLE) {
        return false;
    }
    
    // Create descriptor set layout based on shader type
    std::vector<VkDescriptorType> binding_types;
    
    switch (shader_type) {
        case ShaderType::MATRIX_MULTIPLY:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        case ShaderType::CONVOLUTION_2D:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        case ShaderType::ACTIVATION_RELU:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::ACTIVATION_SOFTMAX:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::BATCH_NORMALIZATION:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::GRADIENT_DESCENT:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::GAN_GENERATOR:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        case ShaderType::GAN_DISCRIMINATOR:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        case ShaderType::RL_Q_LEARNING:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::RL_POLICY_GRADIENT:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::SNN_SPIKE_UPDATE:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::SPARSE_ATTENTION:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::SPARSE_MEMORY_MANAGER:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::GAN_LOSS_COMPUTATION:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
        case ShaderType::GAN_PROGRESSIVE_TRAINING:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        case ShaderType::NEURAL_VISUALIZATION:
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
            info.local_size_x = 16; info.local_size_y = 16; info.local_size_z = 1;
            break;
        default:
            // Generic 2-buffer setup for other shaders
            binding_types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            info.local_size_x = 256; info.local_size_y = 1; info.local_size_z = 1;
            break;
    }
    
    info.descriptor_layout = createDescriptorSetLayout(binding_types);
    if (info.descriptor_layout == VK_NULL_HANDLE) {
        destroyShaderModule(info.shader_module);
        return false;
    }
    
    info.layout = createPipelineLayout(info.descriptor_layout);
    if (info.layout == VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_device, info.descriptor_layout, nullptr);
        destroyShaderModule(info.shader_module);
        return false;
    }
    
    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = info.layout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = info.shader_module;
    pipelineInfo.stage.pName = "main";
    
    if (vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &info.pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline for shader: " << shader_path << std::endl;
        vkDestroyPipelineLayout(_device, info.layout, nullptr);
        vkDestroyDescriptorSetLayout(_device, info.descriptor_layout, nullptr);
        destroyShaderModule(info.shader_module);
        return false;
    }
    
    info.shader_path = shader_path;
    _pipelines[shader_type] = info;
    
    std::cout << "Created compute pipeline for: " << shader_path << std::endl;
    return true;
}

bool ComputePipelineManager::createAllPipelines() {
    bool all_success = true;
    
    for (const auto& pair : _shader_paths) {
        if (!createComputePipeline(pair.first, pair.second)) {
            std::cerr << "Failed to create pipeline for shader type: " << static_cast<int>(pair.first) << std::endl;
            all_success = false;
        }
    }
    
    return all_success;
}

VkDescriptorSet ComputePipelineManager::allocateDescriptorSet(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptor_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;
    
    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(_device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return descriptorSet;
}

void ComputePipelineManager::updateDescriptorSet(VkDescriptorSet descriptor_set, 
                                                const std::vector<VkBuffer>& buffers,
                                                const std::vector<VkDescriptorType>& types) {
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    
    bufferInfos.resize(buffers.size());
    descriptorWrites.resize(buffers.size());
    
    for (size_t i = 0; i < buffers.size(); i++) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptor_set;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = types[i];
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    
    vkUpdateDescriptorSets(_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void ComputePipelineManager::dispatchCompute(ShaderType shader_type,
                                           VkCommandBuffer cmd_buffer,
                                           VkDescriptorSet descriptor_set,
                                           const PushConstantData& push_constants,
                                           uint32_t group_count_x,
                                           uint32_t group_count_y,
                                           uint32_t group_count_z) {
    auto it = _pipelines.find(shader_type);
    if (it == _pipelines.end()) {
        std::cerr << "Pipeline not found for shader type: " << static_cast<int>(shader_type) << std::endl;
        return;
    }
    
    const ComputePipelineInfo& info = it->second;
    
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, info.pipeline);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, info.layout, 0, 1, &descriptor_set, 0, nullptr);
    
    if (push_constants.size > 0) {
        vkCmdPushConstants(cmd_buffer, info.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants.size, push_constants.data);
    }
    
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, group_count_z);
}

ComputePipelineInfo* ComputePipelineManager::getPipelineInfo(ShaderType shader_type) {
    auto it = _pipelines.find(shader_type);
    return (it != _pipelines.end()) ? &it->second : nullptr;
}

const std::string& ComputePipelineManager::getShaderPath(ShaderType shader_type) const {
    static const std::string empty_string;
    auto it = _shader_paths.find(shader_type);
    return (it != _shader_paths.end()) ? it->second : empty_string;
}

void ComputePipelineManager::destroyShaderModule(VkShaderModule module) {
    if (module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(_device, module, nullptr);
    }
}

void ComputePipelineManager::destroyPipeline(ComputePipelineInfo& info) {
    if (info.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(_device, info.pipeline, nullptr);
        info.pipeline = VK_NULL_HANDLE;
    }
    
    if (info.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(_device, info.layout, nullptr);
        info.layout = VK_NULL_HANDLE;
    }
    
    if (info.descriptor_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_device, info.descriptor_layout, nullptr);
        info.descriptor_layout = VK_NULL_HANDLE;
    }
    
    destroyShaderModule(info.shader_module);
    info.shader_module = VK_NULL_HANDLE;
}

} // namespace AI
} // namespace CARL