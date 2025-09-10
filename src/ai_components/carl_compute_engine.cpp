#include "carl_compute_engine.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

namespace CARL {
namespace AI {

CarlComputeEngine::CarlComputeEngine(NovaCore* nova_core) 
    : _nova_core(nova_core), _device(VK_NULL_HANDLE), _physical_device(VK_NULL_HANDLE) {
    _queue_utilization.resize(8, 0.0f); // 8 total queues available
    _queue_operation_counts.resize(8, 0);
    _queue_last_submission.resize(8);
}

CarlComputeEngine::~CarlComputeEngine() {
    shutdown();
}

bool CarlComputeEngine::initialize() {
    if (!_nova_core) {
        std::cerr << "CARL Compute Engine: Nova core not provided!" << std::endl;
        return false;
    }
    
    _device = _nova_core->getLogicalDevice();
    _physical_device = _nova_core->getPhysicalDevice();
    
    if (_device == VK_NULL_HANDLE || _physical_device == VK_NULL_HANDLE) {
        std::cerr << "CARL Compute Engine: Invalid Vulkan devices!" << std::endl;
        return false;
    }
    
    // Verify multi-queue support
    const auto& queues = _nova_core->getQueues();
    if (queues.total_compute_queues() < 4) {
        std::cerr << "CARL Compute Engine: Expected 4+ compute queues, found " 
                  << queues.total_compute_queues() << std::endl;
        return false;
    }
    
    // Create descriptor pool for AI operations
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 1000; // Support many buffers
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 100;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 500;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    
    if (vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "CARL Compute Engine: Failed to create descriptor pool!" << std::endl;
        return false;
    }
    
    // Initialize queue utilization tracking
    for (size_t i = 0; i < _queue_last_submission.size(); i++) {
        _queue_last_submission[i] = std::chrono::steady_clock::now();
    }
    
    std::cout << "CARL Compute Engine: Initialized with " << queues.total_compute_queues() 
              << " compute queues" << std::endl;
    
    return true;
}

void CarlComputeEngine::shutdown() {
    if (_device != VK_NULL_HANDLE) {
        // Clean up descriptor pool
        if (_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
            _descriptor_pool = VK_NULL_HANDLE;
        }
        
        // Clean up compute pipelines
        for (auto pipeline : _compute_pipelines) {
            vkDestroyPipeline(_device, pipeline, nullptr);
        }
        _compute_pipelines.clear();
        
        // Clean up pipeline layouts
        for (auto layout : _pipeline_layouts) {
            vkDestroyPipelineLayout(_device, layout, nullptr);
        }
        _pipeline_layouts.clear();
        
        // Clean up descriptor set layouts
        for (auto layout : _descriptor_layouts) {
            vkDestroyDescriptorSetLayout(_device, layout, nullptr);
        }
        _descriptor_layouts.clear();
        
        // Clean up buffer pool
        for (auto& buffer : _buffer_pool) {
            if (buffer) {
                destroyBuffer(buffer.get());
            }
        }
        _buffer_pool.clear();
    }
}

ComputeBuffer* CarlComputeEngine::createBuffer(size_t size_bytes, VkBufferUsageFlags usage) {
    auto buffer = std::make_unique<ComputeBuffer>();
    buffer->size_bytes = size_bytes;
    buffer->element_count = size_bytes / sizeof(float); // Assume float elements
    buffer->mapped_data = nullptr;
    
    // Create Vulkan buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_bytes;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer->buffer) != VK_SUCCESS) {
        std::cerr << "CARL Compute Engine: Failed to create buffer!" << std::endl;
        return nullptr;
    }
    
    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device, buffer->buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (vkAllocateMemory(_device, &allocInfo, nullptr, &buffer->memory) != VK_SUCCESS) {
        std::cerr << "CARL Compute Engine: Failed to allocate buffer memory!" << std::endl;
        vkDestroyBuffer(_device, buffer->buffer, nullptr);
        return nullptr;
    }
    
    vkBindBufferMemory(_device, buffer->buffer, buffer->memory, 0);
    
    // Create descriptor set
    VkDescriptorSetLayout layouts[] = {createDescriptorSetLayout({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER})};
    VkDescriptorSetAllocateInfo allocInfoDesc{};
    allocInfoDesc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoDesc.descriptorPool = _descriptor_pool;
    allocInfoDesc.descriptorSetCount = 1;
    allocInfoDesc.pSetLayouts = layouts;
    
    if (vkAllocateDescriptorSets(_device, &allocInfoDesc, &buffer->descriptor_set) != VK_SUCCESS) {
        std::cerr << "CARL Compute Engine: Failed to allocate descriptor set!" << std::endl;
    }
    
    // Update descriptor set
    VkDescriptorBufferInfo bufferInfoDesc{};
    bufferInfoDesc.buffer = buffer->buffer;
    bufferInfoDesc.offset = 0;
    bufferInfoDesc.range = size_bytes;
    
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = buffer->descriptor_set;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfoDesc;
    
    vkUpdateDescriptorSets(_device, 1, &descriptorWrite, 0, nullptr);
    
    ComputeBuffer* raw_buffer = buffer.get();
    _buffer_pool.push_back(std::move(buffer));
    return raw_buffer;
}

void CarlComputeEngine::destroyBuffer(ComputeBuffer* buffer) {
    if (!buffer || _device == VK_NULL_HANDLE) return;
    
    if (buffer->mapped_data) {
        vkUnmapMemory(_device, buffer->memory);
        buffer->mapped_data = nullptr;
    }
    
    if (buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(_device, buffer->buffer, nullptr);
        buffer->buffer = VK_NULL_HANDLE;
    }
    
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(_device, buffer->memory, nullptr);
        buffer->memory = VK_NULL_HANDLE;
    }
}

void CarlComputeEngine::uploadData(ComputeBuffer* buffer, const void* data, size_t size) {
    if (!buffer || !data) return;
    
    void* mappedData;
    vkMapMemory(_device, buffer->memory, 0, size, 0, &mappedData);
    memcpy(mappedData, data, size);
    vkUnmapMemory(_device, buffer->memory);
}

void CarlComputeEngine::downloadData(ComputeBuffer* buffer, void* data, size_t size) {
    if (!buffer || !data) return;
    
    void* mappedData;
    vkMapMemory(_device, buffer->memory, 0, size, 0, &mappedData);
    memcpy(data, mappedData, size);
    vkUnmapMemory(_device, buffer->memory);
}

std::future<void> CarlComputeEngine::matrixMultiply(ComputeBuffer* matrix_a, 
                                                  ComputeBuffer* matrix_b, 
                                                  ComputeBuffer* result,
                                                  uint32_t rows_a, uint32_t cols_a, uint32_t cols_b) {
    return std::async(std::launch::async, [this, matrix_a, matrix_b, result, rows_a, cols_a, cols_b]() {
        AIOperation operation;
        operation.type = AIOperationType::MATRIX_MULTIPLY;
        operation.shader_path = "shaders/matrix_multiply.comp.spv";
        operation.input_buffers = {matrix_a, matrix_b};
        operation.output_buffers = {result};
        operation.dispatch_x = (cols_b + 15) / 16;
        operation.dispatch_y = (rows_a + 15) / 16;
        operation.dispatch_z = 1;
        operation.assigned_queue = selectOptimalQueue(AIOperationType::MATRIX_MULTIPLY, 
                                                     rows_a * cols_a * cols_b);
        operation.debug_name = "Matrix Multiply";
        
        executeOperation(operation, operation.assigned_queue);
    });
}

std::future<void> CarlComputeEngine::convolution2D(ComputeBuffer* input, 
                                                  ComputeBuffer* kernel, 
                                                  ComputeBuffer* output,
                                                  uint32_t input_width, uint32_t input_height,
                                                  uint32_t kernel_size, uint32_t stride) {
    return std::async(std::launch::async, [this, input, kernel, output, input_width, input_height, kernel_size, stride]() {
        AIOperation operation;
        operation.type = AIOperationType::CONVOLUTION_2D;
        operation.shader_path = "shaders/convolution2d.comp.spv";
        operation.input_buffers = {input, kernel};
        operation.output_buffers = {output};
        operation.dispatch_x = (input_width + 15) / 16;
        operation.dispatch_y = (input_height + 15) / 16;
        operation.dispatch_z = 1;
        operation.assigned_queue = selectOptimalQueue(AIOperationType::CONVOLUTION_2D, 
                                                     input_width * input_height * kernel_size * kernel_size);
        operation.debug_name = "2D Convolution";
        
        executeOperation(operation, operation.assigned_queue);
    });
}

std::future<void> CarlComputeEngine::activationReLU(ComputeBuffer* input, 
                                                   ComputeBuffer* output, 
                                                   uint32_t element_count) {
    return std::async(std::launch::async, [this, input, output, element_count]() {
        AIOperation operation;
        operation.type = AIOperationType::ACTIVATION_RELU;
        operation.shader_path = "shaders/activation_relu.comp.spv";
        operation.input_buffers = {input};
        operation.output_buffers = {output};
        operation.dispatch_x = (element_count + 255) / 256;
        operation.dispatch_y = 1;
        operation.dispatch_z = 1;
        operation.assigned_queue = selectOptimalQueue(AIOperationType::ACTIVATION_RELU, element_count);
        operation.debug_name = "ReLU Activation";
        
        executeOperation(operation, operation.assigned_queue);
    });
}

uint32_t CarlComputeEngine::selectOptimalQueue(AIOperationType operation, size_t workload_size) {
    const auto& queues = _nova_core->getQueues();
    uint32_t total_compute = queues.total_compute_queues();
    
    if (total_compute == 0) return 0;
    
    // CARL Queue Assignment Strategy:
    // Queue 0: Matrix multiplication operations
    // Queue 1: Convolution forward passes  
    // Queue 2: Convolution backward passes
    // Queue 3: Activation and pooling operations
    
    switch (operation) {
        case AIOperationType::MATRIX_MULTIPLY:
            return 0 % total_compute;
        case AIOperationType::CONVOLUTION_2D:
            return 1 % total_compute;
        case AIOperationType::ACTIVATION_RELU:
        case AIOperationType::ACTIVATION_SOFTMAX:
        case AIOperationType::POOLING_MAX:
        case AIOperationType::POOLING_AVERAGE:
            return 3 % total_compute;
        case AIOperationType::GRADIENT_DESCENT:
            return 2 % total_compute;
        default:
            // Load balance across all queues for other operations
            uint32_t min_queue = 0;
            float min_utilization = _queue_utilization[0];
            for (uint32_t i = 1; i < total_compute; i++) {
                if (_queue_utilization[i] < min_utilization) {
                    min_utilization = _queue_utilization[i];
                    min_queue = i;
                }
            }
            return min_queue;
    }
}

void CarlComputeEngine::executeOperation(const AIOperation& operation, uint32_t queue_index) {
    const auto& queues = _nova_core->getQueues();
    
    if (queue_index >= queues.total_compute_queues()) {
        std::cerr << "CARL Compute Engine: Invalid queue index " << queue_index << std::endl;
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create command pool if needed
    VkCommandPool commandPool;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queues.indices.compute_family.value();
    
    vkCreateCommandPool(_device, &poolInfo, nullptr, &commandPool);
    
    // Allocate command buffer
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);
    
    // Record command buffer
    recordCommandBuffer(commandBuffer, operation);
    
    // Submit to selected compute queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    VkQueue selected_queue = queues.all_compute_queues[queue_index];
    vkQueueSubmit(selected_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(selected_queue);
    
    // Update performance tracking
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time).count();
    
    updateQueueStats(queue_index, execution_time);
    
    // Cleanup
    vkDestroyCommandPool(_device, commandPool, nullptr);
}

void CarlComputeEngine::recordCommandBuffer(VkCommandBuffer cmd_buffer, const AIOperation& operation) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(cmd_buffer, &beginInfo);
    
    // TODO: Bind compute pipeline and descriptor sets based on operation type
    // This would require shader compilation and pipeline creation
    // For now, just record a memory barrier as placeholder
    
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(cmd_buffer, 
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
    
    vkEndCommandBuffer(cmd_buffer);
}

void CarlComputeEngine::updateQueueStats(uint32_t queue_index, uint64_t execution_time_ns) {
    if (queue_index >= _queue_operation_counts.size()) return;
    
    _queue_operation_counts[queue_index]++;
    _queue_last_submission[queue_index] = std::chrono::steady_clock::now();
    
    // Update utilization (simple moving average)
    float execution_ms = execution_time_ns / 1000000.0f;
    _queue_utilization[queue_index] = (_queue_utilization[queue_index] * 0.9f) + (execution_ms * 0.1f);
}

std::vector<CarlComputeEngine::QueuePerformance> CarlComputeEngine::getQueuePerformanceStats() {
    std::vector<QueuePerformance> stats;
    const auto& queues = _nova_core->getQueues();
    
    for (uint32_t i = 0; i < queues.total_compute_queues(); i++) {
        QueuePerformance perf;
        perf.queue_index = i;
        perf.operations_completed = _queue_operation_counts[i];
        perf.average_execution_time_ms = _queue_utilization[i];
        perf.utilization_percent = std::min(100.0f, _queue_utilization[i] / 10.0f * 100.0f);
        
        stats.push_back(perf);
    }
    
    return stats;
}

void CarlComputeEngine::printPerformanceReport() {
    auto stats = getQueuePerformanceStats();
    
    std::cout << "\n=== CARL Compute Engine Performance Report ===" << std::endl;
    std::cout << "Queue | Operations | Avg Time (ms) | Utilization %" << std::endl;
    std::cout << "------|------------|---------------|---------------" << std::endl;
    
    for (const auto& stat : stats) {
        std::cout << "  " << stat.queue_index 
                  << "   |     " << stat.operations_completed
                  << "     |      " << std::fixed << std::setprecision(2) << stat.average_execution_time_ms
                  << "     |     " << std::setprecision(1) << stat.utilization_percent << "%" << std::endl;
    }
    
    const auto& queues = _nova_core->getQueues();
    std::cout << "\nTotal Compute Queues: " << queues.total_compute_queues() << std::endl;
    std::cout << "Sparse Binding Support: " << (queues.has_sparse_binding() ? "Yes" : "No") << std::endl;
}

uint32_t CarlComputeEngine::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(_physical_device, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    return 0; // Fallback
}

VkDescriptorSetLayout CarlComputeEngine::createDescriptorSetLayout(const std::vector<VkDescriptorType>& types) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    
    for (size_t i = 0; i < types.size(); i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = static_cast<uint32_t>(i);
        binding.descriptorCount = 1;
        binding.descriptorType = types[i];
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
        std::cerr << "CARL Compute Engine: Failed to create descriptor set layout!" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    _descriptor_layouts.push_back(layout);
    return layout;
}

} // namespace AI
} // namespace CARL