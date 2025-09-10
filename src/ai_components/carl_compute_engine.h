#pragma once

#include "../nova/Core/core.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <future>

/**
 * CARL AI Compute Engine
 * 
 * Provides high-level AI operations using Nova-CARL enhanced multi-queue system:
 * - Matrix operations accelerated across 4 compute queues
 * - Neural network layers with parallel execution
 * - Memory management with sparse binding support
 * - Integration with graphics queue for hybrid operations
 */

namespace CARL {
namespace AI {

enum class AIOperationType {
    MATRIX_MULTIPLY,
    CONVOLUTION_2D,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX,
    POOLING_MAX,
    POOLING_AVERAGE,
    BATCH_NORMALIZATION,
    GRADIENT_DESCENT,
    SNN_SPIKE_UPDATE,
    SPARSE_ATTENTION
};

struct ComputeBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorSet descriptor_set;
    size_t size_bytes;
    uint32_t element_count;
    void* mapped_data;
};

struct AIOperation {
    AIOperationType type;
    std::string shader_path;
    std::vector<ComputeBuffer*> input_buffers;
    std::vector<ComputeBuffer*> output_buffers;
    uint32_t dispatch_x, dispatch_y, dispatch_z;
    uint32_t assigned_queue;
    std::string debug_name;
};

class CarlComputeEngine {
public:
    CarlComputeEngine(NovaCore* nova_core);
    ~CarlComputeEngine();
    
    bool initialize();
    void shutdown();
    
    // Buffer Management
    ComputeBuffer* createBuffer(size_t size_bytes, VkBufferUsageFlags usage);
    void destroyBuffer(ComputeBuffer* buffer);
    void uploadData(ComputeBuffer* buffer, const void* data, size_t size);
    void downloadData(ComputeBuffer* buffer, void* data, size_t size);
    
    // AI Operations - Matrix Operations
    std::future<void> matrixMultiply(ComputeBuffer* matrix_a, 
                                   ComputeBuffer* matrix_b, 
                                   ComputeBuffer* result,
                                   uint32_t rows_a, uint32_t cols_a, uint32_t cols_b);
    
    // AI Operations - Neural Network Layers
    std::future<void> convolution2D(ComputeBuffer* input, 
                                  ComputeBuffer* kernel, 
                                  ComputeBuffer* output,
                                  uint32_t input_width, uint32_t input_height,
                                  uint32_t kernel_size, uint32_t stride);
    
    std::future<void> activationReLU(ComputeBuffer* input, 
                                   ComputeBuffer* output, 
                                   uint32_t element_count);
    
    std::future<void> maxPooling2D(ComputeBuffer* input, 
                                 ComputeBuffer* output,
                                 uint32_t input_width, uint32_t input_height,
                                 uint32_t pool_size, uint32_t stride);
    
    // AI Operations - Advanced
    std::future<void> batchNormalization(ComputeBuffer* input, 
                                        ComputeBuffer* output,
                                        ComputeBuffer* mean, 
                                        ComputeBuffer* variance,
                                        uint32_t batch_size, uint32_t features);
    
    // Parallel AI Pipeline
    struct NeuralNetworkLayer {
        AIOperationType operation;
        std::vector<ComputeBuffer*> inputs;
        std::vector<ComputeBuffer*> outputs;
        std::vector<ComputeBuffer*> weights;
        uint32_t width, height, channels;
    };
    
    std::future<void> executeNeuralNetworkParallel(const std::vector<NeuralNetworkLayer>& layers);
    
    // Queue Distribution Strategy
    uint32_t selectOptimalQueue(AIOperationType operation, size_t workload_size);
    void balanceQueueLoad();
    
    // Performance Monitoring
    struct QueuePerformance {
        uint32_t queue_index;
        uint64_t operations_completed;
        uint64_t total_execution_time_ns;
        float average_execution_time_ms;
        float utilization_percent;
    };
    
    std::vector<QueuePerformance> getQueuePerformanceStats();
    void printPerformanceReport();
    
    // Sparse Memory Management (Family 4 queue)
    struct SparseBuffer {
        VkBuffer sparse_buffer;
        VkDeviceMemory sparse_memory;
        size_t virtual_size_bytes;
        size_t committed_size_bytes;
        std::vector<VkSparseMemoryBind> memory_binds;
    };
    
    SparseBuffer* createSparseBuffer(size_t virtual_size_bytes);
    void commitSparseMemory(SparseBuffer* buffer, size_t offset, size_t size);
    void releaseSparseMemory(SparseBuffer* buffer, size_t offset, size_t size);
    
    // Graphics-Compute Hybrid Operations (Family 0 queue)
    std::future<void> neuralVisualization(ComputeBuffer* weights, 
                                         VkImage output_texture,
                                         uint32_t width, uint32_t height);
    
    std::future<void> renderToTextureTraining(VkImage input_texture,
                                             ComputeBuffer* feature_output,
                                             uint32_t width, uint32_t height);
    
private:
    NovaCore* _nova_core;
    VkDevice _device;
    VkPhysicalDevice _physical_device;
    
    // Compute pipeline management
    std::vector<VkPipeline> _compute_pipelines;
    std::vector<VkPipelineLayout> _pipeline_layouts;
    std::vector<VkDescriptorSetLayout> _descriptor_layouts;
    VkDescriptorPool _descriptor_pool;
    
    // Queue utilization tracking
    std::vector<float> _queue_utilization; // Per queue load factor
    std::vector<uint64_t> _queue_operation_counts;
    std::vector<std::chrono::steady_clock::time_point> _queue_last_submission;
    
    // Buffer pool management
    std::vector<std::unique_ptr<ComputeBuffer>> _buffer_pool;
    
    // Shader compilation and pipeline creation
    VkShaderModule loadComputeShader(const std::string& shader_path);
    VkPipeline createComputePipeline(VkShaderModule shader, VkPipelineLayout layout);
    VkDescriptorSetLayout createDescriptorSetLayout(const std::vector<VkDescriptorType>& types);
    
    // Operation execution
    void executeOperation(const AIOperation& operation, uint32_t queue_index);
    void recordCommandBuffer(VkCommandBuffer cmd_buffer, const AIOperation& operation);
    
    // Performance tracking
    void updateQueueStats(uint32_t queue_index, uint64_t execution_time_ns);
    
    // Memory management helpers
    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory);
};

// High-level AI Model Templates
namespace Models {
    
    class ConvolutionalNeuralNetwork {
    public:
        ConvolutionalNeuralNetwork(CarlComputeEngine* engine, 
                                  uint32_t input_width, uint32_t input_height, uint32_t channels);
        
        void addConvolutionalLayer(uint32_t filters, uint32_t kernel_size, uint32_t stride = 1);
        void addPoolingLayer(uint32_t pool_size, uint32_t stride = 2);
        void addFullyConnectedLayer(uint32_t units);
        
        std::future<void> forward(ComputeBuffer* input, ComputeBuffer* output);
        std::future<void> backward(ComputeBuffer* gradients);
        
    private:
        CarlComputeEngine* _engine;
        std::vector<NeuralNetworkLayer> _layers;
        uint32_t _input_width, _input_height, _channels;
    };
    
    class GenerativeAdversarialNetwork {
    public:
        GenerativeAdversarialNetwork(CarlComputeEngine* engine);
        
        void setGenerator(const std::vector<NeuralNetworkLayer>& layers);
        void setDiscriminator(const std::vector<NeuralNetworkLayer>& layers);
        
        std::future<void> trainStep(ComputeBuffer* real_data, ComputeBuffer* noise);
        std::future<void> generate(ComputeBuffer* noise, ComputeBuffer* output);
        
    private:
        CarlComputeEngine* _engine;
        std::vector<NeuralNetworkLayer> _generator_layers;
        std::vector<NeuralNetworkLayer> _discriminator_layers;
    };
    
    class SpikingNeuralNetwork {
    public:
        SpikingNeuralNetwork(CarlComputeEngine* engine, uint32_t neurons, uint32_t timesteps);
        
        std::future<void> simulateTimestep(ComputeBuffer* input_spikes, 
                                          ComputeBuffer* membrane_potentials,
                                          ComputeBuffer* output_spikes);
        
        void setMemoryManager(SparseBuffer* sparse_memory);
        
    private:
        CarlComputeEngine* _engine;
        uint32_t _neurons, _timesteps;
        SparseBuffer* _sparse_memory;
    };
}

} // namespace AI
} // namespace CARL

/**
 * CARL AI Compute Engine leverages Nova-CARL multi-queue architecture:
 * 
 * Queue Utilization:
 * - Compute Queue 0: Matrix multiplication operations
 * - Compute Queue 1: Convolution forward passes  
 * - Compute Queue 2: Convolution backward passes
 * - Compute Queue 3: Activation and pooling operations
 * - Graphics Queue: Neural network visualization, hybrid operations
 * - Sparse Queue: Large model memory management
 * - Video Queues: Computer vision preprocessing and output
 * 
 * Expected Performance:
 * - 4x parallelization of neural network training
 * - Real-time inference with visualization
 * - Support for models >16GB with sparse binding
 * - Integrated computer vision pipeline with video queues
 */