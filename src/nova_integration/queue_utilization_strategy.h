#pragma once

#include "nova_context.h"
#include "../nova/Core/core.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

/**
 * CARL AI Queue Utilization Strategy
 * 
 * Leverages ALL 6 AMD RX 6800 XT queue types for maximum AI throughput:
 * 
 * QUEUE SPECIALIZATION ANALYSIS:
 * ==============================
 * 
 * Family 0: Graphics+Compute+Transfer+Sparse (1 queue)
 * - CARL Usage: Hybrid operations requiring both graphics and compute
 * - AI Applications: Neural network visualization, render-to-texture training data
 * - Memory: Can handle large buffer operations with graphics pipeline integration
 * 
 * Family 1: Dedicated Compute+Transfer+Sparse (4 queues) ⭐ PRIMARY AI WORKLOAD
 * - CARL Usage: Pure neural network computations, matrix operations
 * - AI Applications: CNN forward/backward passes, RL policy updates, GAN training
 * - Memory: Optimal for large tensor operations, can use sparse binding for huge models
 * 
 * Family 2: Video Decode (1 queue)
 * - CARL Usage: Video preprocessing for computer vision models
 * - AI Applications: Frame extraction, video augmentation, temporal feature extraction
 * - Memory: Hardware-accelerated video memory management
 * 
 * Family 3: Video Encode (1 queue) 
 * - CARL Usage: Generated video output, model visualization recording
 * - AI Applications: GAN video generation, training progress recordings
 * - Memory: Optimized for compressed video output
 * 
 * Family 4: Dedicated Sparse Binding (1 queue)
 * - CARL Usage: Ultra-large model memory management
 * - AI Applications: Sparse attention mechanisms, massive parameter matrices
 * - Memory: Virtual memory management for models > 16GB VRAM
 * 
 * TOTAL PARALLEL POTENTIAL: 8 concurrent operations vs Nova's 1
 */

namespace CARL {
namespace GPU {

enum class AIWorkloadType {
    // Pure compute workloads (Family 1: 4 dedicated compute queues)
    MATRIX_MULTIPLICATION,
    CONVOLUTION_FORWARD,
    CONVOLUTION_BACKWARD,
    ACTIVATION_FUNCTIONS,
    LOSS_COMPUTATION,
    GRADIENT_DESCENT,
    SNN_SPIKE_PROCESSING,
    
    // Hybrid graphics-compute (Family 0: graphics+compute queue)
    NEURAL_VISUALIZATION,
    TRAINING_DATA_RENDERING,
    COMPUTE_TO_TEXTURE,
    GRAPHICS_FEEDBACK_LOOP,
    
    // Video processing (Family 2: video decode)
    VIDEO_FRAME_EXTRACTION,
    VIDEO_PREPROCESSING,
    TEMPORAL_FEATURE_EXTRACTION,
    OPTICAL_FLOW_ANALYSIS,
    
    // Video generation (Family 3: video encode)
    GAN_VIDEO_GENERATION,
    TRAINING_VISUALIZATION,
    MODEL_OUTPUT_RECORDING,
    
    // Sparse memory operations (Family 4: sparse binding)
    SPARSE_ATTENTION_SETUP,
    LARGE_MODEL_PAGING,
    DYNAMIC_MEMORY_MAPPING,
    VIRTUAL_TENSOR_MANAGEMENT
};

struct QueueUtilizationPlan {
    AIWorkloadType workload;
    uint32_t queue_family;
    uint32_t queue_index;
    std::string rationale;
    float expected_performance_multiplier;
    size_t memory_requirements_mb;
};

class AIQueueScheduler {
public:
    AIQueueScheduler(NovaCore* nova_core);
    
    // Queue Assignment Strategy
    QueueUtilizationPlan selectOptimalQueue(AIWorkloadType workload, 
                                           size_t memory_size,
                                           bool requires_sparse = false);
    
    // Workload Distribution
    std::vector<QueueUtilizationPlan> distributeWorkload(
        const std::vector<AIWorkloadType>& workloads);
    
    // Specialized Scheduling
    std::vector<QueueUtilizationPlan> scheduleNeuralNetworkTraining(
        size_t batch_size, 
        size_t model_parameters,
        bool has_video_input = false);
    
    std::vector<QueueUtilizationPlan> scheduleGANTraining(
        bool generate_video = false,
        size_t discriminator_params = 0,
        size_t generator_params = 0);
    
    std::vector<QueueUtilizationPlan> scheduleReinforcementLearning(
        bool has_visual_environment = false,
        size_t policy_network_size = 0);
    
    // Sparse Binding Analysis
    bool shouldUseSparseBinding(size_t model_size_mb) const;
    std::vector<QueueUtilizationPlan> createSparseBindingStrategy(
        size_t total_model_size_mb,
        size_t available_vram_mb);
    
    // Performance Analysis
    struct PerformanceProjection {
        float nova_baseline_time;
        float carl_optimized_time;
        float speedup_factor;
        uint32_t parallel_operations;
        std::string bottleneck_analysis;
    };
    
    PerformanceProjection analyzePerformanceGains(
        const std::vector<AIWorkloadType>& workloads) const;
    
    // Real-world AI Scenarios
    std::vector<QueueUtilizationPlan> optimizeForCNN(
        size_t input_resolution,
        size_t num_layers,
        size_t batch_size);
    
    std::vector<QueueUtilizationPlan> optimizeForRNN(
        size_t sequence_length,
        size_t hidden_size,
        bool bidirectional = false);
    
    std::vector<QueueUtilizationPlan> optimizeForTransformer(
        size_t sequence_length,
        size_t attention_heads,
        size_t model_dimension,
        bool sparse_attention = false);
    
    // Queue Load Balancing
    float getQueueUtilization(uint32_t family, uint32_t index) const;
    void rebalanceQueues();
    
    // Debug and Analysis
    void printQueueUtilizationReport() const;
    void printAIWorkloadDistribution() const;
    
private:
    NovaCore* _nova_core;
    
    // Queue state tracking
    std::vector<float> _queue_utilization; // Per queue load (0.0-1.0)
    std::unordered_map<AIWorkloadType, uint32_t> _workload_assignments;
    
    // Performance modeling
    float _estimateExecutionTime(AIWorkloadType workload, uint32_t queue_family) const;
    bool _canUseGraphicsQueue(AIWorkloadType workload) const;
    bool _requiresVideoQueue(AIWorkloadType workload) const;
    bool _benefitsFromSparseBinding(AIWorkloadType workload, size_t memory_size) const;
    
    // Queue family capabilities
    struct QueueCapabilities {
        bool supports_compute;
        bool supports_graphics; 
        bool supports_video_decode;
        bool supports_video_encode;
        bool supports_sparse_binding;
        bool supports_transfer;
        uint32_t queue_count;
        float base_performance_factor;
    };
    
    QueueCapabilities _analyzeQueueCapabilities(uint32_t family) const;
};

// Pre-defined optimization strategies for common AI architectures
namespace OptimizationPresets {
    
    // Convolutional Neural Networks
    std::vector<QueueUtilizationPlan> ResNet50Training(AIQueueScheduler& scheduler);
    std::vector<QueueUtilizationPlan> EfficientNetInference(AIQueueScheduler& scheduler);
    
    // Generative Models
    std::vector<QueueUtilizationPlan> GANTraining(AIQueueScheduler& scheduler);
    std::vector<QueueUtilizationPlan> VAETraining(AIQueueScheduler& scheduler);
    std::vector<QueueUtilizationPlan> DiffusionModelInference(AIQueueScheduler& scheduler);
    
    // Reinforcement Learning
    std::vector<QueueUtilizationPlan> PPOTraining(AIQueueScheduler& scheduler);
    std::vector<QueueUtilizationPlan> DQNTraining(AIQueueScheduler& scheduler);
    
    // Large Language Models (requires sparse binding)
    std::vector<QueueUtilizationPlan> TransformerInference(AIQueueScheduler& scheduler, 
                                                          size_t parameters_billions);
    
    // Spiking Neural Networks
    std::vector<QueueUtilizationPlan> SNNSimulation(AIQueueScheduler& scheduler,
                                                   size_t neurons,
                                                   size_t timesteps);
}

} // namespace GPU
} // namespace CARL

/**
 * QUEUE UTILIZATION SCENARIOS FOR CARL AI:
 * 
 * Scenario 1: CNN Training with Video Input
 * - Family 2 (Video Decode): Extract and preprocess video frames
 * - Family 1 (Compute 0-3): Parallel CNN forward passes on 4 batches  
 * - Family 0 (Graphics+Compute): Render augmented training data
 * - Family 4 (Sparse): Manage large dataset virtual memory
 * Result: 7x parallelization vs Nova's 1 queue
 * 
 * Scenario 2: GAN Video Generation
 * - Family 1 (Compute 0-1): Generator network parallel execution
 * - Family 1 (Compute 2-3): Discriminator network parallel execution
 * - Family 3 (Video Encode): Real-time video encoding of generated frames
 * - Family 0 (Graphics+Compute): Final compositing and effects
 * Result: 6x parallelization vs Nova's 1 queue
 * 
 * Scenario 3: Large Transformer Model (>16GB)
 * - Family 4 (Sparse): Virtual memory management for 50B+ parameters
 * - Family 1 (Compute 0-3): Parallel attention head computations
 * - Family 0 (Graphics+Compute): Token embedding visualization
 * Result: 6x parallelization + sparse memory management vs Nova's 1 queue
 * 
 * Expected Overall Performance: 4-8x speedup depending on workload complexity
 */