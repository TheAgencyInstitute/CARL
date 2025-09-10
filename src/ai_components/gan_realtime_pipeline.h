#pragma once

#include "neural_network_models.h"
#include "carl_compute_engine.h"
#include "compute_pipeline_manager.h"
#include "../nova/Core/core.h"
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>

/**
 * Real-time GAN Image Generation Pipeline for CARL AI System
 * 
 * Features:
 * - Multi-threaded generation pipeline
 * - Nova-CARL multi-queue utilization
 * - Real-time parameter interpolation
 * - Live performance monitoring
 * - Streaming output to graphics pipeline
 */

namespace CARL {
namespace AI {

class GANRealtimePipeline {
public:
    struct GenerationRequest {
        uint32_t request_id;
        std::vector<float> noise_vector;
        uint32_t output_width;
        uint32_t output_height;
        uint32_t channels;
        float style_interpolation; // 0.0 to 1.0 for style mixing
        std::promise<ComputeBuffer*> result_promise;
    };
    
    struct PipelineConfig {
        uint32_t max_concurrent_generations = 4;
        uint32_t noise_dimension = 128;
        uint32_t default_resolution = 256;
        uint32_t channels = 3;
        bool enable_style_mixing = true;
        bool enable_real_time_metrics = true;
        float target_fps = 30.0f;
    };
    
    struct RealtimeMetrics {
        float current_fps;
        float average_generation_time_ms;
        uint32_t completed_generations;
        uint32_t pending_requests;
        uint32_t failed_generations;
        float gpu_utilization_percent;
        float memory_usage_mb;
        
        // Queue-specific metrics
        struct QueueMetrics {
            uint32_t queue_id;
            float utilization_percent;
            uint32_t operations_per_second;
            float average_latency_ms;
        };
        
        std::vector<QueueMetrics> queue_metrics;
    };

public:
    GANRealtimePipeline(CarlComputeEngine* engine, 
                       ComputePipelineManager* pipeline_manager,
                       NovaCore* nova_core);
    ~GANRealtimePipeline();
    
    // Pipeline Management
    bool initialize(const PipelineConfig& config);
    void shutdown();
    bool isRunning() const { return _running.load(); }
    
    // GAN Integration
    void setGAN(std::unique_ptr<GenerativeAdversarialNetwork> gan);
    void updateGANWeights(); // Hot-swap trained weights
    
    // Generation Interface
    std::future<ComputeBuffer*> generateAsync(const std::vector<float>& noise);
    std::future<ComputeBuffer*> generateWithStyleMixing(const std::vector<float>& noise1,
                                                        const std::vector<float>& noise2,
                                                        float mix_factor);
    
    // Interactive Generation
    void startInteractiveMode(); // Continuous generation for real-time demos
    void stopInteractiveMode();
    void setInteractiveNoise(const std::vector<float>& noise);
    void interpolateToNoise(const std::vector<float>& target_noise, float duration_seconds);
    
    // Multi-Queue Configuration
    void setQueueStrategy(uint32_t primary_queue, uint32_t secondary_queue, uint32_t graphics_queue);
    void enableAdaptiveLoadBalancing(bool enable);
    
    // Performance Monitoring
    RealtimeMetrics getCurrentMetrics() const;
    void resetMetrics();
    void printPerformanceReport();
    
    // Nova Integration
    void setRenderTarget(VkImage target_image); // For direct rendering
    void enableGraphicsOutput(bool enable); // Stream to Nova graphics pipeline
    
    // Advanced Features
    void enableProgressiveGeneration(bool enable); // Multi-resolution pipeline
    void setLatencyTarget(float target_ms); // Adaptive quality for latency
    void enableMemoryOptimization(bool enable); // Dynamic buffer management

private:
    // Core components
    CarlComputeEngine* _engine;
    ComputePipelineManager* _pipeline_manager;
    NovaCore* _nova_core;
    std::unique_ptr<GenerativeAdversarialNetwork> _gan;
    
    // Pipeline configuration
    PipelineConfig _config;
    std::atomic<bool> _running;
    std::atomic<bool> _interactive_mode;
    
    // Queue management
    uint32_t _primary_queue;
    uint32_t _secondary_queue;
    uint32_t _graphics_queue;
    bool _adaptive_load_balancing;
    
    // Worker threads
    std::vector<std::thread> _worker_threads;
    std::queue<GenerationRequest> _request_queue;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    
    // Interactive generation
    std::thread _interactive_thread;
    std::vector<float> _current_noise;
    std::vector<float> _target_noise;
    std::atomic<bool> _interpolating;
    float _interpolation_progress;
    
    // Buffer pools
    struct BufferPool {
        std::queue<ComputeBuffer*> available_buffers;
        std::mutex pool_mutex;
        size_t buffer_size;
        uint32_t max_buffers;
        uint32_t allocated_buffers;
    };
    
    BufferPool _noise_buffer_pool;
    BufferPool _output_buffer_pool;
    
    // Performance tracking
    mutable std::mutex _metrics_mutex;
    RealtimeMetrics _metrics;
    std::chrono::steady_clock::time_point _last_fps_update;
    uint32_t _frames_since_last_update;
    
    // Nova integration
    VkImage _render_target;
    bool _graphics_output_enabled;
    
    // Advanced features
    bool _progressive_generation_enabled;
    float _latency_target_ms;
    bool _memory_optimization_enabled;
    
    // Internal methods
    void workerThreadMain();
    void interactiveThreadMain();
    void processGenerationRequest(const GenerationRequest& request);
    
    // Buffer management
    bool initializeBufferPools();
    void shutdownBufferPools();
    ComputeBuffer* acquireNoiseBuffer();
    ComputeBuffer* acquireOutputBuffer();
    void releaseBuffer(ComputeBuffer* buffer, BufferPool& pool);
    
    // Queue load balancing
    uint32_t selectOptimalQueue() const;
    void updateQueueMetrics();
    
    // Performance optimization
    void adaptiveQualityControl();
    void memoryOptimization();
    
    // Interactive generation helpers
    void updateInteractiveNoise();
    std::vector<float> interpolateNoise(const std::vector<float>& from, 
                                       const std::vector<float>& to, 
                                       float factor);
    
    // Nova integration helpers
    void copyToRenderTarget(ComputeBuffer* generated_image);
    void streamToGraphicsPipeline(ComputeBuffer* generated_image);
    
    // Metrics collection
    void updatePerformanceMetrics();
    void recordGenerationTime(float time_ms);
    void recordQueueOperation(uint32_t queue_id, float time_ms);
};

// Utility functions for real-time generation
namespace RealtimeUtils {
    
    // Generate smooth noise transitions for animation
    std::vector<std::vector<float>> generateNoiseSequence(uint32_t noise_dim, 
                                                          uint32_t sequence_length,
                                                          float smoothness = 0.8f);
    
    // Create noise from user input (mouse position, audio, etc.)
    std::vector<float> inputToNoise(float x, float y, float z, uint32_t noise_dim);
    
    // Style mixing utilities
    std::vector<float> mixNoise(const std::vector<float>& noise1,
                               const std::vector<float>& noise2,
                               float mix_factor);
    
    // Performance profiling
    void benchmarkGenerationPipeline(GANRealtimePipeline* pipeline, 
                                    uint32_t num_generations = 100);
}

} // namespace AI
} // namespace CARL