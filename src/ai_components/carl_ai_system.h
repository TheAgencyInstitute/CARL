#pragma once

#include "carl_compute_engine.h"
#include "snn_integration.h"
#include "neural_network_models.h"
#include "../nova_integration/queue_manager.h"
#include "../core/CNN/CNN.h"
#include "../core/GAN/GAN.h"
#include "../core/RL/RL.h"
#include "../core/SNN/SNN.h"
#include <unordered_map>
#include <memory>
#include <future>
#include <mutex>
#include <vector>
#include <string>

/**
 * CARL AI System - Unified Integration Layer
 * 
 * Central orchestration system for CNN+RL+GAN+SNN integration:
 * - Unified API for all AI components
 * - Cross-component training workflows
 * - Multi-queue resource management across all 8 queues
 * - System-wide performance monitoring
 * - End-to-end application frameworks
 * 
 * Target Performance:
 * - 8x queue utilization vs Nova's 1 queue (800% performance gain)
 * - Real-time cross-component communication
 * - Support for massive models (>16GB) via sparse binding
 * - Sub-10ms inference with parallel execution
 */

namespace CARL {
namespace AI {

class CarlAISystem {
public:
    CarlAISystem();
    ~CarlAISystem();
    
    // System Initialization and Configuration
    bool initialize();
    void shutdown();
    
    // Component Management
    bool registerCNNModel(const std::string& name, std::shared_ptr<Models::ConvolutionalNeuralNetwork> model);
    bool registerGANModel(const std::string& name, std::shared_ptr<Models::GenerativeAdversarialNetwork> model);
    bool registerRLAgent(const std::string& name, std::shared_ptr<RL> agent);
    bool registerSNNNetwork(const std::string& name, std::shared_ptr<SpikingNeuralNetwork> network);
    
    // Model Retrieval
    std::shared_ptr<Models::ConvolutionalNeuralNetwork> getCNNModel(const std::string& name);
    std::shared_ptr<Models::GenerativeAdversarialNetwork> getGANModel(const std::string& name);
    std::shared_ptr<RL> getRLAgent(const std::string& name);
    std::shared_ptr<SpikingNeuralNetwork> getSNNNetwork(const std::string& name);
    
    // Unified Training Workflows
    
    // CNN+RL Training: Feature extraction with reinforcement learning
    struct CNNRLTrainingConfig {
        std::string cnn_name;
        std::string rl_name;
        uint32_t training_episodes = 1000;
        float learning_rate = 0.001f;
        uint32_t batch_size = 32;
        bool use_experience_replay = true;
        uint32_t replay_buffer_size = 10000;
        float exploration_rate = 0.1f;
        float discount_factor = 0.99f;
    };
    
    std::future<TrainingResult> trainCNNRL(const CNNRLTrainingConfig& config);
    
    // GAN+SNN Training: Memory-enhanced generative learning  
    struct GANSNNTrainingConfig {
        std::string gan_name;
        std::string snn_name;
        uint32_t training_iterations = 50000;
        float generator_lr = 0.0002f;
        float discriminator_lr = 0.0002f;
        uint32_t batch_size = 64;
        bool use_memory_augmentation = true;
        float memory_influence = 0.2f;
        uint32_t snn_timesteps = 100;
    };
    
    std::future<TrainingResult> trainGANSNN(const GANSNNTrainingConfig& config);
    
    // CNN+GAN Training: Feature-guided generation
    struct CNNGANTrainingConfig {
        std::string cnn_name;
        std::string gan_name;
        uint32_t training_iterations = 30000;
        float feature_loss_weight = 0.1f;
        float adversarial_loss_weight = 1.0f;
        uint32_t batch_size = 32;
        bool use_perceptual_loss = true;
        std::vector<uint32_t> feature_layers = {2, 5, 8}; // CNN layers for feature matching
    };
    
    std::future<TrainingResult> trainCNNGAN(const CNNGANTrainingConfig& config);
    
    // All-Component Integration: Full CARL training protocol
    struct CARLIntegratedTrainingConfig {
        std::string cnn_name;
        std::string gan_name; 
        std::string rl_name;
        std::string snn_name;
        uint32_t total_epochs = 100;
        float component_sync_frequency = 10; // Sync every N epochs
        bool enable_cross_component_learning = true;
        bool use_hierarchical_learning = true;
        float hierarchy_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f}; // CNN, GAN, RL, SNN
    };
    
    std::future<TrainingResult> trainCARLIntegrated(const CARLIntegratedTrainingConfig& config);
    
    // Cross-Component Communication Protocols
    
    // CNN -> RL: Feature extraction for decision making
    std::future<void> extractCNNFeaturesForRL(const std::string& cnn_name,
                                              const std::string& rl_name,
                                              ComputeBuffer* image_input,
                                              ComputeBuffer* rl_state_output);
    
    // RL -> GAN: Action-guided generation
    std::future<void> generateFromRLAction(const std::string& rl_name,
                                           const std::string& gan_name,
                                           ComputeBuffer* action_input,
                                           ComputeBuffer* generated_output);
    
    // GAN -> CNN: Synthetic data augmentation
    std::future<void> augmentDataWithGAN(const std::string& gan_name,
                                         const std::string& cnn_name,
                                         uint32_t augmentation_count,
                                         ComputeBuffer* training_data_output);
    
    // SNN -> All: Memory-enhanced operations
    std::future<void> enhanceWithMemory(const std::string& snn_name,
                                        const std::vector<std::string>& target_models,
                                        ComputeBuffer* memory_enhancement_data);
    
    // Performance Monitoring and Resource Management
    
    struct SystemPerformanceMetrics {
        // Queue utilization across all 8 queues
        float queue_utilization[8];
        uint64_t operations_per_queue[8];
        
        // Component-specific metrics
        float cnn_inference_time_ms;
        float gan_generation_time_ms;
        float rl_decision_time_ms;
        float snn_recall_time_ms;
        
        // Cross-component communication metrics
        uint64_t cross_component_operations;
        float average_protocol_latency_ms;
        
        // Memory usage
        size_t vram_used_bytes;
        size_t sparse_memory_committed_bytes;
        float memory_utilization_percent;
        
        // System-wide throughput
        float operations_per_second;
        float effective_speedup_factor; // Compared to single-queue Nova
    };
    
    SystemPerformanceMetrics getSystemMetrics();
    void resetMetrics();
    void printSystemReport();
    
    // Advanced Resource Management
    
    // Queue Load Balancing
    void enableAutomaticLoadBalancing(bool enable = true);
    void setQueueAffinities(const std::unordered_map<std::string, uint32_t>& model_queue_map);
    uint32_t getOptimalQueue(const std::string& model_name, AIOperationType operation);
    
    // Memory Management
    bool enableSparseMemory(const std::string& model_name, uint32_t virtual_size_gb);
    void optimizeMemoryUsage();
    void defragmentMemory();
    
    // Error Handling and Recovery
    struct SystemHealthStatus {
        bool all_queues_operational;
        bool memory_healthy;
        bool components_synchronized;
        std::vector<std::string> error_messages;
        std::vector<std::string> warning_messages;
        float overall_health_score; // 0.0 to 1.0
    };
    
    SystemHealthStatus getSystemHealth();
    bool attemptRecovery();
    void enableWatchdog(bool enable = true);
    
    // End-to-End Application Frameworks
    
    // Computer Vision + Reinforcement Learning
    class CVRLFramework {
    public:
        CVRLFramework(CarlAISystem* system);
        
        bool setupForTask(const std::string& task_name,
                         uint32_t input_width, uint32_t input_height,
                         uint32_t action_space_size);
        
        std::future<void> trainOnEnvironment(const std::string& environment_name,
                                           uint32_t training_episodes);
        
        std::future<uint32_t> inference(ComputeBuffer* image_input);
        
    private:
        CarlAISystem* _system;
        std::string _cnn_name, _rl_name;
    };
    
    // Generative AI with Memory
    class GenerativeMemoryFramework {
    public:
        GenerativeMemoryFramework(CarlAISystem* system);
        
        bool setupForDomain(const std::string& domain_name,
                           uint32_t data_dimensions);
        
        std::future<void> trainOnDataset(ComputeBuffer* training_data,
                                       uint32_t data_count);
        
        std::future<void> generateWithMemory(ComputeBuffer* memory_context,
                                           ComputeBuffer* generated_output);
        
    private:
        CarlAISystem* _system;
        std::string _gan_name, _snn_name;
    };
    
    // Create application frameworks
    std::unique_ptr<CVRLFramework> createCVRLFramework();
    std::unique_ptr<GenerativeMemoryFramework> createGenerativeMemoryFramework();
    
    // Configuration and Persistence
    
    struct SystemConfiguration {
        bool enable_auto_load_balancing = true;
        bool enable_cross_component_learning = true;
        bool enable_sparse_memory = true;
        bool enable_performance_monitoring = true;
        bool enable_automatic_recovery = true;
        
        float default_learning_rate = 0.001f;
        uint32_t default_batch_size = 32;
        uint32_t max_concurrent_operations = 16;
        
        std::string checkpoint_directory = "./checkpoints/";
        uint32_t auto_save_interval_minutes = 10;
    };
    
    void setConfiguration(const SystemConfiguration& config);
    SystemConfiguration getConfiguration() const;
    
    bool saveSystemState(const std::string& filepath);
    bool loadSystemState(const std::string& filepath);
    
    // Debug and Development Tools
    
    void enableDebugMode(bool enable = true);
    void setVerboseLogging(bool enable = true);
    
    // Export system state for analysis
    bool exportSystemVisualization(const std::string& output_path);
    bool exportPerformanceLogs(const std::string& output_path);
    
    // Benchmarking suite
    struct BenchmarkResults {
        float single_component_fps[4]; // CNN, GAN, RL, SNN
        float cross_component_fps[6];  // All pairwise combinations
        float full_system_fps;
        float queue_efficiency_score;
        float memory_efficiency_score;
        float overall_performance_score;
    };
    
    std::future<BenchmarkResults> runSystemBenchmark();

private:
    // Core engine and managers
    std::unique_ptr<CarlComputeEngine> _compute_engine;
    std::unique_ptr<GPU::QueueManager> _queue_manager;
    std::unique_ptr<SNNIntegration> _snn_integration;
    
    // Component registries
    std::unordered_map<std::string, std::shared_ptr<Models::ConvolutionalNeuralNetwork>> _cnn_models;
    std::unordered_map<std::string, std::shared_ptr<Models::GenerativeAdversarialNetwork>> _gan_models;
    std::unordered_map<std::string, std::shared_ptr<RL>> _rl_agents;
    std::unordered_map<std::string, std::shared_ptr<SpikingNeuralNetwork>> _snn_networks;
    
    // Cross-component protocols
    std::unique_ptr<Protocol::CNNSNNProtocol> _cnn_snn_protocol;
    std::unique_ptr<Protocol::RLSNNProtocol> _rl_snn_protocol;
    std::unique_ptr<Protocol::GANSNNProtocol> _gan_snn_protocol;
    
    // Performance monitoring
    SystemPerformanceMetrics _current_metrics;
    std::mutex _metrics_mutex;
    std::chrono::steady_clock::time_point _metrics_start_time;
    
    // Resource management
    SystemConfiguration _config;
    std::vector<float> _queue_load_balancing_weights;
    bool _auto_load_balancing_enabled = true;
    
    // Health monitoring
    std::unique_ptr<SystemHealthMonitor> _health_monitor;
    bool _watchdog_enabled = false;
    
    // Internal helper methods
    bool _initializeAllComponents();
    bool _setupCrossComponentProtocols(); 
    void _updateSystemMetrics();
    uint32_t _calculateOptimalQueue(const std::string& component_name, 
                                   AIOperationType operation_type,
                                   size_t workload_size);
    
    // Training orchestration helpers
    std::future<void> _synchronizeComponents(const std::vector<std::string>& component_names);
    void _balanceTrainingLoads(const std::vector<std::string>& component_names);
    
    // Error handling helpers
    bool _validateSystemState();
    void _logSystemEvent(const std::string& event, const std::string& details);
};

// System Health Monitoring (Internal)
class SystemHealthMonitor {
public:
    SystemHealthMonitor(CarlAISystem* system);
    ~SystemHealthMonitor();
    
    void startMonitoring();
    void stopMonitoring();
    
    CarlAISystem::SystemHealthStatus getCurrentHealth();
    
private:
    CarlAISystem* _system;
    std::atomic<bool> _monitoring_active;
    std::thread _monitoring_thread;
    
    void _monitoringLoop();
    void _checkQueueHealth();
    void _checkMemoryHealth();
    void _checkComponentSynchronization();
};

// Training Results Structure
struct TrainingResult {
    bool success;
    uint32_t epochs_completed;
    float final_loss;
    float training_time_seconds;
    std::vector<float> loss_history;
    std::unordered_map<std::string, float> component_metrics;
    std::string error_message;
};

} // namespace AI
} // namespace CARL

/**
 * CARL AI System Usage Example:
 * 
 * auto carl_system = std::make_unique<CARL::AI::CarlAISystem>();
 * carl_system->initialize();
 * 
 * // Register components
 * auto cnn = std::make_shared<CARL::AI::Models::ConvolutionalNeuralNetwork>(engine, 224, 224, 3);
 * auto gan = std::make_shared<CARL::AI::Models::GenerativeAdversarialNetwork>(engine);  
 * auto rl = std::make_shared<RL>();
 * auto snn = std::make_shared<SpikingNeuralNetwork>(engine, 1000, 100);
 * 
 * carl_system->registerCNNModel("vision_model", cnn);
 * carl_system->registerGANModel("generator", gan);
 * carl_system->registerRLAgent("decision_agent", rl);
 * carl_system->registerSNNNetwork("memory_network", snn);
 * 
 * // Cross-component training
 * CARL::AI::CarlAISystem::CARLIntegratedTrainingConfig config;
 * config.cnn_name = "vision_model";
 * config.gan_name = "generator"; 
 * config.rl_name = "decision_agent";
 * config.snn_name = "memory_network";
 * 
 * auto training_result = carl_system->trainCARLIntegrated(config);
 * 
 * // Real-time inference with 8x queue utilization
 * auto cvrl_framework = carl_system->createCVRLFramework();
 * auto action = cvrl_framework->inference(image_buffer);
 * 
 * Expected Performance:
 * - 800% performance increase vs Nova (8 queues vs 1)
 * - Real-time cross-component communication
 * - Sub-10ms inference latency
 * - Support for >16GB models via sparse binding
 */