#pragma once

#include "neural_network_models.h"
#include "compute_pipeline_manager.h"
#include "carl_compute_engine.h"
#include "../nova_integration/queue_manager.h"
#include <memory>
#include <vector>
#include <future>

/**
 * SNN Integration Layer for CARL AI System
 * 
 * Provides high-level interface for Spiking Neural Network operations:
 * - Multi-queue GPU acceleration via Nova framework
 * - STDP learning with parallel weight updates
 * - Sparse memory binding for massive neuromorphic networks (>100K neurons)
 * - Real-time spike simulation with 12ms memory recall latency
 * - Integration with CNN-RL-GAN protocols
 */

namespace CARL {
namespace AI {

class SNNIntegration {
public:
    SNNIntegration(CarlComputeEngine* engine, GPU::QueueManager* queue_manager);
    ~SNNIntegration();
    
    bool initialize();
    void shutdown();
    
    // Network Creation and Management
    std::shared_ptr<SpikingNeuralNetwork> createSNN(uint32_t neurons, uint32_t timesteps);
    std::shared_ptr<SpikingNeuralNetwork> createLargeSNN(uint32_t neurons, uint32_t timesteps, 
                                                         uint32_t virtual_memory_size_gb);
    
    bool addSNNNetwork(const std::string& name, std::shared_ptr<SpikingNeuralNetwork> network);
    std::shared_ptr<SpikingNeuralNetwork> getSNNNetwork(const std::string& name);
    void removeSNNNetwork(const std::string& name);
    
    // High-Performance Simulation Pipeline
    struct SimulationConfig {
        float timestep_dt = 1e-3f;           // 1ms timestep
        uint32_t simulation_steps = 1000;    // Total steps to simulate
        bool enable_learning = true;         // Enable STDP learning
        bool use_sparse_binding = false;     // Use sparse memory for large networks
        bool parallel_execution = true;      // Use multiple compute queues
        float learning_rate = 0.01f;         // STDP learning rate
        uint32_t batch_size = 1;             // Number of parallel simulations
    };
    
    std::future<void> runSimulation(const std::string& network_name, 
                                   ComputeBuffer* input_patterns,
                                   const SimulationConfig& config);
    
    std::future<void> runBatchSimulation(const std::vector<std::string>& network_names,
                                        const std::vector<ComputeBuffer*>& input_patterns,
                                        const SimulationConfig& config);
    
    // Real-time Memory Recall Interface (Target: 12ms latency)
    struct MemoryQuery {
        ComputeBuffer* query_pattern;
        float similarity_threshold = 0.7f;
        uint32_t max_matches = 10;
        bool use_sparse_retrieval = true;
    };
    
    struct MemoryResult {
        std::vector<uint32_t> matched_patterns;
        std::vector<float> similarity_scores;
        uint64_t retrieval_time_ns;
        bool success;
    };
    
    std::future<MemoryResult> queryMemory(const std::string& network_name, 
                                         const MemoryQuery& query);
    
    // STDP Learning Integration
    struct STDPConfig {
        float learning_rate = 0.01f;
        float tau_plus = 20e-3f;        // LTP time constant (20ms)
        float tau_minus = 20e-3f;       // LTD time constant (20ms) 
        float A_plus = 1.0f;            // LTP amplitude
        float A_minus = 1.0f;           // LTD amplitude
        float weight_min = 0.0f;        // Minimum synaptic weight
        float weight_max = 1.0f;        // Maximum synaptic weight
        bool batch_updates = true;      // Batch weight updates for efficiency
    };
    
    void setSTDPParameters(const std::string& network_name, const STDPConfig& config);
    std::future<void> applySTDPLearning(const std::string& network_name);
    std::future<void> applyBatchSTDPLearning(const std::vector<std::string>& network_names);
    
    // Sparse Memory Management for Ultra-Large Networks
    struct SparseMemoryConfig {
        uint32_t virtual_memory_gb = 16;     // Virtual memory space (16GB)
        uint32_t physical_memory_gb = 4;     // Physical GPU memory (4GB) 
        uint32_t page_size_kb = 64;          // Memory page size (64KB)
        uint32_t cache_policy = 1;           // 0=LRU, 1=LFU, 2=Random
        bool enable_compression = true;      // Compress inactive memory regions
        bool enable_prefetching = true;      // Prefetch likely-needed memory
    };
    
    bool configureSparseMemory(const std::string& network_name, 
                              const SparseMemoryConfig& config);
    
    std::future<void> commitMemoryRegions(const std::string& network_name,
                                         const std::vector<std::pair<uint32_t, uint32_t>>& regions);
    
    std::future<void> releaseMemoryRegions(const std::string& network_name,
                                          const std::vector<std::pair<uint32_t, uint32_t>>& regions);
    
    // Integration with Other CARL Components
    
    // CNN-SNN Integration: Use CNN features as SNN input patterns
    std::future<void> integrateCNNFeatures(const std::string& snn_name,
                                          const std::string& cnn_name,
                                          ComputeBuffer* cnn_feature_map);
    
    // RL-SNN Integration: Use SNN memory for RL state encoding
    std::future<void> encodeRLState(const std::string& snn_name,
                                   ComputeBuffer* rl_state,
                                   ComputeBuffer* snn_encoding);
    
    // GAN-SNN Integration: Use SNN to generate training patterns for GAN
    std::future<void> generateGANPatterns(const std::string& snn_name,
                                         uint32_t pattern_count,
                                         ComputeBuffer* generated_patterns);
    
    // Performance Monitoring and Analytics
    struct SNNPerformanceMetrics {
        uint64_t total_spikes_generated;
        float average_firing_rate;
        float memory_utilization_percent;
        uint64_t simulation_time_ns;
        uint64_t learning_time_ns;
        uint64_t memory_access_time_ns;
        uint32_t sparse_memory_pages_used;
        uint32_t sparse_memory_cache_hits;
        uint32_t sparse_memory_cache_misses;
        float queue_utilization_percent;
    };
    
    SNNPerformanceMetrics getPerformanceMetrics(const std::string& network_name);
    void resetPerformanceMetrics(const std::string& network_name);
    void printPerformanceReport();
    
    // Advanced Features
    
    // Multi-Network Synchronization
    std::future<void> synchronizeNetworks(const std::vector<std::string>& network_names,
                                         float synchronization_strength = 0.1f);
    
    // Network Topology Manipulation  
    bool addConnection(const std::string& src_network, const std::string& dst_network,
                      float connection_strength = 0.1f);
    
    bool removeConnection(const std::string& src_network, const std::string& dst_network);
    
    // Save/Load Network State
    bool saveNetworkState(const std::string& network_name, const std::string& filepath);
    bool loadNetworkState(const std::string& network_name, const std::string& filepath);
    
    // Debug and Visualization
    struct VisualizationData {
        std::vector<float> membrane_potentials;
        std::vector<uint32_t> spike_times;
        std::vector<float> synaptic_weights;
        uint32_t timestep;
        uint32_t neuron_count;
    };
    
    VisualizationData getVisualizationData(const std::string& network_name);
    std::future<void> exportVisualizationToTexture(const std::string& network_name,
                                                   VkImage output_texture,
                                                   uint32_t width, uint32_t height);
    
private:
    CarlComputeEngine* _engine;
    GPU::QueueManager* _queue_manager;
    
    // Network registry
    std::unordered_map<std::string, std::shared_ptr<SpikingNeuralNetwork>> _snn_networks;
    std::unordered_map<std::string, STDPConfig> _stdp_configs;
    std::unordered_map<std::string, SparseMemoryConfig> _sparse_configs;
    std::unordered_map<std::string, SNNPerformanceMetrics> _performance_metrics;
    
    // Compute pipeline resources
    std::unique_ptr<ComputePipelineManager> _pipeline_manager;
    
    // Synchronization resources
    std::vector<VkSemaphore> _simulation_semaphores;
    std::vector<VkFence> _completion_fences;
    
    // Resource pools for high-frequency operations
    std::vector<ComputeBuffer*> _temp_buffer_pool;
    std::mutex _buffer_pool_mutex;
    
    // Performance tracking
    std::mutex _metrics_mutex;
    std::chrono::steady_clock::time_point _last_performance_reset;
    
    // Internal helper methods
    bool _initializePipelines();
    ComputeBuffer* _getTempBuffer(size_t size_bytes);
    void _returnTempBuffer(ComputeBuffer* buffer);
    
    void _updatePerformanceMetrics(const std::string& network_name,
                                  uint64_t simulation_time,
                                  uint64_t learning_time);
    
    uint32_t _selectOptimalQueue(const std::string& network_name, 
                                GPU::WorkloadType workload);
    
    std::future<void> _executeOnQueue(uint32_t queue_index,
                                     std::function<void()> operation);
};

// Specialized SNN protocols for CARL integration
namespace Protocol {

// CNN-SNN Protocol: Self-reflection through spiking memory
class CNNSNNProtocol {
public:
    CNNSNNProtocol(SNNIntegration* snn_integration);
    
    // Convert CNN feature maps to spike patterns
    std::future<void> encodeFeatureMap(ComputeBuffer* cnn_features,
                                      ComputeBuffer* spike_patterns,
                                      uint32_t width, uint32_t height, uint32_t channels);
    
    // Retrieve similar patterns from SNN memory
    std::future<void> retrieveSimilarFeatures(ComputeBuffer* query_spikes,
                                             ComputeBuffer* retrieved_features,
                                             float similarity_threshold = 0.8f);
    
private:
    SNNIntegration* _snn_integration;
};

// RL-SNN Protocol: Cognitive memory loop with spike-based state encoding
class RLSNNProtocol {
public:
    RLSNNProtocol(SNNIntegration* snn_integration);
    
    // Encode RL state as spike patterns
    std::future<void> encodeState(ComputeBuffer* rl_state,
                                 ComputeBuffer* spike_encoding);
    
    // Decode spike patterns back to RL state
    std::future<void> decodeState(ComputeBuffer* spike_patterns,
                                 ComputeBuffer* rl_state);
    
    // Associate states with rewards using STDP
    std::future<void> associateReward(ComputeBuffer* state_spikes,
                                     float reward_value);
    
private:
    SNNIntegration* _snn_integration;
};

// GAN-SNN Protocol: Imagination engine using neuromorphic memory
class GANSNNProtocol {
public:
    GANSNNProtocol(SNNIntegration* snn_integration);
    
    // Generate novel patterns using SNN memory associations
    std::future<void> generateNovelPatterns(uint32_t pattern_count,
                                           ComputeBuffer* generated_patterns);
    
    // Evaluate pattern novelty using SNN memory
    std::future<float> evaluateNovelty(ComputeBuffer* candidate_pattern);
    
private:
    SNNIntegration* _snn_integration;
};

} // namespace Protocol

} // namespace AI
} // namespace CARL