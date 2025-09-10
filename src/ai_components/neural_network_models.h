#pragma once

#include "compute_pipeline_manager.h"
#include "carl_compute_engine.h"
#include <vector>
#include <memory>
#include <future>

/**
 * Neural Network Model Implementations for CARL AI System
 * Complete model definitions for CNN, GAN, RL, and SNN
 */

namespace CARL {
namespace AI {

// Forward declarations
class CarlComputeEngine;
class ComputePipelineManager;

// Base Neural Network Layer
struct NeuralLayer {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t width, height, channels;
    
    std::vector<ComputeBuffer*> weights;
    std::vector<ComputeBuffer*> biases;
    std::vector<ComputeBuffer*> activations;
    std::vector<ComputeBuffer*> gradients;
    
    ShaderType activation_type;
    float learning_rate;
    
    virtual ~NeuralLayer() = default;
    virtual void forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) = 0;
    virtual void backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) = 0;
};

// Convolutional Layer
class ConvolutionalLayer : public NeuralLayer {
public:
    ConvolutionalLayer(uint32_t input_width, uint32_t input_height, uint32_t input_channels,
                      uint32_t filter_count, uint32_t filter_size, uint32_t stride = 1, uint32_t padding = 0);
    
    void forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) override;
    void backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) override;
    
    void initializeWeights(CarlComputeEngine* engine);
    
private:
    uint32_t _filter_count;
    uint32_t _filter_size;
    uint32_t _stride;
    uint32_t _padding;
    uint32_t _output_width;
    uint32_t _output_height;
};

// Fully Connected Layer
class FullyConnectedLayer : public NeuralLayer {
public:
    FullyConnectedLayer(uint32_t input_size, uint32_t output_size);
    
    void forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) override;
    void backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) override;
    
    void initializeWeights(CarlComputeEngine* engine);

private:
    ComputeBuffer* _weight_matrix;
    ComputeBuffer* _bias_vector;
};

// CNN Model
class ConvolutionalNeuralNetwork {
public:
    ConvolutionalNeuralNetwork(CarlComputeEngine* engine, 
                              uint32_t input_width, uint32_t input_height, uint32_t channels);
    ~ConvolutionalNeuralNetwork();
    
    // Architecture Building
    void addConvolutionalLayer(uint32_t filters, uint32_t kernel_size, uint32_t stride = 1);
    void addPoolingLayer(uint32_t pool_size, uint32_t stride = 2, bool max_pool = true);
    void addFullyConnectedLayer(uint32_t units);
    void addActivationLayer(ShaderType activation_type);
    void addBatchNormalizationLayer();
    
    // Training and Inference
    std::future<void> forward(ComputeBuffer* input, ComputeBuffer* output);
    std::future<void> backward(ComputeBuffer* gradients);
    std::future<void> updateWeights(float learning_rate);
    
    // Model Management
    void initializeNetwork();
    void setTrainingMode(bool training);
    float calculateLoss(ComputeBuffer* predictions, ComputeBuffer* targets);
    
    // Getters
    uint32_t getLayerCount() const { return _layers.size(); }
    const NeuralLayer* getLayer(uint32_t index) const;
    
private:
    CarlComputeEngine* _engine;
    std::vector<std::unique_ptr<NeuralLayer>> _layers;
    
    uint32_t _input_width, _input_height, _channels;
    uint32_t _current_width, _current_height, _current_channels;
    bool _training_mode;
    
    std::vector<ComputeBuffer*> _intermediate_buffers;
    
    void allocateIntermediateBuffers();
    void deallocateIntermediateBuffers();
    uint32_t calculateBufferSize(uint32_t width, uint32_t height, uint32_t channels);
};

// GAN Model
class GenerativeAdversarialNetwork {
public:
    GenerativeAdversarialNetwork(CarlComputeEngine* engine, uint32_t noise_dim, 
                                uint32_t output_width, uint32_t output_height, uint32_t channels);
    ~GenerativeAdversarialNetwork();
    
    // Architecture Setup
    void buildGenerator();
    void buildDiscriminator();
    
    // Training
    std::future<void> trainDiscriminator(ComputeBuffer* real_data, ComputeBuffer* fake_data);
    std::future<void> trainGenerator(ComputeBuffer* noise);
    std::future<void> trainStep(ComputeBuffer* real_data, ComputeBuffer* noise);
    
    // Generation
    std::future<void> generate(ComputeBuffer* noise, ComputeBuffer* output);
    std::future<void> discriminate(ComputeBuffer* input, ComputeBuffer* output);
    
    // Training Configuration
    void setLearningRates(float generator_lr, float discriminator_lr);
    void setLossWeights(float adversarial_weight, float content_weight = 0.0f);
    
    // Progressive Training
    void enableProgressiveTraining(uint32_t start_resolution = 32);
    void updateProgressiveStage();
    float getProgressiveBlendFactor() const { return _progressive_blend_factor; }
    uint32_t getCurrentResolution() const { return _current_resolution; }
    
    // Multi-Queue Training  
    void setQueueStrategy(uint32_t generator_queue, uint32_t discriminator_queue);
    void setPipelineManager(ComputePipelineManager* pipeline_manager);
    
    // Advanced Training Methods
    std::future<void> trainWithLossComputation(ComputeBuffer* real_data, ComputeBuffer* noise);
    std::future<void> spectralNormalization();
    std::future<void> featureMatching(ComputeBuffer* real_features, ComputeBuffer* fake_features);
    
    // Performance Metrics
    struct TrainingMetrics {
        float discriminator_loss;
        float generator_loss;
        float inception_score;
        float fid_score;
        uint32_t training_steps;
        float avg_real_score;
        float avg_fake_score;
    };
    
    TrainingMetrics getTrainingMetrics() const { return _training_metrics; }
    
private:
    CarlComputeEngine* _engine;
    
    // Generator Network
    std::unique_ptr<ConvolutionalNeuralNetwork> _generator;
    uint32_t _noise_dim;
    
    // Discriminator Network
    std::unique_ptr<ConvolutionalNeuralNetwork> _discriminator;
    
    // Training Parameters
    float _generator_lr;
    float _discriminator_lr;
    float _adversarial_weight;
    float _content_weight;
    
    // Buffers
    ComputeBuffer* _generated_data;
    ComputeBuffer* _discriminator_real_output;
    ComputeBuffer* _discriminator_fake_output;
    ComputeBuffer* _generator_gradients;
    ComputeBuffer* _discriminator_gradients;
    
    uint32_t _output_width, _output_height, _channels;
    
    // Progressive Training State
    bool _progressive_training_enabled;
    uint32_t _current_resolution;
    uint32_t _target_resolution;
    float _progressive_blend_factor;
    uint32_t _training_steps;
    uint32_t _steps_per_stage;
    
    // Multi-Queue Configuration
    uint32_t _generator_queue;
    uint32_t _discriminator_queue;
    ComputePipelineManager* _pipeline_manager;
    
    // Progressive Training Buffers
    ComputeBuffer* _low_res_buffer;
    ComputeBuffer* _high_res_buffer;
    ComputeBuffer* _blended_output_buffer;
    ComputeBuffer* _interpolation_weights;
    
    // Loss Computation Buffers
    ComputeBuffer* _loss_statistics_buffer;
    ComputeBuffer* _d_loss_buffer;
    ComputeBuffer* _g_loss_buffer;
    
    // Training Metrics
    TrainingMetrics _training_metrics;
    
    void allocateTrainingBuffers();
    void deallocateTrainingBuffers();
    void allocateProgressiveBuffers();
    void deallocateProgressiveBuffers();
    void initializeProgressiveTraining();
    
    float calculateDiscriminatorLoss(ComputeBuffer* real_output, ComputeBuffer* fake_output);
    float calculateGeneratorLoss(ComputeBuffer* fake_output);
    
    // GPU-based loss computation
    std::future<void> computeLossesGPU(ComputeBuffer* real_output, ComputeBuffer* fake_output);
    
    // Progressive training helpers
    void updateProgressiveBuffers();
    bool shouldProgressToNextStage() const;
};

// Advanced Reinforcement Learning Agent with GPU Acceleration
class ReinforcementLearningAgent {
public:
    ReinforcementLearningAgent(CarlComputeEngine* engine, uint32_t state_dim, uint32_t action_dim);
    ~ReinforcementLearningAgent();
    
    // Network Architecture
    void buildQNetwork();
    void buildPolicyNetwork();
    void buildDuelingQNetwork();
    
    // Q-Learning Algorithms
    std::future<uint32_t> selectAction(ComputeBuffer* state, float epsilon = 0.1f);
    std::future<void> updateQValues(ComputeBuffer* states, ComputeBuffer* actions, 
                                   ComputeBuffer* rewards, ComputeBuffer* next_states, 
                                   ComputeBuffer* done_flags);
    
    // Policy Gradient Algorithms
    std::future<void> updatePolicy(ComputeBuffer* states, ComputeBuffer* actions, 
                                  ComputeBuffer* rewards);
    std::future<void> updateActorCritic(ComputeBuffer* states, ComputeBuffer* actions,
                                       ComputeBuffer* rewards, ComputeBuffer* next_states);
    
    // Experience Replay Systems
    void addExperience(const float* state, uint32_t action, float reward, 
                      const float* next_state, bool done);
    void addExperienceWithPriority(const float* state, uint32_t action, float reward,
                                  const float* next_state, bool done, float priority);
    std::future<void> replayExperience(uint32_t batch_size);
    std::future<void> prioritizedReplayExperience(uint32_t batch_size);
    
    // Multi-Agent Support
    void enableMultiAgent(uint32_t num_agents);
    std::future<void> trainMultiAgentParallel(const std::vector<ComputeBuffer*>& agent_states,
                                             const std::vector<ComputeBuffer*>& agent_actions,
                                             const std::vector<ComputeBuffer*>& agent_rewards);
    
    // Action Space Exploration
    struct ExplorationStrategy {
        enum Type { EPSILON_GREEDY, UCB, THOMPSON_SAMPLING, NOISY_NETWORKS };
        Type type;
        float exploration_param;
        uint32_t exploration_steps;
    };
    
    void setExplorationStrategy(const ExplorationStrategy& strategy);
    std::future<uint32_t> selectActionWithExploration(ComputeBuffer* state, uint32_t agent_id = 0);
    
    // Curriculum Learning
    struct CurriculumLevel {
        uint32_t level;
        float reward_threshold;
        float epsilon_override;
        uint32_t max_steps;
        std::vector<float> environment_params;
    };
    
    void enableCurriculumLearning(const std::vector<CurriculumLevel>& levels);
    void updateCurriculumProgress(float average_reward);
    uint32_t getCurrentCurriculumLevel() const { return _current_curriculum_level; }
    
    // Advanced Training Features
    void enableDoubleDQN(bool enable) { _use_double_dqn = enable; }
    void enableDuelingDQN(bool enable) { _use_dueling_dqn = enable; }
    void enablePrioritizedReplay(bool enable) { _use_prioritized_replay = enable; }
    void enableNoisyNetworks(bool enable) { _use_noisy_networks = enable; }
    
    // Multi-Environment Training
    void setupMultiEnvironment(uint32_t num_environments);
    std::future<void> trainParallelEnvironments(const std::vector<ComputeBuffer*>& env_states,
                                               const std::vector<ComputeBuffer*>& env_actions,
                                               const std::vector<ComputeBuffer*>& env_rewards);
    
    // Real-time Training Control
    void pauseTraining() { _training_paused = true; }
    void resumeTraining() { _training_paused = false; }
    bool isTraining() const { return !_training_paused; }
    
    // Performance Metrics
    struct RLMetrics {
        float average_reward;
        float episode_length;
        float exploration_rate;
        float q_value_mean;
        float policy_entropy;
        uint32_t training_steps;
        float loss_magnitude;
    };
    
    RLMetrics getTrainingMetrics() const { return _metrics; }
    
    // Configuration
    void setHyperparameters(float learning_rate, float gamma, float epsilon_decay);
    void setAlgorithm(bool use_dqn, bool use_policy_gradient);
    void setTargetUpdateFrequency(uint32_t frequency) { _target_update_frequency = frequency; }
    
private:
    CarlComputeEngine* _engine;
    std::unique_ptr<ComputePipelineManager> _pipeline_manager;
    
    uint32_t _state_dim;
    uint32_t _action_dim;
    uint32_t _batch_size;
    
    // Neural Networks
    std::unique_ptr<ConvolutionalNeuralNetwork> _q_network;
    std::unique_ptr<ConvolutionalNeuralNetwork> _target_q_network;
    std::unique_ptr<ConvolutionalNeuralNetwork> _policy_network;
    std::unique_ptr<ConvolutionalNeuralNetwork> _value_network;
    std::unique_ptr<ConvolutionalNeuralNetwork> _dueling_advantage_network;
    
    // Hyperparameters
    float _learning_rate;
    float _gamma;
    float _epsilon;
    float _epsilon_decay;
    float _entropy_coefficient;
    
    // Training Control
    uint32_t _target_update_frequency;
    uint32_t _steps_since_target_update;
    uint32_t _training_step;
    bool _training_paused;
    
    // Algorithm Configuration
    bool _use_double_dqn;
    bool _use_dueling_dqn;
    bool _use_prioritized_replay;
    bool _use_noisy_networks;
    
    // Multi-Agent Configuration
    bool _multi_agent_enabled;
    uint32_t _num_agents;
    std::vector<std::unique_ptr<ConvolutionalNeuralNetwork>> _agent_q_networks;
    std::vector<std::unique_ptr<ConvolutionalNeuralNetwork>> _agent_policy_networks;
    
    // Experience Replay Buffers
    struct Experience {
        std::vector<float> state;
        uint32_t action;
        float reward;
        std::vector<float> next_state;
        bool done;
        float priority; // For prioritized replay
        uint32_t agent_id; // For multi-agent
    };
    
    std::vector<Experience> _experience_buffer;
    uint32_t _buffer_capacity;
    uint32_t _buffer_index;
    
    // Priority Buffer (for prioritized experience replay)
    std::vector<float> _priority_buffer;
    float _priority_alpha;
    float _priority_beta;
    
    // GPU Buffers
    ComputeBuffer* _q_values;
    ComputeBuffer* _target_q_values;
    ComputeBuffer* _policy_output;
    ComputeBuffer* _value_estimates;
    ComputeBuffer* _policy_weights;
    ComputeBuffer* _policy_gradients;
    ComputeBuffer* _q_learning_targets;
    
    // Experience Replay Buffers
    ComputeBuffer* _experience_states;
    ComputeBuffer* _experience_actions;
    ComputeBuffer* _experience_rewards;
    ComputeBuffer* _experience_next_states;
    ComputeBuffer* _experience_done;
    ComputeBuffer* _experience_priorities;
    
    // Multi-Agent Buffers
    std::vector<ComputeBuffer*> _multi_agent_states;
    std::vector<ComputeBuffer*> _multi_agent_actions;
    std::vector<ComputeBuffer*> _multi_agent_rewards;
    
    // Exploration Strategy
    ExplorationStrategy _exploration_strategy;
    std::vector<float> _exploration_noise;
    ComputeBuffer* _exploration_buffer;
    
    // Curriculum Learning
    bool _curriculum_learning_enabled;
    uint32_t _current_curriculum_level;
    std::vector<CurriculumLevel> _curriculum_levels;
    std::vector<float> _episode_rewards;
    
    // Multi-Environment Support
    uint32_t _num_environments;
    std::vector<ComputeBuffer*> _environment_states;
    std::vector<ComputeBuffer*> _environment_actions;
    std::vector<ComputeBuffer*> _environment_rewards;
    
    // Performance Metrics
    RLMetrics _metrics;
    std::vector<float> _recent_rewards;
    std::vector<float> _recent_losses;
    
    // Buffer Management
    void allocateBuffers();
    void deallocateBuffers();
    void allocateMultiAgentBuffers();
    void allocateMultiEnvironmentBuffers();
    
    // Network Management
    void updateTargetNetwork();
    void copyNetworkWeights(ConvolutionalNeuralNetwork* source, ConvolutionalNeuralNetwork* target);
    
    // Action Selection
    uint32_t sampleAction(ComputeBuffer* policy_output);
    uint32_t selectActionEpsilonGreedy(ComputeBuffer* q_values, float epsilon);
    uint32_t selectActionUCB(ComputeBuffer* q_values, const std::vector<uint32_t>& action_counts);
    uint32_t selectActionThompsonSampling(ComputeBuffer* q_values);
    
    // Experience Replay Helpers
    void updatePriorities(const std::vector<uint32_t>& indices, const std::vector<float>& td_errors);
    std::vector<uint32_t> samplePrioritizedIndices(uint32_t batch_size);
    float calculateImportanceSamplingWeight(float priority, uint32_t buffer_size);
    
    // Curriculum Learning Helpers
    void initializeActionSpaceExploration();
    void setupMultiEnvironmentSupport();
    bool shouldAdvanceCurriculum() const;
    void updateMetrics();
    
    // Multi-Agent Helpers
    void synchronizeAgentNetworks();
    void aggregateAgentExperiences();
};

// Spiking Neural Network
class SpikingNeuralNetwork {
public:
    SpikingNeuralNetwork(CarlComputeEngine* engine, uint32_t neurons, uint32_t timesteps);
    ~SpikingNeuralNetwork();
    
    // Network Configuration
    void setNeuronParameters(float membrane_resistance, float membrane_capacitance, 
                           float threshold_voltage, float reset_voltage, float refractory_period);
    void initializeNetwork();
    
    // Simulation
    std::future<void> simulateTimestep(ComputeBuffer* input_currents, float dt);
    std::future<void> reset();
    
    // Sparse Memory Management
    void enableSparseBinding(uint32_t virtual_memory_size);
    std::future<void> commitMemoryRegion(uint32_t offset, uint32_t size);
    std::future<void> releaseMemoryRegion(uint32_t offset, uint32_t size);
    
    // Output Access
    ComputeBuffer* getMembranePotentials() { return _membrane_potentials; }
    ComputeBuffer* getSpikeTimes() { return _spike_times; }
    ComputeBuffer* getOutputSpikes() { return _output_spikes; }
    
    // Monitoring
    uint32_t getSpikeCount() const;
    float getAverageFireRate() const;
    
private:
    CarlComputeEngine* _engine;
    
    uint32_t _neurons;
    uint32_t _timesteps;
    uint32_t _current_timestep;
    
    // Neuron Parameters
    float _membrane_resistance;
    float _membrane_capacitance;
    float _threshold_voltage;
    float _reset_voltage;
    float _refractory_period;
    
    // Neuron State Buffers
    ComputeBuffer* _membrane_potentials;
    ComputeBuffer* _input_currents;
    ComputeBuffer* _spike_times;
    ComputeBuffer* _output_spikes;
    ComputeBuffer* _refractory_timers;
    
    // Sparse Memory
    bool _sparse_enabled;
    uint32_t _virtual_memory_size;
    ComputeBuffer* _virtual_memory_map;
    ComputeBuffer* _physical_memory_pool;
    
    void allocateBuffers();
    void deallocateBuffers();
    void allocateSparseBuffers();
};

} // namespace AI
} // namespace CARL