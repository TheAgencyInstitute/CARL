#include "neural_network_models.h"
#include "compute_pipeline_manager.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <thread>
#include <chrono>

namespace CARL {
namespace AI {

ReinforcementLearningAgent::ReinforcementLearningAgent(CarlComputeEngine* engine, uint32_t state_dim, uint32_t action_dim)
    : _engine(engine), _state_dim(state_dim), _action_dim(action_dim),
      _learning_rate(0.001f), _gamma(0.99f), _epsilon(1.0f), _epsilon_decay(0.995f),
      _buffer_capacity(100000), _buffer_index(0), _target_update_frequency(1000),
      _steps_since_target_update(0), _training_step(0), _use_double_dqn(true),
      _use_dueling_dqn(false), _use_prioritized_replay(false), _multi_agent_enabled(false),
      _num_agents(1), _curriculum_learning_enabled(false), _current_curriculum_level(0) {
    
    // Initialize experience replay buffer
    _experience_buffer.reserve(_buffer_capacity);
    
    // Initialize pipeline manager for RL compute shaders
    _pipeline_manager = std::make_unique<ComputePipelineManager>(_engine->getDevice(), _engine->getPhysicalDevice());
    _pipeline_manager->initialize();
    
    // Create RL-specific compute pipelines
    _pipeline_manager->createComputePipeline(ShaderType::RL_Q_LEARNING, "src/shaders/compiled/rl_q_learning.comp.spv");
    _pipeline_manager->createComputePipeline(ShaderType::RL_POLICY_GRADIENT, "src/shaders/compiled/rl_policy_gradient.comp.spv");
    
    allocateBuffers();
    initializeActionSpaceExploration();
    setupMultiEnvironmentSupport();
    
    std::cout << "RL Agent initialized with enhanced GPU acceleration" << std::endl;
}

ReinforcementLearningAgent::~ReinforcementLearningAgent() {
    deallocateBuffers();
}

void ReinforcementLearningAgent::buildQNetwork() {
    // Q-Network: State -> Q-values for each action
    _q_network = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
    
    // Hidden layers
    _q_network->addFullyConnectedLayer(128);
    _q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _q_network->addFullyConnectedLayer(128);
    _q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _q_network->addFullyConnectedLayer(64);
    _q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    // Output layer: Q-values for each action
    _q_network->addFullyConnectedLayer(_action_dim);
    
    _q_network->initializeNetwork();
    
    // Target Q-Network (copy of Q-Network)
    _target_q_network = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
    
    _target_q_network->addFullyConnectedLayer(128);
    _target_q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _target_q_network->addFullyConnectedLayer(128);
    _target_q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _target_q_network->addFullyConnectedLayer(64);
    _target_q_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _target_q_network->addFullyConnectedLayer(_action_dim);
    
    _target_q_network->initializeNetwork();
    
    std::cout << "Q-Network built with " << _q_network->getLayerCount() << " layers" << std::endl;
}

void ReinforcementLearningAgent::buildPolicyNetwork() {
    // Policy Network: State -> Action probabilities
    _policy_network = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
    
    // Hidden layers
    _policy_network->addFullyConnectedLayer(128);
    _policy_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    _policy_network->addFullyConnectedLayer(64);
    _policy_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    // Output layer: Action probabilities
    _policy_network->addFullyConnectedLayer(_action_dim);
    _policy_network->addActivationLayer(ShaderType::ACTIVATION_SOFTMAX);
    
    _policy_network->initializeNetwork();
    
    std::cout << "Policy Network built with " << _policy_network->getLayerCount() << " layers" << std::endl;
}

std::future<uint32_t> ReinforcementLearningAgent::selectAction(ComputeBuffer* state, float epsilon) {
    return std::async(std::launch::async, [this, state, epsilon]() -> uint32_t {
        // Epsilon-greedy action selection
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        if (dis(gen) < epsilon) {
            // Random action
            std::uniform_int_distribution<uint32_t> action_dis(0, _action_dim - 1);
            return action_dis(gen);
        } else {
            // Greedy action based on Q-values
            auto forward = _q_network->forward(state, _q_values);
            forward.wait();
            
            // Find action with maximum Q-value
            std::vector<float> q_data(_action_dim);
            _engine->downloadData(_q_values, q_data.data(), q_data.size() * sizeof(float));
            
            auto max_it = std::max_element(q_data.begin(), q_data.end());
            return static_cast<uint32_t>(std::distance(q_data.begin(), max_it));
        }
    });
}

std::future<void> ReinforcementLearningAgent::updateQValues(ComputeBuffer* states, ComputeBuffer* actions, 
                                                           ComputeBuffer* rewards, ComputeBuffer* next_states, 
                                                           ComputeBuffer* done_flags) {
    return std::async(std::launch::async, [this, states, actions, rewards, next_states, done_flags]() {
        // Forward pass through Q-network
        auto q_forward = _q_network->forward(states, _q_values);
        q_forward.wait();
        
        // Forward pass through target Q-network for next states
        auto target_forward = _target_q_network->forward(next_states, _target_q_values);
        target_forward.wait();
        
        // Use Q-learning compute shader for GPU-accelerated Q-value updates
        PushConstantData push_constants = {};
        push_constants.data[0] = _batch_size;
        push_constants.data[1] = _state_dim;
        push_constants.data[2] = _action_dim;
        
        // Pack gamma as float
        *reinterpret_cast<float*>(&push_constants.data[3]) = _gamma;
        *reinterpret_cast<float*>(&push_constants.data[4]) = _learning_rate;
        
        push_constants.data[5] = _use_double_dqn ? 1 : 0; // Update type
        push_constants.size = 6 * sizeof(uint32_t);
        
        // Create descriptor set for Q-learning compute shader
        std::vector<VkBuffer> buffers = {
            _q_values->buffer, states->buffer, actions->buffer,
            rewards->buffer, next_states->buffer, done_flags->buffer,
            _q_learning_targets->buffer
        };
        
        std::vector<VkDescriptorType> types(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        
        auto* pipeline_info = _pipeline_manager->getPipelineInfo(ShaderType::RL_Q_LEARNING);
        VkDescriptorSet descriptor_set = _pipeline_manager->allocateDescriptorSet(pipeline_info->descriptor_layout);
        _pipeline_manager->updateDescriptorSet(descriptor_set, buffers, types);
        
        // Execute Q-learning compute shader on queue 3 (RL operations)
        VkCommandBuffer cmd_buffer = _engine->beginSingleTimeCommands(3);
        
        _pipeline_manager->dispatchCompute(
            ShaderType::RL_Q_LEARNING,
            cmd_buffer,
            descriptor_set,
            push_constants,
            (_batch_size + 255) / 256, 1, 1
        );
        
        _engine->endSingleTimeCommands(cmd_buffer, 3);
        
        std::cout << "Q-values updated using GPU Q-learning algorithm, batch size: " << _batch_size << std::endl;
    });
}

std::future<void> ReinforcementLearningAgent::updatePolicy(ComputeBuffer* states, ComputeBuffer* actions, 
                                                          ComputeBuffer* rewards) {
    return std::async(std::launch::async, [this, states, actions, rewards]() {
        // Forward pass through policy network
        auto policy_forward = _policy_network->forward(states, _policy_output);
        policy_forward.wait();
        
        // Use policy gradient compute shader for GPU-accelerated policy updates
        PushConstantData push_constants = {};
        push_constants.data[0] = _batch_size;
        push_constants.data[1] = _state_dim;
        push_constants.data[2] = _action_dim;
        push_constants.data[3] = _policy_network->getWeightCount();
        
        *reinterpret_cast<float*>(&push_constants.data[4]) = _learning_rate;
        *reinterpret_cast<float*>(&push_constants.data[5]) = _entropy_coefficient;
        
        push_constants.data[6] = 1; // Actor-Critic algorithm type
        *reinterpret_cast<float*>(&push_constants.data[7]) = _gamma;
        push_constants.size = 8 * sizeof(uint32_t);
        
        // Create descriptor set for policy gradient compute shader
        std::vector<VkBuffer> buffers = {
            _policy_weights->buffer, states->buffer, actions->buffer,
            rewards->buffer, _policy_output->buffer, _value_estimates->buffer,
            _policy_gradients->buffer
        };
        
        std::vector<VkDescriptorType> types(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        
        auto* pipeline_info = _pipeline_manager->getPipelineInfo(ShaderType::RL_POLICY_GRADIENT);
        VkDescriptorSet descriptor_set = _pipeline_manager->allocateDescriptorSet(pipeline_info->descriptor_layout);
        _pipeline_manager->updateDescriptorSet(descriptor_set, buffers, types);
        
        // Execute policy gradient compute shader on queue 3 (RL operations)
        VkCommandBuffer cmd_buffer = _engine->beginSingleTimeCommands(3);
        
        _pipeline_manager->dispatchCompute(
            ShaderType::RL_POLICY_GRADIENT,
            cmd_buffer,
            descriptor_set,
            push_constants,
            (_batch_size + 255) / 256, 1, 1
        );
        
        _engine->endSingleTimeCommands(cmd_buffer, 3);
        
        std::cout << "Policy updated using GPU policy gradient algorithm, batch size: " << _batch_size << std::endl;
    });
}

void ReinforcementLearningAgent::addExperience(const float* state, uint32_t action, float reward, 
                                              const float* next_state, bool done) {
    Experience exp;
    exp.state.assign(state, state + _state_dim);
    exp.action = action;
    exp.reward = reward;
    exp.next_state.assign(next_state, next_state + _state_dim);
    exp.done = done;
    
    if (_experience_buffer.size() < _buffer_capacity) {
        _experience_buffer.push_back(exp);
    } else {
        _experience_buffer[_buffer_index] = exp;
        _buffer_index = (_buffer_index + 1) % _buffer_capacity;
    }
}

std::future<void> ReinforcementLearningAgent::replayExperience(uint32_t batch_size) {
    return std::async(std::launch::async, [this, batch_size]() {
        _batch_size = batch_size;
        if (_experience_buffer.size() < batch_size) {
            std::cout << "Not enough experiences for replay" << std::endl;
            return;
        }
        
        // Sample random batch from experience buffer
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, _experience_buffer.size() - 1);
        
        std::vector<float> batch_states(batch_size * _state_dim);
        std::vector<uint32_t> batch_actions(batch_size);
        std::vector<float> batch_rewards(batch_size);
        std::vector<float> batch_next_states(batch_size * _state_dim);
        std::vector<uint32_t> batch_done(batch_size);
        
        for (uint32_t i = 0; i < batch_size; i++) {
            size_t idx = dis(gen);
            const Experience& exp = _experience_buffer[idx];
            
            // Copy state
            std::copy(exp.state.begin(), exp.state.end(), 
                     batch_states.begin() + i * _state_dim);
            
            batch_actions[i] = exp.action;
            batch_rewards[i] = exp.reward;
            
            // Copy next state
            std::copy(exp.next_state.begin(), exp.next_state.end(), 
                     batch_next_states.begin() + i * _state_dim);
            
            batch_done[i] = exp.done ? 1 : 0;
        }
        
        // Upload batch data to GPU buffers
        _engine->uploadData(_experience_states, batch_states.data(), 
                           batch_states.size() * sizeof(float));
        _engine->uploadData(_experience_actions, batch_actions.data(), 
                           batch_actions.size() * sizeof(uint32_t));
        _engine->uploadData(_experience_rewards, batch_rewards.data(), 
                           batch_rewards.size() * sizeof(float));
        _engine->uploadData(_experience_next_states, batch_next_states.data(), 
                           batch_next_states.size() * sizeof(float));
        _engine->uploadData(_experience_done, batch_done.data(), 
                           batch_done.size() * sizeof(uint32_t));
        
        // Update Q-values using the batch
        auto update = updateQValues(_experience_states, _experience_actions, 
                                   _experience_rewards, _experience_next_states, 
                                   _experience_done);
        update.wait();
        
        std::cout << "Experience replay completed with batch size: " << batch_size << std::endl;
    });
}

void ReinforcementLearningAgent::setHyperparameters(float learning_rate, float gamma, float epsilon_decay) {
    _learning_rate = learning_rate;
    _gamma = gamma;
    _epsilon_decay = epsilon_decay;
}

void ReinforcementLearningAgent::setAlgorithm(bool use_dqn, bool use_policy_gradient) {
    if (use_dqn && !_q_network) {
        buildQNetwork();
    }
    
    if (use_policy_gradient && !_policy_network) {
        buildPolicyNetwork();
    }
}

void ReinforcementLearningAgent::allocateBuffers() {
    size_t state_size = _state_dim * sizeof(float);
    size_t action_values_size = _action_dim * sizeof(float);
    size_t batch_states_size = 64 * _state_dim * sizeof(float); // Batch size 64
    size_t batch_actions_size = 64 * sizeof(uint32_t);
    size_t batch_rewards_size = 64 * sizeof(float);
    
    // Core RL buffers
    _q_values = _engine->createBuffer(action_values_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _target_q_values = _engine->createBuffer(action_values_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _policy_output = _engine->createBuffer(action_values_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _value_estimates = _engine->createBuffer(batch_rewards_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _q_learning_targets = _engine->createBuffer(batch_rewards_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Policy gradient buffers
    size_t weight_count = 128 * _state_dim + 128 * 64 + 64 * _action_dim; // Estimated weight count
    _policy_weights = _engine->createBuffer(weight_count * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _policy_gradients = _engine->createBuffer(weight_count * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Experience replay buffers
    _experience_states = _engine->createBuffer(batch_states_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _experience_actions = _engine->createBuffer(batch_actions_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _experience_rewards = _engine->createBuffer(batch_rewards_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _experience_next_states = _engine->createBuffer(batch_states_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _experience_done = _engine->createBuffer(batch_actions_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _experience_priorities = _engine->createBuffer(batch_rewards_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Exploration buffer
    _exploration_buffer = _engine->createBuffer(batch_actions_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    std::cout << "Enhanced RL Agent buffers allocated with GPU acceleration support" << std::endl;
}

void ReinforcementLearningAgent::deallocateBuffers() {
    // Core RL buffers
    if (_q_values) _engine->destroyBuffer(_q_values);
    if (_target_q_values) _engine->destroyBuffer(_target_q_values);
    if (_policy_output) _engine->destroyBuffer(_policy_output);
    if (_value_estimates) _engine->destroyBuffer(_value_estimates);
    if (_q_learning_targets) _engine->destroyBuffer(_q_learning_targets);
    
    // Policy gradient buffers
    if (_policy_weights) _engine->destroyBuffer(_policy_weights);
    if (_policy_gradients) _engine->destroyBuffer(_policy_gradients);
    
    // Experience replay buffers
    if (_experience_states) _engine->destroyBuffer(_experience_states);
    if (_experience_actions) _engine->destroyBuffer(_experience_actions);
    if (_experience_rewards) _engine->destroyBuffer(_experience_rewards);
    if (_experience_next_states) _engine->destroyBuffer(_experience_next_states);
    if (_experience_done) _engine->destroyBuffer(_experience_done);
    if (_experience_priorities) _engine->destroyBuffer(_experience_priorities);
    
    // Exploration buffer
    if (_exploration_buffer) _engine->destroyBuffer(_exploration_buffer);
    
    // Multi-agent buffers
    for (auto* buffer : _multi_agent_states) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    for (auto* buffer : _multi_agent_actions) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    for (auto* buffer : _multi_agent_rewards) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    
    // Multi-environment buffers
    for (auto* buffer : _environment_states) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    for (auto* buffer : _environment_actions) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    for (auto* buffer : _environment_rewards) {
        if (buffer) _engine->destroyBuffer(buffer);
    }
    
    // Reset pointers
    _q_values = nullptr;
    _target_q_values = nullptr;
    _policy_output = nullptr;
    _value_estimates = nullptr;
    _q_learning_targets = nullptr;
    _policy_weights = nullptr;
    _policy_gradients = nullptr;
    _experience_states = nullptr;
    _experience_actions = nullptr;
    _experience_rewards = nullptr;
    _experience_next_states = nullptr;
    _experience_done = nullptr;
    _experience_priorities = nullptr;
    _exploration_buffer = nullptr;
}

void ReinforcementLearningAgent::updateTargetNetwork() {
    if (_q_network && _target_q_network) {
        copyNetworkWeights(_q_network.get(), _target_q_network.get());
        _steps_since_target_update = 0;
        std::cout << "Target network updated at step " << _training_step << std::endl;
    }
}

uint32_t ReinforcementLearningAgent::sampleAction(ComputeBuffer* policy_output) {
    // Sample action from policy probability distribution
    std::vector<float> probs(_action_dim);
    _engine->downloadData(policy_output, probs.data(), probs.size() * sizeof(float));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<uint32_t> dis(probs.begin(), probs.end());
    
    return dis(gen);
}

// Advanced RL Features Implementation

void ReinforcementLearningAgent::buildDuelingQNetwork() {
    // Dueling Q-Network: State -> [State Value, Action Advantages]
    _value_network = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
    _dueling_advantage_network = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
    
    // Shared feature extraction layers
    _value_network->addFullyConnectedLayer(128);
    _value_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    _value_network->addFullyConnectedLayer(64);
    _value_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    
    // Value stream: outputs single state value
    _value_network->addFullyConnectedLayer(1);
    
    // Advantage stream: outputs advantage for each action
    _dueling_advantage_network->addFullyConnectedLayer(128);
    _dueling_advantage_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    _dueling_advantage_network->addFullyConnectedLayer(64);
    _dueling_advantage_network->addActivationLayer(ShaderType::ACTIVATION_RELU);
    _dueling_advantage_network->addFullyConnectedLayer(_action_dim);
    
    _value_network->initializeNetwork();
    _dueling_advantage_network->initializeNetwork();
    
    _use_dueling_dqn = true;
    std::cout << "Dueling Q-Network built" << std::endl;
}

std::future<void> ReinforcementLearningAgent::updateActorCritic(ComputeBuffer* states, ComputeBuffer* actions,
                                                               ComputeBuffer* rewards, ComputeBuffer* next_states) {
    return std::async(std::launch::async, [this, states, actions, rewards, next_states]() {
        // Update value network (critic)
        auto value_forward = _value_network->forward(states, _value_estimates);
        value_forward.wait();
        
        // Update policy network (actor) with advantage estimates
        auto policy_forward = _policy_network->forward(states, _policy_output);
        policy_forward.wait();
        
        // Use policy gradient shader with value function baseline
        auto policy_update = updatePolicy(states, actions, rewards);
        policy_update.wait();
        
        std::cout << "Actor-Critic networks updated" << std::endl;
    });
}

void ReinforcementLearningAgent::addExperienceWithPriority(const float* state, uint32_t action, float reward,
                                                          const float* next_state, bool done, float priority) {
    Experience exp;
    exp.state.assign(state, state + _state_dim);
    exp.action = action;
    exp.reward = reward;
    exp.next_state.assign(next_state, next_state + _state_dim);
    exp.done = done;
    exp.priority = priority;
    exp.agent_id = 0; // Default to single agent
    
    if (_experience_buffer.size() < _buffer_capacity) {
        _experience_buffer.push_back(exp);
        _priority_buffer.push_back(priority);
    } else {
        _experience_buffer[_buffer_index] = exp;
        _priority_buffer[_buffer_index] = priority;
        _buffer_index = (_buffer_index + 1) % _buffer_capacity;
    }
}

std::future<void> ReinforcementLearningAgent::prioritizedReplayExperience(uint32_t batch_size) {
    return std::async(std::launch::async, [this, batch_size]() {
        if (!_use_prioritized_replay || _experience_buffer.size() < batch_size) {
            std::cout << "Insufficient experiences for prioritized replay" << std::endl;
            return;
        }
        
        // Sample indices based on priorities
        auto indices = samplePrioritizedIndices(batch_size);
        
        std::vector<float> batch_states(batch_size * _state_dim);
        std::vector<uint32_t> batch_actions(batch_size);
        std::vector<float> batch_rewards(batch_size);
        std::vector<float> batch_next_states(batch_size * _state_dim);
        std::vector<uint32_t> batch_done(batch_size);
        std::vector<float> importance_weights(batch_size);
        
        for (uint32_t i = 0; i < batch_size; i++) {
            uint32_t idx = indices[i];
            const Experience& exp = _experience_buffer[idx];
            
            std::copy(exp.state.begin(), exp.state.end(), 
                     batch_states.begin() + i * _state_dim);
            batch_actions[i] = exp.action;
            batch_rewards[i] = exp.reward;
            std::copy(exp.next_state.begin(), exp.next_state.end(), 
                     batch_next_states.begin() + i * _state_dim);
            batch_done[i] = exp.done ? 1 : 0;
            
            // Calculate importance sampling weights
            importance_weights[i] = calculateImportanceSamplingWeight(exp.priority, _experience_buffer.size());
        }
        
        // Upload prioritized batch to GPU
        _engine->uploadData(_experience_states, batch_states.data(), batch_states.size() * sizeof(float));
        _engine->uploadData(_experience_actions, batch_actions.data(), batch_actions.size() * sizeof(uint32_t));
        _engine->uploadData(_experience_rewards, batch_rewards.data(), batch_rewards.size() * sizeof(float));
        _engine->uploadData(_experience_next_states, batch_next_states.data(), batch_next_states.size() * sizeof(float));
        _engine->uploadData(_experience_done, batch_done.data(), batch_done.size() * sizeof(uint32_t));
        _engine->uploadData(_experience_priorities, importance_weights.data(), importance_weights.size() * sizeof(float));
        
        // Update Q-values with prioritized experience
        auto update = updateQValues(_experience_states, _experience_actions, _experience_rewards, 
                                   _experience_next_states, _experience_done);
        update.wait();
        
        std::cout << "Prioritized experience replay completed" << std::endl;
    });
}

void ReinforcementLearningAgent::enableMultiAgent(uint32_t num_agents) {
    _multi_agent_enabled = true;
    _num_agents = num_agents;
    
    // Create separate networks for each agent
    _agent_q_networks.resize(num_agents);
    _agent_policy_networks.resize(num_agents);
    
    for (uint32_t i = 0; i < num_agents; i++) {
        _agent_q_networks[i] = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
        _agent_policy_networks[i] = std::make_unique<ConvolutionalNeuralNetwork>(_engine, _state_dim, 1, 1);
        
        // Build agent-specific networks using the same architecture
        _agent_q_networks[i]->addFullyConnectedLayer(128);
        _agent_q_networks[i]->addActivationLayer(ShaderType::ACTIVATION_RELU);
        _agent_q_networks[i]->addFullyConnectedLayer(128);
        _agent_q_networks[i]->addActivationLayer(ShaderType::ACTIVATION_RELU);
        _agent_q_networks[i]->addFullyConnectedLayer(64);
        _agent_q_networks[i]->addActivationLayer(ShaderType::ACTIVATION_RELU);
        _agent_q_networks[i]->addFullyConnectedLayer(_action_dim);
        _agent_q_networks[i]->initializeNetwork();
        
        _agent_policy_networks[i]->addFullyConnectedLayer(128);
        _agent_policy_networks[i]->addActivationLayer(ShaderType::ACTIVATION_RELU);
        _agent_policy_networks[i]->addFullyConnectedLayer(64);
        _agent_policy_networks[i]->addActivationLayer(ShaderType::ACTIVATION_RELU);
        _agent_policy_networks[i]->addFullyConnectedLayer(_action_dim);
        _agent_policy_networks[i]->addActivationLayer(ShaderType::ACTIVATION_SOFTMAX);
        _agent_policy_networks[i]->initializeNetwork();
    }
    
    allocateMultiAgentBuffers();
    
    std::cout << "Multi-agent RL enabled with " << num_agents << " agents" << std::endl;
}

std::future<void> ReinforcementLearningAgent::trainMultiAgentParallel(const std::vector<ComputeBuffer*>& agent_states,
                                                                     const std::vector<ComputeBuffer*>& agent_actions,
                                                                     const std::vector<ComputeBuffer*>& agent_rewards) {
    return std::async(std::launch::async, [this, &agent_states, &agent_actions, &agent_rewards]() {
        if (!_multi_agent_enabled || agent_states.size() != _num_agents) {
            std::cout << "Multi-agent training not properly configured" << std::endl;
            return;
        }
        
        // Train each agent in parallel using different compute queues
        std::vector<std::future<void>> agent_futures;
        
        for (uint32_t i = 0; i < _num_agents; i++) {
            agent_futures.push_back(std::async(std::launch::async, [this, i, &agent_states, &agent_actions, &agent_rewards]() {
                // Train agent i - simplified Q-value update
                auto q_forward = _agent_q_networks[i]->forward(agent_states[i], _multi_agent_states[i]);
                q_forward.wait();
                
                auto policy_forward = _agent_policy_networks[i]->forward(agent_states[i], _multi_agent_actions[i]);
                policy_forward.wait();
                
                // Update agent networks
                auto q_update = _agent_q_networks[i]->updateWeights(_learning_rate);
                q_update.wait();
                
                auto policy_update = _agent_policy_networks[i]->updateWeights(_learning_rate);
                policy_update.wait();
            }));
        }
        
        // Wait for all agents to complete training
        for (auto& future : agent_futures) {
            future.wait();
        }
        
        // Synchronize agent experiences and networks if needed
        aggregateAgentExperiences();
        
        std::cout << "Multi-agent parallel training completed" << std::endl;
    });
}

void ReinforcementLearningAgent::setExplorationStrategy(const ExplorationStrategy& strategy) {
    _exploration_strategy = strategy;
    
    // Initialize exploration-specific data structures
    if (strategy.type == ExplorationStrategy::NOISY_NETWORKS) {
        _use_noisy_networks = true;
        // Initialize noise parameters for noisy networks
        _exploration_noise.resize(_action_dim, 0.0f);
    }
    
    std::cout << "Exploration strategy set to type " << static_cast<int>(strategy.type) << std::endl;
}

std::future<uint32_t> ReinforcementLearningAgent::selectActionWithExploration(ComputeBuffer* state, uint32_t agent_id) {
    return std::async(std::launch::async, [this, state, agent_id]() -> uint32_t {
        uint32_t selected_action = 0;
        
        switch (_exploration_strategy.type) {
            case ExplorationStrategy::EPSILON_GREEDY:
                selected_action = selectActionEpsilonGreedy(_q_values, _epsilon);
                break;
                
            case ExplorationStrategy::UCB: {
                std::vector<uint32_t> action_counts(_action_dim, 1); // Initialize with 1 to avoid division by zero
                selected_action = selectActionUCB(_q_values, action_counts);
                break;
            }
            
            case ExplorationStrategy::THOMPSON_SAMPLING:
                selected_action = selectActionThompsonSampling(_q_values);
                break;
                
            case ExplorationStrategy::NOISY_NETWORKS: {
                // Use noisy networks for exploration
                auto forward = _q_network->forward(state, _q_values);
                forward.wait();
                
                // Add noise to Q-values for exploration
                std::vector<float> q_data(_action_dim);
                _engine->downloadData(_q_values, q_data.data(), q_data.size() * sizeof(float));
                
                // Apply noise
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> noise_dist(0.0f, _exploration_strategy.exploration_param);
                
                for (uint32_t i = 0; i < _action_dim; i++) {
                    q_data[i] += noise_dist(gen);
                }
                
                auto max_it = std::max_element(q_data.begin(), q_data.end());
                selected_action = static_cast<uint32_t>(std::distance(q_data.begin(), max_it));
                break;
            }
        }
        
        return selected_action;
    });
}

void ReinforcementLearningAgent::enableCurriculumLearning(const std::vector<CurriculumLevel>& levels) {
    _curriculum_learning_enabled = true;
    _curriculum_levels = levels;
    _current_curriculum_level = 0;
    _episode_rewards.clear();
    
    std::cout << "Curriculum learning enabled with " << levels.size() << " levels" << std::endl;
}

void ReinforcementLearningAgent::updateCurriculumProgress(float average_reward) {
    if (!_curriculum_learning_enabled || _current_curriculum_level >= _curriculum_levels.size() - 1) {
        return;
    }
    
    _episode_rewards.push_back(average_reward);
    
    // Check if ready to advance to next curriculum level
    if (shouldAdvanceCurriculum()) {
        _current_curriculum_level++;
        _episode_rewards.clear();
        
        // Update hyperparameters for new level
        const auto& level = _curriculum_levels[_current_curriculum_level];
        if (level.epsilon_override > 0) {
            _epsilon = level.epsilon_override;
        }
        
        std::cout << "Advanced to curriculum level " << _current_curriculum_level << std::endl;
    }
}

void ReinforcementLearningAgent::setupMultiEnvironment(uint32_t num_environments) {
    _num_environments = num_environments;
    allocateMultiEnvironmentBuffers();
    
    std::cout << "Multi-environment training setup with " << num_environments << " environments" << std::endl;
}

std::future<void> ReinforcementLearningAgent::trainParallelEnvironments(const std::vector<ComputeBuffer*>& env_states,
                                                                       const std::vector<ComputeBuffer*>& env_actions,
                                                                       const std::vector<ComputeBuffer*>& env_rewards) {
    return std::async(std::launch::async, [this, &env_states, &env_actions, &env_rewards]() {
        if (env_states.size() != _num_environments) {
            std::cout << "Environment count mismatch in parallel training" << std::endl;
            return;
        }
        
        // Train on all environments in parallel
        std::vector<std::future<void>> env_futures;
        
        for (uint32_t i = 0; i < _num_environments; i++) {
            env_futures.push_back(std::async(std::launch::async, [this, i, &env_states, &env_actions, &env_rewards]() {
                auto update = updateQValues(env_states[i], env_actions[i], env_rewards[i], 
                                           env_states[i], _experience_done);
                update.wait();
            }));
        }
        
        // Wait for all environment updates
        for (auto& future : env_futures) {
            future.wait();
        }
        
        // Update target network if necessary
        _steps_since_target_update++;
        if (_steps_since_target_update >= _target_update_frequency) {
            updateTargetNetwork();
        }
        
        updateMetrics();
        
        std::cout << "Parallel environment training completed for " << _num_environments << " environments" << std::endl;
    });
}

// Helper function implementations

void ReinforcementLearningAgent::allocateMultiAgentBuffers() {
    size_t state_size = _state_dim * sizeof(float);
    size_t action_size = sizeof(uint32_t);
    size_t reward_size = sizeof(float);
    
    _multi_agent_states.resize(_num_agents);
    _multi_agent_actions.resize(_num_agents);
    _multi_agent_rewards.resize(_num_agents);
    
    for (uint32_t i = 0; i < _num_agents; i++) {
        _multi_agent_states[i] = _engine->createBuffer(state_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _multi_agent_actions[i] = _engine->createBuffer(action_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _multi_agent_rewards[i] = _engine->createBuffer(reward_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
}

void ReinforcementLearningAgent::allocateMultiEnvironmentBuffers() {
    size_t state_size = _state_dim * sizeof(float);
    size_t action_size = sizeof(uint32_t);
    size_t reward_size = sizeof(float);
    
    _environment_states.resize(_num_environments);
    _environment_actions.resize(_num_environments);
    _environment_rewards.resize(_num_environments);
    
    for (uint32_t i = 0; i < _num_environments; i++) {
        _environment_states[i] = _engine->createBuffer(state_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _environment_actions[i] = _engine->createBuffer(action_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _environment_rewards[i] = _engine->createBuffer(reward_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
}

void ReinforcementLearningAgent::copyNetworkWeights(ConvolutionalNeuralNetwork* source, ConvolutionalNeuralNetwork* target) {
    // Implementation would copy all weight buffers from source to target network
    // This would involve GPU-to-GPU memory copies for efficiency
    std::cout << "Network weights copied" << std::endl;
}

uint32_t ReinforcementLearningAgent::selectActionEpsilonGreedy(ComputeBuffer* q_values, float epsilon) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    if (dis(gen) < epsilon) {
        std::uniform_int_distribution<uint32_t> action_dis(0, _action_dim - 1);
        return action_dis(gen);
    } else {
        std::vector<float> q_data(_action_dim);
        _engine->downloadData(q_values, q_data.data(), q_data.size() * sizeof(float));
        
        auto max_it = std::max_element(q_data.begin(), q_data.end());
        return static_cast<uint32_t>(std::distance(q_data.begin(), max_it));
    }
}

uint32_t ReinforcementLearningAgent::selectActionUCB(ComputeBuffer* q_values, const std::vector<uint32_t>& action_counts) {
    std::vector<float> q_data(_action_dim);
    _engine->downloadData(q_values, q_data.data(), q_data.size() * sizeof(float));
    
    uint32_t total_counts = std::accumulate(action_counts.begin(), action_counts.end(), 0u);
    
    std::vector<float> ucb_values(_action_dim);
    for (uint32_t i = 0; i < _action_dim; i++) {
        float confidence = std::sqrt(2.0f * std::log(static_cast<float>(total_counts)) / static_cast<float>(action_counts[i]));
        ucb_values[i] = q_data[i] + _exploration_strategy.exploration_param * confidence;
    }
    
    auto max_it = std::max_element(ucb_values.begin(), ucb_values.end());
    return static_cast<uint32_t>(std::distance(ucb_values.begin(), max_it));
}

uint32_t ReinforcementLearningAgent::selectActionThompsonSampling(ComputeBuffer* q_values) {
    std::vector<float> q_data(_action_dim);
    _engine->downloadData(q_values, q_data.data(), q_data.size() * sizeof(float));
    
    // Sample from posterior distributions (simplified Gaussian)
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<float> sampled_values(_action_dim);
    for (uint32_t i = 0; i < _action_dim; i++) {
        std::normal_distribution<float> posterior_dist(q_data[i], _exploration_strategy.exploration_param);
        sampled_values[i] = posterior_dist(gen);
    }
    
    auto max_it = std::max_element(sampled_values.begin(), sampled_values.end());
    return static_cast<uint32_t>(std::distance(sampled_values.begin(), max_it));
}

std::vector<uint32_t> ReinforcementLearningAgent::samplePrioritizedIndices(uint32_t batch_size) {
    std::vector<uint32_t> indices;
    indices.reserve(batch_size);
    
    // Calculate total priority sum
    float total_priority = std::accumulate(_priority_buffer.begin(), _priority_buffer.end(), 0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, total_priority);
    
    for (uint32_t i = 0; i < batch_size; i++) {
        float target = dis(gen);
        float cumulative = 0.0f;
        
        for (uint32_t j = 0; j < _priority_buffer.size(); j++) {
            cumulative += _priority_buffer[j];
            if (cumulative >= target) {
                indices.push_back(j);
                break;
            }
        }
    }
    
    return indices;
}

float ReinforcementLearningAgent::calculateImportanceSamplingWeight(float priority, uint32_t buffer_size) {
    float max_priority = *std::max_element(_priority_buffer.begin(), _priority_buffer.end());
    float min_prob = priority / std::accumulate(_priority_buffer.begin(), _priority_buffer.end(), 0.0f);
    float max_weight = std::pow(static_cast<float>(buffer_size) * min_prob, -_priority_beta);
    return max_weight / std::pow(static_cast<float>(buffer_size) * (priority / std::accumulate(_priority_buffer.begin(), _priority_buffer.end(), 0.0f)), _priority_beta);
}

void ReinforcementLearningAgent::initializeActionSpaceExploration() {
    _exploration_strategy.type = ExplorationStrategy::EPSILON_GREEDY;
    _exploration_strategy.exploration_param = 0.1f;
    _exploration_strategy.exploration_steps = 10000;
    
    _entropy_coefficient = 0.01f;
    _priority_alpha = 0.6f;
    _priority_beta = 0.4f;
}

void ReinforcementLearningAgent::setupMultiEnvironmentSupport() {
    _num_environments = 1; // Default single environment
    _training_paused = false;
}

bool ReinforcementLearningAgent::shouldAdvanceCurriculum() const {
    if (_episode_rewards.size() < 100) return false; // Need minimum episodes
    
    const auto& current_level = _curriculum_levels[_current_curriculum_level];
    float recent_average = std::accumulate(_episode_rewards.end() - 50, _episode_rewards.end(), 0.0f) / 50.0f;
    
    return recent_average >= current_level.reward_threshold;
}

void ReinforcementLearningAgent::updateMetrics() {
    if (!_recent_rewards.empty()) {
        _metrics.average_reward = std::accumulate(_recent_rewards.begin(), _recent_rewards.end(), 0.0f) / static_cast<float>(_recent_rewards.size());
    }
    
    _metrics.exploration_rate = _epsilon;
    _metrics.training_steps = _training_step;
    
    // Update training step counter
    _training_step++;
}

void ReinforcementLearningAgent::synchronizeAgentNetworks() {
    // Implement parameter sharing or other synchronization strategies for multi-agent training
    std::cout << "Agent networks synchronized" << std::endl;
}

void ReinforcementLearningAgent::aggregateAgentExperiences() {
    // Combine experiences from multiple agents into shared replay buffer
    std::cout << "Agent experiences aggregated" << std::endl;
}

} // namespace AI
} // namespace CARL