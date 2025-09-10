#include "../ai_components/neural_network_models.h"
#include "../ai_components/carl_compute_engine.h"
#include "../nova_integration/carl_gpu.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

/**
 * Reinforcement Learning Training Example
 * 
 * Demonstrates the advanced RL agent capabilities:
 * - GPU-accelerated Q-learning and Policy Gradient algorithms
 * - Multi-agent parallel training
 * - Curriculum learning progression
 * - Advanced exploration strategies
 * - Real-time performance monitoring
 */

using namespace CARL::AI;

class SimpleEnvironment {
public:
    SimpleEnvironment(uint32_t state_dim, uint32_t action_dim)
        : _state_dim(state_dim), _action_dim(action_dim), _episode_step(0), _max_steps(1000) {
        
        // Initialize random environment state
        std::random_device rd;
        _gen.seed(rd());
        _state_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
        
        reset();
    }
    
    void reset() {
        _current_state.resize(_state_dim);
        for (uint32_t i = 0; i < _state_dim; i++) {
            _current_state[i] = _state_dist(_gen);
        }
        _episode_step = 0;
    }
    
    struct StepResult {
        std::vector<float> next_state;
        float reward;
        bool done;
    };
    
    StepResult step(uint32_t action) {
        _episode_step++;
        
        StepResult result;
        result.next_state.resize(_state_dim);
        
        // Simple reward function: closer to center state = higher reward
        float distance_to_center = 0.0f;
        for (uint32_t i = 0; i < _state_dim; i++) {
            distance_to_center += _current_state[i] * _current_state[i];
        }
        
        result.reward = std::exp(-distance_to_center); // Gaussian reward centered at origin
        
        // Action affects state transition
        float action_effect = static_cast<float>(action) / static_cast<float>(_action_dim) - 0.5f;
        for (uint32_t i = 0; i < _state_dim; i++) {
            result.next_state[i] = _current_state[i] + 0.1f * action_effect + 0.05f * _state_dist(_gen);
            result.next_state[i] = std::clamp(result.next_state[i], -2.0f, 2.0f);
        }
        
        result.done = (_episode_step >= _max_steps) || (distance_to_center < 0.01f);
        
        _current_state = result.next_state;
        return result;
    }
    
    const std::vector<float>& getCurrentState() const { return _current_state; }
    
private:
    uint32_t _state_dim;
    uint32_t _action_dim;
    uint32_t _episode_step;
    uint32_t _max_steps;
    
    std::vector<float> _current_state;
    std::mt19937 _gen;
    std::uniform_real_distribution<float> _state_dist;
};

int main() {
    std::cout << "=== CARL Reinforcement Learning Training Example ===" << std::endl;
    
    // Initialize Nova-CARL GPU framework
    auto nova_core = std::make_unique<NovaCore>();
    if (!nova_core->initialize()) {
        std::cerr << "Failed to initialize Nova core" << std::endl;
        return -1;
    }
    
    // Initialize CARL compute engine
    auto compute_engine = std::make_unique<CarlComputeEngine>(nova_core.get());
    if (!compute_engine->initialize()) {
        std::cerr << "Failed to initialize CARL compute engine" << std::endl;
        return -1;
    }
    
    // Environment configuration
    constexpr uint32_t STATE_DIM = 4;
    constexpr uint32_t ACTION_DIM = 8;
    constexpr uint32_t NUM_EPISODES = 1000;
    constexpr uint32_t BATCH_SIZE = 32;
    
    // Create RL agent with enhanced features
    auto rl_agent = std::make_unique<ReinforcementLearningAgent>(
        compute_engine.get(), STATE_DIM, ACTION_DIM);
    
    // Configure advanced RL features
    std::cout << "Configuring advanced RL features..." << std::endl;
    
    // Enable Double DQN and Dueling DQN
    rl_agent->enableDoubleDQN(true);
    rl_agent->buildDuelingQNetwork();
    
    // Enable prioritized experience replay
    rl_agent->enablePrioritizedReplay(true);
    
    // Setup exploration strategy (UCB for better exploration)
    ReinforcementLearningAgent::ExplorationStrategy exploration;
    exploration.type = ReinforcementLearningAgent::ExplorationStrategy::UCB;
    exploration.exploration_param = 1.5f;
    exploration.exploration_steps = 5000;
    rl_agent->setExplorationStrategy(exploration);
    
    // Setup curriculum learning
    std::vector<ReinforcementLearningAgent::CurriculumLevel> curriculum = {
        {0, 0.3f, 0.9f, 500, {1.0f}},   // Easy: high exploration, low target
        {1, 0.6f, 0.5f, 750, {1.0f}},   // Medium: moderate exploration
        {2, 0.8f, 0.1f, 1000, {1.0f}}   // Hard: low exploration, high target
    };
    rl_agent->enableCurriculumLearning(curriculum);
    
    // Build neural networks
    std::cout << "Building Q-Network and Policy Network..." << std::endl;
    rl_agent->buildQNetwork();
    rl_agent->buildPolicyNetwork();
    
    // Configure hyperparameters
    rl_agent->setHyperparameters(0.001f, 0.99f, 0.995f);
    rl_agent->setTargetUpdateFrequency(100);
    rl_agent->setAlgorithm(true, true); // Use both DQN and policy gradient
    
    // Create training environment
    auto environment = std::make_unique<SimpleEnvironment>(STATE_DIM, ACTION_DIM);
    
    // Create GPU buffers for state representation
    ComputeBuffer* state_buffer = compute_engine->createBuffer(
        STATE_DIM * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Training loop
    std::cout << "Starting training loop..." << std::endl;
    std::vector<float> episode_rewards;
    auto training_start = std::chrono::high_resolution_clock::now();
    
    for (uint32_t episode = 0; episode < NUM_EPISODES; episode++) {
        environment->reset();
        float episode_reward = 0.0f;
        uint32_t step_count = 0;
        
        while (true) {
            step_count++;
            
            // Upload current state to GPU
            const auto& current_state = environment->getCurrentState();
            compute_engine->uploadData(state_buffer, current_state.data(), 
                                     current_state.size() * sizeof(float));
            
            // Select action using advanced exploration strategy
            auto action_future = rl_agent->selectActionWithExploration(state_buffer);
            uint32_t action = action_future.get();
            
            // Take environment step
            auto step_result = environment->step(action);
            episode_reward += step_result.reward;
            
            // Add experience to replay buffer with priority
            float priority = std::abs(step_result.reward) + 0.01f; // Priority based on reward magnitude
            rl_agent->addExperienceWithPriority(
                current_state.data(), action, step_result.reward,
                step_result.next_state.data(), step_result.done, priority);
            
            // Perform experience replay training
            if (episode > 10 && step_count % 4 == 0) {
                if (rl_agent->getCurrentCurriculumLevel() < 2) {
                    // Use regular experience replay for early curriculum levels
                    auto replay_future = rl_agent->replayExperience(BATCH_SIZE);
                    replay_future.wait();
                } else {
                    // Use prioritized experience replay for advanced curriculum
                    auto prioritized_replay_future = rl_agent->prioritizedReplayExperience(BATCH_SIZE);
                    prioritized_replay_future.wait();
                }
            }
            
            if (step_result.done) {
                break;
            }
        }
        
        episode_rewards.push_back(episode_reward);
        
        // Update curriculum learning progress
        if (episode_rewards.size() >= 50) {
            float avg_reward = std::accumulate(episode_rewards.end() - 50, episode_rewards.end(), 0.0f) / 50.0f;
            rl_agent->updateCurriculumProgress(avg_reward);
        }
        
        // Print progress
        if (episode % 100 == 0) {
            auto metrics = rl_agent->getTrainingMetrics();
            std::cout << "Episode " << episode 
                      << ", Reward: " << episode_reward 
                      << ", Steps: " << step_count
                      << ", Avg Reward: " << metrics.average_reward
                      << ", Exploration Rate: " << metrics.exploration_rate
                      << ", Curriculum Level: " << rl_agent->getCurrentCurriculumLevel()
                      << std::endl;
        }
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);
    
    // Final performance metrics
    std::cout << "\n=== Training Completed ===" << std::endl;
    std::cout << "Total training time: " << training_duration.count() << " ms" << std::endl;
    
    auto final_metrics = rl_agent->getTrainingMetrics();
    std::cout << "Final Metrics:" << std::endl;
    std::cout << "  Average Reward: " << final_metrics.average_reward << std::endl;
    std::cout << "  Exploration Rate: " << final_metrics.exploration_rate << std::endl;
    std::cout << "  Training Steps: " << final_metrics.training_steps << std::endl;
    std::cout << "  Final Curriculum Level: " << rl_agent->getCurrentCurriculumLevel() << std::endl;
    
    // Demonstrate multi-agent training capabilities
    std::cout << "\n=== Multi-Agent Training Demo ===" << std::endl;
    
    constexpr uint32_t NUM_AGENTS = 4;
    rl_agent->enableMultiAgent(NUM_AGENTS);
    
    // Create multiple environment states for parallel training
    std::vector<ComputeBuffer*> agent_states(NUM_AGENTS);
    std::vector<ComputeBuffer*> agent_actions(NUM_AGENTS);
    std::vector<ComputeBuffer*> agent_rewards(NUM_AGENTS);
    
    for (uint32_t i = 0; i < NUM_AGENTS; i++) {
        agent_states[i] = compute_engine->createBuffer(STATE_DIM * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        agent_actions[i] = compute_engine->createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        agent_rewards[i] = compute_engine->createBuffer(sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Upload random training data for demo
        std::vector<float> random_state(STATE_DIM);
        for (uint32_t j = 0; j < STATE_DIM; j++) {
            random_state[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        compute_engine->uploadData(agent_states[i], random_state.data(), random_state.size() * sizeof(float));
        
        uint32_t random_action = rand() % ACTION_DIM;
        compute_engine->uploadData(agent_actions[i], &random_action, sizeof(uint32_t));
        
        float random_reward = static_cast<float>(rand()) / RAND_MAX;
        compute_engine->uploadData(agent_rewards[i], &random_reward, sizeof(float));
    }
    
    // Perform multi-agent parallel training
    auto multi_agent_start = std::chrono::high_resolution_clock::now();
    auto multi_agent_future = rl_agent->trainMultiAgentParallel(agent_states, agent_actions, agent_rewards);
    multi_agent_future.wait();
    auto multi_agent_end = std::chrono::high_resolution_clock::now();
    
    auto multi_agent_duration = std::chrono::duration_cast<std::chrono::microseconds>(multi_agent_end - multi_agent_start);
    std::cout << "Multi-agent parallel training completed in " << multi_agent_duration.count() << " μs" << std::endl;
    
    // Clean up buffers
    compute_engine->destroyBuffer(state_buffer);
    for (uint32_t i = 0; i < NUM_AGENTS; i++) {
        compute_engine->destroyBuffer(agent_states[i]);
        compute_engine->destroyBuffer(agent_actions[i]);
        compute_engine->destroyBuffer(agent_rewards[i]);
    }
    
    // Display GPU performance statistics
    std::cout << "\n=== GPU Performance Statistics ===" << std::endl;
    compute_engine->printPerformanceReport();
    
    std::cout << "\n=== RL Training Example Complete ===" << std::endl;
    return 0;
}