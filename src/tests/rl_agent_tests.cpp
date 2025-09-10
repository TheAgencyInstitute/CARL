#include "../ai_components/neural_network_models.h"
#include "../ai_components/carl_compute_engine.h"
#include "../nova_integration/carl_gpu.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <random>
#include <chrono>

/**
 * Comprehensive Test Suite for Reinforcement Learning Agent
 * 
 * Tests all advanced RL features:
 * - GPU-accelerated Q-learning and Policy Gradient
 * - Multi-agent parallel training
 * - Curriculum learning progression
 * - Advanced exploration strategies
 * - Experience replay systems
 */

using namespace CARL::AI;

class RLAgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Nova-CARL GPU framework
        nova_core = std::make_unique<NovaCore>();
        ASSERT_TRUE(nova_core->initialize()) << "Failed to initialize Nova core";
        
        // Initialize CARL compute engine
        compute_engine = std::make_unique<CarlComputeEngine>(nova_core.get());
        ASSERT_TRUE(compute_engine->initialize()) << "Failed to initialize CARL compute engine";
        
        // Create RL agent
        constexpr uint32_t STATE_DIM = 4;
        constexpr uint32_t ACTION_DIM = 8;
        
        rl_agent = std::make_unique<ReinforcementLearningAgent>(
            compute_engine.get(), STATE_DIM, ACTION_DIM);
        
        // Create test buffers
        state_buffer = compute_engine->createBuffer(
            STATE_DIM * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        action_buffer = compute_engine->createBuffer(
            sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        reward_buffer = compute_engine->createBuffer(
            sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    
    void TearDown() override {
        if (state_buffer) compute_engine->destroyBuffer(state_buffer);
        if (action_buffer) compute_engine->destroyBuffer(action_buffer);
        if (reward_buffer) compute_engine->destroyBuffer(reward_buffer);
        
        rl_agent.reset();
        compute_engine->shutdown();
        compute_engine.reset();
        nova_core.reset();
    }
    
    std::unique_ptr<NovaCore> nova_core;
    std::unique_ptr<CarlComputeEngine> compute_engine;
    std::unique_ptr<ReinforcementLearningAgent> rl_agent;
    
    ComputeBuffer* state_buffer;
    ComputeBuffer* action_buffer;
    ComputeBuffer* reward_buffer;
};

TEST_F(RLAgentTest, NetworkInitialization) {
    // Test Q-Network building
    ASSERT_NO_THROW(rl_agent->buildQNetwork());
    
    // Test Policy Network building
    ASSERT_NO_THROW(rl_agent->buildPolicyNetwork());
    
    // Test Dueling Q-Network building
    ASSERT_NO_THROW(rl_agent->buildDuelingQNetwork());
    
    std::cout << "Network initialization tests passed" << std::endl;
}

TEST_F(RLAgentTest, HyperparameterConfiguration) {
    // Test hyperparameter setting
    float learning_rate = 0.001f;
    float gamma = 0.99f;
    float epsilon_decay = 0.995f;
    
    ASSERT_NO_THROW(rl_agent->setHyperparameters(learning_rate, gamma, epsilon_decay));
    
    // Test algorithm configuration
    ASSERT_NO_THROW(rl_agent->setAlgorithm(true, true));
    
    // Test advanced features
    ASSERT_NO_THROW(rl_agent->enableDoubleDQN(true));
    ASSERT_NO_THROW(rl_agent->enableDuelingDQN(true));
    ASSERT_NO_THROW(rl_agent->enablePrioritizedReplay(true));
    ASSERT_NO_THROW(rl_agent->enableNoisyNetworks(true));
    
    std::cout << "Hyperparameter configuration tests passed" << std::endl;
}

TEST_F(RLAgentTest, BasicTraining) {
    // Build networks for training
    rl_agent->buildQNetwork();
    rl_agent->buildPolicyNetwork();
    
    // Create test data
    std::vector<float> test_state = {0.5f, -0.3f, 0.8f, -0.1f};
    uint32_t test_action = 3;
    float test_reward = 1.0f;
    std::vector<float> test_next_state = {0.4f, -0.2f, 0.7f, 0.0f};
    bool done = false;
    
    // Test experience addition
    ASSERT_NO_THROW(rl_agent->addExperience(
        test_state.data(), test_action, test_reward, test_next_state.data(), done));
    
    // Add multiple experiences for batch training
    for (int i = 0; i < 100; i++) {
        std::vector<float> random_state(4);
        std::vector<float> random_next_state(4);
        for (int j = 0; j < 4; j++) {
            random_state[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            random_next_state[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        uint32_t random_action = rand() % 8;
        float random_reward = static_cast<float>(rand()) / RAND_MAX;
        
        rl_agent->addExperience(random_state.data(), random_action, random_reward, 
                               random_next_state.data(), rand() % 2 == 0);
    }
    
    // Test experience replay
    auto replay_future = rl_agent->replayExperience(32);
    ASSERT_NO_THROW(replay_future.wait());
    
    std::cout << "Basic training tests passed" << std::endl;
}

TEST_F(RLAgentTest, ActionSelection) {
    // Build Q-Network for action selection
    rl_agent->buildQNetwork();
    
    // Upload test state
    std::vector<float> test_state = {0.1f, 0.2f, 0.3f, 0.4f};
    compute_engine->uploadData(state_buffer, test_state.data(), test_state.size() * sizeof(float));
    
    // Test epsilon-greedy action selection
    auto action_future = rl_agent->selectAction(state_buffer, 0.1f);
    uint32_t selected_action = action_future.get();
    
    ASSERT_LT(selected_action, 8u) << "Selected action should be within action space";
    
    std::cout << "Action selection test passed, selected action: " << selected_action << std::endl;
}

TEST_F(RLAgentTest, ExplorationStrategies) {
    rl_agent->buildQNetwork();
    
    // Test different exploration strategies
    ReinforcementLearningAgent::ExplorationStrategy strategies[] = {
        {ReinforcementLearningAgent::ExplorationStrategy::EPSILON_GREEDY, 0.1f, 1000},
        {ReinforcementLearningAgent::ExplorationStrategy::UCB, 1.5f, 1000},
        {ReinforcementLearningAgent::ExplorationStrategy::THOMPSON_SAMPLING, 0.2f, 1000},
        {ReinforcementLearningAgent::ExplorationStrategy::NOISY_NETWORKS, 0.3f, 1000}
    };
    
    std::vector<float> test_state = {0.1f, 0.2f, 0.3f, 0.4f};
    compute_engine->uploadData(state_buffer, test_state.data(), test_state.size() * sizeof(float));
    
    for (const auto& strategy : strategies) {
        ASSERT_NO_THROW(rl_agent->setExplorationStrategy(strategy));
        
        auto action_future = rl_agent->selectActionWithExploration(state_buffer, 0);
        uint32_t selected_action = action_future.get();
        
        ASSERT_LT(selected_action, 8u) << "Action should be within action space for strategy " 
                                       << static_cast<int>(strategy.type);
    }
    
    std::cout << "Exploration strategy tests passed" << std::endl;
}

TEST_F(RLAgentTest, PrioritizedExperienceReplay) {
    rl_agent->buildQNetwork();
    rl_agent->enablePrioritizedReplay(true);
    
    // Add experiences with different priorities
    std::vector<float> high_priority_state = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> low_priority_state = {0.1f, 0.1f, 0.1f, 0.1f};
    std::vector<float> next_state = {0.5f, 0.5f, 0.5f, 0.5f};
    
    // High priority experience (large reward)
    ASSERT_NO_THROW(rl_agent->addExperienceWithPriority(
        high_priority_state.data(), 1, 10.0f, next_state.data(), false, 1.0f));
    
    // Low priority experience (small reward)  
    ASSERT_NO_THROW(rl_agent->addExperienceWithPriority(
        low_priority_state.data(), 2, 0.1f, next_state.data(), false, 0.1f));
    
    // Add more experiences for batch training
    for (int i = 0; i < 50; i++) {
        std::vector<float> random_state(4);
        std::vector<float> random_next_state(4);
        for (int j = 0; j < 4; j++) {
            random_state[j] = static_cast<float>(rand()) / RAND_MAX;
            random_next_state[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        float priority = static_cast<float>(rand()) / RAND_MAX;
        rl_agent->addExperienceWithPriority(random_state.data(), rand() % 8, 
                                           static_cast<float>(rand()) / RAND_MAX, 
                                           random_next_state.data(), false, priority);
    }
    
    // Test prioritized experience replay
    auto prioritized_replay_future = rl_agent->prioritizedReplayExperience(16);
    ASSERT_NO_THROW(prioritized_replay_future.wait());
    
    std::cout << "Prioritized experience replay tests passed" << std::endl;
}

TEST_F(RLAgentTest, CurriculumLearning) {
    // Setup curriculum learning levels
    std::vector<ReinforcementLearningAgent::CurriculumLevel> curriculum = {
        {0, 0.5f, 0.8f, 500, {1.0f}},   // Easy level
        {1, 0.7f, 0.4f, 750, {1.0f}},   // Medium level
        {2, 0.9f, 0.1f, 1000, {1.0f}}   // Hard level
    };
    
    ASSERT_NO_THROW(rl_agent->enableCurriculumLearning(curriculum));
    ASSERT_EQ(rl_agent->getCurrentCurriculumLevel(), 0u);
    
    // Simulate progress through curriculum levels
    float good_performance = 0.8f;  // Above threshold for level 0
    ASSERT_NO_THROW(rl_agent->updateCurriculumProgress(good_performance));
    
    std::cout << "Curriculum learning tests passed, current level: " 
              << rl_agent->getCurrentCurriculumLevel() << std::endl;
}

TEST_F(RLAgentTest, MultiAgentTraining) {
    constexpr uint32_t NUM_AGENTS = 4;
    
    // Enable multi-agent mode
    ASSERT_NO_THROW(rl_agent->enableMultiAgent(NUM_AGENTS));
    
    // Create buffers for multi-agent training
    std::vector<ComputeBuffer*> agent_states(NUM_AGENTS);
    std::vector<ComputeBuffer*> agent_actions(NUM_AGENTS);
    std::vector<ComputeBuffer*> agent_rewards(NUM_AGENTS);
    
    for (uint32_t i = 0; i < NUM_AGENTS; i++) {
        agent_states[i] = compute_engine->createBuffer(4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        agent_actions[i] = compute_engine->createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        agent_rewards[i] = compute_engine->createBuffer(sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Upload test data
        std::vector<float> test_state = {
            static_cast<float>(i) * 0.1f, 
            static_cast<float>(i) * 0.2f, 
            static_cast<float>(i) * 0.3f, 
            static_cast<float>(i) * 0.4f
        };
        compute_engine->uploadData(agent_states[i], test_state.data(), test_state.size() * sizeof(float));
        
        uint32_t test_action = i % 8;
        compute_engine->uploadData(agent_actions[i], &test_action, sizeof(uint32_t));
        
        float test_reward = static_cast<float>(i) * 0.25f;
        compute_engine->uploadData(agent_rewards[i], &test_reward, sizeof(float));
    }
    
    // Test multi-agent parallel training
    auto multi_agent_future = rl_agent->trainMultiAgentParallel(agent_states, agent_actions, agent_rewards);
    ASSERT_NO_THROW(multi_agent_future.wait());
    
    // Clean up agent buffers
    for (uint32_t i = 0; i < NUM_AGENTS; i++) {
        compute_engine->destroyBuffer(agent_states[i]);
        compute_engine->destroyBuffer(agent_actions[i]);
        compute_engine->destroyBuffer(agent_rewards[i]);
    }
    
    std::cout << "Multi-agent training tests passed" << std::endl;
}

TEST_F(RLAgentTest, MultiEnvironmentTraining) {
    constexpr uint32_t NUM_ENVIRONMENTS = 3;
    
    rl_agent->buildQNetwork();
    
    // Setup multi-environment training
    ASSERT_NO_THROW(rl_agent->setupMultiEnvironment(NUM_ENVIRONMENTS));
    
    // Create environment buffers
    std::vector<ComputeBuffer*> env_states(NUM_ENVIRONMENTS);
    std::vector<ComputeBuffer*> env_actions(NUM_ENVIRONMENTS);
    std::vector<ComputeBuffer*> env_rewards(NUM_ENVIRONMENTS);
    
    for (uint32_t i = 0; i < NUM_ENVIRONMENTS; i++) {
        env_states[i] = compute_engine->createBuffer(4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        env_actions[i] = compute_engine->createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        env_rewards[i] = compute_engine->createBuffer(sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Upload diverse environment data
        std::vector<float> env_state = {
            static_cast<float>(i) * 0.3f - 0.5f,
            static_cast<float>(i) * 0.2f - 0.3f,
            static_cast<float>(i) * 0.4f - 0.6f,
            static_cast<float>(i) * 0.1f - 0.1f
        };
        compute_engine->uploadData(env_states[i], env_state.data(), env_state.size() * sizeof(float));
        
        uint32_t env_action = (i * 2) % 8;
        compute_engine->uploadData(env_actions[i], &env_action, sizeof(uint32_t));
        
        float env_reward = 1.0f - static_cast<float>(i) * 0.2f;
        compute_engine->uploadData(env_rewards[i], &env_reward, sizeof(float));
    }
    
    // Test parallel environment training
    auto env_training_future = rl_agent->trainParallelEnvironments(env_states, env_actions, env_rewards);
    ASSERT_NO_THROW(env_training_future.wait());
    
    // Clean up environment buffers
    for (uint32_t i = 0; i < NUM_ENVIRONMENTS; i++) {
        compute_engine->destroyBuffer(env_states[i]);
        compute_engine->destroyBuffer(env_actions[i]);
        compute_engine->destroyBuffer(env_rewards[i]);
    }
    
    std::cout << "Multi-environment training tests passed" << std::endl;
}

TEST_F(RLAgentTest, TrainingControlAndMetrics) {
    // Test training control
    ASSERT_FALSE(rl_agent->isTraining());  // Should start paused
    
    rl_agent->resumeTraining();
    ASSERT_TRUE(rl_agent->isTraining());
    
    rl_agent->pauseTraining();
    ASSERT_FALSE(rl_agent->isTraining());
    
    // Test metrics retrieval
    auto metrics = rl_agent->getTrainingMetrics();
    ASSERT_GE(metrics.exploration_rate, 0.0f);
    ASSERT_LE(metrics.exploration_rate, 1.0f);
    ASSERT_GE(metrics.training_steps, 0u);
    
    std::cout << "Training control and metrics tests passed" << std::endl;
    std::cout << "  Exploration rate: " << metrics.exploration_rate << std::endl;
    std::cout << "  Training steps: " << metrics.training_steps << std::endl;
}

TEST_F(RLAgentTest, ActorCriticTraining) {
    // Build both Q-network (critic) and policy network (actor)
    rl_agent->buildQNetwork();
    rl_agent->buildPolicyNetwork();
    
    // Create test data
    std::vector<float> states_data = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<uint32_t> actions_data = {2};
    std::vector<float> rewards_data = {1.5f};
    std::vector<float> next_states_data = {0.2f, 0.3f, 0.4f, 0.5f};
    
    // Create and upload buffers
    ComputeBuffer* states_buffer = compute_engine->createBuffer(states_data.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ComputeBuffer* actions_buffer = compute_engine->createBuffer(actions_data.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ComputeBuffer* rewards_buffer = compute_engine->createBuffer(rewards_data.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ComputeBuffer* next_states_buffer = compute_engine->createBuffer(next_states_data.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    compute_engine->uploadData(states_buffer, states_data.data(), states_data.size() * sizeof(float));
    compute_engine->uploadData(actions_buffer, actions_data.data(), actions_data.size() * sizeof(uint32_t));
    compute_engine->uploadData(rewards_buffer, rewards_data.data(), rewards_data.size() * sizeof(float));
    compute_engine->uploadData(next_states_buffer, next_states_data.data(), next_states_data.size() * sizeof(float));
    
    // Test Actor-Critic update
    auto actor_critic_future = rl_agent->updateActorCritic(states_buffer, actions_buffer, rewards_buffer, next_states_buffer);
    ASSERT_NO_THROW(actor_critic_future.wait());
    
    // Clean up
    compute_engine->destroyBuffer(states_buffer);
    compute_engine->destroyBuffer(actions_buffer);
    compute_engine->destroyBuffer(rewards_buffer);
    compute_engine->destroyBuffer(next_states_buffer);
    
    std::cout << "Actor-Critic training tests passed" << std::endl;
}

// Performance benchmark test
TEST_F(RLAgentTest, PerformanceBenchmark) {
    rl_agent->buildQNetwork();
    rl_agent->buildPolicyNetwork();
    
    // Add training experiences
    for (int i = 0; i < 1000; i++) {
        std::vector<float> random_state(4);
        std::vector<float> random_next_state(4);
        for (int j = 0; j < 4; j++) {
            random_state[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            random_next_state[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        uint32_t random_action = rand() % 8;
        float random_reward = static_cast<float>(rand()) / RAND_MAX;
        
        rl_agent->addExperience(random_state.data(), random_action, random_reward, 
                               random_next_state.data(), rand() % 2 == 0);
    }
    
    // Benchmark training speed
    auto start_time = std::chrono::high_resolution_clock::now();
    
    constexpr int NUM_TRAINING_STEPS = 100;
    for (int i = 0; i < NUM_TRAINING_STEPS; i++) {
        auto replay_future = rl_agent->replayExperience(32);
        replay_future.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Performance benchmark completed:" << std::endl;
    std::cout << "  " << NUM_TRAINING_STEPS << " training steps in " << duration.count() << " ms" << std::endl;
    std::cout << "  Average: " << (static_cast<float>(duration.count()) / NUM_TRAINING_STEPS) << " ms per step" << std::endl;
    
    // Performance should be reasonable (less than 100ms per training step)
    ASSERT_LT(duration.count() / NUM_TRAINING_STEPS, 100) 
        << "Training performance is too slow: " << (duration.count() / NUM_TRAINING_STEPS) << " ms per step";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Starting CARL Reinforcement Learning Agent Test Suite..." << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "CARL RL Agent Test Suite completed with result: " << result << std::endl;
    
    return result;
}