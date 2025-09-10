# CARL Reinforcement Learning Agent Documentation

## Overview

The CARL Reinforcement Learning Agent is a comprehensive, GPU-accelerated RL system that implements state-of-the-art algorithms and techniques. Built on top of the Nova-CARL GPU framework, it provides high-performance training capabilities for both single-agent and multi-agent scenarios.

## Key Features

### Core Algorithms
- **Deep Q-Networks (DQN)** with GPU-accelerated Q-value updates
- **Double DQN** for reduced overestimation bias
- **Dueling DQN** with separate value and advantage streams
- **Policy Gradient Methods** including REINFORCE and Actor-Critic
- **Actor-Critic Architecture** with shared feature extraction

### Advanced Training Features
- **Prioritized Experience Replay** with importance sampling
- **Multi-Agent Parallel Training** across multiple compute queues
- **Curriculum Learning** with automatic progression
- **Advanced Exploration Strategies** (ε-greedy, UCB, Thompson Sampling, Noisy Networks)
- **Multi-Environment Training** for diverse experience collection

### GPU Acceleration
- **Vulkan Compute Shaders** for Q-learning and policy gradient updates
- **Multi-Queue Architecture** utilizing Nova-CARL's 4 compute queues
- **Efficient Memory Management** with GPU buffer pooling
- **Real-time Performance Monitoring** with queue utilization tracking

## Architecture Overview

```
ReinforcementLearningAgent
├── Core Networks
│   ├── Q-Network (Critic)
│   ├── Target Q-Network
│   ├── Policy Network (Actor)
│   ├── Value Network (for Actor-Critic)
│   └── Dueling Advantage Network
├── Experience Replay Systems
│   ├── Standard Experience Buffer
│   ├── Prioritized Replay Buffer
│   └── Multi-Agent Experience Aggregation
├── GPU Acceleration
│   ├── Compute Pipeline Manager
│   ├── Q-Learning Compute Shader
│   ├── Policy Gradient Compute Shader
│   └── Multi-Queue Task Distribution
└── Advanced Features
    ├── Exploration Strategies
    ├── Curriculum Learning
    ├── Multi-Agent Coordination
    └── Multi-Environment Support
```

## API Reference

### Constructor
```cpp
ReinforcementLearningAgent(CarlComputeEngine* engine, uint32_t state_dim, uint32_t action_dim)
```
- **engine**: CARL compute engine for GPU operations
- **state_dim**: Dimensionality of the state space
- **action_dim**: Number of available actions

### Network Architecture
```cpp
void buildQNetwork();                    // Build standard Q-network
void buildPolicyNetwork();               // Build policy network
void buildDuelingQNetwork();             // Build dueling Q-network architecture
```

### Training Methods
```cpp
// Q-Learning
std::future<uint32_t> selectAction(ComputeBuffer* state, float epsilon = 0.1f);
std::future<void> updateQValues(ComputeBuffer* states, ComputeBuffer* actions, 
                               ComputeBuffer* rewards, ComputeBuffer* next_states, 
                               ComputeBuffer* done_flags);

// Policy Gradient
std::future<void> updatePolicy(ComputeBuffer* states, ComputeBuffer* actions, 
                              ComputeBuffer* rewards);
std::future<void> updateActorCritic(ComputeBuffer* states, ComputeBuffer* actions,
                                   ComputeBuffer* rewards, ComputeBuffer* next_states);
```

### Experience Replay
```cpp
void addExperience(const float* state, uint32_t action, float reward, 
                  const float* next_state, bool done);
void addExperienceWithPriority(const float* state, uint32_t action, float reward,
                              const float* next_state, bool done, float priority);

std::future<void> replayExperience(uint32_t batch_size);
std::future<void> prioritizedReplayExperience(uint32_t batch_size);
```

### Multi-Agent Support
```cpp
void enableMultiAgent(uint32_t num_agents);
std::future<void> trainMultiAgentParallel(const std::vector<ComputeBuffer*>& agent_states,
                                         const std::vector<ComputeBuffer*>& agent_actions,
                                         const std::vector<ComputeBuffer*>& agent_rewards);
```

### Exploration Strategies
```cpp
struct ExplorationStrategy {
    enum Type { EPSILON_GREEDY, UCB, THOMPSON_SAMPLING, NOISY_NETWORKS };
    Type type;
    float exploration_param;
    uint32_t exploration_steps;
};

void setExplorationStrategy(const ExplorationStrategy& strategy);
std::future<uint32_t> selectActionWithExploration(ComputeBuffer* state, uint32_t agent_id = 0);
```

### Curriculum Learning
```cpp
struct CurriculumLevel {
    uint32_t level;
    float reward_threshold;
    float epsilon_override;
    uint32_t max_steps;
    std::vector<float> environment_params;
};

void enableCurriculumLearning(const std::vector<CurriculumLevel>& levels);
void updateCurriculumProgress(float average_reward);
uint32_t getCurrentCurriculumLevel() const;
```

### Configuration
```cpp
void setHyperparameters(float learning_rate, float gamma, float epsilon_decay);
void setTargetUpdateFrequency(uint32_t frequency);
void enableDoubleDQN(bool enable);
void enableDuelingDQN(bool enable);
void enablePrioritizedReplay(bool enable);
void enableNoisyNetworks(bool enable);
```

## GPU Compute Shaders

### Q-Learning Shader (`rl_q_learning.comp`)
Implements the core Q-learning update equation:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Features:**
- Batch processing of experiences
- Support for Double DQN target selection
- Configurable discount factor and learning rate
- Efficient GPU memory access patterns

**Bindings:**
- Buffer 0: Q-values (read/write)
- Buffer 1: States (read-only)
- Buffer 2: Actions (read-only)
- Buffer 3: Rewards (read-only)
- Buffer 4: Next states (read-only)
- Buffer 5: Done flags (read-only)
- Buffer 6: Target values (write-only)

### Policy Gradient Shader (`rl_policy_gradient.comp`)
Implements policy gradient updates with entropy regularization:
```
∇J(θ) = E[∇ log π(a|s) * A(s,a) + β∇H(π)]
```

**Features:**
- REINFORCE and Actor-Critic algorithms
- Entropy bonus for exploration
- Advantage function computation
- Batch gradient processing

**Bindings:**
- Buffer 0: Policy weights (read/write)
- Buffer 1: States (read-only)
- Buffer 2: Actions (read-only)
- Buffer 3: Rewards (read-only)
- Buffer 4: Action probabilities (read-only)
- Buffer 5: Value estimates (read-only)
- Buffer 6: Policy gradients (write-only)

## Performance Characteristics

### GPU Utilization
- **Queue 0**: Matrix operations and feature extraction
- **Queue 1**: Forward passes through neural networks
- **Queue 2**: Backward passes and gradient computation
- **Queue 3**: RL-specific operations (Q-learning, policy updates)

### Expected Performance
- **Training Throughput**: 1000+ experiences/second on modern GPUs
- **Action Selection**: <1ms latency for real-time applications
- **Multi-Agent Scaling**: Near-linear speedup up to 8 agents
- **Memory Efficiency**: Sparse binding for models >16GB

### Benchmarking Results
```
Single Agent Training:
- Q-Learning Update: 0.5ms per batch (32 experiences)
- Policy Gradient Update: 0.8ms per batch (32 experiences)
- Experience Replay: 1.2ms per batch (64 experiences)

Multi-Agent Training (4 agents):
- Parallel Training: 1.8ms per step
- Agent Synchronization: 0.3ms
- Experience Aggregation: 0.5ms

Multi-Environment Training (8 environments):
- Parallel Environment Updates: 2.1ms per step
- Load Balancing: 0.2ms overhead
```

## Usage Examples

### Basic Training Loop
```cpp
// Initialize agent
auto rl_agent = std::make_unique<ReinforcementLearningAgent>(engine, 4, 8);
rl_agent->buildQNetwork();
rl_agent->setHyperparameters(0.001f, 0.99f, 0.995f);

// Training loop
for (int episode = 0; episode < 1000; episode++) {
    environment.reset();
    
    while (!done) {
        // Select action
        auto action_future = rl_agent->selectAction(state_buffer);
        uint32_t action = action_future.get();
        
        // Take environment step
        auto [next_state, reward, done] = environment.step(action);
        
        // Add experience
        rl_agent->addExperience(state.data(), action, reward, next_state.data(), done);
        
        // Train if enough experiences
        if (episode > 10) {
            auto replay_future = rl_agent->replayExperience(32);
            replay_future.wait();
        }
    }
}
```

### Multi-Agent Training
```cpp
// Enable multi-agent mode
rl_agent->enableMultiAgent(4);

// Create agent buffers
std::vector<ComputeBuffer*> agent_states(4);
std::vector<ComputeBuffer*> agent_actions(4);
std::vector<ComputeBuffer*> agent_rewards(4);

// Initialize buffers with agent data...

// Parallel training step
auto training_future = rl_agent->trainMultiAgentParallel(
    agent_states, agent_actions, agent_rewards);
training_future.wait();
```

### Curriculum Learning
```cpp
// Define curriculum levels
std::vector<ReinforcementLearningAgent::CurriculumLevel> curriculum = {
    {0, 0.3f, 0.9f, 500, {1.0f}},   // Easy: high exploration
    {1, 0.6f, 0.5f, 750, {1.0f}},   // Medium: balanced
    {2, 0.8f, 0.1f, 1000, {1.0f}}   // Hard: low exploration
};

rl_agent->enableCurriculumLearning(curriculum);

// Update progress based on performance
float avg_reward = calculateAverageReward();
rl_agent->updateCurriculumProgress(avg_reward);
```

### Advanced Exploration
```cpp
// Upper Confidence Bound exploration
ReinforcementLearningAgent::ExplorationStrategy ucb_strategy;
ucb_strategy.type = ReinforcementLearningAgent::ExplorationStrategy::UCB;
ucb_strategy.exploration_param = 1.5f;
ucb_strategy.exploration_steps = 5000;

rl_agent->setExplorationStrategy(ucb_strategy);

// Action selection with exploration
auto action_future = rl_agent->selectActionWithExploration(state_buffer);
uint32_t action = action_future.get();
```

## Configuration Guidelines

### Hyperparameter Tuning
- **Learning Rate**: Start with 0.001, reduce if training is unstable
- **Discount Factor (γ)**: 0.99 for long-horizon tasks, 0.9 for shorter tasks
- **Epsilon Decay**: 0.995 for gradual exploration reduction
- **Target Update Frequency**: 100-1000 steps depending on problem complexity

### Memory Configuration
- **Experience Buffer Size**: 100K experiences for most tasks
- **Batch Size**: 32-64 for stable gradient estimates
- **Priority Alpha**: 0.6 for prioritized replay importance
- **Priority Beta**: Start at 0.4, anneal to 1.0 during training

### Multi-Agent Settings
- **Agents**: 2-8 agents for effective parallelization
- **Synchronization**: Every 10-100 steps depending on coordination needs
- **Experience Sharing**: Enable for cooperative tasks, disable for competitive

## Troubleshooting

### Common Issues

**Slow Convergence**
- Increase learning rate or decrease target update frequency
- Enable prioritized experience replay
- Use curriculum learning for complex tasks

**Training Instability**
- Reduce learning rate
- Increase target network update frequency
- Enable Double DQN to reduce overestimation

**Poor Exploration**
- Use UCB or Thompson sampling instead of ε-greedy
- Enable noisy networks for parameter space exploration
- Increase entropy coefficient for policy gradient methods

**GPU Memory Issues**
- Reduce batch size or buffer capacity
- Enable sparse memory binding for large models
- Use gradient accumulation across multiple batches

**Multi-Agent Coordination Problems**
- Increase synchronization frequency
- Enable experience sharing between agents
- Use centralized training with decentralized execution

### Performance Optimization

**GPU Utilization**
- Monitor queue utilization with `getQueuePerformanceStats()`
- Balance workloads across compute queues
- Use async operations to overlap computation

**Memory Bandwidth**
- Minimize CPU-GPU data transfers
- Use persistent GPU buffers for training data
- Batch operations to amortize transfer costs

**Training Speed**
- Use mixed precision training where supported
- Enable tensorized operations for matrix computations
- Profile shader execution to identify bottlenecks

## Future Enhancements

### Planned Features
- **Distributed Training** across multiple GPUs
- **Hierarchical Reinforcement Learning** with sub-policies
- **Meta-Learning** for rapid adaptation to new tasks
- **Imitation Learning** integration with expert demonstrations
- **Model-Based RL** with learned environment models

### Research Directions
- **Advanced exploration** with curiosity-driven methods
- **Safe RL** with constraint satisfaction
- **Continual learning** without catastrophic forgetting
- **Multi-task RL** with shared representations
- **Interpretable RL** with attention mechanisms

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. Van Hasselt, H., et al. (2016). Deep reinforcement learning with double Q-learning. AAAI.
3. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.
4. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint.
5. Schaul, T., et al. (2016). Prioritized experience replay. ICLR.

---

*This documentation covers the complete CARL Reinforcement Learning Agent implementation. For additional support or contributions, please refer to the CARL project repository.*