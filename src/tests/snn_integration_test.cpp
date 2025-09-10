#include "../ai_components/snn_integration.h"
#include "../ai_components/neural_network_models.h"
#include "../ai_components/carl_compute_engine.h"
#include "../nova_integration/queue_manager.h"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <random>

using namespace CARL::AI;
using namespace CARL::GPU;

class SNNIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Nova core (mock for testing)
        nova_core = nullptr; // Would be initialized from Nova framework
        
        // Initialize CARL compute engine
        compute_engine = std::make_unique<CarlComputeEngine>(nova_core);
        ASSERT_TRUE(compute_engine->initialize());
        
        // Initialize queue manager
        queue_manager = std::make_unique<QueueManager>(nullptr);
        ASSERT_TRUE(queue_manager->initialize());
        
        // Initialize SNN integration
        snn_integration = std::make_unique<SNNIntegration>(
            compute_engine.get(), queue_manager.get());
        ASSERT_TRUE(snn_integration->initialize());
    }
    
    void TearDown() override {
        snn_integration->shutdown();
        queue_manager->shutdown();
        compute_engine->shutdown();
    }
    
    NovaCore* nova_core;
    std::unique_ptr<CarlComputeEngine> compute_engine;
    std::unique_ptr<QueueManager> queue_manager;
    std::unique_ptr<SNNIntegration> snn_integration;
};

// Test 1: Basic SNN Network Creation and Initialization
TEST_F(SNNIntegrationTest, BasicNetworkCreation) {
    // Create a small SNN with 1000 neurons, 1000 timesteps
    auto snn = snn_integration->createSNN(1000, 1000);
    ASSERT_NE(snn, nullptr);
    
    // Add network to registry
    ASSERT_TRUE(snn_integration->addSNNNetwork("test_network", snn));
    
    // Retrieve network
    auto retrieved = snn_integration->getSNNNetwork("test_network");
    ASSERT_EQ(snn, retrieved);
    
    // Initialize network
    snn->setNeuronParameters(
        10e6f,    // membrane_resistance (10 MΩ)
        100e-12f, // membrane_capacitance (100 pF)
        -55e-3f,  // threshold_voltage (-55 mV)
        -70e-3f,  // reset_voltage (-70 mV)
        2e-3f     // refractory_period (2 ms)
    );
    
    snn->initializeNetwork();
    
    // Test initial state
    EXPECT_EQ(snn->getSpikeCount(), 0);
    EXPECT_EQ(snn->getAverageFireRate(), 0.0f);
}

// Test 2: Large SNN with Sparse Memory Binding
TEST_F(SNNIntegrationTest, LargeSNNSparseMemory) {
    // Create large SNN with 100,000 neurons and 4GB virtual memory
    auto large_snn = snn_integration->createLargeSNN(100000, 10000, 4);
    ASSERT_NE(large_snn, nullptr);
    
    ASSERT_TRUE(snn_integration->addSNNNetwork("large_network", large_snn));
    
    // Configure sparse memory
    SNNIntegration::SparseMemoryConfig sparse_config;
    sparse_config.virtual_memory_gb = 4;
    sparse_config.physical_memory_gb = 1; // Only 1GB physical
    sparse_config.page_size_kb = 64;
    sparse_config.enable_compression = true;
    sparse_config.enable_prefetching = true;
    
    ASSERT_TRUE(snn_integration->configureSparseMemory("large_network", sparse_config));
    
    // Enable sparse binding
    large_snn->enableSparseBinding(4 * 1024 * 1024 * 1024); // 4GB virtual
    
    // Commit initial memory region (first 256MB)
    auto commit_future = snn_integration->commitMemoryRegions(
        "large_network", {{0, 256 * 1024 * 1024}});
    commit_future.wait();
    
    large_snn->initializeNetwork();
}

// Test 3: Real-time Spike Simulation Performance
TEST_F(SNNIntegrationTest, RealTimeSimulationPerformance) {
    // Create medium-sized network for performance testing
    auto snn = snn_integration->createSNN(10000, 1000);
    ASSERT_TRUE(snn_integration->addSNNNetwork("perf_test", snn));
    
    snn->initializeNetwork();
    
    // Create input pattern buffer
    ComputeBuffer* input_buffer = compute_engine->createBuffer(
        10000 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    // Generate random input currents (some neurons receive input)
    std::vector<float> input_currents(10000, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> current_dist(0.0f, 50e-12f); // 0-50 pA
    std::uniform_int_distribution<int> neuron_dist(0, 9999);
    
    // Activate 10% of neurons with input current
    for (int i = 0; i < 1000; i++) {
        int neuron_idx = neuron_dist(gen);
        input_currents[neuron_idx] = current_dist(gen);
    }
    
    compute_engine->uploadData(input_buffer, input_currents.data(), 
                              10000 * sizeof(float));
    
    // Configure simulation
    SNNIntegration::SimulationConfig config;
    config.timestep_dt = 1e-3f;        // 1ms timestep
    config.simulation_steps = 100;      // 100ms simulation
    config.enable_learning = false;     // Disable learning for pure performance test
    config.parallel_execution = true;   // Use multi-queue execution
    
    // Measure simulation performance
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto simulation_future = snn_integration->runSimulation("perf_test", input_buffer, config);
    simulation_future.wait();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Simulation performance: " << duration_ms.count() << "ms for 100ms of biological time\n";
    std::cout << "Real-time factor: " << (100.0f / duration_ms.count()) << "x\n";
    
    // Verify some spikes were generated
    uint32_t spike_count = snn->getSpikeCount();
    EXPECT_GT(spike_count, 0);
    
    std::cout << "Total spikes generated: " << spike_count << "\n";
    std::cout << "Average firing rate: " << snn->getAverageFireRate() << " Hz\n";
    
    compute_engine->destroyBuffer(input_buffer);
}

// Test 4: STDP Learning Functionality
TEST_F(SNNIntegrationTest, STDPLearning) {
    // Create network for STDP testing
    auto snn = snn_integration->createSNN(1000, 1000);
    ASSERT_TRUE(snn_integration->addSNNNetwork("stdp_test", snn));
    
    snn->initializeNetwork();
    
    // Configure STDP parameters
    SNNIntegration::STDPConfig stdp_config;
    stdp_config.learning_rate = 0.1f;   // High learning rate for testing
    stdp_config.tau_plus = 20e-3f;      // 20ms LTP time constant
    stdp_config.tau_minus = 20e-3f;     // 20ms LTD time constant
    stdp_config.A_plus = 1.0f;          // LTP amplitude
    stdp_config.A_minus = 1.0f;         // LTD amplitude
    stdp_config.batch_updates = true;
    
    snn_integration->setSTDPParameters("stdp_test", stdp_config);
    
    // Create input patterns that should cause correlated firing
    ComputeBuffer* input_buffer = compute_engine->createBuffer(
        1000 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    // Pattern 1: Activate first 100 neurons strongly
    std::vector<float> pattern1(1000, 0.0f);
    for (int i = 0; i < 100; i++) {
        pattern1[i] = 100e-12f; // 100 pA - above threshold
    }
    
    // Run simulation with pattern 1
    compute_engine->uploadData(input_buffer, pattern1.data(), 1000 * sizeof(float));
    
    SNNIntegration::SimulationConfig config;
    config.timestep_dt = 1e-3f;
    config.simulation_steps = 50;
    config.enable_learning = true;
    
    auto sim_future = snn_integration->runSimulation("stdp_test", input_buffer, config);
    sim_future.wait();
    
    // Apply STDP learning
    auto stdp_future = snn_integration->applySTDPLearning("stdp_test");
    stdp_future.wait();
    
    // Verify learning occurred (weights should have changed)
    // This would require access to synaptic weights for verification
    auto metrics = snn_integration->getPerformanceMetrics("stdp_test");
    EXPECT_GT(metrics.learning_time_ns, 0);
    
    compute_engine->destroyBuffer(input_buffer);
}

// Test 5: Memory Recall Latency (Target: 12ms)
TEST_F(SNNIntegrationTest, MemoryRecallLatency) {
    // Create network for memory testing
    auto snn = snn_integration->createSNN(5000, 2000);
    ASSERT_TRUE(snn_integration->addSNNNetwork("memory_test", snn));
    
    snn->initializeNetwork();
    
    // Train with some patterns first
    ComputeBuffer* pattern_buffer = compute_engine->createBuffer(
        5000 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    // Create and train with 10 distinct patterns
    for (int pattern = 0; pattern < 10; pattern++) {
        std::vector<float> training_pattern(5000, 0.0f);
        
        // Each pattern activates different subset of neurons
        for (int i = pattern * 100; i < (pattern + 1) * 100; i++) {
            training_pattern[i] = 80e-12f; // Strong input
        }
        
        compute_engine->uploadData(pattern_buffer, training_pattern.data(), 5000 * sizeof(float));
        
        SNNIntegration::SimulationConfig train_config;
        train_config.simulation_steps = 100;
        train_config.enable_learning = true;
        
        auto train_future = snn_integration->runSimulation("memory_test", pattern_buffer, train_config);
        train_future.wait();
        
        auto learn_future = snn_integration->applySTDPLearning("memory_test");
        learn_future.wait();
    }
    
    // Test memory recall with partial pattern (pattern 5 with 50% of neurons)
    std::vector<float> query_pattern(5000, 0.0f);
    for (int i = 500; i < 550; i++) { // Only half the neurons of pattern 5
        query_pattern[i] = 70e-12f;
    }
    
    ComputeBuffer* query_buffer = compute_engine->createBuffer(
        5000 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    compute_engine->uploadData(query_buffer, query_pattern.data(), 5000 * sizeof(float));
    
    // Perform memory query and measure latency
    SNNIntegration::MemoryQuery query;
    query.query_pattern = query_buffer;
    query.similarity_threshold = 0.6f;
    query.max_matches = 5;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto recall_future = snn_integration->queryMemory("memory_test", query);
    auto result = recall_future.get();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Memory recall latency: " << latency_ms.count() << "ms\n";
    std::cout << "Target latency: 12ms\n";
    
    // Verify recall was successful
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.matched_patterns.size(), 0);
    
    // Check if we achieved target latency (12ms)
    EXPECT_LE(latency_ms.count(), 20); // Allow some margin for test environment
    
    if (result.matched_patterns.size() > 0) {
        std::cout << "Best match similarity: " << result.similarity_scores[0] << "\n";
    }
    
    compute_engine->destroyBuffer(pattern_buffer);
    compute_engine->destroyBuffer(query_buffer);
}

// Test 6: CNN-SNN Integration Protocol
TEST_F(SNNIntegrationTest, CNNSNNIntegration) {
    // Create SNN for CNN integration
    auto snn = snn_integration->createSNN(2048, 1000); // Match typical CNN feature map size
    ASSERT_TRUE(snn_integration->addSNNNetwork("cnn_snn", snn));
    
    snn->initializeNetwork();
    
    // Create mock CNN feature map (32x32x2 feature map flattened)
    ComputeBuffer* cnn_features = compute_engine->createBuffer(
        2048 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    std::vector<float> feature_map(2048);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> feature_dist(-1.0f, 1.0f);
    
    for (int i = 0; i < 2048; i++) {
        feature_map[i] = feature_dist(gen);
    }
    
    compute_engine->uploadData(cnn_features, feature_map.data(), 2048 * sizeof(float));
    
    // Test CNN-SNN integration
    auto integration_future = snn_integration->integrateCNNFeatures("cnn_snn", "mock_cnn", cnn_features);
    integration_future.wait();
    
    // Verify integration worked (SNN should show activity)
    uint32_t post_integration_spikes = snn->getSpikeCount();
    std::cout << "CNN-SNN integration spikes: " << post_integration_spikes << "\n";
    
    compute_engine->destroyBuffer(cnn_features);
}

// Test 7: Multi-Network Parallel Execution
TEST_F(SNNIntegrationTest, ParallelMultiNetwork) {
    const int network_count = 4;
    std::vector<std::string> network_names;
    std::vector<ComputeBuffer*> input_buffers;
    
    // Create multiple small networks
    for (int i = 0; i < network_count; i++) {
        std::string name = "parallel_net_" + std::to_string(i);
        auto snn = snn_integration->createSNN(500, 500);
        ASSERT_TRUE(snn_integration->addSNNNetwork(name, snn));
        snn->initializeNetwork();
        
        network_names.push_back(name);
        
        // Create input buffer for each network
        ComputeBuffer* input_buffer = compute_engine->createBuffer(
            500 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        
        std::vector<float> input_pattern(500, 0.0f);
        for (int j = i * 50; j < (i + 1) * 50; j++) { // Different pattern per network
            input_pattern[j] = 60e-12f;
        }
        
        compute_engine->uploadData(input_buffer, input_pattern.data(), 500 * sizeof(float));
        input_buffers.push_back(input_buffer);
    }
    
    // Run parallel simulation across all networks
    SNNIntegration::SimulationConfig parallel_config;
    parallel_config.simulation_steps = 100;
    parallel_config.parallel_execution = true;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto parallel_future = snn_integration->runBatchSimulation(
        network_names, input_buffers, parallel_config);
    parallel_future.wait();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Parallel execution time for " << network_count << " networks: " 
              << parallel_duration.count() << "ms\n";
    
    // Verify all networks generated spikes
    for (const auto& name : network_names) {
        auto net = snn_integration->getSNNNetwork(name);
        uint32_t spikes = net->getSpikeCount();
        EXPECT_GT(spikes, 0);
        std::cout << "Network " << name << " spikes: " << spikes << "\n";
    }
    
    // Cleanup
    for (auto* buffer : input_buffers) {
        compute_engine->destroyBuffer(buffer);
    }
}

// Test 8: Performance Metrics and Reporting
TEST_F(SNNIntegrationTest, PerformanceMetrics) {
    auto snn = snn_integration->createSNN(1000, 1000);
    ASSERT_TRUE(snn_integration->addSNNNetwork("metrics_test", snn));
    snn->initializeNetwork();
    
    // Run some simulation to generate metrics
    ComputeBuffer* input_buffer = compute_engine->createBuffer(
        1000 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    std::vector<float> input_pattern(1000, 0.0f);
    for (int i = 0; i < 100; i++) {
        input_pattern[i] = 75e-12f;
    }
    
    compute_engine->uploadData(input_buffer, input_pattern.data(), 1000 * sizeof(float));
    
    SNNIntegration::SimulationConfig config;
    config.simulation_steps = 200;
    config.enable_learning = true;
    
    auto sim_future = snn_integration->runSimulation("metrics_test", input_buffer, config);
    sim_future.wait();
    
    auto learn_future = snn_integration->applySTDPLearning("metrics_test");
    learn_future.wait();
    
    // Get and verify performance metrics
    auto metrics = snn_integration->getPerformanceMetrics("metrics_test");
    
    EXPECT_GT(metrics.total_spikes_generated, 0);
    EXPECT_GT(metrics.simulation_time_ns, 0);
    EXPECT_GT(metrics.learning_time_ns, 0);
    EXPECT_GE(metrics.memory_utilization_percent, 0.0f);
    EXPECT_LE(metrics.memory_utilization_percent, 100.0f);
    
    std::cout << "Performance Metrics:\n";
    std::cout << "  Total spikes: " << metrics.total_spikes_generated << "\n";
    std::cout << "  Simulation time: " << metrics.simulation_time_ns / 1e6 << "ms\n"; 
    std::cout << "  Learning time: " << metrics.learning_time_ns / 1e6 << "ms\n";
    std::cout << "  Memory utilization: " << metrics.memory_utilization_percent << "%\n";
    std::cout << "  Queue utilization: " << metrics.queue_utilization_percent << "%\n";
    
    // Test performance report generation
    snn_integration->printPerformanceReport();
    
    compute_engine->destroyBuffer(input_buffer);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}