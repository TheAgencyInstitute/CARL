#include "neural_network_models.h"
#include "carl_compute_engine.h"
#include "compute_pipeline_manager.h"
#include "gan_realtime_pipeline.h"
#include <gtest/gtest.h>
#include <random>
#include <chrono>

/**
 * Comprehensive GAN Integration Tests for CARL AI System
 * 
 * Tests all aspects of the GAN implementation:
 * - Basic training workflow
 * - Progressive training stages
 * - Multi-queue utilization
 * - Real-time generation pipeline
 * - Performance benchmarks
 */

namespace CARL {
namespace AI {
namespace Tests {

class GANIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize core components
        _nova_core = std::make_unique<NovaCore>();
        ASSERT_TRUE(_nova_core->initialize());
        
        _engine = std::make_unique<CarlComputeEngine>(_nova_core.get());
        ASSERT_TRUE(_engine->initialize());
        
        _pipeline_manager = std::make_unique<ComputePipelineManager>(
            _nova_core->getDevice(), 
            _nova_core->getPhysicalDevice()
        );
        ASSERT_TRUE(_pipeline_manager->initialize());
        
        // Test configuration
        _noise_dim = 64; // Smaller for faster tests
        _image_size = 64; // Smaller for faster tests
        _channels = 3;
        _batch_size = 8;
    }
    
    void TearDown() override {
        _pipeline_manager.reset();
        _engine.reset();
        _nova_core.reset();
    }
    
    std::unique_ptr<NovaCore> _nova_core;
    std::unique_ptr<CarlComputeEngine> _engine;
    std::unique_ptr<ComputePipelineManager> _pipeline_manager;
    
    uint32_t _noise_dim;
    uint32_t _image_size;
    uint32_t _channels;
    uint32_t _batch_size;
};

TEST_F(GANIntegrationTest, BasicGANCreation) {
    // Test basic GAN instantiation
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    ASSERT_NE(gan.get(), nullptr);
    
    // Configure GAN
    gan->setPipelineManager(_pipeline_manager.get());
    gan->setQueueStrategy(1, 2);
    gan->setLearningRates(0.001f, 0.001f);
    
    // Build networks
    ASSERT_NO_THROW(gan->buildGenerator());
    ASSERT_NO_THROW(gan->buildDiscriminator());
    
    std::cout << "Basic GAN creation: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, ProgressiveTrainingSetup) {
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    gan->setPipelineManager(_pipeline_manager.get());
    
    // Enable progressive training
    gan->enableProgressiveTraining(16); // Start at 16x16
    
    EXPECT_EQ(gan->getCurrentResolution(), 16);
    EXPECT_EQ(gan->getProgressiveBlendFactor(), 0.0f);
    
    gan->buildGenerator();
    gan->buildDiscriminator();
    
    std::cout << "Progressive training setup: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, TrainingStepExecution) {
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    gan->setPipelineManager(_pipeline_manager.get());
    gan->setQueueStrategy(1, 2);
    gan->buildGenerator();
    gan->buildDiscriminator();
    
    // Create test data
    size_t real_data_size = _batch_size * _image_size * _image_size * _channels * sizeof(float);
    size_t noise_size = _batch_size * _noise_dim * sizeof(float);
    
    auto real_data = _engine->createBuffer(real_data_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto noise_data = _engine->createBuffer(noise_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Generate random test data
    std::vector<float> real_images(_batch_size * _image_size * _image_size * _channels, 0.5f);
    std::vector<float> noise_vector(_batch_size * _noise_dim);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : noise_vector) {
        val = dist(gen);
    }
    
    _engine->uploadData(real_data, real_images.data(), real_images.size() * sizeof(float));
    _engine->uploadData(noise_data, noise_vector.data(), noise_vector.size() * sizeof(float));
    
    // Execute training step
    auto training_start = std::chrono::steady_clock::now();
    
    ASSERT_NO_THROW({
        auto train_future = gan->trainWithLossComputation(real_data, noise_data);
        train_future.wait();
    });
    
    auto training_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);
    
    std::cout << "Training step completed in " << duration.count() << "ms" << std::endl;
    
    // Verify metrics are updated
    auto metrics = gan->getTrainingMetrics();
    EXPECT_GT(metrics.training_steps, 0);
    
    // Cleanup
    _engine->destroyBuffer(real_data);
    _engine->destroyBuffer(noise_data);
    
    std::cout << "Training step execution: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, MultiQueueUtilization) {
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    gan->setPipelineManager(_pipeline_manager.get());
    
    // Test different queue configurations
    ASSERT_NO_THROW(gan->setQueueStrategy(0, 1)); // Graphics + Compute 1
    ASSERT_NO_THROW(gan->setQueueStrategy(1, 2)); // Compute 1 + Compute 2
    ASSERT_NO_THROW(gan->setQueueStrategy(2, 3)); // Compute 2 + Compute 3
    
    gan->buildGenerator();
    gan->buildDiscriminator();
    
    // Verify queue performance stats are available
    auto queue_stats = _engine->getQueuePerformanceStats();
    EXPECT_GT(queue_stats.size(), 0);
    
    std::cout << "Multi-queue utilization: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, RealtimePipelineIntegration) {
    // Create GAN for real-time pipeline
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    gan->setPipelineManager(_pipeline_manager.get());
    gan->buildGenerator();
    gan->buildDiscriminator();
    
    // Initialize real-time pipeline
    GANRealtimePipeline pipeline(_engine.get(), _pipeline_manager.get(), _nova_core.get());
    
    GANRealtimePipeline::PipelineConfig config;
    config.max_concurrent_generations = 2;
    config.noise_dimension = _noise_dim;
    config.default_resolution = _image_size;
    config.channels = _channels;
    config.target_fps = 10.0f; // Lower for testing
    
    ASSERT_TRUE(pipeline.initialize(config));
    
    // Set GAN
    pipeline.setGAN(std::move(gan));
    
    // Test async generation
    std::vector<float> test_noise(_noise_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : test_noise) {
        val = dist(gen);
    }
    
    auto generation_start = std::chrono::steady_clock::now();
    
    auto result_future = pipeline.generateAsync(test_noise);
    
    // Wait for generation to complete
    auto result = result_future.get();
    ASSERT_NE(result, nullptr);
    
    auto generation_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(generation_end - generation_start);
    
    std::cout << "Real-time generation completed in " << duration.count() << "ms" << std::endl;
    
    // Test metrics
    auto metrics = pipeline.getCurrentMetrics();
    EXPECT_GT(metrics.completed_generations, 0);
    
    pipeline.shutdown();
    
    std::cout << "Real-time pipeline integration: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, PerformanceBenchmark) {
    auto gan = std::make_unique<GenerativeAdversarialNetwork>(
        _engine.get(), _noise_dim, _image_size, _image_size, _channels
    );
    
    gan->setPipelineManager(_pipeline_manager.get());
    gan->setQueueStrategy(1, 2);
    gan->buildGenerator();
    gan->buildDiscriminator();
    
    // Benchmark training performance
    const uint32_t benchmark_steps = 10;
    
    size_t real_data_size = _batch_size * _image_size * _image_size * _channels * sizeof(float);
    size_t noise_size = _batch_size * _noise_dim * sizeof(float);
    
    auto real_data = _engine->createBuffer(real_data_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto noise_data = _engine->createBuffer(noise_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    // Prepare data
    std::vector<float> real_images(_batch_size * _image_size * _image_size * _channels, 0.5f);
    std::vector<float> noise_vector(_batch_size * _noise_dim, 0.0f);
    
    _engine->uploadData(real_data, real_images.data(), real_images.size() * sizeof(float));
    _engine->uploadData(noise_data, noise_vector.data(), noise_vector.size() * sizeof(float));
    
    // Benchmark
    auto benchmark_start = std::chrono::steady_clock::now();
    
    for (uint32_t step = 0; step < benchmark_steps; step++) {
        auto train_future = gan->trainWithLossComputation(real_data, noise_data);
        train_future.wait();
    }
    
    auto benchmark_end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(benchmark_end - benchmark_start);
    
    float avg_time_per_step = static_cast<float>(total_time.count()) / benchmark_steps;
    float steps_per_second = 1000.0f / avg_time_per_step;
    
    std::cout << "Performance Benchmark Results:" << std::endl;
    std::cout << "  Total time: " << total_time.count() << "ms" << std::endl;
    std::cout << "  Average time per step: " << avg_time_per_step << "ms" << std::endl;
    std::cout << "  Steps per second: " << steps_per_second << std::endl;
    
    // Performance targets (adjust based on hardware)
    EXPECT_LT(avg_time_per_step, 1000.0f); // Less than 1 second per step
    EXPECT_GT(steps_per_second, 0.1f); // At least 0.1 steps per second
    
    // Cleanup
    _engine->destroyBuffer(real_data);
    _engine->destroyBuffer(noise_data);
    
    std::cout << "Performance benchmark: PASSED" << std::endl;
}

TEST_F(GANIntegrationTest, MemoryManagement) {
    // Test memory allocation and cleanup
    {
        auto gan = std::make_unique<GenerativeAdversarialNetwork>(
            _engine.get(), _noise_dim, _image_size, _image_size, _channels
        );
        
        gan->setPipelineManager(_pipeline_manager.get());
        gan->enableProgressiveTraining(16);
        gan->buildGenerator();
        gan->buildDiscriminator();
        
        // GAN should automatically manage its buffers
        auto metrics_before = gan->getTrainingMetrics();
        
        // Force some memory operations
        size_t data_size = _batch_size * _image_size * _image_size * _channels * sizeof(float);
        auto test_buffer = _engine->createBuffer(data_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        ASSERT_NE(test_buffer, nullptr);
        
        _engine->destroyBuffer(test_buffer);
        
        // GAN destructor should clean up properly
    }
    
    std::cout << "Memory management: PASSED" << std::endl;
}

// Integration test suite
class GANFullWorkflowTest : public GANIntegrationTest {
protected:
    void runFullWorkflow() {
        std::cout << "\n=== Running Full GAN Workflow Test ===" << std::endl;
        
        // 1. Initialize GAN with all features
        auto gan = std::make_unique<GenerativeAdversarialNetwork>(
            _engine.get(), _noise_dim, _image_size, _image_size, _channels
        );
        
        gan->setPipelineManager(_pipeline_manager.get());
        gan->enableProgressiveTraining(16);
        gan->setQueueStrategy(1, 2);
        gan->setLearningRates(0.002f, 0.002f);
        
        gan->buildGenerator();
        gan->buildDiscriminator();
        
        std::cout << "✓ GAN initialized with progressive training" << std::endl;
        
        // 2. Prepare training data
        setupTrainingData();
        std::cout << "✓ Training data prepared" << std::endl;
        
        // 3. Run training iterations
        runTrainingIterations(gan.get(), 5);
        std::cout << "✓ Training iterations completed" << std::endl;
        
        // 4. Test generation
        testGeneration(gan.get());
        std::cout << "✓ Generation testing completed" << std::endl;
        
        // 5. Performance analysis
        analyzePerformance(gan.get());
        std::cout << "✓ Performance analysis completed" << std::endl;
        
        // 6. Cleanup
        cleanupTrainingData();
        std::cout << "✓ Cleanup completed" << std::endl;
        
        std::cout << "\n=== Full GAN Workflow Test PASSED ===" << std::endl;
    }

private:
    ComputeBuffer* _real_data_buffer;
    ComputeBuffer* _noise_buffer;
    
    void setupTrainingData() {
        size_t real_data_size = _batch_size * _image_size * _image_size * _channels * sizeof(float);
        size_t noise_size = _batch_size * _noise_dim * sizeof(float);
        
        _real_data_buffer = _engine->createBuffer(real_data_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _noise_buffer = _engine->createBuffer(noise_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Fill with test data
        std::vector<float> real_data(real_data_size / sizeof(float), 0.5f);
        std::vector<float> noise_data(noise_size / sizeof(float), 0.0f);
        
        _engine->uploadData(_real_data_buffer, real_data.data(), real_data.size() * sizeof(float));
        _engine->uploadData(_noise_buffer, noise_data.data(), noise_data.size() * sizeof(float));
    }
    
    void runTrainingIterations(GenerativeAdversarialNetwork* gan, uint32_t iterations) {
        for (uint32_t i = 0; i < iterations; i++) {
            auto train_future = gan->trainWithLossComputation(_real_data_buffer, _noise_buffer);
            train_future.wait();
            
            if (i % 2 == 0) {
                auto metrics = gan->getTrainingMetrics();
                std::cout << "  Step " << i << ": D_loss=" << metrics.discriminator_loss 
                          << ", G_loss=" << metrics.generator_loss 
                          << ", Resolution=" << gan->getCurrentResolution() << std::endl;
            }
        }
    }
    
    void testGeneration(GenerativeAdversarialNetwork* gan) {
        size_t output_size = _image_size * _image_size * _channels * sizeof(float);
        auto output_buffer = _engine->createBuffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        auto generate_future = gan->generate(_noise_buffer, output_buffer);
        generate_future.wait();
        
        // Verify generation succeeded
        std::vector<float> generated_data(output_size / sizeof(float));
        _engine->downloadData(output_buffer, generated_data.data(), output_size);
        
        // Basic sanity check - data should not be all zeros
        bool has_non_zero = false;
        for (float val : generated_data) {
            if (std::abs(val) > 1e-6f) {
                has_non_zero = true;
                break;
            }
        }
        
        EXPECT_TRUE(has_non_zero) << "Generated data appears to be all zeros";
        
        _engine->destroyBuffer(output_buffer);
    }
    
    void analyzePerformance(GenerativeAdversarialNetwork* gan) {
        auto metrics = gan->getTrainingMetrics();
        auto queue_stats = _engine->getQueuePerformanceStats();
        
        std::cout << "  Training steps: " << metrics.training_steps << std::endl;
        std::cout << "  Final D_loss: " << metrics.discriminator_loss << std::endl;
        std::cout << "  Final G_loss: " << metrics.generator_loss << std::endl;
        std::cout << "  Queue utilization: " << queue_stats.size() << " queues tracked" << std::endl;
    }
    
    void cleanupTrainingData() {
        if (_real_data_buffer) _engine->destroyBuffer(_real_data_buffer);
        if (_noise_buffer) _engine->destroyBuffer(_noise_buffer);
        
        _real_data_buffer = nullptr;
        _noise_buffer = nullptr;
    }
};

TEST_F(GANFullWorkflowTest, CompleteWorkflow) {
    runFullWorkflow();
}

} // namespace Tests
} // namespace AI
} // namespace CARL

// Test runner main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== CARL GAN Integration Tests ===" << std::endl;
    std::cout << "Testing complete GAN training workflows and integration" << std::endl;
    
    return RUN_ALL_TESTS();
}