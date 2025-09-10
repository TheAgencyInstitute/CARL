#pragma once

#include "carl_ai_system.h"
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <fstream>
#include <thread>

/**
 * CARL Performance Benchmark Framework
 * 
 * Comprehensive benchmarking system for CARL AI performance measurement:
 * - Individual component benchmarks (CNN, GAN, RL, SNN)
 * - Cross-component integration benchmarks
 * - Queue utilization efficiency measurement
 * - Memory performance and sparse binding tests
 * - Comparison against Nova baseline performance
 * - Real-time performance monitoring
 * - Detailed reporting and analytics
 */

namespace CARL {
namespace AI {
namespace Benchmark {

struct BenchmarkMetrics {
    // Timing metrics
    float total_execution_time_ms;
    float average_operation_time_ms;
    float min_operation_time_ms;
    float max_operation_time_ms;
    
    // Throughput metrics
    float operations_per_second;
    float data_throughput_mbps;
    
    // Resource utilization
    float average_cpu_usage_percent;
    float peak_memory_usage_mb;
    float average_gpu_utilization_percent;
    std::vector<float> queue_utilization_percent; // Per queue
    
    // Quality metrics
    float accuracy_score;
    float precision_score;
    float recall_score;
    float f1_score;
    
    // System metrics
    uint64_t total_operations;
    uint32_t failed_operations;
    float success_rate_percent;
    
    // Efficiency metrics
    float performance_per_watt;
    float performance_per_dollar;
    float queue_efficiency_score;
    float memory_efficiency_score;
};

struct ComparisonResults {
    BenchmarkMetrics carl_performance;
    BenchmarkMetrics nova_baseline;
    float speedup_factor;
    float efficiency_improvement_percent;
    float resource_utilization_improvement_percent;
    std::string performance_summary;
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark(CarlAISystem* carl_system);
    ~PerformanceBenchmark();
    
    // Component-Specific Benchmarks
    
    // CNN Performance Benchmarks
    struct CNNBenchmarkConfig {
        uint32_t input_width = 224;
        uint32_t input_height = 224;
        uint32_t input_channels = 3;
        uint32_t batch_size = 32;
        uint32_t num_iterations = 1000;
        bool measure_training = true;
        bool measure_inference = true;
        std::vector<uint32_t> layer_sizes = {64, 128, 256, 512};
    };
    
    BenchmarkMetrics benchmarkCNN(const std::string& model_name, const CNNBenchmarkConfig& config);
    
    // GAN Performance Benchmarks
    struct GANBenchmarkConfig {
        uint32_t latent_dim = 100;
        uint32_t output_width = 64;
        uint32_t output_height = 64;
        uint32_t output_channels = 3;
        uint32_t batch_size = 64;
        uint32_t num_iterations = 1000;
        bool benchmark_generator = true;
        bool benchmark_discriminator = true;
        bool benchmark_adversarial_training = true;
    };
    
    BenchmarkMetrics benchmarkGAN(const std::string& model_name, const GANBenchmarkConfig& config);
    
    // RL Performance Benchmarks
    struct RLBenchmarkConfig {
        uint32_t state_space_size = 128;
        uint32_t action_space_size = 8;
        uint32_t episode_length = 200;
        uint32_t num_episodes = 1000;
        bool benchmark_policy_updates = true;
        bool benchmark_value_function = true;
        bool benchmark_exploration = true;
        float exploration_rate = 0.1f;
    };
    
    BenchmarkMetrics benchmarkRL(const std::string& agent_name, const RLBenchmarkConfig& config);
    
    // SNN Performance Benchmarks
    struct SNNBenchmarkConfig {
        uint32_t num_neurons = 10000;
        uint32_t simulation_timesteps = 1000;
        uint32_t num_simulations = 100;
        float simulation_dt = 1e-3f; // 1ms timestep
        bool benchmark_spike_generation = true;
        bool benchmark_learning = true;
        bool benchmark_memory_recall = true;
        uint32_t memory_query_count = 1000;
        bool use_sparse_memory = true;
    };
    
    BenchmarkMetrics benchmarkSNN(const std::string& network_name, const SNNBenchmarkConfig& config);
    
    // Cross-Component Integration Benchmarks
    
    struct IntegrationBenchmarkConfig {
        std::string cnn_name;
        std::string gan_name;
        std::string rl_name;
        std::string snn_name;
        uint32_t integration_iterations = 500;
        bool benchmark_cnn_rl = true;
        bool benchmark_gan_snn = true;
        bool benchmark_cnn_gan = true;
        bool benchmark_full_integration = true;
    };
    
    struct IntegrationBenchmarkResults {
        BenchmarkMetrics cnn_rl_metrics;
        BenchmarkMetrics gan_snn_metrics;
        BenchmarkMetrics cnn_gan_metrics;
        BenchmarkMetrics full_integration_metrics;
        float integration_overhead_percent;
        float synchronization_latency_ms;
        float cross_component_throughput_ops_sec;
    };
    
    IntegrationBenchmarkResults benchmarkIntegration(const IntegrationBenchmarkConfig& config);
    
    // Queue Utilization Benchmarks
    
    struct QueueBenchmarkConfig {
        uint32_t workload_duration_seconds = 60;
        std::vector<uint32_t> target_queues = {0, 1, 2, 3, 4, 5, 6, 7}; // All 8 queues
        bool simulate_mixed_workload = true;
        bool measure_load_balancing = true;
        float target_utilization_percent = 80.0f;
    };
    
    struct QueueBenchmarkResults {
        std::vector<float> individual_queue_utilization; // Per queue utilization
        float total_system_utilization;
        float load_balancing_efficiency;
        float queue_synchronization_overhead_ms;
        uint64_t total_queue_operations;
        float operations_per_queue_per_second[8];
        float queue_efficiency_score;
        std::string bottleneck_analysis;
    };
    
    QueueBenchmarkResults benchmarkQueueUtilization(const QueueBenchmarkConfig& config);
    
    // Memory Performance Benchmarks
    
    struct MemoryBenchmarkConfig {
        size_t test_data_size_mb = 1024; // 1GB test data
        uint32_t memory_operations = 10000;
        bool test_sparse_binding = true;
        uint32_t sparse_virtual_size_gb = 32;
        bool test_memory_fragmentation = true;
        bool test_garbage_collection = true;
        bool stress_test_memory = true;
    };
    
    struct MemoryBenchmarkResults {
        float memory_bandwidth_gbps;
        float sparse_memory_efficiency_percent;
        float fragmentation_overhead_percent;
        float garbage_collection_pause_ms;
        size_t peak_memory_usage_mb;
        float memory_utilization_efficiency;
        uint32_t memory_allocation_failures;
        float memory_access_latency_ns;
    };
    
    MemoryBenchmarkResults benchmarkMemoryPerformance(const MemoryBenchmarkConfig& config);
    
    // Real-World Application Benchmarks
    
    // Computer Vision + Reinforcement Learning Application
    struct CVRLApplicationBenchmark {
        uint32_t image_width = 224;
        uint32_t image_height = 224;
        uint32_t action_space_size = 8;
        uint32_t training_episodes = 1000;
        uint32_t inference_iterations = 10000;
        float target_accuracy = 0.85f;
    };
    
    BenchmarkMetrics benchmarkCVRLApplication(const CVRLApplicationBenchmark& config);
    
    // Generative AI + Memory Application
    struct GenerativeMemoryApplicationBenchmark {
        uint32_t generation_width = 256;
        uint32_t generation_height = 256;
        uint32_t generation_count = 1000;
        uint32_t memory_context_size = 100;
        float generation_quality_threshold = 0.8f;
    };
    
    BenchmarkMetrics benchmarkGenerativeMemoryApplication(const GenerativeMemoryApplicationBenchmark& config);
    
    // Comprehensive System Benchmarks
    
    struct SystemBenchmarkConfig {
        CNNBenchmarkConfig cnn_config;
        GANBenchmarkConfig gan_config;
        RLBenchmarkConfig rl_config;
        SNNBenchmarkConfig snn_config;
        IntegrationBenchmarkConfig integration_config;
        QueueBenchmarkConfig queue_config;
        MemoryBenchmarkConfig memory_config;
        
        bool run_all_benchmarks = true;
        bool run_stress_tests = true;
        bool run_endurance_tests = false;
        uint32_t endurance_duration_hours = 24;
        bool generate_detailed_report = true;
    };
    
    CarlAISystem::BenchmarkResults runSystemBenchmark(const SystemBenchmarkConfig& config);
    
    // Nova Baseline Comparison
    
    struct NovaComparisonConfig {
        bool compare_single_queue_performance = true;
        bool compare_memory_usage = true;
        bool compare_power_efficiency = true;
        bool simulate_nova_limitations = true;
        uint32_t comparison_iterations = 1000;
    };
    
    ComparisonResults compareWithNova(const NovaComparisonConfig& config);
    
    // Real-Time Performance Monitoring
    
    class RealTimeMonitor {
    public:
        RealTimeMonitor(CarlAISystem* system);
        ~RealTimeMonitor();
        
        void startMonitoring();
        void stopMonitoring();
        
        BenchmarkMetrics getCurrentMetrics();
        std::vector<BenchmarkMetrics> getHistoricalMetrics(uint32_t last_n_samples = 100);
        
        void setAlertThresholds(float min_fps, float max_memory_mb, float min_queue_utilization);
        void enableAlerts(bool enable = true);
        
    private:
        CarlAISystem* _system;
        std::atomic<bool> _monitoring_active;
        std::thread _monitoring_thread;
        std::vector<BenchmarkMetrics> _metrics_history;
        std::mutex _metrics_mutex;
        
        // Alert system
        bool _alerts_enabled = false;
        float _min_fps_threshold = 10.0f;
        float _max_memory_mb_threshold = 8192.0f; // 8GB
        float _min_queue_utilization_threshold = 0.5f;
        
        void _monitoringLoop();
        void _checkAlerts(const BenchmarkMetrics& metrics);
    };
    
    std::unique_ptr<RealTimeMonitor> createRealTimeMonitor();
    
    // Reporting and Analytics
    
    struct BenchmarkReport {
        std::string system_configuration;
        std::string timestamp;
        std::vector<BenchmarkMetrics> component_results;
        IntegrationBenchmarkResults integration_results;
        QueueBenchmarkResults queue_results;
        MemoryBenchmarkResults memory_results;
        ComparisonResults nova_comparison;
        
        std::string performance_summary;
        std::vector<std::string> recommendations;
        std::vector<std::string> identified_bottlenecks;
        float overall_performance_score; // 0.0 to 100.0
    };
    
    BenchmarkReport generateComprehensiveReport(const SystemBenchmarkConfig& config);
    bool exportReportToJSON(const BenchmarkReport& report, const std::string& filepath);
    bool exportReportToHTML(const BenchmarkReport& report, const std::string& filepath);
    bool exportReportToCSV(const BenchmarkReport& report, const std::string& filepath);
    
    // Performance Optimization Recommendations
    
    struct OptimizationRecommendation {
        std::string category; // "queue", "memory", "component", "integration"
        std::string description;
        float expected_improvement_percent;
        std::string implementation_difficulty; // "low", "medium", "high"
        std::vector<std::string> implementation_steps;
    };
    
    std::vector<OptimizationRecommendation> generateOptimizationRecommendations(const BenchmarkReport& report);
    
    // Statistical Analysis
    
    struct PerformanceStatistics {
        float mean;
        float median;
        float standard_deviation;
        float min_value;
        float max_value;
        float percentile_95;
        float percentile_99;
        float coefficient_of_variation;
    };
    
    PerformanceStatistics calculateStatistics(const std::vector<float>& measurements);
    
    // A/B Testing Framework
    
    struct ABTestConfig {
        std::string test_name;
        std::string baseline_configuration;
        std::string test_configuration;
        uint32_t sample_size = 1000;
        float significance_level = 0.05f;
        std::string primary_metric = "operations_per_second";
    };
    
    struct ABTestResult {
        bool statistically_significant;
        float p_value;
        float effect_size;
        float confidence_interval_lower;
        float confidence_interval_upper;
        std::string recommendation;
    };
    
    ABTestResult runABTest(const ABTestConfig& config);

private:
    CarlAISystem* _carl_system;
    
    // Timing utilities
    std::chrono::steady_clock::time_point _benchmark_start_time;
    
    // Resource monitoring
    std::unique_ptr<SystemResourceMonitor> _resource_monitor;
    
    // Test data generation
    ComputeBuffer* generateTestData(size_t size_bytes, uint32_t pattern_type = 0);
    void releaseTestData(ComputeBuffer* buffer);
    
    // Statistical calculations
    float calculateMean(const std::vector<float>& values);
    float calculateStandardDeviation(const std::vector<float>& values, float mean);
    float calculatePercentile(const std::vector<float>& values, float percentile);
    
    // Benchmark execution helpers
    void warmupSystem();
    void cooldownSystem();
    BenchmarkMetrics executeBenchmarkLoop(const std::function<void()>& operation, uint32_t iterations);
    
    // Report generation helpers
    std::string formatPerformanceMetrics(const BenchmarkMetrics& metrics);
    std::string generateBottleneckAnalysis(const BenchmarkReport& report);
    std::vector<std::string> generatePerformanceRecommendations(const BenchmarkReport& report);
};

// System Resource Monitor (Internal utility)
class SystemResourceMonitor {
public:
    SystemResourceMonitor();
    ~SystemResourceMonitor();
    
    void startMonitoring();
    void stopMonitoring();
    
    float getCurrentCPUUsage();
    size_t getCurrentMemoryUsage();
    float getCurrentGPUUtilization();
    float getCurrentPowerUsage();
    
    struct ResourceUsageHistory {
        std::vector<float> cpu_usage_history;
        std::vector<size_t> memory_usage_history;
        std::vector<float> gpu_utilization_history;
        std::vector<float> power_usage_history;
        std::vector<std::chrono::steady_clock::time_point> timestamps;
    };
    
    ResourceUsageHistory getUsageHistory();
    
private:
    std::atomic<bool> _monitoring_active;
    std::thread _monitoring_thread;
    ResourceUsageHistory _usage_history;
    std::mutex _history_mutex;
    
    void _monitoringLoop();
    void _updateResourceUsage();
};

} // namespace Benchmark
} // namespace AI
} // namespace CARL

/**
 * Usage Example:
 * 
 * auto carl_system = std::make_unique<CARL::AI::CarlAISystem>();
 * carl_system->initialize();
 * 
 * // Create benchmark suite
 * CARL::AI::Benchmark::PerformanceBenchmark benchmark(carl_system.get());
 * 
 * // Run comprehensive system benchmark
 * CARL::AI::Benchmark::PerformanceBenchmark::SystemBenchmarkConfig config;
 * config.run_all_benchmarks = true;
 * config.generate_detailed_report = true;
 * 
 * auto results = benchmark.runSystemBenchmark(config);
 * auto report = benchmark.generateComprehensiveReport(config);
 * 
 * // Export results
 * benchmark.exportReportToHTML(report, "./carl_performance_report.html");
 * 
 * // Compare with Nova
 * CARL::AI::Benchmark::PerformanceBenchmark::NovaComparisonConfig nova_config;
 * auto comparison = benchmark.compareWithNova(nova_config);
 * 
 * std::cout << "Performance speedup vs Nova: " << comparison.speedup_factor << "x" << std::endl;
 * 
 * Expected Results:
 * - 6-8x performance improvement vs Nova baseline
 * - 100% queue utilization (8/8 queues) vs Nova's 12.5% (1/8 queues)
 * - Support for >16GB models via sparse binding
 * - Sub-10ms inference latency for real-time applications
 * - Comprehensive performance analytics and optimization recommendations
 */