#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <fstream>
#include <mutex>

/**
 * CARL System Configuration Management
 * 
 * Centralized configuration system for the CARL AI platform:
 * - Component-specific configuration (CNN, GAN, RL, SNN)
 * - Queue allocation and load balancing settings
 * - Memory management and sparse binding configuration
 * - Performance tuning and optimization parameters
 * - Cross-component integration settings
 * - Runtime configuration updates
 * - Configuration validation and error handling
 * - Profile management for different use cases
 */

namespace CARL {
namespace AI {
namespace Config {

// Base Configuration Structure
struct BaseConfig {
    std::string config_version = "1.0.0";
    std::string created_timestamp;
    std::string last_modified_timestamp;
    std::string description;
    bool enabled = true;
    
    virtual ~BaseConfig() = default;
    virtual bool validate() const = 0;
    virtual std::string serialize() const = 0;
    virtual bool deserialize(const std::string& data) = 0;
};

// Component Configuration Structures

struct CNNConfig : public BaseConfig {
    // Network Architecture
    uint32_t input_width = 224;
    uint32_t input_height = 224;
    uint32_t input_channels = 3;
    std::vector<uint32_t> layer_filters = {64, 128, 256, 512};
    std::vector<uint32_t> kernel_sizes = {3, 3, 3, 3};
    std::vector<uint32_t> strides = {1, 1, 1, 1};
    std::vector<std::string> activation_functions = {"relu", "relu", "relu", "relu"};
    
    // Training Parameters
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 1e-4f;
    uint32_t batch_size = 32;
    uint32_t max_epochs = 100;
    float dropout_rate = 0.5f;
    
    // Optimization Settings
    std::string optimizer = "adam"; // "sgd", "adam", "rmsprop"
    bool use_batch_normalization = true;
    bool use_data_augmentation = true;
    std::string loss_function = "categorical_crossentropy";
    
    // Queue Assignment
    uint32_t forward_pass_queue = 1;
    uint32_t backward_pass_queue = 2;
    uint32_t data_loading_queue = 5; // Video decode queue for image preprocessing
    
    // Memory Management
    bool use_mixed_precision = true;
    size_t max_memory_mb = 4096;
    bool enable_gradient_checkpointing = false;
    
    bool validate() const override;
    std::string serialize() const override;
    bool deserialize(const std::string& data) override;
};

struct GANConfig : public BaseConfig {
    // Generator Configuration
    struct GeneratorConfig {
        uint32_t latent_dim = 100;
        std::vector<uint32_t> hidden_layers = {256, 512, 1024};
        std::string activation = "leaky_relu";
        std::string output_activation = "tanh";
        float dropout_rate = 0.3f;
        bool use_batch_norm = true;
    } generator;
    
    // Discriminator Configuration
    struct DiscriminatorConfig {
        std::vector<uint32_t> hidden_layers = {1024, 512, 256};
        std::string activation = "leaky_relu";
        float leaky_relu_alpha = 0.2f;
        float dropout_rate = 0.3f;
        bool use_spectral_norm = true;
    } discriminator;
    
    // Training Parameters
    float generator_learning_rate = 0.0002f;
    float discriminator_learning_rate = 0.0002f;
    float beta1 = 0.5f;  // Adam optimizer parameter
    float beta2 = 0.999f; // Adam optimizer parameter
    uint32_t batch_size = 64;
    uint32_t max_iterations = 50000;
    
    // Loss Configuration
    std::string loss_type = "wgan-gp"; // "vanilla", "lsgan", "wgan", "wgan-gp"
    float gradient_penalty_weight = 10.0f;
    uint32_t critic_iterations = 5; // For WGAN
    
    // Queue Assignment
    uint32_t generator_queue = 3;
    uint32_t discriminator_queue = 4;
    uint32_t data_loading_queue = 5;
    uint32_t output_generation_queue = 6; // Video encode for output
    
    // Memory and Performance
    bool enable_progressive_growing = false;
    bool use_self_attention = false;
    size_t max_memory_mb = 6144;
    
    bool validate() const override;
    std::string serialize() const override;
    bool deserialize(const std::string& data) override;
};

struct RLConfig : public BaseConfig {
    // Environment Configuration
    uint32_t state_space_size = 128;
    uint32_t action_space_size = 8;
    bool continuous_action_space = false;
    float reward_scale = 1.0f;
    float discount_factor = 0.99f;
    
    // Algorithm Configuration
    std::string algorithm = "ppo"; // "dqn", "ddpg", "a3c", "ppo", "sac"
    
    // DQN Specific
    struct DQNConfig {
        uint32_t replay_buffer_size = 100000;
        uint32_t target_update_frequency = 1000;
        float epsilon_start = 1.0f;
        float epsilon_end = 0.01f;
        float epsilon_decay = 0.995f;
        bool use_double_dqn = true;
        bool use_dueling_dqn = true;
        bool use_prioritized_replay = true;
    } dqn;
    
    // PPO Specific
    struct PPOConfig {
        float clip_ratio = 0.2f;
        uint32_t ppo_epochs = 4;
        uint32_t batch_size = 64;
        float value_function_coeff = 0.5f;
        float entropy_coeff = 0.01f;
        bool use_gae = true;
        float gae_lambda = 0.95f;
    } ppo;
    
    // Network Architecture
    std::vector<uint32_t> policy_hidden_layers = {256, 256};
    std::vector<uint32_t> value_hidden_layers = {256, 256};
    std::string activation = "relu";
    
    // Training Parameters
    float learning_rate = 3e-4f;
    uint32_t max_episodes = 10000;
    uint32_t max_steps_per_episode = 1000;
    uint32_t training_frequency = 4;
    
    // Queue Assignment
    uint32_t policy_update_queue = 1;
    uint32_t value_update_queue = 2;
    uint32_t experience_processing_queue = 3;
    
    // Memory Management
    size_t max_memory_mb = 2048;
    bool use_gpu_buffers = true;
    
    bool validate() const override;
    std::string serialize() const override;
    bool deserialize(const std::string& data) override;
};

struct SNNConfig : public BaseConfig {
    // Network Topology
    uint32_t num_neurons = 10000;
    uint32_t simulation_timesteps = 1000;
    float simulation_dt = 1e-3f; // 1ms timestep
    
    // Neuron Model Parameters
    struct NeuronModelConfig {
        std::string model_type = "lif"; // "lif", "izhikevich", "hodgkin_huxley"
        float resting_potential = -70.0f; // mV
        float threshold_potential = -55.0f; // mV
        float reset_potential = -80.0f; // mV
        float membrane_resistance = 10.0f; // MOhm
        float membrane_capacitance = 1.0f; // nF
        float refractory_period = 2e-3f; // 2ms
    } neuron_model;
    
    // Synapse Configuration
    struct SynapseConfig {
        float max_weight = 1.0f;
        float min_weight = 0.0f;
        float initial_weight_mean = 0.5f;
        float initial_weight_std = 0.1f;
        float transmission_delay_ms = 1.0f;
        bool enable_plasticity = true;
    } synapse;
    
    // STDP Learning Parameters
    struct STDPConfig {
        bool enabled = true;
        float learning_rate = 0.01f;
        float tau_plus = 20e-3f;  // LTP time constant (20ms)
        float tau_minus = 20e-3f; // LTD time constant (20ms)
        float A_plus = 1.0f;      // LTP amplitude
        float A_minus = 1.0f;     // LTD amplitude
        float weight_dependence = 1.0f; // 0=additive, 1=multiplicative
    } stdp;
    
    // Memory Configuration
    struct MemoryConfig {
        bool use_sparse_memory = true;
        uint32_t virtual_memory_gb = 16;
        uint32_t physical_memory_gb = 4;
        uint32_t page_size_kb = 64;
        std::string cache_policy = "lfu"; // "lru", "lfu", "random"
        bool enable_compression = true;
        float compression_ratio = 0.3f;
        bool enable_prefetching = true;
        uint32_t prefetch_distance = 16;
    } memory;
    
    // Queue Assignment
    uint32_t simulation_queue = 1;
    uint32_t learning_queue = 2;
    uint32_t memory_management_queue = 7; // Sparse binding queue
    
    // Performance Optimization
    uint32_t parallel_simulation_batches = 4;
    bool use_event_driven_simulation = true;
    float spike_sparsity_threshold = 0.1f;
    
    bool validate() const override;
    std::string serialize() const override;
    bool deserialize(const std::string& data) override;
};

// System-Wide Configuration

struct SystemConfig : public BaseConfig {
    // Queue Management
    struct QueueConfig {
        bool enable_auto_load_balancing = true;
        std::vector<float> queue_priority_weights = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        uint32_t max_concurrent_operations = 16;
        float queue_overload_threshold = 0.9f;
        uint32_t load_balancing_update_interval_ms = 100;
        
        // Queue-specific settings
        struct QueueSettings {
            bool enabled = true;
            float utilization_target = 0.8f;
            uint32_t max_operations_per_queue = 4;
            std::string workload_profile = "mixed"; // "compute", "memory", "mixed"
        };
        
        std::vector<QueueSettings> queue_settings; // 8 queues
    } queue_management;
    
    // Memory Management
    struct MemoryConfig {
        size_t system_memory_limit_gb = 32;
        size_t gpu_memory_limit_gb = 16;
        float memory_pressure_threshold = 0.85f;
        bool enable_automatic_garbage_collection = true;
        uint32_t gc_interval_seconds = 60;
        
        // Sparse Memory Settings
        bool enable_sparse_binding = true;
        uint32_t sparse_virtual_memory_gb = 64;
        float sparse_commitment_ratio = 0.25f;
        std::string sparse_allocation_strategy = "on_demand"; // "eager", "on_demand", "predictive"
        
        // Buffer Pool Settings
        uint32_t buffer_pool_size_mb = 512;
        uint32_t max_buffer_size_mb = 64;
        bool enable_buffer_reuse = true;
    } memory_management;
    
    // Performance Monitoring
    struct MonitoringConfig {
        bool enable_performance_monitoring = true;
        uint32_t metrics_collection_interval_ms = 1000;
        uint32_t metrics_history_size = 1000;
        bool enable_real_time_alerts = true;
        
        // Alert Thresholds
        float min_fps_threshold = 10.0f;
        float max_memory_usage_threshold = 0.9f;
        float min_queue_utilization_threshold = 0.5f;
        float max_temperature_threshold = 85.0f;
        
        // Logging Configuration
        std::string log_level = "info"; // "debug", "info", "warning", "error"
        bool enable_performance_logging = true;
        std::string log_file_path = "./logs/carl_performance.log";
        uint32_t max_log_file_size_mb = 100;
        uint32_t max_log_files = 10;
    } monitoring;
    
    // Cross-Component Integration
    struct IntegrationConfig {
        bool enable_cross_component_learning = true;
        float component_synchronization_frequency = 10.0f; // Every N epochs
        float cross_component_learning_rate = 0.1f;
        
        // Protocol-specific settings
        struct ProtocolConfig {
            bool enabled = true;
            float communication_latency_target_ms = 5.0f;
            uint32_t max_data_transfer_size_mb = 64;
            bool enable_data_compression = true;
            float compression_level = 0.5f;
        };
        
        ProtocolConfig cnn_rl_protocol;
        ProtocolConfig gan_snn_protocol;
        ProtocolConfig cnn_gan_protocol;
        ProtocolConfig rl_snn_protocol;
    } integration;
    
    // Security and Validation
    struct SecurityConfig {
        bool enable_input_validation = true;
        bool enable_memory_protection = true;
        bool enable_secure_random = true;
        std::string random_seed_source = "hardware"; // "hardware", "os", "manual"
        uint64_t manual_seed = 12345;
        
        // Model Protection
        bool enable_model_checksum_validation = true;
        bool enable_gradient_clipping = true;
        float gradient_clip_norm = 1.0f;
        
        // Privacy Settings
        bool enable_differential_privacy = false;
        float privacy_epsilon = 1.0f;
        float privacy_delta = 1e-5f;
    } security;
    
    bool validate() const override;
    std::string serialize() const override;
    bool deserialize(const std::string& data) override;
};

// Configuration Management System

class ConfigurationManager {
public:
    ConfigurationManager();
    ~ConfigurationManager();
    
    // Configuration Loading and Saving
    bool loadSystemConfig(const std::string& filepath);
    bool saveSystemConfig(const std::string& filepath);
    
    bool loadComponentConfig(const std::string& component_name, const std::string& filepath);
    bool saveComponentConfig(const std::string& component_name, const std::string& filepath);
    
    // Configuration Access
    SystemConfig& getSystemConfig();
    const SystemConfig& getSystemConfig() const;
    
    CNNConfig& getCNNConfig(const std::string& model_name);
    GANConfig& getGANConfig(const std::string& model_name);
    RLConfig& getRLConfig(const std::string& agent_name);
    SNNConfig& getSNNConfig(const std::string& network_name);
    
    // Configuration Validation
    bool validateAllConfigurations();
    std::vector<std::string> getConfigurationErrors();
    
    // Configuration Profiles
    bool saveConfigurationProfile(const std::string& profile_name);
    bool loadConfigurationProfile(const std::string& profile_name);
    std::vector<std::string> getAvailableProfiles();
    bool deleteConfigurationProfile(const std::string& profile_name);
    
    // Built-in Profiles
    void loadOptimalPerformanceProfile();
    void loadLowMemoryProfile();
    void loadMaxAccuracyProfile();
    void loadRealTimeProfile();
    void loadDebugProfile();
    
    // Dynamic Configuration Updates
    void enableRuntimeUpdates(bool enable = true);
    void registerUpdateCallback(const std::string& config_path, std::function<void()> callback);
    void updateConfiguration(const std::string& config_path, const std::string& new_value);
    
    // Configuration Monitoring
    void enableConfigurationMonitoring(bool enable = true);
    void setConfigurationChangeCallback(std::function<void(const std::string&)> callback);
    
    // Export and Import
    bool exportConfigurationToJSON(const std::string& filepath);
    bool importConfigurationFromJSON(const std::string& filepath);
    bool exportConfigurationToYAML(const std::string& filepath);
    bool importConfigurationFromYAML(const std::string& filepath);
    
    // Configuration Templates
    void createCNNTemplate(const std::string& template_name, const CNNConfig& config);
    void createGANTemplate(const std::string& template_name, const GANConfig& config);
    void createRLTemplate(const std::string& template_name, const RLConfig& config);
    void createSNNTemplate(const std::string& template_name, const SNNConfig& config);
    
    CNNConfig createCNNFromTemplate(const std::string& template_name);
    GANConfig createGANFromTemplate(const std::string& template_name);
    RLConfig createRLFromTemplate(const std::string& template_name);
    SNNConfig createSNNFromTemplate(const std::string& template_name);
    
    // Configuration Statistics and Analytics
    struct ConfigurationStats {
        uint32_t total_configurations;
        uint32_t active_configurations;
        uint32_t configuration_errors;
        float average_memory_usage_mb;
        float average_queue_utilization;
        std::unordered_map<std::string, uint32_t> configuration_usage_counts;
    };
    
    ConfigurationStats getConfigurationStatistics();
    
private:
    SystemConfig _system_config;
    std::unordered_map<std::string, CNNConfig> _cnn_configs;
    std::unordered_map<std::string, GANConfig> _gan_configs;
    std::unordered_map<std::string, RLConfig> _rl_configs;
    std::unordered_map<std::string, SNNConfig> _snn_configs;
    
    // Configuration templates
    std::unordered_map<std::string, CNNConfig> _cnn_templates;
    std::unordered_map<std::string, GANConfig> _gan_templates;
    std::unordered_map<std::string, RLConfig> _rl_templates;
    std::unordered_map<std::string, SNNConfig> _snn_templates;
    
    // Runtime update system
    bool _runtime_updates_enabled = false;
    std::unordered_map<std::string, std::function<void()>> _update_callbacks;
    std::mutex _config_mutex;
    
    // Configuration monitoring
    bool _monitoring_enabled = false;
    std::function<void(const std::string&)> _change_callback;
    
    // Validation errors
    std::vector<std::string> _validation_errors;
    
    // Helper methods
    bool _validateConfig(const BaseConfig& config);
    std::string _getProfilePath(const std::string& profile_name);
    bool _fileExists(const std::string& filepath);
    std::string _readFileContents(const std::string& filepath);
    bool _writeFileContents(const std::string& filepath, const std::string& contents);
    
    // JSON/YAML serialization helpers
    std::string _serializeToJSON(const BaseConfig& config);
    bool _deserializeFromJSON(BaseConfig& config, const std::string& json_data);
    std::string _serializeToYAML(const BaseConfig& config);
    bool _deserializeFromYAML(BaseConfig& config, const std::string& yaml_data);
};

// Global Configuration Access
ConfigurationManager& getGlobalConfigManager();

} // namespace Config
} // namespace AI
} // namespace CARL

/**
 * Usage Example:
 * 
 * // Initialize configuration manager
 * auto& config_mgr = CARL::AI::Config::getGlobalConfigManager();
 * 
 * // Load system configuration
 * config_mgr.loadSystemConfig("./config/carl_system.json");
 * 
 * // Configure CNN model
 * auto& cnn_config = config_mgr.getCNNConfig("vision_model");
 * cnn_config.learning_rate = 0.0005f;
 * cnn_config.batch_size = 64;
 * cnn_config.forward_pass_queue = 1;
 * cnn_config.backward_pass_queue = 2;
 * 
 * // Configure system-wide settings
 * auto& sys_config = config_mgr.getSystemConfig();
 * sys_config.queue_management.enable_auto_load_balancing = true;
 * sys_config.memory_management.enable_sparse_binding = true;
 * 
 * // Save configuration profile
 * config_mgr.saveConfigurationProfile("high_performance");
 * 
 * // Load optimal performance profile
 * config_mgr.loadOptimalPerformanceProfile();
 * 
 * // Validate all configurations
 * bool configs_valid = config_mgr.validateAllConfigurations();
 * 
 * Expected Benefits:
 * - Centralized configuration management for all CARL components
 * - Easy performance tuning and optimization
 * - Configuration profiles for different use cases
 * - Runtime configuration updates without restart
 * - Comprehensive validation and error handling
 */