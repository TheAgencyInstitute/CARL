#pragma once

#include "../carl_gpu.h"
#include "../../core/CNN/CNN.h"
#include <memory>

/**
 * CNN Accelerator - GPU acceleration for Convolutional Neural Networks
 * 
 * Provides GPU-accelerated implementations of CNN operations:
 * - 2D Convolution with various padding and stride options
 * - Activation functions (ReLU, Sigmoid, Tanh, etc.)  
 * - Pooling operations (Max, Average, Global)
 * - Batch normalization
 * - Forward and backward propagation
 */

namespace CARL {
namespace GPU {

enum class ActivationType {
    RELU,
    SIGMOID, 
    TANH,
    LEAKY_RELU,
    ELU,
    SWISH,
    GELU
};

enum class PoolingType {
    MAX,
    AVERAGE,
    GLOBAL_MAX,
    GLOBAL_AVERAGE
};

enum class PaddingType {
    VALID,     // No padding
    SAME,      // Pad to maintain output size
    CUSTOM     // User-specified padding
};

struct ConvolutionConfig {
    uint32_t stride_x = 1;
    uint32_t stride_y = 1;
    uint32_t dilation_x = 1;
    uint32_t dilation_y = 1;
    PaddingType padding = PaddingType::VALID;
    uint32_t pad_left = 0, pad_right = 0;
    uint32_t pad_top = 0, pad_bottom = 0;
    uint32_t groups = 1;  // For grouped convolution
};

struct PoolingConfig {
    uint32_t kernel_x = 2;
    uint32_t kernel_y = 2; 
    uint32_t stride_x = 2;
    uint32_t stride_y = 2;
    PaddingType padding = PaddingType::VALID;
};

struct BatchNormConfig {
    float epsilon = 1e-5f;
    float momentum = 0.9f;
    bool training = true;
    bool affine = true;  // Apply scale and bias
};

class CNNAccelerator {
public:
    CNNAccelerator(NovaContext* context, ComputeWrapper* compute, MemoryManager* memory);
    ~CNNAccelerator();
    
    bool initialize();
    void shutdown();
    
    // Core CNN Operations
    ComputeBuffer* convolution2D(ComputeBuffer* input,          // [batch, height, width, in_channels]
                                ComputeBuffer* kernel,          // [out_channels, kernel_h, kernel_w, in_channels]
                                ComputeBuffer* bias = nullptr,  // [out_channels] (optional)
                                const ConvolutionConfig& config = {});
    
    ComputeBuffer* activation(ComputeBuffer* input, ActivationType type);
    
    ComputeBuffer* pooling2D(ComputeBuffer* input, PoolingType type, 
                            const PoolingConfig& config = {});
    
    ComputeBuffer* batchNorm(ComputeBuffer* input,
                            ComputeBuffer* scale,       // [channels]
                            ComputeBuffer* bias,        // [channels] 
                            ComputeBuffer* running_mean,// [channels]
                            ComputeBuffer* running_var, // [channels]
                            const BatchNormConfig& config = {});
    
    // Layer Operations
    ComputeBuffer* fullyConnected(ComputeBuffer* input,    // [batch, input_size]
                                 ComputeBuffer* weights,   // [output_size, input_size]
                                 ComputeBuffer* bias = nullptr); // [output_size]
    
    ComputeBuffer* dropout(ComputeBuffer* input, float dropout_rate, bool training = true);
    
    // Composite Operations
    ComputeBuffer* convBlock(ComputeBuffer* input,
                           ComputeBuffer* kernel,
                           ComputeBuffer* bias = nullptr,
                           ActivationType activation = ActivationType::RELU,
                           const ConvolutionConfig& conv_config = {},
                           const PoolingConfig& pool_config = {});
    
    // Gradient Operations (for training)
    ComputeBuffer* convolutionBackward(ComputeBuffer* grad_output,
                                      ComputeBuffer* input,
                                      ComputeBuffer* kernel,
                                      const ConvolutionConfig& config = {});
    
    // Memory Management Helpers
    ComputeBuffer* allocateFeatureMap(uint32_t batch, uint32_t height, 
                                     uint32_t width, uint32_t channels,
                                     DataType type = DataType::FLOAT32);
    
    ComputeBuffer* allocateKernel(uint32_t out_channels, uint32_t kernel_h,
                                 uint32_t kernel_w, uint32_t in_channels,
                                 DataType type = DataType::FLOAT32);
    
    // Performance Utilities
    void prefetchWeights(const std::vector<ComputeBuffer*>& weights);
    void optimizeMemoryLayout(ComputeBuffer* buffer, const std::string& layout = "NHWC");
    
    // Debugging and Profiling
    void enableProfiling(bool enable) { _profiling_enabled = enable; }
    struct LayerProfile {
        std::string layer_name;
        uint64_t compute_time_ns;
        size_t memory_used_bytes;
        float gflops;
    };
    std::vector<LayerProfile> getProfilingResults() const;
    
private:
    NovaContext* _context;
    ComputeWrapper* _compute;
    MemoryManager* _memory;
    
    // Compute kernels for CNN operations
    ComputeKernel* _conv2d_kernel;
    ComputeKernel* _conv2d_depthwise_kernel;
    ComputeKernel* _activation_kernel;
    ComputeKernel* _pooling_kernel;
    ComputeKernel* _batchnorm_kernel;
    ComputeKernel* _fully_connected_kernel;
    ComputeKernel* _dropout_kernel;
    
    // Temporary buffers for complex operations
    std::vector<ComputeBuffer*> _temp_buffers;
    size_t _temp_buffer_index;
    
    bool _profiling_enabled;
    std::vector<LayerProfile> _profile_results;
    
    // Kernel creation and management
    bool _createKernels();
    void _destroyKernels();
    ComputeBuffer* _getTempBuffer(size_t size_bytes, DataType type);
    void _resetTempBuffers();
    
    // Operation implementations
    std::vector<uint32_t> _calculateOutputDims(const std::vector<uint32_t>& input_dims,
                                              const std::vector<uint32_t>& kernel_dims,
                                              const ConvolutionConfig& config);
    
    void _setupConvolutionDescriptors(ComputeKernel* kernel,
                                     ComputeBuffer* input,
                                     ComputeBuffer* weights,
                                     ComputeBuffer* output,
                                     ComputeBuffer* bias = nullptr);
    
    // Performance profiling
    void _startProfile(const std::string& operation_name);
    void _endProfile();
    std::string _current_operation;
    std::chrono::steady_clock::time_point _profile_start;
};

// Utility functions
std::vector<uint32_t> calculateConvOutputShape(const std::vector<uint32_t>& input_shape,
                                              const std::vector<uint32_t>& kernel_shape,
                                              const ConvolutionConfig& config);

size_t calculateConvFLOPs(const std::vector<uint32_t>& input_shape,
                          const std::vector<uint32_t>& kernel_shape,
                          const std::vector<uint32_t>& output_shape);

} // namespace GPU
} // namespace CARL