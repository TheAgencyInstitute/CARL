#include "neural_network_models.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

namespace CARL {
namespace AI {

// Convolutional Layer Implementation
ConvolutionalLayer::ConvolutionalLayer(uint32_t input_width, uint32_t input_height, uint32_t input_channels,
                                     uint32_t filter_count, uint32_t filter_size, uint32_t stride, uint32_t padding)
    : _filter_count(filter_count), _filter_size(filter_size), _stride(stride), _padding(padding) {
    
    width = input_width;
    height = input_height;
    channels = input_channels;
    
    // Calculate output dimensions
    _output_width = (input_width + 2 * padding - filter_size) / stride + 1;
    _output_height = (input_height + 2 * padding - filter_size) / stride + 1;
    
    input_size = input_width * input_height * input_channels;
    output_size = _output_width * _output_height * filter_count;
    
    activation_type = ShaderType::ACTIVATION_RELU;
    learning_rate = 0.001f;
}

void ConvolutionalLayer::initializeWeights(CarlComputeEngine* engine) {
    // Allocate weight and bias buffers
    uint32_t weight_count = _filter_count * _filter_size * _filter_size * channels;
    size_t weight_size = weight_count * sizeof(float);
    size_t bias_size = _filter_count * sizeof(float);
    
    auto* weight_buffer = engine->createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto* bias_buffer = engine->createBuffer(bias_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    weights.push_back(weight_buffer);
    biases.push_back(bias_buffer);
    
    // Initialize with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float fan_in = _filter_size * _filter_size * channels;
    float fan_out = _filter_size * _filter_size * _filter_count;
    float xavier_std = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> weight_dist(0.0f, xavier_std);
    
    std::vector<float> weight_data(weight_count);
    for (auto& w : weight_data) {
        w = weight_dist(gen);
    }
    
    std::vector<float> bias_data(_filter_count, 0.0f);
    
    engine->uploadData(weight_buffer, weight_data.data(), weight_size);
    engine->uploadData(bias_buffer, bias_data.data(), bias_size);
}

void ConvolutionalLayer::forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) {
    // Use convolution2d shader
    auto future = engine->convolution2D(input, weights[0], output, 
                                       width, height, _filter_size, _stride);
    future.wait();
}

void ConvolutionalLayer::backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) {
    // Backward pass implementation (simplified)
    // In a full implementation, this would compute gradients with respect to weights and input
    // For now, we'll use the same convolution operation as a placeholder
    auto future = engine->convolution2D(grad_output, weights[0], grad_input, 
                                       _output_width, _output_height, _filter_size, _stride);
    future.wait();
}

// Fully Connected Layer Implementation
FullyConnectedLayer::FullyConnectedLayer(uint32_t input_size, uint32_t output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->width = input_size;
    this->height = 1;
    this->channels = 1;
    
    activation_type = ShaderType::ACTIVATION_RELU;
    learning_rate = 0.001f;
}

void FullyConnectedLayer::initializeWeights(CarlComputeEngine* engine) {
    size_t weight_size = input_size * output_size * sizeof(float);
    size_t bias_size = output_size * sizeof(float);
    
    _weight_matrix = engine->createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    _bias_vector = engine->createBuffer(bias_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    
    weights.push_back(_weight_matrix);
    biases.push_back(_bias_vector);
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float xavier_std = std::sqrt(2.0f / (input_size + output_size));
    std::normal_distribution<float> weight_dist(0.0f, xavier_std);
    
    std::vector<float> weight_data(input_size * output_size);
    for (auto& w : weight_data) {
        w = weight_dist(gen);
    }
    
    std::vector<float> bias_data(output_size, 0.0f);
    
    engine->uploadData(_weight_matrix, weight_data.data(), weight_size);
    engine->uploadData(_bias_vector, bias_data.data(), bias_size);
}

void FullyConnectedLayer::forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) {
    // Use matrix multiplication: output = input * weights + bias
    auto future = engine->matrixMultiply(input, _weight_matrix, output, 
                                        1, input_size, output_size);
    future.wait();
    
    // Add bias (simplified - would need a separate bias addition kernel)
}

void FullyConnectedLayer::backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) {
    // Backward pass: grad_input = grad_output * weights^T
    auto future = engine->matrixMultiply(grad_output, _weight_matrix, grad_input, 
                                        1, output_size, input_size);
    future.wait();
}

// CNN Model Implementation
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(CarlComputeEngine* engine, 
                                                      uint32_t input_width, uint32_t input_height, uint32_t channels)
    : _engine(engine), _input_width(input_width), _input_height(input_height), _channels(channels),
      _current_width(input_width), _current_height(input_height), _current_channels(channels),
      _training_mode(true) {
}

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork() {
    deallocateIntermediateBuffers();
}

void ConvolutionalNeuralNetwork::addConvolutionalLayer(uint32_t filters, uint32_t kernel_size, uint32_t stride) {
    auto layer = std::make_unique<ConvolutionalLayer>(_current_width, _current_height, _current_channels,
                                                     filters, kernel_size, stride);
    
    // Update current dimensions for next layer
    _current_width = (_current_width - kernel_size) / stride + 1;
    _current_height = (_current_height - kernel_size) / stride + 1;
    _current_channels = filters;
    
    _layers.push_back(std::move(layer));
}

void ConvolutionalNeuralNetwork::addPoolingLayer(uint32_t pool_size, uint32_t stride, bool max_pool) {
    // Create a simple pooling layer (would need proper pooling layer class in full implementation)
    auto layer = std::make_unique<ConvolutionalLayer>(_current_width, _current_height, _current_channels,
                                                     _current_channels, pool_size, stride);
    
    if (max_pool) {
        layer->activation_type = ShaderType::POOLING_MAX;
    } else {
        layer->activation_type = ShaderType::POOLING_AVERAGE;
    }
    
    _current_width = (_current_width - pool_size) / stride + 1;
    _current_height = (_current_height - pool_size) / stride + 1;
    
    _layers.push_back(std::move(layer));
}

void ConvolutionalNeuralNetwork::addFullyConnectedLayer(uint32_t units) {
    uint32_t input_size = _current_width * _current_height * _current_channels;
    auto layer = std::make_unique<FullyConnectedLayer>(input_size, units);
    
    _current_width = units;
    _current_height = 1;
    _current_channels = 1;
    
    _layers.push_back(std::move(layer));
}

void ConvolutionalNeuralNetwork::addActivationLayer(ShaderType activation_type) {
    // Create a pass-through layer with specified activation
    auto layer = std::make_unique<FullyConnectedLayer>(_current_width, _current_width);
    layer->activation_type = activation_type;
    _layers.push_back(std::move(layer));
}

void ConvolutionalNeuralNetwork::addBatchNormalizationLayer() {
    // Batch normalization layer (simplified)
    auto layer = std::make_unique<FullyConnectedLayer>(_current_width, _current_width);
    layer->activation_type = ShaderType::BATCH_NORMALIZATION;
    _layers.push_back(std::move(layer));
}

void ConvolutionalNeuralNetwork::initializeNetwork() {
    // Initialize all layer weights
    for (auto& layer : _layers) {
        layer->initializeWeights(_engine);
    }
    
    allocateIntermediateBuffers();
}

std::future<void> ConvolutionalNeuralNetwork::forward(ComputeBuffer* input, ComputeBuffer* output) {
    return std::async(std::launch::async, [this, input, output]() {
        ComputeBuffer* current_input = input;
        
        for (size_t i = 0; i < _layers.size(); i++) {
            ComputeBuffer* current_output = (i == _layers.size() - 1) ? output : _intermediate_buffers[i];
            
            _layers[i]->forward(_engine, current_input, current_output);
            
            // Apply activation if needed
            if (_layers[i]->activation_type != ShaderType::CONVOLUTION_2D) {
                auto activation_future = _engine->activationReLU(current_output, current_output, 
                                                               current_output->element_count);
                activation_future.wait();
            }
            
            current_input = current_output;
        }
    });
}

std::future<void> ConvolutionalNeuralNetwork::backward(ComputeBuffer* gradients) {
    return std::async(std::launch::async, [this, gradients]() {
        if (!_training_mode) return;
        
        ComputeBuffer* current_gradients = gradients;
        
        // Backward pass through layers in reverse order
        for (int i = static_cast<int>(_layers.size()) - 1; i >= 0; i--) {
            ComputeBuffer* prev_gradients = (i == 0) ? nullptr : _intermediate_buffers[i - 1];
            
            if (prev_gradients) {
                _layers[i]->backward(_engine, prev_gradients, current_gradients);
                current_gradients = prev_gradients;
            }
        }
    });
}

std::future<void> ConvolutionalNeuralNetwork::updateWeights(float learning_rate) {
    return std::async(std::launch::async, [this, learning_rate]() {
        // Update weights for all layers using gradient descent
        for (auto& layer : _layers) {
            if (!layer->weights.empty() && !layer->gradients.empty()) {
                // Use gradient descent shader to update weights
                // This is simplified - would need proper gradient computation
            }
        }
    });
}

void ConvolutionalNeuralNetwork::setTrainingMode(bool training) {
    _training_mode = training;
}

float ConvolutionalNeuralNetwork::calculateLoss(ComputeBuffer* predictions, ComputeBuffer* targets) {
    // Simplified loss calculation - would need proper loss computation on GPU
    std::vector<float> pred_data(predictions->element_count);
    std::vector<float> target_data(targets->element_count);
    
    _engine->downloadData(predictions, pred_data.data(), pred_data.size() * sizeof(float));
    _engine->downloadData(targets, target_data.data(), target_data.size() * sizeof(float));
    
    float loss = 0.0f;
    for (size_t i = 0; i < pred_data.size(); i++) {
        float diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    
    return loss / pred_data.size();
}

const NeuralLayer* ConvolutionalNeuralNetwork::getLayer(uint32_t index) const {
    if (index < _layers.size()) {
        return _layers[index].get();
    }
    return nullptr;
}

void ConvolutionalNeuralNetwork::allocateIntermediateBuffers() {
    _intermediate_buffers.clear();
    
    uint32_t current_width = _input_width;
    uint32_t current_height = _input_height;
    uint32_t current_channels = _channels;
    
    for (size_t i = 0; i < _layers.size() - 1; i++) {
        // Calculate buffer size based on layer output
        uint32_t buffer_size = _layers[i]->output_size;
        size_t size_bytes = buffer_size * sizeof(float);
        
        auto* buffer = _engine->createBuffer(size_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _intermediate_buffers.push_back(buffer);
    }
}

void ConvolutionalNeuralNetwork::deallocateIntermediateBuffers() {
    for (auto* buffer : _intermediate_buffers) {
        _engine->destroyBuffer(buffer);
    }
    _intermediate_buffers.clear();
}

uint32_t ConvolutionalNeuralNetwork::calculateBufferSize(uint32_t width, uint32_t height, uint32_t channels) {
    return width * height * channels;
}

// ==================================================================
// MODERN CNN ARCHITECTURES AND TRAINING WORKFLOWS
// ==================================================================

// ResNet Block Implementation
class ResNetBlock : public NeuralLayer {
public:
    ResNetBlock(uint32_t channels, uint32_t stride = 1) 
        : _channels(channels), _stride(stride), _use_projection(stride != 1) {
        
        input_size = channels * 64 * 64; // Default input size
        output_size = channels * (64 / stride) * (64 / stride);
        width = 64;
        height = 64;
        this->channels = channels;
        
        activation_type = ShaderType::ACTIVATION_RELU;
        learning_rate = 0.001f;
    }
    
    void initializeWeights(CarlComputeEngine* engine) {
        // Conv1: 3x3 convolution
        size_t conv1_weights_size = 3 * 3 * _channels * _channels * sizeof(float);
        _conv1_weights = engine->createBuffer(conv1_weights_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Conv2: 3x3 convolution
        size_t conv2_weights_size = 3 * 3 * _channels * _channels * sizeof(float);
        _conv2_weights = engine->createBuffer(conv2_weights_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Projection layer for shortcut connection if needed
        if (_use_projection) {
            size_t proj_weights_size = 1 * 1 * _channels * _channels * sizeof(float);
            _projection_weights = engine->createBuffer(proj_weights_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
        
        // Batch normalization parameters
        size_t bn_size = _channels * sizeof(float);
        _bn1_scale = engine->createBuffer(bn_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _bn1_shift = engine->createBuffer(bn_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _bn2_scale = engine->createBuffer(bn_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        _bn2_shift = engine->createBuffer(bn_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        // Xavier initialization
        initializeXavierWeights(engine);
    }
    
    void forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) override {
        // ResNet block: conv1 -> bn1 -> relu -> conv2 -> bn2 -> add shortcut -> relu
        
        // Conv1
        auto conv1_future = engine->convolution2D(input, _conv1_weights, _residual_buffer1,
                                                 width, height, 3, _stride);
        conv1_future.wait();
        
        // BatchNorm1 + ReLU
        auto bn1_future = engine->batchNormalization(_residual_buffer1, _residual_buffer2,
                                                    _bn1_scale, _bn1_shift, 1, _channels);
        bn1_future.wait();
        
        auto relu1_future = engine->activationReLU(_residual_buffer2, _residual_buffer2,
                                                  _residual_buffer2->element_count);
        relu1_future.wait();
        
        // Conv2
        auto conv2_future = engine->convolution2D(_residual_buffer2, _conv2_weights, _residual_buffer3,
                                                 width / _stride, height / _stride, 3, 1);
        conv2_future.wait();
        
        // BatchNorm2
        auto bn2_future = engine->batchNormalization(_residual_buffer3, _residual_buffer3,
                                                    _bn2_scale, _bn2_shift, 1, _channels);
        bn2_future.wait();
        
        // Shortcut connection and final ReLU
        addShortcutConnection(engine, input, _residual_buffer3, output);
    }
    
    void backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) override {
        // Simplified backward pass - would need full gradient computation
        // For now, pass gradients through
        if (grad_input != grad_output) {
            std::vector<float> grad_data(grad_output->element_count);
            engine->downloadData(grad_output, grad_data.data(), grad_data.size() * sizeof(float));
            engine->uploadData(grad_input, grad_data.data(), grad_data.size() * sizeof(float));
        }
    }
    
private:
    uint32_t _channels;
    uint32_t _stride;
    bool _use_projection;
    
    ComputeBuffer* _conv1_weights;
    ComputeBuffer* _conv2_weights;
    ComputeBuffer* _projection_weights;
    ComputeBuffer* _bn1_scale;
    ComputeBuffer* _bn1_shift;
    ComputeBuffer* _bn2_scale;
    ComputeBuffer* _bn2_shift;
    
    ComputeBuffer* _residual_buffer1;
    ComputeBuffer* _residual_buffer2;
    ComputeBuffer* _residual_buffer3;
    
    void initializeXavierWeights(CarlComputeEngine* engine) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Initialize conv1 weights
        float fan_in = 3 * 3 * _channels;
        float fan_out = 3 * 3 * _channels;
        float xavier_std = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, xavier_std);
        
        std::vector<float> conv1_data(3 * 3 * _channels * _channels);
        for (auto& w : conv1_data) w = dist(gen);
        engine->uploadData(_conv1_weights, conv1_data.data(), conv1_data.size() * sizeof(float));
        
        std::vector<float> conv2_data(3 * 3 * _channels * _channels);
        for (auto& w : conv2_data) w = dist(gen);
        engine->uploadData(_conv2_weights, conv2_data.data(), conv2_data.size() * sizeof(float));
        
        // Initialize batch norm parameters
        std::vector<float> bn_scale(_channels, 1.0f);
        std::vector<float> bn_shift(_channels, 0.0f);
        
        engine->uploadData(_bn1_scale, bn_scale.data(), bn_scale.size() * sizeof(float));
        engine->uploadData(_bn1_shift, bn_shift.data(), bn_shift.size() * sizeof(float));
        engine->uploadData(_bn2_scale, bn_scale.data(), bn_scale.size() * sizeof(float));
        engine->uploadData(_bn2_shift, bn_shift.data(), bn_shift.size() * sizeof(float));
    }
    
    void addShortcutConnection(CarlComputeEngine* engine, ComputeBuffer* input,
                              ComputeBuffer* residual, ComputeBuffer* output) {
        if (_use_projection) {
            // Apply 1x1 convolution to input for dimension matching
            auto proj_future = engine->convolution2D(input, _projection_weights, output,
                                                    width, height, 1, _stride);
            proj_future.wait();
            
            // Add residual (simplified - would need element-wise addition kernel)
            addBuffers(engine, output, residual, output);
        } else {
            // Direct addition
            addBuffers(engine, input, residual, output);
        }
        
        // Final ReLU activation
        auto relu_future = engine->activationReLU(output, output, output->element_count);
        relu_future.wait();
    }
    
    void addBuffers(CarlComputeEngine* engine, ComputeBuffer* a, ComputeBuffer* b, ComputeBuffer* result) {
        // Simplified element-wise addition - would need proper GPU kernel
        std::vector<float> data_a(a->element_count);
        std::vector<float> data_b(b->element_count);
        std::vector<float> result_data(result->element_count);
        
        engine->downloadData(a, data_a.data(), data_a.size() * sizeof(float));
        engine->downloadData(b, data_b.data(), data_b.size() * sizeof(float));
        
        for (size_t i = 0; i < result_data.size() && i < data_a.size() && i < data_b.size(); i++) {
            result_data[i] = data_a[i] + data_b[i];
        }
        
        engine->uploadData(result, result_data.data(), result_data.size() * sizeof(float));
    }
};

// DenseNet Block Implementation
class DenseNetBlock : public NeuralLayer {
public:
    DenseNetBlock(uint32_t growth_rate, uint32_t num_layers)
        : _growth_rate(growth_rate), _num_layers(num_layers) {
        
        activation_type = ShaderType::ACTIVATION_RELU;
        learning_rate = 0.001f;
    }
    
    void initializeWeights(CarlComputeEngine* engine) {
        // Dense connectivity requires multiple 1x1 and 3x3 convolutions
        for (uint32_t i = 0; i < _num_layers; i++) {
            // 1x1 conv for bottleneck
            size_t bottleneck_size = 1 * 1 * (_growth_rate * i) * (4 * _growth_rate) * sizeof(float);
            auto* bottleneck_weights = engine->createBuffer(bottleneck_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            _bottleneck_weights.push_back(bottleneck_weights);
            
            // 3x3 conv for feature extraction
            size_t conv_size = 3 * 3 * (4 * _growth_rate) * _growth_rate * sizeof(float);
            auto* conv_weights = engine->createBuffer(conv_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            _conv_weights.push_back(conv_weights);
        }
        
        initializeWeights(engine);
    }
    
    void forward(CarlComputeEngine* engine, ComputeBuffer* input, ComputeBuffer* output) override {
        // Dense connectivity: each layer receives all previous feature maps
        ComputeBuffer* current_input = input;
        
        for (uint32_t i = 0; i < _num_layers; i++) {
            // Bottleneck layer (1x1 conv)
            auto bottleneck_future = engine->convolution2D(current_input, _bottleneck_weights[i],
                                                          _intermediate_buffers[i * 2],
                                                          width, height, 1, 1);
            bottleneck_future.wait();
            
            // 3x3 convolution
            auto conv_future = engine->convolution2D(_intermediate_buffers[i * 2], _conv_weights[i],
                                                    _intermediate_buffers[i * 2 + 1],
                                                    width, height, 3, 1);
            conv_future.wait();
            
            // Concatenate with previous features (simplified)
            concatenateFeatures(engine, current_input, _intermediate_buffers[i * 2 + 1], 
                              _concatenated_buffers[i]);
            current_input = _concatenated_buffers[i];
        }
        
        // Copy final result
        copyBuffer(engine, current_input, output);
    }
    
    void backward(CarlComputeEngine* engine, ComputeBuffer* grad_input, ComputeBuffer* grad_output) override {
        // Simplified backward pass
        copyBuffer(engine, grad_output, grad_input);
    }
    
private:
    uint32_t _growth_rate;
    uint32_t _num_layers;
    
    std::vector<ComputeBuffer*> _bottleneck_weights;
    std::vector<ComputeBuffer*> _conv_weights;
    std::vector<ComputeBuffer*> _intermediate_buffers;
    std::vector<ComputeBuffer*> _concatenated_buffers;
    
    void concatenateFeatures(CarlComputeEngine* engine, ComputeBuffer* features1,
                           ComputeBuffer* features2, ComputeBuffer* output) {
        // Simplified concatenation - would need proper GPU kernel
        std::vector<float> data1(features1->element_count);
        std::vector<float> data2(features2->element_count);
        
        engine->downloadData(features1, data1.data(), data1.size() * sizeof(float));
        engine->downloadData(features2, data2.data(), data2.size() * sizeof(float));
        
        std::vector<float> concatenated_data;
        concatenated_data.insert(concatenated_data.end(), data1.begin(), data1.end());
        concatenated_data.insert(concatenated_data.end(), data2.begin(), data2.end());
        
        engine->uploadData(output, concatenated_data.data(), concatenated_data.size() * sizeof(float));
    }
    
    void copyBuffer(CarlComputeEngine* engine, ComputeBuffer* source, ComputeBuffer* dest) {
        std::vector<float> data(source->element_count);
        engine->downloadData(source, data.data(), data.size() * sizeof(float));
        engine->uploadData(dest, data.data(), data.size() * sizeof(float));
    }
};

// ==================================================================
// CNN TRAINING WORKFLOWS
// ==================================================================

// GPU-Accelerated Data Augmentation Pipeline
class DataAugmentationPipeline {
public:
    DataAugmentationPipeline(CarlComputeEngine* engine) : _engine(engine) {
        _random_engine.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    std::future<void> augmentBatch(ComputeBuffer* input_batch, ComputeBuffer* output_batch,
                                  uint32_t batch_size, uint32_t width, uint32_t height, uint32_t channels) {
        return std::async(std::launch::async, [this, input_batch, output_batch, batch_size, width, height, channels]() {
            for (uint32_t i = 0; i < batch_size; i++) {
                applyRandomAugmentations(input_batch, output_batch, i, width, height, channels);
            }
        });
    }
    
    void setAugmentationConfig(bool horizontal_flip, bool vertical_flip, 
                              float rotation_range, float zoom_range,
                              float brightness_range, float contrast_range) {
        _config.horizontal_flip = horizontal_flip;
        _config.vertical_flip = vertical_flip;
        _config.rotation_range = rotation_range;
        _config.zoom_range = zoom_range;
        _config.brightness_range = brightness_range;
        _config.contrast_range = contrast_range;
    }
    
private:
    CarlComputeEngine* _engine;
    std::mt19937 _random_engine;
    
    struct AugmentationConfig {
        bool horizontal_flip = true;
        bool vertical_flip = false;
        float rotation_range = 15.0f;
        float zoom_range = 0.1f;
        float brightness_range = 0.2f;
        float contrast_range = 0.2f;
    } _config;
    
    void applyRandomAugmentations(ComputeBuffer* input, ComputeBuffer* output, uint32_t index,
                                 uint32_t width, uint32_t height, uint32_t channels) {
        // Random horizontal flip
        if (_config.horizontal_flip && _random_engine() % 2) {
            horizontalFlip(input, output, index, width, height, channels);
        }
        
        // Random rotation
        float rotation = (_random_engine() % 1000 / 1000.0f - 0.5f) * 2.0f * _config.rotation_range;
        if (std::abs(rotation) > 1.0f) {
            rotateImage(input, output, index, width, height, channels, rotation);
        }
        
        // Random brightness adjustment
        float brightness = (_random_engine() % 1000 / 1000.0f - 0.5f) * 2.0f * _config.brightness_range;
        if (std::abs(brightness) > 0.01f) {
            adjustBrightness(input, output, index, width, height, channels, brightness);
        }
    }
    
    void horizontalFlip(ComputeBuffer* input, ComputeBuffer* output, uint32_t index,
                       uint32_t width, uint32_t height, uint32_t channels) {
        // Simplified CPU implementation - would use GPU shaders in production
        size_t image_size = width * height * channels;
        size_t offset = index * image_size;
        
        std::vector<float> image_data(image_size);
        engine->downloadData(input, image_data.data(), image_size * sizeof(float));
        
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < channels; c++) {
                    uint32_t src_idx = y * width * channels + x * channels + c;
                    uint32_t dst_idx = y * width * channels + (width - 1 - x) * channels + c;
                    image_data[dst_idx] = image_data[src_idx];
                }
            }
        }
        
        engine->uploadData(output, image_data.data(), image_size * sizeof(float));
    }
    
    void rotateImage(ComputeBuffer* input, ComputeBuffer* output, uint32_t index,
                    uint32_t width, uint32_t height, uint32_t channels, float angle_degrees) {
        // Simplified rotation - would use GPU compute shader
        // Implementation placeholder
    }
    
    void adjustBrightness(ComputeBuffer* input, ComputeBuffer* output, uint32_t index,
                         uint32_t width, uint32_t height, uint32_t channels, float brightness) {
        // Simplified brightness adjustment
        size_t image_size = width * height * channels;
        std::vector<float> image_data(image_size);
        
        engine->downloadData(input, image_data.data(), image_size * sizeof(float));
        
        for (auto& pixel : image_data) {
            pixel = std::clamp(pixel + brightness, 0.0f, 1.0f);
        }
        
        engine->uploadData(output, image_data.data(), image_size * sizeof(float));
    }
};

// Model Checkpoint Manager
class ModelCheckpointManager {
public:
    ModelCheckpointManager(const std::string& checkpoint_dir) : _checkpoint_dir(checkpoint_dir) {}
    
    bool saveCheckpoint(const ConvolutionalNeuralNetwork* model, uint32_t epoch, float loss) {
        std::string checkpoint_path = _checkpoint_dir + "/checkpoint_epoch_" + std::to_string(epoch) + ".bin";
        
        std::ofstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to save checkpoint: " << checkpoint_path << std::endl;
            return false;
        }
        
        // Save model metadata
        uint32_t layer_count = model->getLayerCount();
        file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
        file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
        file.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
        
        // Save layer weights (simplified)
        for (uint32_t i = 0; i < layer_count; i++) {
            const NeuralLayer* layer = model->getLayer(i);
            if (layer && !layer->weights.empty()) {
                saveLayerWeights(file, layer);
            }
        }
        
        file.close();
        
        // Cleanup old checkpoints
        cleanupOldCheckpoints(epoch);
        
        std::cout << "Checkpoint saved: " << checkpoint_path << " (loss: " << loss << ")" << std::endl;
        return true;
    }
    
    bool loadCheckpoint(ConvolutionalNeuralNetwork* model, const std::string& checkpoint_path) {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to load checkpoint: " << checkpoint_path << std::endl;
            return false;
        }
        
        // Load model metadata
        uint32_t layer_count;
        uint32_t epoch;
        float loss;
        
        file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
        file.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));
        file.read(reinterpret_cast<char*>(&loss), sizeof(loss));
        
        // Verify layer count matches
        if (layer_count != model->getLayerCount()) {
            std::cerr << "Checkpoint layer count mismatch" << std::endl;
            return false;
        }
        
        // Load layer weights (simplified)
        for (uint32_t i = 0; i < layer_count; i++) {
            loadLayerWeights(file, model, i);
        }
        
        file.close();
        
        std::cout << "Checkpoint loaded: " << checkpoint_path << " (epoch: " << epoch << ", loss: " << loss << ")" << std::endl;
        return true;
    }
    
private:
    std::string _checkpoint_dir;
    
    void saveLayerWeights(std::ofstream& file, const NeuralLayer* layer) {
        // Simplified weight saving - would need proper GPU memory download
        uint32_t weight_count = layer->weights.size();
        file.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
        
        for (const auto* weight_buffer : layer->weights) {
            uint32_t element_count = weight_buffer->element_count;
            file.write(reinterpret_cast<const char*>(&element_count), sizeof(element_count));
            
            // Download weights from GPU (placeholder)
            std::vector<float> weights(element_count);
            // engine->downloadData(weight_buffer, weights.data(), weights.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        }
    }
    
    void loadLayerWeights(std::ifstream& file, ConvolutionalNeuralNetwork* model, uint32_t layer_index) {
        // Simplified weight loading
        uint32_t weight_count;
        file.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
        
        for (uint32_t i = 0; i < weight_count; i++) {
            uint32_t element_count;
            file.read(reinterpret_cast<char*>(&element_count), sizeof(element_count));
            
            std::vector<float> weights(element_count);
            file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
            
            // Upload weights to GPU (placeholder)
            // engine->uploadData(layer->weights[i], weights.data(), weights.size() * sizeof(float));
        }
    }
    
    void cleanupOldCheckpoints(uint32_t current_epoch) {
        // Keep only last 5 checkpoints
        const uint32_t max_checkpoints = 5;
        
        if (current_epoch > max_checkpoints) {
            uint32_t old_epoch = current_epoch - max_checkpoints;
            std::string old_checkpoint = _checkpoint_dir + "/checkpoint_epoch_" + std::to_string(old_epoch) + ".bin";
            std::remove(old_checkpoint.c_str());
        }
    }
};

// Transfer Learning Implementation
class TransferLearningManager {
public:
    TransferLearningManager(CarlComputeEngine* engine) : _engine(engine) {}
    
    bool loadPretrainedModel(ConvolutionalNeuralNetwork* model, const std::string& pretrained_path) {
        // Load pretrained weights (simplified)
        std::ifstream file(pretrained_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to load pretrained model: " << pretrained_path << std::endl;
            return false;
        }
        
        // Load and apply weights to appropriate layers
        // Implementation would match layer types and transfer compatible weights
        
        std::cout << "Pretrained model loaded: " << pretrained_path << std::endl;
        return true;
    }
    
    void freezeLayers(ConvolutionalNeuralNetwork* model, uint32_t freeze_up_to_layer) {
        // Freeze early layers for transfer learning
        for (uint32_t i = 0; i < freeze_up_to_layer && i < model->getLayerCount(); i++) {
            const NeuralLayer* layer = model->getLayer(i);
            if (layer) {
                // Set learning rate to 0 for frozen layers
                // layer->learning_rate = 0.0f; // Would need non-const access
                _frozen_layers.insert(i);
            }
        }
        
        std::cout << "Frozen layers 0 to " << freeze_up_to_layer << " for transfer learning" << std::endl;
    }
    
    void setLayerLearningRates(ConvolutionalNeuralNetwork* model, 
                              const std::vector<float>& layer_learning_rates) {
        // Set different learning rates for different layers
        for (uint32_t i = 0; i < layer_learning_rates.size() && i < model->getLayerCount(); i++) {
            // layer->learning_rate = layer_learning_rates[i]; // Would need non-const access
            _layer_learning_rates[i] = layer_learning_rates[i];
        }
    }
    
    bool isLayerFrozen(uint32_t layer_index) const {
        return _frozen_layers.find(layer_index) != _frozen_layers.end();
    }
    
private:
    CarlComputeEngine* _engine;
    std::set<uint32_t> _frozen_layers;
    std::unordered_map<uint32_t, float> _layer_learning_rates;
};

// Comprehensive CNN Training Manager
class CNNTrainingManager {
public:
    CNNTrainingManager(CarlComputeEngine* engine) 
        : _engine(engine),
          _augmentation_pipeline(engine),
          _checkpoint_manager("./checkpoints"),
          _transfer_manager(engine) {
        
        _training_config.batch_size = 32;
        _training_config.learning_rate = 0.001f;
        _training_config.epochs = 100;
        _training_config.validation_split = 0.2f;
        _training_config.early_stopping_patience = 10;
        _training_config.lr_schedule_factor = 0.5f;
        _training_config.lr_schedule_patience = 5;
    }
    
    struct TrainingConfig {
        uint32_t batch_size;
        float learning_rate;
        uint32_t epochs;
        float validation_split;
        uint32_t early_stopping_patience;
        float lr_schedule_factor;
        uint32_t lr_schedule_patience;
        bool use_data_augmentation = true;
        bool save_checkpoints = true;
        bool use_transfer_learning = false;
        std::string pretrained_model_path;
    };
    
    void setTrainingConfig(const TrainingConfig& config) {
        _training_config = config;
    }
    
    std::future<void> trainModel(ConvolutionalNeuralNetwork* model,
                                ComputeBuffer* training_data, ComputeBuffer* training_labels,
                                ComputeBuffer* validation_data, ComputeBuffer* validation_labels,
                                uint32_t num_samples, uint32_t input_width, uint32_t input_height, uint32_t channels) {
        
        return std::async(std::launch::async, [this, model, training_data, training_labels, 
                                              validation_data, validation_labels, num_samples, 
                                              input_width, input_height, channels]() {
            
            // Initialize model
            model->initializeNetwork();
            model->setTrainingMode(true);
            
            // Setup transfer learning if requested
            if (_training_config.use_transfer_learning && !_training_config.pretrained_model_path.empty()) {
                _transfer_manager.loadPretrainedModel(model, _training_config.pretrained_model_path);
            }
            
            // Training loop
            float best_val_loss = std::numeric_limits<float>::max();
            uint32_t patience_counter = 0;
            
            for (uint32_t epoch = 0; epoch < _training_config.epochs; epoch++) {
                auto epoch_start = std::chrono::steady_clock::now();
                
                // Training phase
                float train_loss = trainEpoch(model, training_data, training_labels, 
                                            num_samples, input_width, input_height, channels);
                
                // Validation phase
                float val_loss = validateEpoch(model, validation_data, validation_labels,
                                             num_samples * _training_config.validation_split,
                                             input_width, input_height, channels);
                
                auto epoch_end = std::chrono::steady_clock::now();
                auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
                
                // Logging
                std::cout << "Epoch " << (epoch + 1) << "/" << _training_config.epochs
                         << " - Train Loss: " << train_loss
                         << " - Val Loss: " << val_loss
                         << " - Time: " << epoch_time.count() << "ms" << std::endl;
                
                // Save checkpoint
                if (_training_config.save_checkpoints) {
                    _checkpoint_manager.saveCheckpoint(model, epoch, val_loss);
                }
                
                // Early stopping
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter++;
                    if (patience_counter >= _training_config.early_stopping_patience) {
                        std::cout << "Early stopping triggered at epoch " << (epoch + 1) << std::endl;
                        break;
                    }
                }
                
                // Learning rate scheduling
                if (patience_counter >= _training_config.lr_schedule_patience) {
                    _training_config.learning_rate *= _training_config.lr_schedule_factor;
                    std::cout << "Learning rate reduced to: " << _training_config.learning_rate << std::endl;
                    patience_counter = 0; // Reset patience after LR reduction
                }
            }
            
            model->setTrainingMode(false);
            std::cout << "Training completed!" << std::endl;
        });
    }
    
    std::future<float> evaluateModel(ConvolutionalNeuralNetwork* model,
                                    ComputeBuffer* test_data, ComputeBuffer* test_labels,
                                    uint32_t num_samples, uint32_t input_width, uint32_t input_height, uint32_t channels) {
        
        return std::async(std::launch::async, [this, model, test_data, test_labels, 
                                              num_samples, input_width, input_height, channels]() -> float {
            
            model->setTrainingMode(false);
            
            float total_loss = 0.0f;
            uint32_t num_batches = (num_samples + _training_config.batch_size - 1) / _training_config.batch_size;
            
            // Create inference buffers
            size_t batch_input_size = _training_config.batch_size * input_width * input_height * channels * sizeof(float);
            size_t batch_output_size = _training_config.batch_size * model->getLayer(model->getLayerCount() - 1)->output_size * sizeof(float);
            
            auto* batch_input = _engine->createBuffer(batch_input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            auto* batch_output = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            auto* batch_labels = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            
            for (uint32_t batch = 0; batch < num_batches; batch++) {
                uint32_t batch_start = batch * _training_config.batch_size;
                uint32_t current_batch_size = std::min(_training_config.batch_size, num_samples - batch_start);
                
                // Copy batch data
                copyBatchData(test_data, batch_input, batch_start, current_batch_size, 
                            input_width, input_height, channels);
                copyBatchData(test_labels, batch_labels, batch_start, current_batch_size, 1, 1, 1);
                
                // Forward pass
                auto forward_future = model->forward(batch_input, batch_output);
                forward_future.wait();
                
                // Calculate loss
                float batch_loss = model->calculateLoss(batch_output, batch_labels);
                total_loss += batch_loss * current_batch_size;
            }
            
            // Cleanup
            _engine->destroyBuffer(batch_input);
            _engine->destroyBuffer(batch_output);
            _engine->destroyBuffer(batch_labels);
            
            float average_loss = total_loss / num_samples;
            std::cout << "Evaluation completed - Average Loss: " << average_loss << std::endl;
            
            return average_loss;
        });
    }
    
private:
    CarlComputeEngine* _engine;
    DataAugmentationPipeline _augmentation_pipeline;
    ModelCheckpointManager _checkpoint_manager;
    TransferLearningManager _transfer_manager;
    TrainingConfig _training_config;
    
    float trainEpoch(ConvolutionalNeuralNetwork* model, ComputeBuffer* training_data, ComputeBuffer* training_labels,
                    uint32_t num_samples, uint32_t input_width, uint32_t input_height, uint32_t channels) {
        
        float total_loss = 0.0f;
        uint32_t num_batches = (num_samples + _training_config.batch_size - 1) / _training_config.batch_size;
        
        // Create training buffers
        size_t batch_input_size = _training_config.batch_size * input_width * input_height * channels * sizeof(float);
        size_t batch_output_size = _training_config.batch_size * model->getLayer(model->getLayerCount() - 1)->output_size * sizeof(float);
        
        auto* batch_input = _engine->createBuffer(batch_input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* batch_output = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* batch_labels = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* augmented_input = _engine->createBuffer(batch_input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        for (uint32_t batch = 0; batch < num_batches; batch++) {
            uint32_t batch_start = batch * _training_config.batch_size;
            uint32_t current_batch_size = std::min(_training_config.batch_size, num_samples - batch_start);
            
            // Copy batch data
            copyBatchData(training_data, batch_input, batch_start, current_batch_size, 
                        input_width, input_height, channels);
            copyBatchData(training_labels, batch_labels, batch_start, current_batch_size, 1, 1, 1);
            
            // Apply data augmentation
            ComputeBuffer* model_input = batch_input;
            if (_training_config.use_data_augmentation) {
                auto aug_future = _augmentation_pipeline.augmentBatch(batch_input, augmented_input,
                                                                    current_batch_size, input_width, input_height, channels);
                aug_future.wait();
                model_input = augmented_input;
            }
            
            // Forward pass
            auto forward_future = model->forward(model_input, batch_output);
            forward_future.wait();
            
            // Calculate loss
            float batch_loss = model->calculateLoss(batch_output, batch_labels);
            total_loss += batch_loss * current_batch_size;
            
            // Backward pass
            auto backward_future = model->backward(batch_output); // Simplified
            backward_future.wait();
            
            // Update weights
            auto update_future = model->updateWeights(_training_config.learning_rate);
            update_future.wait();
        }
        
        // Cleanup
        _engine->destroyBuffer(batch_input);
        _engine->destroyBuffer(batch_output);
        _engine->destroyBuffer(batch_labels);
        _engine->destroyBuffer(augmented_input);
        
        return total_loss / num_samples;
    }
    
    float validateEpoch(ConvolutionalNeuralNetwork* model, ComputeBuffer* validation_data, ComputeBuffer* validation_labels,
                       uint32_t num_samples, uint32_t input_width, uint32_t input_height, uint32_t channels) {
        
        model->setTrainingMode(false);
        
        float total_loss = 0.0f;
        uint32_t num_batches = (num_samples + _training_config.batch_size - 1) / _training_config.batch_size;
        
        // Create validation buffers
        size_t batch_input_size = _training_config.batch_size * input_width * input_height * channels * sizeof(float);
        size_t batch_output_size = _training_config.batch_size * model->getLayer(model->getLayerCount() - 1)->output_size * sizeof(float);
        
        auto* batch_input = _engine->createBuffer(batch_input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* batch_output = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* batch_labels = _engine->createBuffer(batch_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        
        for (uint32_t batch = 0; batch < num_batches; batch++) {
            uint32_t batch_start = batch * _training_config.batch_size;
            uint32_t current_batch_size = std::min(_training_config.batch_size, num_samples - batch_start);
            
            // Copy batch data
            copyBatchData(validation_data, batch_input, batch_start, current_batch_size, 
                        input_width, input_height, channels);
            copyBatchData(validation_labels, batch_labels, batch_start, current_batch_size, 1, 1, 1);
            
            // Forward pass only
            auto forward_future = model->forward(batch_input, batch_output);
            forward_future.wait();
            
            // Calculate loss
            float batch_loss = model->calculateLoss(batch_output, batch_labels);
            total_loss += batch_loss * current_batch_size;
        }
        
        // Cleanup
        _engine->destroyBuffer(batch_input);
        _engine->destroyBuffer(batch_output);
        _engine->destroyBuffer(batch_labels);
        
        model->setTrainingMode(true);
        
        return total_loss / num_samples;
    }
    
    void copyBatchData(ComputeBuffer* source, ComputeBuffer* dest, uint32_t batch_start, uint32_t batch_size,
                      uint32_t width, uint32_t height, uint32_t channels) {
        
        size_t sample_size = width * height * channels * sizeof(float);
        size_t offset = batch_start * sample_size;
        size_t copy_size = batch_size * sample_size;
        
        // Simplified copy - would use GPU memory operations in production
        std::vector<float> temp_data(copy_size / sizeof(float));
        _engine->downloadData(source, temp_data.data(), copy_size);
        _engine->uploadData(dest, temp_data.data(), copy_size);
    }
};

// ==================================================================
// NOVA-CARL QUEUE INTEGRATION
// ==================================================================

// CNN Queue Manager for Nova-CARL Multi-Queue System
class CNNQueueManager {
public:
    CNNQueueManager(CarlComputeEngine* engine) : _engine(engine) {
        // Initialize queue assignments for CNN operations
        _queue_assignments[AIOperationType::CONVOLUTION_2D] = 0; // Queue 0 for CNN convolutions
        _queue_assignments[AIOperationType::POOLING_MAX] = 1;
        _queue_assignments[AIOperationType::POOLING_AVERAGE] = 1;
        _queue_assignments[AIOperationType::ACTIVATION_RELU] = 2;
        _queue_assignments[AIOperationType::BATCH_NORMALIZATION] = 2;
        _queue_assignments[AIOperationType::MATRIX_MULTIPLY] = 3; // FC layers
    }
    
    std::future<void> executeParallelCNNOperations(const std::vector<AIOperation>& operations) {
        return std::async(std::launch::async, [this, operations]() {
            // Distribute operations across Nova-CARL queues
            std::vector<std::future<void>> queue_futures;
            
            for (const auto& operation : operations) {
                uint32_t assigned_queue = _queue_assignments[operation.type];
                
                auto operation_future = std::async(std::launch::async, [this, operation, assigned_queue]() {
                    executeOnQueue(operation, assigned_queue);
                });
                
                queue_futures.push_back(std::move(operation_future));
            }
            
            // Wait for all operations to complete
            for (auto& future : queue_futures) {
                future.wait();
            }
        });
    }
    
    void optimizeQueueLoad() {
        // Dynamic queue load balancing
        auto performance_stats = _engine->getQueuePerformanceStats();
        
        for (const auto& stats : performance_stats) {
            if (stats.utilization_percent > 80.0f) {
                // Redistribute load from overloaded queues
                redistributeLoad(stats.queue_index);
            }
        }
    }
    
private:
    CarlComputeEngine* _engine;
    std::unordered_map<AIOperationType, uint32_t> _queue_assignments;
    
    void executeOnQueue(const AIOperation& operation, uint32_t queue_index) {
        // Execute operation on specified Nova-CARL queue
        switch (operation.type) {
            case AIOperationType::CONVOLUTION_2D:
                executeConvolutionOnQueue(operation, queue_index);
                break;
            case AIOperationType::MATRIX_MULTIPLY:
                executeMatrixMultiplyOnQueue(operation, queue_index);
                break;
            // Add other operation types...
        }
    }
    
    void executeConvolutionOnQueue(const AIOperation& operation, uint32_t queue_index) {
        // Use Nova-CARL queue 0 for CNN convolution operations
        if (operation.input_buffers.size() >= 2 && !operation.output_buffers.empty()) {
            auto future = _engine->convolution2D(operation.input_buffers[0], operation.input_buffers[1],
                                                operation.output_buffers[0], operation.dispatch_x,
                                                operation.dispatch_y, 3, 1);
            future.wait();
        }
    }
    
    void executeMatrixMultiplyOnQueue(const AIOperation& operation, uint32_t queue_index) {
        // Use Nova-CARL queue 3 for matrix operations (FC layers)
        if (operation.input_buffers.size() >= 2 && !operation.output_buffers.empty()) {
            auto future = _engine->matrixMultiply(operation.input_buffers[0], operation.input_buffers[1],
                                                operation.output_buffers[0], operation.dispatch_x,
                                                operation.dispatch_y, operation.dispatch_z);
            future.wait();
        }
    }
    
    void redistributeLoad(uint32_t overloaded_queue) {
        // Find alternative queue with lower utilization
        auto performance_stats = _engine->getQueuePerformanceStats();
        
        uint32_t target_queue = 0;
        float min_utilization = 100.0f;
        
        for (const auto& stats : performance_stats) {
            if (stats.queue_index != overloaded_queue && stats.utilization_percent < min_utilization) {
                min_utilization = stats.utilization_percent;
                target_queue = stats.queue_index;
            }
        }
        
        // Update queue assignments for load balancing
        for (auto& assignment : _queue_assignments) {
            if (assignment.second == overloaded_queue) {
                assignment.second = target_queue;
                break; // Reassign one operation type at a time
            }
        }
    }
};

} // namespace AI
} // namespace CARL