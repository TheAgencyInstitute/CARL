#!/bin/bash

# CARL AI Compute Shader Compilation Script
# Compiles GLSL compute shaders to SPIR-V for Nova-CARL

SHADER_DIR="/home/persist/repos/work/vazio/CARL/src/shaders"
OUTPUT_DIR="$SHADER_DIR/compiled"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "CARL Shader Compilation - Nova-CARL Integration"
echo "==============================================="

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Install Vulkan SDK."
    exit 1
fi

# Compile compute shaders
shaders=(
    # Core AI Operations
    "matrix_multiply.comp"
    "convolution2d.comp" 
    "activation_relu.comp"
    "activation_softmax.comp"
    "pooling_max.comp"
    "pooling_average.comp"
    "batch_normalization.comp"
    "gradient_descent.comp"
    
    # Spiking Neural Networks
    "snn_spike_update.comp"
    "sparse_attention.comp"
    
    # Generative Adversarial Networks
    "gan_generator.comp"
    "gan_discriminator.comp"
    
    # Reinforcement Learning
    "rl_q_learning.comp"
    "rl_policy_gradient.comp"
    
    # Hybrid Graphics-Compute Operations
    "neural_visualization.comp"
    "sparse_memory_manager.comp"
)

for shader in "${shaders[@]}"; do
    input_file="$SHADER_DIR/$shader"
    output_file="$OUTPUT_DIR/${shader}.spv"
    
    echo "Compiling $shader..."
    
    if glslc "$input_file" -o "$output_file"; then
        echo "✅ Successfully compiled $shader"
        file_size=$(stat -c%s "$output_file")
        echo "   Output: $output_file ($file_size bytes)"
    else
        echo "❌ Failed to compile $shader"
        exit 1
    fi
done

echo ""
echo "🚀 All CARL compute shaders compiled successfully!"
echo "   Located in: $OUTPUT_DIR"
echo ""
echo "Nova-CARL shader integration ready for AI workloads."