# CLAUDE.md - CARL AI System Project

## Project Information
- **Name**: CARL (CNN+RL+GAN+SNN)
- **Path**: /home/persist/repos/work/vazio/CARL
- **Description**: Modular AI system integrating Reinforcement Learning, Generative Adversarial Networks, Convolutional Neural Networks with Spiking Neural Network memory subsystem
- **Status**: Development Phase - GPU Framework Integration
- **Repository**: Initialized with Nova GPU compute framework submodule

## Overview
CARL is a foundational AI system built from mathematical and neural primitives, leveraging Vulkan GPU acceleration through the Nova graphics engine. The system combines multiple AI architectures with a neuromorphic memory subsystem for advanced learning and adaptation capabilities.

## Core Architecture Components
1. **Reinforcement Learning (RL)**: Policy optimization and decision-making
2. **Generative Adversarial Networks (GAN)**: Data generation and synthetic training
3. **Convolutional Neural Networks (CNN)**: Feature extraction and pattern recognition  
4. **Spiking Neural Networks (SNN)**: Neuromorphic memory and comparison engine
5. **Nova GPU Framework**: Vulkan-based compute acceleration and rendering

## Current Phase: GPU Framework Integration
- **Active Framework**: Nova (TheAgencyInstitute/Nova) - Vulkan-based GPU compute
- **Integration Focus**: Replacing custom Vulkan implementation with Nova abstraction
- **Target**: Streamlined GPU compute for all AI components

## Development Stack
- **Low-Level**: C/Vulkan via Nova framework
- **Mid-Layer**: Go/CGO for API and IO orchestration  
- **High-Level**: Modular AI component assembly
- **Build**: Visual Studio solution with cross-platform support

## Agent Responsibilities

### Core Process Agents
- **@agent-coordinator**: Session continuity, workflow orchestration, Nova integration management
- **@agent-analyzer**: CARL system analysis, performance benchmarking, GPU utilization
- **@agent-researcher**: AI theory research, Nova framework capabilities, optimization strategies
- **@agent-planner**: Development roadmaps, integration phases, testing protocols

### Development Specialists  
- **@agent-backend_developer**: Core AI implementations, Nova integration, C/Go bindings
- **@agent-gpu_engineer**: Vulkan compute shaders via Nova, performance optimization
- **@agent-ai_architect**: RL-GAN-CNN-SNN protocols, neural network designs
- **@agent-memory_specialist**: SNN implementation, spike encoding, neuromorphic computing

### Quality & Infrastructure
- **@agent-test_engineer**: AI model validation, GPU compute testing, benchmark suites
- **@agent-performance_engineer**: Profiling, optimization, memory management
- **@agent-integration_specialist**: Nova framework integration, cross-platform compatibility

## Available Tools
- **Nova Framework**: GPU compute acceleration, Vulkan abstraction, rendering pipeline
- **mcp__context7__**: AI library documentation and dependency research  
- **mcp__worktree__**: Feature branch management for parallel AI component development
- **mcp__shamash__**: Security analysis for AI model implementations

## Technical Standards
- **File Structure**: Maximum 500 lines per file, modular component design
- **GPU Compute**: Nova framework for all Vulkan operations
- **Neural Primitives**: Shared tensor operations, activation functions, loss functions
- **Memory Management**: Efficient GPU/CPU data transfer via Nova abstractions
- **Cross-Platform**: Support for AMD and NVIDIA GPUs through Vulkan

## Integration Protocols
- **CNN-RL Protocol**: Self-reflection through feature extraction and decision loops
- **RL-GAN Protocol**: Reward-driven optimization of generated data
- **GAN-CNN Protocol**: Imagination engine for synthetic data analysis
- **SNN-RL Protocol**: Cognitive memory loop with spike-based state encoding
- **Nova Integration**: GPU acceleration for all neural network computations

## Development Phases
1. **Phase 1**: Nova framework integration (Current)
2. **Phase 2**: Mathematical primitives via Nova compute shaders  
3. **Phase 3**: Core AI model implementations
4. **Phase 4**: SNN memory subsystem integration
5. **Phase 5**: Protocol implementations and testing
6. **Phase 6**: Performance optimization and benchmarking

## Performance Targets
- Memory recall latency: 12ms (Vulkan via Nova)
- Pattern match accuracy: 92% (SNN)  
- Memory density: 1GB/4M engrams
- Cross-vendor GPU support (AMD/NVIDIA)

## Project Structure
```
/home/persist/repos/work/vazio/CARL/
├── .claude/                 # Project coordination
├── nova/                    # Nova GPU framework (submodule)
├── src/                     # Source code organization
│   ├── core/               # Core AI components
│   ├── nova_integration/   # Nova framework bindings
│   ├── bindings/          # Go/C language bindings
│   └── tests/             # Test suites
├── docs/                   # Consolidated documentation  
├── Alpha_Dev/             # Development components (legacy)
├── Concept/               # Proof of concept implementations
└── [build artifacts and configs]
```

## Next Steps
1. Integrate Nova framework into existing CARL components
2. Replace custom Vulkan code with Nova abstractions
3. Implement AI mathematical primitives using Nova compute
4. Develop SNN memory subsystem with Nova acceleration
5. Create comprehensive testing suite for GPU-accelerated AI components