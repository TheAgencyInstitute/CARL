# CARL AI System

A modular AI system integrating Reinforcement Learning (RL), Generative Adversarial Networks (GANs), Convolutional Neural Networks (CNNs), and Spiking Neural Networks (SNNs) with GPU acceleration via the Nova framework.

## Architecture Overview

CARL combines multiple AI paradigms with a neuromorphic memory subsystem:
- **RL**: Policy optimization and decision-making
- **GAN**: Data generation and synthetic training  
- **CNN**: Feature extraction and pattern recognition
- **SNN**: Neuromorphic memory and comparison engine

## GPU Acceleration

Leverages the [Nova Graphics Engine](https://github.com/TheAgencyInstitute/Nova) for Vulkan-based GPU compute acceleration, providing cross-vendor GPU support (AMD/NVIDIA) with reduced boilerplate complexity.

## Project Structure

```
CARL/
├── .claude/                 # Project coordination and documentation
├── nova/                    # Nova GPU framework (submodule)  
├── src/                     # Source code
│   ├── core/               # AI components (RL, GAN, CNN, SNN)
│   ├── nova_integration/   # Nova framework bindings
│   ├── bindings/          # Language bindings (Go/C)  
│   └── tests/             # Test suites
├── docs/                   # Documentation
│   ├── architecture.md    # System architecture
│   ├── theory.md         # AI theory and applications
│   └── nova_integration.md # GPU integration guide
├── Alpha_Dev/             # Development components
└── Concept/              # Proof of concept implementations
```

## Development Stack

- **Low-Level**: C/Vulkan via Nova framework
- **Mid-Layer**: Go/CGO for API and IO
- **High-Level**: Modular AI component assembly  
- **Build**: Visual Studio solution with cross-platform support

## Performance Targets

- Memory recall latency: <15ms (GPU-accelerated)
- Pattern match accuracy: >90% (SNN)
- Cross-vendor GPU compatibility
- Modular, maintainable architecture

## Getting Started

### Prerequisites
- Vulkan SDK
- Nova framework dependencies
- Go compiler
- C/C++ build tools
- GPU with Vulkan support

### Setup
```bash
# Clone with submodules
git clone --recursive <repository-url>

# Initialize Nova submodule if not cloned recursively  
git submodule update --init --recursive

# Build Nova framework
cd nova
# Follow Nova build instructions

# Build CARL components
# (Build instructions in development)
```

## Integration Protocols

- **CNN-RL**: Self-reflection through feature extraction loops
- **RL-GAN**: Reward-driven optimization of generated data  
- **GAN-CNN**: Imagination engine for synthetic data analysis
- **SNN-RL**: Cognitive memory loop with spike-based encoding

## Documentation

See `docs/` directory for detailed documentation:
- [Architecture Guide](docs/architecture.md) - System design and components
- [AI Theory](docs/theory.md) - Theoretical foundations and applications  
- [Nova Integration](docs/nova_integration.md) - GPU acceleration implementation

## Current Status

**Phase 1**: GPU Framework Integration
- [x] Nova submodule integration
- [x] Project structure reorganization  
- [x] Documentation alignment
- [ ] Basic Nova compute bindings
- [ ] Primitive operation implementations

## License

Licensed under the Ancillary License - see the [LICENSE](LICENSE) file for details.

This project is open-source for non-commercial use. Commercial exploitation requires prior written agreement.

## Contributing

[Contributing guidelines to be added]