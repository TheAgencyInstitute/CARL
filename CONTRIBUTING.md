# Contributing to CARL

Thank you for your interest in contributing to CARL! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- **Vulkan SDK**: For GPU compute support
- **C/C++ Compiler**: GCC, Clang, or MSVC
- **Go**: For API and binding development
- **Git**: With submodule support
- **GPU**: Vulkan-compatible (AMD/NVIDIA)

### Environment Setup
```bash
# Clone repository with submodules
git clone --recursive https://github.com/TheAgencyInstitute/CARL.git
cd CARL

# Initialize Nova submodule if needed
git submodule update --init --recursive

# Build Nova framework
cd nova
# Follow Nova-specific build instructions
cd ..

# Set up development environment
# (Build system in development)
```

## Project Structure

```
CARL/
├── .claude/                 # Project coordination
├── nova/                    # Nova GPU framework (submodule)
├── src/                     # Source code
│   ├── core/               # AI components (RL, GAN, CNN, SNN)
│   ├── nova_integration/   # Nova framework bindings
│   ├── bindings/          # Language bindings
│   └── tests/             # Test suites
├── docs/                   # Documentation
├── Alpha_Dev/             # Development components
└── Concept/              # Proof of concept implementations
```

## Coding Standards

### General Guidelines
- **File Size**: Maximum 500 lines per file
- **Function Size**: Maximum 50 lines per function
- **Nesting**: Maximum 3 levels of nesting
- **Memory Safety**: No memory leaks or undefined behavior
- **Error Handling**: Comprehensive error checking

### Code Style
- Use clear, descriptive variable names
- Include header guards in all `.h` files
- Document all public functions
- Use consistent indentation (4 spaces)
- Include appropriate copyright headers

### Example Code Structure
```c
#pragma once

#include "required_headers.h"

/**
 * Brief description of function
 * @param param1 Description of parameter
 * @return Description of return value
 */
ReturnType function_name(ParamType param1);

// Implementation in .cpp/.c file
ReturnType function_name(ParamType param1) {
    // Input validation
    if (param1 == NULL) {
        return ERROR_INVALID_PARAM;
    }
    
    // Function logic
    // ...
    
    return success_value;
}
```

## Component Development

### AI Components (src/core/)
- **RL**: Reinforcement learning algorithms
- **GAN**: Generative adversarial networks
- **CNN**: Convolutional neural networks
- **SNN**: Spiking neural networks

### Integration Guidelines
1. All GPU operations must use Nova framework
2. Implement both CPU and GPU versions where applicable
3. Include comprehensive unit tests
4. Document integration protocols
5. Ensure cross-platform compatibility

### Nova Integration (src/nova_integration/)
- Implement compute shader abstractions
- Manage GPU memory efficiently
- Provide error handling for GPU operations
- Maintain compatibility with Nova updates

## Testing

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: GPU acceleration validation
- **System Tests**: End-to-end functionality

### Running Tests
```bash
# Run all tests (command in development)
make test

# Run specific component tests
make test-rl
make test-gan
make test-cnn
make test-snn

# Run GPU integration tests
make test-gpu
```

### Writing Tests
- Test both success and failure cases
- Include edge cases and boundary conditions
- Validate GPU acceleration benefits
- Test memory management thoroughly

## Documentation

### Required Documentation
- **API Documentation**: All public functions
- **Integration Guides**: Component interaction protocols
- **Performance Analysis**: GPU acceleration benchmarks
- **Examples**: Usage demonstrations

### Documentation Style
- Use clear, concise language
- Include code examples
- Document performance characteristics
- Explain design decisions

## Submission Process

### Pull Request Guidelines
1. **Branch Naming**: `feature/component-description` or `fix/issue-description`
2. **Commit Messages**: Clear, descriptive messages
3. **Testing**: All tests must pass
4. **Documentation**: Update relevant documentation
5. **Code Review**: Address all review comments

### Commit Message Format
```
component: brief description of change

Detailed explanation of what was changed and why.
Include any breaking changes or special considerations.

Closes #issue_number
```

### Pull Request Template
- Description of changes
- Testing performed
- Documentation updates
- Performance impact
- Breaking changes (if any)

## Performance Considerations

### GPU Acceleration
- Minimize CPU-GPU memory transfers
- Use compute shaders for parallelizable operations
- Implement memory pooling for frequent allocations
- Profile GPU utilization regularly

### Memory Management
- No memory leaks
- Efficient GPU memory usage
- Proper cleanup in error cases
- Consider memory alignment for GPU operations

## Issue Reporting

### Bug Reports
Include the following information:
- CARL version
- Operating system and version
- GPU model and drivers
- Detailed steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

### Feature Requests
- Clear description of the feature
- Use cases and benefits
- Proposed implementation approach
- Impact on existing functionality

## Communication

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and design discussions
- **Pull Requests**: Code review and technical discussions

### Code Review Process
1. Automated tests must pass
2. At least one maintainer review required
3. Address all review comments
4. Ensure documentation is updated
5. Maintain backward compatibility when possible

## License

By contributing to CARL, you agree that your contributions will be licensed under the Ancillary License. This means:

- Your contributions will be available for non-commercial use
- Commercial use requires prior written agreement from Ancillary, Inc.
- You retain intellectual property rights as a contributor
- All contributors share in any financial gains from commercial licensing

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search GitHub issues and discussions
3. Open a new discussion for clarification
4. Contact the maintainers directly if needed

Thank you for helping make CARL better!