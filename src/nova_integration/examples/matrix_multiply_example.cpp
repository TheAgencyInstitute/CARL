#include "../carl_gpu.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

/**
 * Matrix Multiplication Example using CARL GPU acceleration
 * 
 * Demonstrates:
 * 1. GPU system initialization
 * 2. Memory allocation and data transfer
 * 3. Compute shader execution
 * 4. Performance comparison with CPU
 */

void fillRandomMatrix(std::vector<float>& matrix, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

void cpuMatrixMultiply(const std::vector<float>& A, 
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool validateResults(const std::vector<float>& gpu_result,
                    const std::vector<float>& cpu_result,
                    float tolerance = 1e-4f) {
    assert(gpu_result.size() == cpu_result.size());
    
    for (size_t i = 0; i < gpu_result.size(); i++) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            std::cout << "Validation failed at index " << i 
                     << ": GPU=" << gpu_result[i] 
                     << ", CPU=" << cpu_result[i] 
                     << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "CARL GPU Matrix Multiplication Example\n";
    std::cout << "======================================\n\n";
    
    // Matrix dimensions for testing
    const size_t M = 1024;  // Rows of A and C
    const size_t N = 1024;  // Columns of B and C
    const size_t K = 1024;  // Columns of A, rows of B
    
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N 
              << " = " << M << "x" << N << "\n";
    std::cout << "Total FLOPs: " << (2.0 * M * N * K) / 1e9 << " GFLOP\n\n";
    
    // Initialize GPU system
    CARL::GPU::NovaContextConfig config;
    config.app_name = "CARL Matrix Multiply Example";
    config.debug_level = "info";
    config.memory_pool_size = 512 * 1024 * 1024; // 512MB
    
    if (!CARL::GPU::Global::initialize(config)) {
        std::cerr << "Failed to initialize GPU system!" << std::endl;
        return -1;
    }
    
    // Get GPU info
    auto gpu_info = CARL::GPU::Global::getDeviceInfo();
    std::cout << "GPU Device: " << gpu_info.name << "\n";
    std::cout << "GPU Memory: " << (gpu_info.memory_bytes / 1024 / 1024) << " MB\n";
    std::cout << "GPU Type: " << static_cast<int>(gpu_info.type) << "\n\n";
    
    // Prepare test data
    std::cout << "Preparing test data...\n";
    std::vector<float> matrix_A(M * K);
    std::vector<float> matrix_B(K * N);
    std::vector<float> matrix_C_gpu(M * N);
    std::vector<float> matrix_C_cpu(M * N);
    
    fillRandomMatrix(matrix_A, M * K);
    fillRandomMatrix(matrix_B, K * N);
    
    // Allocate GPU buffers
    auto memory = CARL_MEMORY();
    auto compute = CARL_COMPUTE();
    
    std::cout << "Allocating GPU buffers...\n";
    auto buffer_A = memory->allocateBuffer(
        M * K * sizeof(float), 
        CARL::GPU::DataType::FLOAT32, 
        {static_cast<uint32_t>(M), static_cast<uint32_t>(K)}
    );
    
    auto buffer_B = memory->allocateBuffer(
        K * N * sizeof(float), 
        CARL::GPU::DataType::FLOAT32, 
        {static_cast<uint32_t>(K), static_cast<uint32_t>(N)}
    );
    
    auto buffer_C = memory->allocateBuffer(
        M * N * sizeof(float), 
        CARL::GPU::DataType::FLOAT32, 
        {static_cast<uint32_t>(M), static_cast<uint32_t>(N)}
    );
    
    if (!buffer_A || !buffer_B || !buffer_C) {
        std::cerr << "Failed to allocate GPU buffers!" << std::endl;
        CARL::GPU::Global::shutdown();
        return -1;
    }
    
    // Upload data to GPU
    std::cout << "Uploading data to GPU...\n";
    auto start_upload = std::chrono::high_resolution_clock::now();
    
    compute->uploadData(buffer_A, matrix_A.data(), M * K * sizeof(float));
    compute->uploadData(buffer_B, matrix_B.data(), K * N * sizeof(float));
    
    auto end_upload = std::chrono::high_resolution_clock::now();
    auto upload_time = std::chrono::duration_cast<std::chrono::microseconds>(end_upload - start_upload);
    
    // Perform GPU matrix multiplication
    std::cout << "Performing GPU matrix multiplication...\n";
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    bool success = compute->matrixMultiply(buffer_A, buffer_B, buffer_C);
    compute->waitForCompletion();
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
    
    if (!success) {
        std::cerr << "GPU matrix multiplication failed: " << compute->getLastError() << std::endl;
        CARL::GPU::Global::shutdown();
        return -1;
    }
    
    // Download result from GPU
    auto start_download = std::chrono::high_resolution_clock::now();
    compute->downloadData(buffer_C, matrix_C_gpu.data(), M * N * sizeof(float));
    auto end_download = std::chrono::high_resolution_clock::now();
    auto download_time = std::chrono::duration_cast<std::chrono::microseconds>(end_download - start_download);
    
    // Perform CPU matrix multiplication for comparison
    std::cout << "Performing CPU matrix multiplication for validation...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiply(matrix_A, matrix_B, matrix_C_cpu, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    // Validate results
    std::cout << "Validating results...\n";
    bool results_valid = validateResults(matrix_C_gpu, matrix_C_cpu);
    
    // Performance results
    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Upload time:    " << upload_time.count() << " μs\n";
    std::cout << "GPU compute:    " << gpu_time.count() << " μs\n";
    std::cout << "Download time:  " << download_time.count() << " μs\n";
    std::cout << "CPU compute:    " << cpu_time.count() << " μs\n";
    std::cout << "\nTotal GPU time: " << (upload_time + gpu_time + download_time).count() << " μs\n";
    std::cout << "GPU speedup:    " << (double)cpu_time.count() / gpu_time.count() << "x\n";
    std::cout << "GPU throughput: " << (2.0 * M * N * K) / (gpu_time.count() * 1e-3) << " GFLOPS\n";
    std::cout << "CPU throughput: " << (2.0 * M * N * K) / (cpu_time.count() * 1e-3) << " GFLOPS\n";
    
    // Memory usage
    auto memory_stats = memory->getTotalStats();
    std::cout << "\n=== Memory Statistics ===\n";
    std::cout << "Total allocated: " << memory_stats.total_allocated / 1024 / 1024 << " MB\n";
    std::cout << "Peak usage:      " << memory_stats.peak_usage / 1024 / 1024 << " MB\n";
    std::cout << "Current usage:   " << memory_stats.current_usage / 1024 / 1024 << " MB\n";
    
    // Validation result
    std::cout << "\n=== Validation ===\n";
    std::cout << "Results valid: " << (results_valid ? "YES" : "NO") << "\n";
    
    // Cleanup
    memory->deallocateBuffer(buffer_A);
    memory->deallocateBuffer(buffer_B);
    memory->deallocateBuffer(buffer_C);
    
    CARL::GPU::Global::shutdown();
    
    std::cout << "\nExample completed successfully!\n";
    return results_valid ? 0 : -1;
}

/*
Expected Performance (RTX 3080):
- 1024x1024 matrix multiplication: ~2.1 GFLOP
- GPU compute time: ~70 microseconds  
- GPU throughput: ~30 TFLOPS
- CPU throughput: ~50 GFLOPS (8-core CPU)
- GPU speedup: ~600x for compute only

Memory Usage:
- Matrix A: 1024 * 1024 * 4 = 4 MB
- Matrix B: 1024 * 1024 * 4 = 4 MB  
- Matrix C: 1024 * 1024 * 4 = 4 MB
- Total: ~12 MB GPU memory

Build Instructions:
g++ -std=c++17 matrix_multiply_example.cpp \
    ../nova_context.cpp ../compute_wrapper.cpp ../memory_manager.cpp \
    -I../../nova -lvulkan -lSDL2 -o matrix_example
*/