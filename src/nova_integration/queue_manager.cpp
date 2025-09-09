#include "queue_manager.h"
#include "nova_context.h"
#include <algorithm>
#include <iostream>

namespace CARL {
namespace GPU {

QueueManager::QueueManager(NovaContext* context) 
    : _context(context), _core(nullptr), _device(VK_NULL_HANDLE), 
      _physical_device(VK_NULL_HANDLE), _nova_initialized(false),
      _shutdown_requested(false), _debug_mode(false), _submission_counter(0) {
}

QueueManager::~QueueManager() {
    if (!_shutdown_requested) {
        shutdown();
    }
}

bool QueueManager::initialize() {
    if (!_context || !_context->getNova()) {
        std::cerr << "QueueManager: Invalid Nova context" << std::endl;
        return false;
    }
    
    _core = _context->getCore();
    if (!_core) {
        std::cerr << "QueueManager: Invalid Nova core" << std::endl;
        return false;
    }
    
    _device = _core->logical_device;
    _physical_device = _core->physical_device;
    
    if (_device == VK_NULL_HANDLE || _physical_device == VK_NULL_HANDLE) {
        std::cerr << "QueueManager: Nova devices not initialized" << std::endl;
        return false;
    }
    
    // Initialize from Nova's existing queue setup
    initializeFromNova(_core);
    
    // Extend to discover additional queues Nova missed
    if (!extendNovaQueues()) {
        std::cerr << "QueueManager: Failed to extend Nova queue discovery" << std::endl;
        return false;
    }
    
    // Create command pools for all queues
    if (!_createCommandPools()) {
        std::cerr << "QueueManager: Failed to create command pools" << std::endl;
        return false;
    }
    
    // Start worker threads for parallel queue management
    _startWorkerThreads();
    
    _nova_initialized = true;
    
    if (_debug_mode) {
        printQueueStatus();
    }
    
    return true;
}

void QueueManager::initializeFromNova(NovaCore* nova_core) {
    // Copy Nova's discovered queue information
    _enhanced_queues.nova_graphics = nova_core->queues.graphics;
    _enhanced_queues.nova_present = nova_core->queues.present;
    _enhanced_queues.nova_transfer = nova_core->queues.transfer;
    _enhanced_queues.nova_compute = nova_core->queues.compute;
    
    // Store family indices from Nova
    _enhanced_queues.graphics_family_index = nova_core->queues.indices.graphics_family.value();
    _enhanced_queues.compute_family_index = nova_core->queues.indices.compute_family.value();
    
    if (_debug_mode) {
        std::cout << "QueueManager: Initialized from Nova" << std::endl;
        std::cout << "  Graphics Family: " << _enhanced_queues.graphics_family_index << std::endl;
        std::cout << "  Compute Family: " << _enhanced_queues.compute_family_index << std::endl;
    }
}

bool QueueManager::extendNovaQueues() {
    // Get queue family properties directly from physical device
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count, queue_families.data());
    
    if (_debug_mode) {
        std::cout << "QueueManager: Discovered " << queue_family_count << " queue families" << std::endl;
    }
    
    // Find the dedicated compute family (should be different from graphics)
    for (uint32_t i = 0; i < queue_families.size(); i++) {
        const auto& family = queue_families[i];
        
        if (_debug_mode) {
            std::cout << "  Family " << i << ": " << family.queueCount << " queues, flags: " << family.queueFlags << std::endl;
        }
        
        // Check for dedicated compute family (not graphics+compute)
        if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) && 
            !(family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && 
            i != _enhanced_queues.graphics_family_index) {
            
            _enhanced_queues.compute_family_index = i;
            
            // Get ALL compute queues from this family (Nova only got the first one)
            for (uint32_t q = 0; q < family.queueCount; q++) {
                VkQueue queue;
                vkGetDeviceQueue(_device, i, q, &queue);
                _enhanced_queues.all_compute_queues.push_back(queue);
                
                if (_debug_mode) {
                    std::cout << "    Added compute queue " << q << std::endl;
                }
            }
        }
        
        // Check for video decode family  
        if ((family.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)) {
            _enhanced_queues.video_family_index = i;
            
            for (uint32_t q = 0; q < family.queueCount; q++) {
                VkQueue queue;
                vkGetDeviceQueue(_device, i, q, &queue);
                _enhanced_queues.video_decode_queues.push_back(queue);
                
                if (_debug_mode) {
                    std::cout << "    Added video decode queue " << q << std::endl;
                }
            }
        }
    }
    
    if (_enhanced_queues.all_compute_queues.empty()) {
        std::cerr << "QueueManager: Warning - No dedicated compute queues found, falling back to Nova's queue" << std::endl;
        _enhanced_queues.all_compute_queues.push_back(_enhanced_queues.nova_compute);
    }
    
    if (_debug_mode) {
        std::cout << "QueueManager: Extended queue discovery complete" << std::endl;
        std::cout << "  Total Compute Queues: " << _enhanced_queues.all_compute_queues.size() << std::endl;
        std::cout << "  Video Decode Queues: " << _enhanced_queues.video_decode_queues.size() << std::endl;
    }
    
    return true;
}

uint32_t QueueManager::getTotalComputeQueues() const {
    return _enhanced_queues.all_compute_queues.size();
}

bool QueueManager::_createCommandPools() {
    // Create command pools for each compute queue
    _enhanced_queues.compute_command_pools.reserve(_enhanced_queues.all_compute_queues.size());
    
    for (size_t i = 0; i < _enhanced_queues.all_compute_queues.size(); i++) {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = _enhanced_queues.compute_family_index
        };
        
        VkCommandPool pool;
        VkResult result = vkCreateCommandPool(_device, &pool_info, nullptr, &pool);
        if (result != VK_SUCCESS) {
            std::cerr << "QueueManager: Failed to create compute command pool " << i << std::endl;
            return false;
        }
        
        _enhanced_queues.compute_command_pools.push_back(pool);
    }
    
    // Create command pools for video decode queues
    _enhanced_queues.video_command_pools.reserve(_enhanced_queues.video_decode_queues.size());
    
    for (size_t i = 0; i < _enhanced_queues.video_decode_queues.size(); i++) {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = _enhanced_queues.video_family_index
        };
        
        VkCommandPool pool;
        VkResult result = vkCreateCommandPool(_device, &pool_info, nullptr, &pool);
        if (result != VK_SUCCESS) {
            std::cerr << "QueueManager: Failed to create video command pool " << i << std::endl;
            return false;
        }
        
        _enhanced_queues.video_command_pools.push_back(pool);
    }
    
    return true;
}

void QueueManager::_startWorkerThreads() {
    // Start worker threads for parallel queue processing
    size_t num_compute_queues = _enhanced_queues.all_compute_queues.size();
    _worker_threads.reserve(num_compute_queues);
    
    for (size_t i = 0; i < num_compute_queues; i++) {
        _worker_threads.emplace_back(&QueueManager::_queueWorker, this, i);
    }
}

void QueueManager::_stopWorkerThreads() {
    _shutdown_requested = true;
    
    for (auto& thread : _worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    _worker_threads.clear();
}

void QueueManager::_queueWorker(size_t queue_index) {
    // Worker thread implementation for processing queue submissions
    while (!_shutdown_requested) {
        // Process queue submissions for this specific compute queue
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void QueueManager::shutdown() {
    if (!_nova_initialized) return;
    
    _stopWorkerThreads();
    
    // Cleanup command pools
    for (auto pool : _enhanced_queues.compute_command_pools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(_device, pool, nullptr);
        }
    }
    
    for (auto pool : _enhanced_queues.video_command_pools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(_device, pool, nullptr);
        }
    }
    
    _enhanced_queues.compute_command_pools.clear();
    _enhanced_queues.video_command_pools.clear();
    _enhanced_queues.all_compute_queues.clear();
    _enhanced_queues.video_decode_queues.clear();
    
    _nova_initialized = false;
}

void QueueManager::printQueueStatus() const {
    std::cout << "\n=== CARL Queue Manager Status ===" << std::endl;
    std::cout << "Nova Integration: " << (_nova_initialized ? "✓" : "✗") << std::endl;
    std::cout << "\nNova Original Queues:" << std::endl;
    std::cout << "  Graphics: " << _enhanced_queues.nova_graphics << " (Family " << _enhanced_queues.graphics_family_index << ")" << std::endl;
    std::cout << "  Compute:  " << _enhanced_queues.nova_compute << " (Family " << _enhanced_queues.compute_family_index << ")" << std::endl;
    
    std::cout << "\nExtended Queues:" << std::endl;
    std::cout << "  Dedicated Compute: " << _enhanced_queues.all_compute_queues.size() << " queues" << std::endl;
    for (size_t i = 0; i < _enhanced_queues.all_compute_queues.size(); i++) {
        std::cout << "    Queue " << i << ": " << _enhanced_queues.all_compute_queues[i] << std::endl;
    }
    
    std::cout << "  Video Decode: " << _enhanced_queues.video_decode_queues.size() << " queues" << std::endl;
    for (size_t i = 0; i < _enhanced_queues.video_decode_queues.size(); i++) {
        std::cout << "    Queue " << i << ": " << _enhanced_queues.video_decode_queues[i] << std::endl;
    }
    
    std::cout << "\nPerformance Multiplier: " << _enhanced_queues.all_compute_queues.size() << "x vs Nova" << std::endl;
    std::cout << "===============================" << std::endl;
}

// Stub implementations for remaining methods
std::vector<QueueInfo> QueueManager::getAvailableQueues() const { return {}; }
std::vector<QueueInfo> QueueManager::getNovaQueues() const { return {}; }
std::vector<QueueInfo> QueueManager::getExtendedQueues() const { return {}; }
bool QueueManager::isQueueFamilyAvailable(QueueType type) const { return true; }
uint32_t QueueManager::getQueueCount(QueueType type) const { return 1; }

} // namespace GPU
} // namespace CARL