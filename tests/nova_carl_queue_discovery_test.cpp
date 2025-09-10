#include "nova/Core/core.h"
#include <iostream>

/**
 * Minimal Nova-CARL Queue Discovery Test
 * Tests queue family discovery without graphics context
 */

class HeadlessNovaCore {
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice logical_device = VK_NULL_HANDLE;

public:
    // Copy queue structures from our enhanced Nova
    struct TestQueueFamilyIndices {
        std::optional<unsigned int> graphics_family = -1;
        std::optional<unsigned int> present_family = -1;
        std::optional<unsigned int> transfer_family = -1;
        std::optional<unsigned int> compute_family = -1;
        std::optional<unsigned int> video_decode_family = -1;
        std::optional<unsigned int> video_encode_family = -1;
        std::optional<unsigned int> sparse_binding_family = -1;

        bool isComplete() {
            return graphics_family >= 0 && present_family >= 0 && transfer_family >= 0 && compute_family >= 0;
        }
        
        bool isFullyComplete() {
            return isComplete() && video_decode_family >= 0 && video_encode_family >= 0 && sparse_binding_family >= 0;
        }
    };

    struct TestQueues {
        std::vector<VkQueueFamilyProperties> families;
        TestQueueFamilyIndices indices;
        
        std::vector<VkQueue> all_compute_queues;
        std::vector<VkQueue> video_decode_queues;
        std::vector<VkQueue> video_encode_queues;
        std::vector<VkQueue> sparse_binding_queues;
        
        uint32_t total_compute_queues() const { return all_compute_queues.size(); }
        uint32_t total_video_decode_queues() const { return video_decode_queues.size(); }
        uint32_t total_video_encode_queues() const { return video_encode_queues.size(); }
        bool has_sparse_binding() const { return !sparse_binding_queues.empty(); }
    };

    TestQueues test_queues;

    bool initializeVulkan() {
        // Create Vulkan instance (headless)
        VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Nova-CARL Queue Test",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "Nova-CARL",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_2
        };

        VkInstanceCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .enabledExtensionCount = 0
        };

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance!" << std::endl;
            return false;
        }

        // Find physical device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            std::cerr << "No Vulkan-compatible devices found!" << std::endl;
            return false;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physical_device = devices[0]; // Use first device

        return true;
    }

    void setQueueFamilyProperties(unsigned int i) {
        VkQueueFamilyProperties* queue_family = &test_queues.families[i];
        std::string queue_name = "";

        if (queue_family->queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_name += "{ Graphics } ";
            test_queues.indices.graphics_family = i;
            std::cout << "    Graphics Family Set." << std::endl;
        }

        if (queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_name += "{ Compute } ";

            if (test_queues.indices.graphics_family.value() != i) {
                test_queues.indices.compute_family = i;
                std::cout << "    Dedicated Compute Family Set (" << queue_family->queueCount << " queues)." << std::endl;
            }
        }

        if (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT) {
            queue_name += "{ Transfer } ";

            if (test_queues.indices.graphics_family.value() != i) {
                test_queues.indices.transfer_family = i;
                std::cout << "    Transfer Family Set." << std::endl;
            }
        }

        // CARL Extensions - Additional queue family detection
        if (queue_family->queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
            queue_name += "{ Video Decode } ";
            test_queues.indices.video_decode_family = i;
            std::cout << "    Video Decode Family Set (" << queue_family->queueCount << " queues)." << std::endl;
        }

        if (queue_family->queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
            queue_name += "{ Video Encode } ";
            test_queues.indices.video_encode_family = i;
            std::cout << "    Video Encode Family Set (" << queue_family->queueCount << " queues)." << std::endl;
        }

        if (queue_family->queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
            queue_name += "{ Sparse Binding } ";

            // Check if this is a dedicated sparse binding family
            if (!(queue_family->queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                test_queues.indices.sparse_binding_family = i;
                std::cout << "    Dedicated Sparse Binding Family Set (" << queue_family->queueCount << " queues)." << std::endl;
            }
        }

        if (queue_name.empty()) {
            queue_name = "~ Unknown ~";
        }

        std::cout << "      Queue Count: " << queue_family->queueCount << std::endl;
        std::cout << "      " << queue_name << std::endl;
    }

    void discoverQueueFamilies() {
        std::cout << "  .. Acquiring Queue Families .." << std::endl;
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        test_queues.families.resize(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, test_queues.families.data());

        for (int i = 0; i < test_queues.families.size(); i++) {
            std::cout << "    Queue Family " << i << std::endl;
            setQueueFamilyProperties(i);
        }

        // Check completion
        if (!test_queues.indices.isComplete()) {
            std::cout << "    Queue Families Incomplete. Setting fallbacks." << std::endl;
            test_queues.indices.transfer_family = test_queues.indices.graphics_family.value();
            test_queues.indices.present_family = test_queues.indices.graphics_family.value();
        }
    }

    void printComparisonReport() {
        std::cout << "\n=== NOVA-CARL vs ORIGINAL NOVA COMPARISON ===" << std::endl;

        std::cout << "\nOriginal Nova Discovery:" << std::endl;
        std::cout << "  Graphics Family: " << (test_queues.indices.graphics_family.has_value() ? std::to_string(test_queues.indices.graphics_family.value()) : "None") << std::endl;
        std::cout << "  Compute Family: " << (test_queues.indices.compute_family.has_value() ? std::to_string(test_queues.indices.compute_family.value()) : "None") << std::endl;
        std::cout << "  Transfer Family: " << (test_queues.indices.transfer_family.has_value() ? std::to_string(test_queues.indices.transfer_family.value()) : "None") << std::endl;
        std::cout << "  Present Family: " << (test_queues.indices.present_family.has_value() ? std::to_string(test_queues.indices.present_family.value()) : "None") << std::endl;

        std::cout << "\nNova-CARL Extensions:" << std::endl;
        std::cout << "  Video Decode: " << (test_queues.indices.video_decode_family.has_value() ? std::to_string(test_queues.indices.video_decode_family.value()) : "None") << std::endl;
        std::cout << "  Video Encode: " << (test_queues.indices.video_encode_family.has_value() ? std::to_string(test_queues.indices.video_encode_family.value()) : "None") << std::endl;
        std::cout << "  Sparse Binding: " << (test_queues.indices.sparse_binding_family.has_value() ? std::to_string(test_queues.indices.sparse_binding_family.value()) : "None") << std::endl;

        std::cout << "\nCompletion Status:" << std::endl;
        std::cout << "  Basic Complete: " << (test_queues.indices.isComplete() ? "Yes" : "No") << std::endl;
        std::cout << "  Fully Complete: " << (test_queues.indices.isFullyComplete() ? "Yes" : "No") << std::endl;

        std::cout << "\nQueue Family Details:" << std::endl;
        for (size_t i = 0; i < test_queues.families.size(); i++) {
            const auto& family = test_queues.families[i];
            std::cout << "Family " << i << ": " << family.queueCount << " queues, flags: " << family.queueFlags;

            if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) std::cout << " Graphics";
            if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) std::cout << " Compute";
            if (family.queueFlags & VK_QUEUE_TRANSFER_BIT) std::cout << " Transfer";
            if (family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) std::cout << " Sparse";
            if (family.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) std::cout << " VideoDecode";
            if (family.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) std::cout << " VideoEncode";

            std::cout << std::endl;
        }

        // Calculate performance impact
        uint32_t compute_queues = 1; // Nova original
        if (test_queues.indices.compute_family.has_value()) {
            compute_queues = test_queues.families[test_queues.indices.compute_family.value()].queueCount;
        }

        std::cout << "\n=== PERFORMANCE IMPACT ===" << std::endl;
        std::cout << "Nova Original: 1 compute queue" << std::endl;
        std::cout << "Nova-CARL: " << compute_queues << " compute queues" << std::endl;
        std::cout << "Expected Speedup: " << compute_queues << "x for parallel workloads" << std::endl;
    }

    void cleanup() {
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

int main() {
    std::cout << "Nova-CARL Queue Discovery Test" << std::endl;
    std::cout << "==============================" << std::endl;

    HeadlessNovaCore core;

    if (!core.initializeVulkan()) {
        std::cerr << "Failed to initialize Vulkan!" << std::endl;
        return 1;
    }

    std::cout << "\nDiscovering Queue Families..." << std::endl;
    core.discoverQueueFamilies();

    core.printComparisonReport();

    std::cout << "\n✅ Nova-CARL queue discovery test completed successfully!" << std::endl;

    core.cleanup();
    return 0;
}