#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS  
#define GLM_FORCE_DEPTH_ZERO_TO_ONE  
// #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/vec4.hpp> // glm::vec4  
#include <glm/mat4x4.hpp> // glm::mat4  
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <GLFW/glfw3.h>
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <optional>
#include <set>
#include <chrono>
#include <memory>

#include "utils.h"
#include "vertex.h"

#define NDEBUG

typedef uint32_t uint32;

// required extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef NDEBUG
    const bool enableValidationLayers = true;
    using std::cout;
    using std::endl;
    static int frameRendered = 0;
#else 
    const bool enableValidationLayers = false;
#endif
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const int MAX_FRAMES_IN_FLIGHT = 2;

// shader filenames
const char* vertShaderFilename = "shaders/cube.vert.spv";
const char* fragShaderFilename = "shaders/cube.frag.spv";

class SpinningCube {
public:
    void run() {
#ifdef NDEBUG
        cout << "cube.run called" << endl;
#endif
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    SpinningCube() {

    }

private:
    GLFWwindow* window;
    const uint32 WIDTH = 800;
    const uint32 HEIGHT = 600;

    // UBO
    UniformBufferObject baseUbo{};


    void initWindow() {
#ifdef NDEBUG
        cout << "initWindow called" << endl;
#endif
        glfwInit();

        // using Vulkan not OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Cube Demo", nullptr, nullptr);
    }
    // Vulkan things
    struct QueueFamilyIndices {
        std::optional<uint32> graphicsFamily;
        std::optional<uint32> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    VkInstance          instance;
    VkSurfaceKHR        surface;

    // device
    VkPhysicalDevice    physicalDevice = VK_NULL_HANDLE;
    VkDevice            device;
    QueueFamilyIndices  queueFamilyIndices;
    VkQueue             graphicsQueue;
    VkQueue             presentQueue;
    
    // swapchain
    VkSwapchainKHR              swapChain;
    std::vector<VkImage>        swapChainImages;
    std::vector<VkImageView>    swapChainImageViews;
    VkFormat                    swapChainImageFormat;
    VkFormat                    depthImageFormat;
    VkExtent2D                  swapChainExtent;
    std::vector<VkFramebuffer>  swapChainFramebuffers; 

    // factory
    VkRenderPass            renderPass;
    VkDescriptorSetLayout   descriptorSetLayout;
    VkPipelineLayout        pipelineLayout;
    VkPipeline              graphicsPipeline;

    // command
    VkCommandPool   commandPool;

    //
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_4_BIT;
    
    VkImage         colorImage;
    VkDeviceMemory  colorImageMemory;
    VkImageView     colorImageView;

    VkImage         depthImage;
    VkDeviceMemory  depthImageMemory;
    VkImageView     depthImageView;

    //
    std::vector<VkCommandBuffer> commandBuffers;

    // geometry
    std::vector<Vertex> vertices;
    std::vector<uint32> vertIndices;
    VkBuffer            vertBuffer;
    VkDeviceMemory      vertBufferMemory;
    VkBuffer            indexBuffer;
    VkDeviceMemory      indexBufferMemory;

    // uniform buffer
    std::vector<VkBuffer>       uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*>          uniformBuffersMapped;

    // descriptor
    VkDescriptorPool             descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // sync
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;

    bool framebufferResized = false;
    uint32 currFrame = 0;

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capanilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "findQueueFamilies called" << endl;
#endif
        QueueFamilyIndices indices{};

        uint32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) break;
            i++;
        }
        return indices;
    }
    bool checkDeviceExtensionSupport(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "checkDeviceExtensionSupport called" << endl;
#endif
        uint32 extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }
    SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "querySwapChainSupport called" << endl;
#endif
        SwapChainSupportDetails details{};

        // query capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capanilities);
        // query supported formats
        uint32 formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        // query present mode
        uint32 presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
#ifdef NDEBUG
        cout << "chooseSwapSurfaceFormat called" << endl;
#endif
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8_SRGB && 
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;        
            }
        }

        return availableFormats[0];
    }
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
#ifdef NDEBUG
        cout << "chooseSwapPresentMode called" << endl;
#endif
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
#ifdef NDEBUG
        cout << "chooseSwapExtent called" << endl;
#endif
        if (capabilities.currentExtent.width != std::numeric_limits<uint32>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { static_cast<uint32>(width), static_cast<uint32>(height) };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
#ifdef NDEBUG
        cout << "findSupportedFormat called" << endl;
#endif
        for (const auto& format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            } 
        }

        throw std::runtime_error("Failed to find supported format!");
    }

    void createInstance() {
#ifdef NDEBUG
        cout << "createInstance called" << endl;
#endif
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Spinning Cube";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // get required extensions for GLFW
        uint32 glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan Instance!");
        }
    }
    void createSurface() {
#ifdef NDEBUG
        cout << "createSurface called" << endl;
#endif
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    bool isDeviceSuitable(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "isDeviceSuitable called" << endl;
#endif
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails details = querySwapChainSupport(device);
            swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }
    void pickupPhysicalDevice() {
#ifdef NDEBUG
        cout << "pickupPhysicalDevice called" << endl;
#endif
        uint32 deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }            
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }
    void createLogicalDevice() {
#ifdef NDEBUG
        cout << "createLogicalDevice called" << endl;
#endif
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        std::set<uint32> uniqueQueueFamilies = {
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value()
        };

        float queuePriority = 1.0f;
        // unique queue
        for (const auto& queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.emplace_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        
        // extensions
        createInfo.enabledExtensionCount = static_cast<uint32>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        
        // validation layer
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to createa logical device!");
        }
    }
    void getQueue() {
#ifdef NDEBUG
        cout << "getQueue called" << endl;
#endif
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &presentQueue);
    }
    void createSwapChain() {
#ifdef NDEBUG
        cout << "createSwapChain called" << endl;
#endif
        SwapChainSupportDetails details = querySwapChainSupport(physicalDevice);

        // select the best format, present mode, resolution
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(details.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(details.presentModes);
        VkExtent2D extent = chooseSwapExtent(details.capanilities);

        // determine image count
        uint32 imageCount = details.capanilities.minImageCount + 1;
        imageCount = std::min(details.capanilities.maxImageCount, imageCount);

        // fill swapchain create info
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32 rawQueueFamilyIndices[] = { 
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value() 
        };

        if (queueFamilyIndices.graphicsFamily.value() != queueFamilyIndices.presentFamily.value()) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = rawQueueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = details.capanilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
    VkImageView createImageView(VkImage image, VkFormat format, 
                                VkImageAspectFlagBits aspectFlags, uint32 mipLevels) {
#ifdef NDEBUG
        cout << "createImageView called" << endl;
#endif
        VkImageViewCreateInfo createInfo{};
        VkImageSubresourceRange range;
        
        range.aspectMask = aspectFlags;
        range.baseMipLevel = 0;
        range.levelCount = mipLevels;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;
        createInfo.subresourceRange = range;

        VkImageView imageView;

        if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image views!");
        }
        return imageView;
    }
    void createImageViews() {
#ifdef NDEBUG
        cout << "createImageViews called" << endl;
#endif
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, 
                                                     VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
    void createImage(uint32 width, uint32 height, uint32 mipLevels, VkSampleCountFlagBits numSamples,
                     VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
#ifdef NDEBUG
        cout << "createImage called" << endl;
#endif
        VkImageCreateInfo imageInfo{};

        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate memory for image!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }
    void createRenderPass() {
#ifdef NDEBUG
        cout << "createRenderPass called" << endl;
#endif
        // color attachment
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = msaaSamples;  // NO MSAA
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // depth attachment
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthImageFormat;
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        
        // color attachment resolve
        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
        // subpass reference
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // subpass description
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;

        // subpass dependency
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        // render pass
        std::array<VkAttachmentDescription, 3> attachments = {
            colorAttachment, depthAttachment, colorAttachmentResolve 
        };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }
    void createDescriptorSetLayout() {
#ifdef NDEBUG
        cout << "createDescriptorSetLayout called" << endl;
#endif
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;

        // only for *.vert
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout)) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }
    VkShaderModule createShaderModule(const std::vector<char>& code) {
#ifdef NDEBUG
        cout << "createShaderModule called" << endl;
#endif
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module");
        }
        return shaderModule;
    }
    void createGraphicsPipeline() {
#ifdef NDEBUG
        cout << "createGraphicsPipeline called" << endl;
#endif
        // 1. load shader
        auto vertShaderCode = utils::read_file(vertShaderFilename);
        auto fragShaderCode = utils::read_file(fragShaderFilename);

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // vertex stage
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // entrypoint name

        // frag stage
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main"; // entrypoint name

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo, fragShaderStageInfo
        };
        
        // 2. binding input state
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        // 3. input assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // 4. dynamic state and viewport & scissor
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateInfo.dynamicStateCount = static_cast<uint32>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        // use dynamic state 
        // ! will be set by command in recordCommanBuffer
        // VkViewport viewport{0, 0, WIDTH, HEIGHT, 0, 1};
        // VkRect2D scissors{{0, 0}, {(uint32) WIDTH, (uint32) HEIGHT}};

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        // viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        // viewportState.pScissors = &scissors;

        // 5. rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        // 6. multisampling
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = msaaSamples;

        // 7. depth test
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;

        // 8. color blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT 
                                        | VK_COLOR_COMPONENT_G_BIT 
                                        | VK_COLOR_COMPONENT_B_BIT 
                                        | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // 9. pipeline layout 
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // 10. pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        // 1
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        // 2
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        // 3
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        // 4
        pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.pViewportState = &viewportState;
        // 5
        pipelineInfo.pRasterizationState = &rasterizer;
        // 6
        pipelineInfo.pMultisampleState = &multisampling;
        // 7
        pipelineInfo.pDepthStencilState = &depthStencil;
        // 8
        pipelineInfo.pColorBlendState = &colorBlending;
        // 9
        pipelineInfo.layout = pipelineLayout;
        // 10 render pass
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;   

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    void createColorResources() {
#ifdef NDEBUG
        cout << "createColorResources called" << endl;
#endif
        VkFormat colorFormat = swapChainImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat,
                    VK_IMAGE_TILING_OPTIMAL, 
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        
        colorImageView = createImageView(colorImage, colorFormat, 
                                         VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
    VkCommandBuffer beginSingleTimeCommands() {
#ifdef NDEBUG
        cout << "beginSingleTimeCommands called" << endl;
#endif
        VkCommandBufferAllocateInfo allocInfo{};

        allocInfo.sType =  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        allocInfo.commandPool = commandPool;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
#ifdef NDEBUG
        cout << "endSingleTimeCommands called" << endl;
#endif
        vkEndCommandBuffer(commandBuffer);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit commands to graphics queue!");
        }
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    void transitionImageLayout(VkImage image, VkFormat format, 
                               VkImageLayout oldLayout, VkImageLayout newLayout, uint32 mipLevels) {
#ifdef NDEBUG
        cout << "transitionImageLayout called" << endl;
#endif
        auto commandBuffer = beginSingleTimeCommands(); {
            VkImageMemoryBarrier barrier;
            VkImageSubresourceRange range;

            range.baseMipLevel = 0;
            range.levelCount = mipLevels;
            range.baseArrayLayer = 0;
            range.layerCount = 1;

            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

                if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
                    range.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
            } else {
                range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }

            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = range;

            VkPipelineStageFlagBits srcStage;   
            VkPipelineStageFlagBits dstStage;

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_NONE;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_NONE;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_NONE;
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT 
                                        | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                
                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            } else {
                throw std::invalid_argument("Unsupported layout transition!");
            }

            vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, VK_DEPENDENCY_BY_REGION_BIT, 
                                 0, nullptr, 0, nullptr, 1, &barrier);
        } endSingleTimeCommands(commandBuffer);
    }
    void createDepthResources() {
#ifdef NDEBUG
        cout << "createDepthResources called" << endl;
#endif
        VkFormat depthFormat = depthImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat,
                    VK_IMAGE_TILING_OPTIMAL, 
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        
        depthImageView = createImageView(depthImage, depthFormat,
                                         VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
    }
    void createCommandPool() {
#ifdef NDEBUG
        cout << "createCommandPool called" << endl;
#endif
        VkCommandPoolCreateInfo createInfo{};

        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }
    void createFrameBuffers() {
#ifdef NDEBUG
        cout << "createFrameBuffers called" << endl;
#endif
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 3> attachments = {
                colorImageView,
                depthImageView,
                swapChainImageViews[i],
            };
            
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
        }
    }
    void allocateCommandBuffers() {
#ifdef NDEBUG
        cout << "allocateCommandBuffers called" << endl;
#endif
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }
    void loadCube() {
#ifdef NDEBUG
        cout << "loadCube called" << endl;
#endif
        vertices.clear();
        vertIndices.clear();
        vertices = {
            // front (Z = 1.0) - red
            {{-0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, // 0
            {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, // 1
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, // 2
            {{-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, // 3

            // back (Z = -0.5) - green
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, // 4
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, // 5
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, // 6
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, // 7

            // left (X = -0.5) - blue
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}}, // 8
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}}, // 9
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, // 10
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}, // 11

            // right (X = 0.5) - yellow
            {{ 0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}}, // 12
            {{ 0.5f,  0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}}, // 13
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}}, // 14
            {{ 0.5f, -0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}}, // 15

            // upside (Y = -0.5) - cyan
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}}, // 16
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}}, // 17
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}}, // 18
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}}, // 19

            // downside (Y = 0.5) - magenta
            {{-0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}}, // 20
            {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}}, // 21
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 1.0f}}, // 22
            {{-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 1.0f}}  // 23
        };

        vertIndices = {
            0, 1, 2,    2, 3, 0,  // front: 0-1-2-3 
            4, 7, 6,    6, 5, 4,  // back: 
            8, 9, 10,   10, 11, 8, // left: 8-9-10-11
            12, 15, 14, 14, 13, 12, // right: 12-15-14-13
            16, 19, 18, 18, 17, 16, // top: 16-19-18-17
            20, 21, 22, 22, 23, 20  // bottom: 20-21-22-23
        };
    }
    void loadTorus() {
#ifdef NDEBUG
        cout << "loadTorus called" << endl;
#endif
        vertices.clear();
        vertIndices.clear();

        const float mainRadius = 0.8f;    
        const float tubeRadius = 0.25f;  
        const int mainSegments = 64;     
        const int tubeSegments = 32;     

        for (int i = 0; i <= mainSegments; ++i) {
            float u = (float)i / mainSegments * 2.0f * glm::pi<float>();
            for (int j = 0; j <= tubeSegments; ++j) {
                float v = (float)j / tubeSegments * 2.0f * glm::pi<float>();

                float x = (mainRadius + tubeRadius * cos(v)) * cos(u);
                float y = (mainRadius + tubeRadius * cos(v)) * sin(u);
                float z = tubeRadius * sin(v);

                Vertex vertex{};
                vertex.pos = {x, y, z};
                // color gradient
                vertex.color = {0.5f + 0.5f * cos(u), 0.5f + 0.5f * sin(v), 0.5f};
                vertices.push_back(vertex);
            }
        }

        for (int i = 0; i < mainSegments; ++i) {
            for (int j = 0; j < tubeSegments; ++j) {
                int first = i * (tubeSegments + 1) + j;
                int second = first + tubeSegments + 1;

                vertIndices.push_back(first);
                vertIndices.push_back(second);
                vertIndices.push_back(first + 1);

                vertIndices.push_back(second);
                vertIndices.push_back(second + 1);
                vertIndices.push_back(first + 1);
            }
        }
    }
    void loadSphere() {
#ifdef NDEBUG
        cout << "loadSphere called" << endl;
#endif
        vertices.clear();
        vertIndices.clear();

        const float radius = 0.7f;
        const int sectorCount = 64; 
        const int stackCount = 64;

        for (int i = 0; i <= stackCount; ++i) {
            float stackAngle = glm::pi<float>() / 2 - i * glm::pi<float>() / stackCount; 
            float z = radius * sin(stackAngle);
            float xy = radius * cos(stackAngle);

            for (int j = 0; j <= sectorCount; ++j) {
                float sectorAngle = j * 2.0f * glm::pi<float>() / sectorCount; 

                Vertex vertex{};
                vertex.pos.x = xy * cos(sectorAngle);
                vertex.pos.y = xy * sin(sectorAngle);
                vertex.pos.z = z;

                vertex.color = {
                    (vertex.pos.x / radius + 1.0f) / 2.0f,
                    (vertex.pos.y / radius + 1.0f) / 2.0f,
                    (vertex.pos.z / radius + 1.0f) / 2.0f
                };
                vertices.push_back(vertex);
            }
        }

        for (int i = 0; i < stackCount; ++i) {
            int k1 = i * (sectorCount + 1);
            int k2 = k1 + sectorCount + 1;

            for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
                if (i != 0) {
                    vertIndices.push_back(k1);
                    vertIndices.push_back(k2);
                    vertIndices.push_back(k1 + 1);
                }
                if (i != (stackCount - 1)) {
                    vertIndices.push_back(k1 + 1);
                    vertIndices.push_back(k2);
                    vertIndices.push_back(k2 + 1);
                }
            }
        }
    }
    void loadMobiusStrip() {
#ifdef NDEBUG
        cout << "loadMobius called" << endl;
#endif
        vertices.clear();
        vertIndices.clear();

        const int uSegments = 128;
        const int vSegments = 20;  
        const float radius = 1.0f; 
        const float width = 0.4f;  

        for (int i = 0; i <= uSegments; ++i) {
            float u = (float)i / uSegments * 2.0f * glm::pi<float>();
            for (int j = 0; j <= vSegments; ++j) {
                float v = -width / 2.0f + (float)j / vSegments * width;

                float x = (radius + v * cos(u / 2.0f)) * cos(u);
                float y = (radius + v * cos(u / 2.0f)) * sin(u);
                float z = v * sin(u / 2.0f);

                Vertex vertex{};
                vertex.pos = {x, y, z};
                vertex.color = {0.5f + 0.5f * cos(u), 0.5f + 0.5f * sin(u), 0.8f};
                vertices.push_back(vertex);
            }
        }

        for (int i = 0; i < uSegments; ++i) {
            for (int j = 0; j < vSegments; ++j) {
                int first = i * (vSegments + 1) + j;
                int second = first + vSegments + 1;

                vertIndices.push_back(first);
                vertIndices.push_back(second);
                vertIndices.push_back(first + 1);

                vertIndices.push_back(second);
                vertIndices.push_back(second + 1);
                vertIndices.push_back(first + 1);
            }
        }
    }
    uint32 findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
#ifdef NDEBUG
        cout << "findMemoryType called" << endl;
#endif
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (int i = 0; i < memProperties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i) 
                && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }
    void createBuffer(VkDeviceSize sz, VkBufferUsageFlags usage, 
                      VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
#ifdef NDEBUG
        cout << "createBuffer called" << endl;
#endif
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sz;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize sz) {
#ifdef NDEBUG
        cout << "copyBuffer called" << endl;
#endif
        auto commandBuffer = beginSingleTimeCommands(); {
            VkBufferCopy copyRegion;

            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = sz;

            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        } endSingleTimeCommands(commandBuffer);
    }
    void createVertexBuffer() {
#ifdef NDEBUG
        cout << "createVertexBuffer called" << endl;
#endif
        size_t sz = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, sz, 0, &data);
        memcpy(data, vertices.data(), sz);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     vertBuffer, vertBufferMemory);
        
        copyBuffer(stagingBuffer, vertBuffer, sz);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    void createIndixBuffer() {
#ifdef NDEBUG
        cout << "createIndixBuffer called" << endl;
#endif
        size_t sz = sizeof(vertIndices[0]) * vertIndices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, sz, 0, &data);
        memcpy(data, vertIndices.data(), sz);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, sz);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    void createUniformBuffers() {
#ifdef NDEBUG
        cout << "createUniformBuffers called" << endl;
#endif
        size_t sz = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, sz, 0, &uniformBuffersMapped[i]);
        }
    }
    void createDescriptorPool() {
#ifdef NDEBUG
        cout << "createDescriptorPool called" << endl;
#endif
        VkDescriptorPoolSize poolSize{};

        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount = 1;
        createInfo.pPoolSizes = &poolSize;
        createInfo.maxSets = static_cast<uint32>(MAX_FRAMES_IN_FLIGHT);

        vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool);
    }
    void createDescriptorSets() {
#ifdef NDEBUG
        cout << "createDescriptorSets called" << endl;
#endif
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts = layouts.data();
        
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            
            descriptorWrite.descriptorCount = 1;

            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr;
            descriptorWrite.pTexelBufferView = nullptr;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }
    void createSyncObjects() {
#ifdef NDEBUG
        cout << "createSyncObjects called" << endl;
#endif
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);
        }
    }
    void initVulkan() {
#ifdef NDEBUG
        cout << "initVulkan called" << endl;
#endif
        createInstance();
        createSurface();
        pickupPhysicalDevice();
        queueFamilyIndices = findQueueFamilies(physicalDevice);
        createLogicalDevice();
        getQueue();

        depthImageFormat = findSupportedFormat({
                                VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT
                                },
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                            );

        createSwapChain();
        createImageViews();
        
        // pipeline
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();

        // command pool
        createCommandPool();

        createColorResources();
        createDepthResources();
        createFrameBuffers();

        allocateCommandBuffers();
        
        // loadCube();
        loadTorus();
        // loadSphere();
        // loadMobiusStrip();

        // copy data to GPU
        createVertexBuffer();
        createIndixBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();

        // sync
        createSyncObjects();
    }
    void recreateSwapChain() {

    }
    void updateUniformBuffer(uint32 currFrame) {
#ifdef NDEBUG
        cout << "updateUniformBuffer called" << endl;
#endif
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currTime = std::chrono::high_resolution_clock::now();

        float time = 0.2 * std::chrono::duration<float, std::chrono::seconds::period>(currTime - startTime).count();
        
        UniformBufferObject ubo{};
        glm::vec3 diagonalAxis = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));

        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(60.0f), diagonalAxis);

        ubo.view = glm::lookAt(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 100.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currFrame], &ubo, sizeof(ubo));
    }
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32 imageIndex, uint32 currFrame) {
#ifdef NDEBUG
        cout << "recordCommandBuffer called" << endl;
#endif
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo); {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkRenderPassBeginInfo renderPassBeginInfo{};
            renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

            std::array<VkClearValue, 2> clearColors;
            clearColors[0].color = {0.1f, 0.1f, 0.1f, 1.0f};
            clearColors[1].depthStencil = {1.0f, 0};

            renderPassBeginInfo.renderPass = renderPass;
            renderPassBeginInfo.framebuffer = swapChainFramebuffers[imageIndex];
            renderPassBeginInfo.renderArea = VkRect2D{{0, 0}, swapChainExtent};
            renderPassBeginInfo.clearValueCount = clearColors.size();
            renderPassBeginInfo.pClearValues = clearColors.data();

            // dynamic state (from graphics pipeline)
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(swapChainExtent.width);
            viewport.height = static_cast<float>(swapChainExtent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE); {
                VkBuffer vertBuffers[] = {vertBuffer};
                VkDeviceSize offsets[] = {0};
                
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currFrame], 0, nullptr);

                vkCmdDrawIndexed(commandBuffer, static_cast<uint32>(vertIndices.size()), 1, 0, 0, 0);
            } vkCmdEndRenderPass(commandBuffer);
        } vkEndCommandBuffer(commandBuffer);
    }
    void drawFrame() {
#ifdef NDEBUG
        frameRendered++;
        cout << "drawFrame called" << endl;
        cout << "totally " << frameRendered << " rendered" << endl;
#endif
        // wait for GPU
        vkWaitForFences(device, 1, &inFlightFences[currFrame], VK_TRUE, UINT64_MAX);

        uint32 imageIndex;
        if (vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, 
            imageAvailableSemaphores[currFrame], VK_NULL_HANDLE, &imageIndex) == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }

        // safe zone
        updateUniformBuffer(currFrame);

        vkResetFences(device, 1, &inFlightFences[currFrame]);

        vkResetCommandBuffer(commandBuffers[currFrame], 0);

        recordCommandBuffer(commandBuffers[currFrame], imageIndex, currFrame);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        // wait for image available semaphore
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currFrame]};

        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }

        currFrame = (currFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    void mainLoop() {
#ifdef NDEBUG
        cout << "mainLoop called" << endl;
#endif
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }
    void cleanup() {
#ifdef NDEBUG
        cout << "cleanup called" << endl;
#endif
        vkDeviceWaitIdle(device);

        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        // swapchain images will be cleaned up by swapchain itself
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertBuffer, nullptr);
        vkFreeMemory(device, vertBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
#ifdef NDEBUG
    cout << "main called" << endl;
#endif
    auto cube = std::make_unique<SpinningCube>(); 
    cube->run();

    return 0;
}