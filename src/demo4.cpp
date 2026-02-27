#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS  
#define GLM_FORCE_DEPTH_ZERO_TO_ONE  
// #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/vec4.hpp> // glm::vec4  
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/type_ptr.hpp>  
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// loader
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

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

// for model
#include "model.h"

// #define NDEBUG

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
const char* vertShaderFilename = "shaders/blinn_phong.vert.spv";
const char* fragShaderFilename = "shaders/blinn_phong.frag.spv";

// model filename
const char* modelGltfFilename = "assets/Sponza/glTF/Sponza.gltf";

// texture sampler max mip levels
const uint32_t MAX_MIP_LEVLES = 10;

// dummy material color
const uint32_t dummyColor = 0xFFFFFFFF;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    // TODO
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Vertex, tangent);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    // MVP mat
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;

    // light and viewer
    glm::vec4 lightPos;
    glm::vec4 lightColor;
    glm::vec4 viewPos;

    // coefficients
    glm::vec4 ka;
    glm::vec4 kdiff;
    glm::vec4 kspec;

    UniformBufferObject() = default;

    UniformBufferObject(const UniformBufferObject& ubo) {
        model = ubo.model;
        view = ubo.view;
        proj = ubo.proj;

        lightPos = ubo.lightPos;
        lightColor = ubo.lightColor;
        viewPos = ubo.viewPos;

        ka = ubo.ka;
        kdiff = ubo.kdiff;
        kspec = ubo.kspec;
    }
};

class PBRRenderer {
public:
    void run() {
#ifdef NDEBUG
        cout << "cube.run called" << "\n";
#endif
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    PBRRenderer() {}

private:
    GLFWwindow* window;
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    // UBO
    UniformBufferObject baseUbo{};

    static void mouseCallback(GLFWwindow *window, double xposIn, double yposIn) {
        auto app = reinterpret_cast<PBRRenderer*>(glfwGetWindowUserPointer(window));
        if (app) {
            app->handleMouseMove(xposIn, yposIn);
        }
    }
    void handleMouseMove(double xposIn, double yposIn) {
        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; 
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        updateCameraVectors();
    }
    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
    }   
    void initWindow() {
#ifdef NDEBUG
        cout << "initWindow called" << "\n";
#endif
        glfwInit();

        // using Vulkan not OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Cube Demo", nullptr, nullptr);

        // ! mouse interaction, but works bad
        // glfwSetWindowUserPointer(window, this);
        // glfwSetCursorPosCallback(window, mouseCallback);

        // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        
        // if (glfwRawMouseMotionSupported()) {
        //     glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        // }
    }
    // Vulkan things
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

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

    // num equals to MAX FRAME
    std::vector<VkCommandBuffer> commandBuffers;

    // model
    std::vector<Primitive> primitives;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> vertIndices;
    VkBuffer            vertBuffer;
    VkDeviceMemory      vertBufferMemory;
    VkBuffer            indexBuffer;
    VkDeviceMemory      indexBufferMemory;
    
    // texture
    Texture dummyTexture;
    std::vector<Texture> textures;
    // sampler
    VkSampler textureSampler;
    // material
    std::vector<Material> materials;

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
    uint32_t currFrame = 0;

    // camera 
    float fov = 45.0f;

    glm::vec3 cameraPos{-10.0f, 0.0f, 0.0f};
    glm::vec3 cameraFront{1.0f, 0.0f, 0.0f};
    glm::vec3 cameraUp{0.0f, 0.0f, 1.0f};
    // glm::vec3 cameraVelocity{0.0f, 0.0f, 0.0f};
    
    // mouse state
    float yaw = -90.0f;
    float pitch = 0.0f;
    float lastX = (float) WIDTH / 2.0f;
    float lastY = (float) HEIGHT / 2.0f;
    bool firstMouse = true; 

    //
    float deltaTime = 0.0f;
    float lastFrameTime = 0.0f;

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capanilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "findQueueFamilies called" << "\n";
#endif
        QueueFamilyIndices indices{};

        uint32_t queueFamilyCount = 0;
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
        cout << "checkDeviceExtensionSupport called" << "\n";
#endif
        uint32_t extensionCount;
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
        cout << "querySwapChainSupport called" << "\n";
#endif
        SwapChainSupportDetails details{};

        // query capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capanilities);
        // query supported formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        // query present mode
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
#ifdef NDEBUG
        cout << "chooseSwapSurfaceFormat called" << "\n";
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
        cout << "chooseSwapPresentMode called" << "\n";
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
        cout << "chooseSwapExtent called" << "\n";
#endif
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
#ifdef NDEBUG
        cout << "findSupportedFormat called" << "\n";
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
        cout << "createInstance called" << "\n";
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
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        // validation layer
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan Instance!");
        }
    }
    void createSurface() {
#ifdef NDEBUG
        cout << "createSurface called" << "\n";
#endif
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    bool isDeviceSuitable(const VkPhysicalDevice& device) {
#ifdef NDEBUG
        cout << "isDeviceSuitable called" << "\n";
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
        cout << "pickupPhysicalDevice called" << "\n";
#endif
        uint32_t deviceCount = 0;
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
        cout << "createLogicalDevice called" << "\n";
#endif
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        std::set<uint32_t> uniqueQueueFamilies = {
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
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        
        // extensions
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to createa logical device!");
        }
    }
    void getQueue() {
#ifdef NDEBUG
        cout << "getQueue called" << "\n";
#endif
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &presentQueue);
    }
    void createSwapChain() {
#ifdef NDEBUG
        cout << "createSwapChain called" << "\n";
#endif
        SwapChainSupportDetails details = querySwapChainSupport(physicalDevice);

        // select the best format, present mode, resolution
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(details.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(details.presentModes);
        VkExtent2D extent = chooseSwapExtent(details.capanilities);

        // determine image count
        uint32_t imageCount = details.capanilities.minImageCount + 1;
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

        uint32_t rawQueueFamilyIndices[] = { 
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
                                VkImageAspectFlagBits aspectFlags, uint32_t mipLevels) {
#ifdef NDEBUG
        cout << "createImageView called" << "\n";
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
        cout << "createImageViews called" << "\n";
#endif
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, 
                                                     VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples,
                     VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
#ifdef NDEBUG
        cout << "createImage called" << "\n";
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
        cout << "createRenderPass called" << "\n";
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
        cout << "createDescriptorSetLayout called" << "\n";
#endif
        // binding 1: ubo
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;

        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        
        // binding 2: texture sampler
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        // combine them
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
            uboLayoutBinding,
            samplerLayoutBinding
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout)) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }
    VkShaderModule createShaderModule(const std::vector<char>& code) {
#ifdef NDEBUG
        cout << "createShaderModule called" << "\n";
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
        cout << "createGraphicsPipeline called" << "\n";
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
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
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
        dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        // use dynamic state 
        // ! will be set by command in recordCommanBuffer
        // VkViewport viewport{0, 0, WIDTH, HEIGHT, 0, 1};
        // VkRect2D scissors{{0, 0}, {(uint32_t) WIDTH, (uint32_t) HEIGHT}};

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
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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
        cout << "createColorResources called" << "\n";
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
        cout << "beginSingleTimeCommands called" << "\n";
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

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        };

        return commandBuffer;
    }
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
#ifdef NDEBUG
        cout << "endSingleTimeCommands called" << "\n";
#endif
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        };
        
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
                               VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
#ifdef NDEBUG
        cout << "transitionImageLayout called" << "\n";
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
        cout << "createDepthResources called" << "\n";
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
        cout << "createCommandPool called" << "\n";
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
        cout << "createFrameBuffers called" << "\n";
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
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
        }
    }
    void allocateCommandBuffers() {
#ifdef NDEBUG
        cout << "allocateCommandBuffers called" << "\n";
#endif
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }
    void loadMeshGeometry(const tinygltf::Model& model) {
#ifdef NDEBUG
    cout << "loadMeshGeometry called" << "\n";
#endif

        for (const auto& mesh : model.meshes) {
            for (const auto& primitive : mesh.primitives) {
                uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
                
                // 1. position
                const tinygltf::Accessor& posAccessor = model.accessors.at(primitive.attributes.at("POSITION"));
                const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
                const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
                const float* posData = reinterpret_cast<const float*>(&(posBuffer.data[posAccessor.byteOffset + posView.byteOffset]));
                int posStride = posAccessor.ByteStride(posView) / sizeof(float);

                // 2. normal
                const float* normData = nullptr;
                int normStride = 0;
                if (primitive.attributes.count("NORMAL")) {
                    const tinygltf::Accessor& normAccessor = model.accessors.at(primitive.attributes.at("NORMAL"));
                    const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
                    normData = reinterpret_cast<const float*>(&(model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
                    normStride = normAccessor.ByteStride(normView) / sizeof(float);
                }

                // 3. texCoord
                const float* uvData = nullptr;
                int uvStride = 0;
                if (primitive.attributes.count("TEXCOORD_0")) {
                    const tinygltf::Accessor& uvAccessor = model.accessors.at(primitive.attributes.at("TEXCOORD_0"));
                    const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.byteOffset];
                    uvData = reinterpret_cast<const float*>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                    uvStride = uvAccessor.ByteStride(uvView) / sizeof(float);
                }

                // fill in vertices
                for (size_t v = 0; v < posAccessor.count; v++) {
                    Vertex vertex{};
                    vertex.pos = glm::make_vec3(&posData[v * posStride]);
                    vertex.normal = normData ? glm::make_vec3(&normData[v * normStride]) : glm::vec3(0.0f, 1.0f, 0.0f);
                    vertex.texCoord = uvData ? glm::make_vec2(&uvData[v * uvStride]) : glm::vec2(0.0f);
                    vertices.push_back(vertex);
                }

                // 4. indices
                const auto& indexAccessor = model.accessors[primitive.indices];
                const auto& indexView = model.bufferViews[indexAccessor.bufferView];
                const auto& indexBuffer = model.buffers[indexView.buffer];
                uint32_t indexStart = static_cast<uint32_t>(vertIndices.size());

                // index type adaptor (5123: uint16, 5125: uint32_t)
                if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(&(indexBuffer.data[indexAccessor.byteOffset + indexView.byteOffset]));
                    for (size_t i = 0; i < indexAccessor.count; i++) vertIndices.push_back(buf[i] + vertexStart);
                } else if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(&(indexBuffer.data[indexAccessor.byteOffset + indexView.byteOffset]));
                    for (size_t i = 0; i < indexAccessor.count; i++) vertIndices.push_back(buf[i] + vertexStart);
                }

                // record primitives
                Primitive part{};
                part.firstIndex = indexStart;
                part.indexCount = static_cast<uint32_t>(indexAccessor.count);
                part.materialIndex = primitive.material;
                primitives.push_back(part);
            }
        }
    }
    void loadTextures(const tinygltf::Model& model) {
#ifdef NDEBUG
    cout << "loadTextures called" << "\n";
#endif
        textures.resize(model.images.size());

        for (size_t i = 0; i < model.images.size(); i++) {
            const tinygltf::Image& gltfImage = model.images[i];

            textures[i].width = gltfImage.width;
            textures[i].height = gltfImage.height;
            textures[i].components = gltfImage.component;

            textures[i].pixels = gltfImage.image; 

#ifdef NDEBUG
        cout << "  Loaded texture: " << gltfImage.name 
            << " (" << gltfImage.width << "x" << gltfImage.height << ")" << "\n";
#endif
        }
    }
    void loadMaterials(const tinygltf::Model& model) {
#ifdef NDEBUG
    cout << "loadMaterials called" << "\n";
#endif
        materials.resize(model.materials.size());

        for (size_t i = 0; i < model.materials.size(); i++) {
            const tinygltf::Material& gltfMat = model.materials[i];
            Material& myMat = materials[i];

            auto& color = gltfMat.pbrMetallicRoughness.baseColorFactor;
            myMat.baseColorFactor = glm::vec4(color[0], color[1], color[2], color[3]);

            if (gltfMat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                int textureIndex = gltfMat.pbrMetallicRoughness.baseColorTexture.index;
                myMat.baseColorTextureIndex = model.textures[textureIndex].source;
            } else {
                myMat.baseColorTextureIndex = -1;
            }

#ifdef NDEBUG
        cout << "  Material " << i << ": " << gltfMat.name 
            << " (Texture Index: " << myMat.baseColorTextureIndex << ")" << "\n";
#endif
        }
    }
    void loadModel() {
#ifdef NDEBUG
    cout << "loadModel called" << "\n";
#endif
tinygltf::TinyGLTF loader;
tinygltf::Model model;
        std::string err, warn;

        if (!loader.LoadASCIIFromFile(&model, &err, &warn, modelGltfFilename)) {
            throw std::runtime_error("Failed to load glTF: " + err);
        }
        // 1. mesh geometry
        loadMeshGeometry(model);
        // 2. texture
        loadTextures(model);
        // 3. material
        loadMaterials(model);
    }
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
#ifdef NDEBUG
        cout << "findMemoryType called" << "\n";
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
        cout << "createBuffer called" << "\n";
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
        cout << "copyBuffer called" << "\n";
#endif
        auto commandBuffer = beginSingleTimeCommands(); {
            VkBufferCopy copyRegion;

            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = sz;

            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        } endSingleTimeCommands(commandBuffer);
    }
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
#ifdef NDEBUG
        cout << "copyBufferToImage called" << "\n";
#endif
        auto commandBuffer = beginSingleTimeCommands(); {
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0; 
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = {0, 0, 0};
            region.imageExtent = { width, height, 1 };

            vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        } endSingleTimeCommands(commandBuffer);
    }
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
#ifdef NDEBUG
        cout << "generateMipmaps called" << "\n";
#endif
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            throw std::runtime_error("Texture image format does not support linear blitting!");
        }

        auto commandBuffer = beginSingleTimeCommands(); {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = image;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.levelCount = 1;

            int32_t mipWidth = texWidth;
            int32_t mipHeight = texHeight;

            for (uint32_t i = 1; i < mipLevels; i++) {
                barrier.subresourceRange.baseMipLevel = i - 1;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    0, nullptr, 0, nullptr, 1, &barrier);

                VkImageBlit blit{};
                blit.srcOffsets[0] = { 0, 0, 0 };
                blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
                blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.mipLevel = i - 1;
                blit.srcSubresource.baseArrayLayer = 0;
                blit.srcSubresource.layerCount = 1;
                blit.dstOffsets[0] = { 0, 0, 0 };
                blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
                blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.dstSubresource.mipLevel = i;
                blit.dstSubresource.baseArrayLayer = 0;
                blit.dstSubresource.layerCount = 1;
                
                // blit image from level(i-1) to level(i)
                vkCmdBlitImage(commandBuffer,
                    image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blit, VK_FILTER_LINEAR);

                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                
                // 
                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                    0, nullptr, 0, nullptr, 1, &barrier);

                if (mipWidth > 1) mipWidth /= 2;
                if (mipHeight > 1) mipHeight /= 2;
            }

            barrier.subresourceRange.baseMipLevel = mipLevels - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr, 0, nullptr, 1, &barrier);

        } endSingleTimeCommands(commandBuffer);
    }
    void createDummyTextureBuffer() {
#ifdef NDEBUG
        cout << "createTextureBuffers called" << "\n";
#endif

        uint32_t whitePixel = dummyColor; 
        VkDeviceSize imageSize = 4;
        dummyTexture.width = 1;
        dummyTexture.height = 1;
        dummyTexture.mipLevels = 1;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                    stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, &whitePixel, 4);
        vkUnmapMemory(device, stagingBufferMemory);

        createImage(1, 1, 1, VK_SAMPLE_COUNT_1_BIT, 
                    VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dummyTexture.image, dummyTexture.imageMemory);

        transitionImageLayout(dummyTexture.image, VK_FORMAT_R8G8B8A8_SRGB, 
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
        copyBufferToImage(stagingBuffer, dummyTexture.image, 1, 1);
        
        transitionImageLayout(dummyTexture.image, VK_FORMAT_R8G8B8A8_SRGB, 
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        dummyTexture.imageView = createImageView(dummyTexture.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
    void createTextureBuffers() {
#ifdef NDEBUG
        cout << "createTextureBuffers called" << "\n";
#endif
        for (auto& tex : textures) {
            if (tex.pixels.empty()) continue;

            // calculate mipmap levels;
            tex.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(tex.width, tex.height)))) + 1;
            
            // the size of a img
            VkDeviceSize imageSize = tex.width * tex.height * 4;

            // staging Buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                        stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
            memcpy(data, tex.pixels.data(), static_cast<size_t>(imageSize));
            vkUnmapMemory(device, stagingBufferMemory);

            createImage(tex.width, tex.height, tex.mipLevels, VK_SAMPLE_COUNT_1_BIT, 
                        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tex.image, tex.imageMemory);

            transitionImageLayout(tex.image, VK_FORMAT_R8G8B8A8_SRGB, 
                                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex.mipLevels);
            copyBufferToImage(stagingBuffer, tex.image, static_cast<uint32_t>(tex.width), static_cast<uint32_t>(tex.height));

            generateMipmaps(tex.image, VK_FORMAT_R8G8B8A8_SRGB, tex.width, tex.height, tex.mipLevels);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            // maybe also free the memory of texture on CPU

            tex.imageView = createImageView(tex.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, tex.mipLevels);
        }
    }
    void createVertexBuffer() {
#ifdef NDEBUG
        cout << "createVertexBuffer called" << "\n";
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
    void createIndexBuffer() {
#ifdef NDEBUG
        cout << "createIndexBuffer called" << "\n";
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
        cout << "createUniformBuffers called" << "\n";
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
        cout << "createDescriptorPool called" << "\n";
#endif
        uint32_t materialCount = static_cast<uint32_t>(materials.size());
        uint32_t totalSets = materialCount * MAX_FRAMES_IN_FLIGHT;

        std::vector<VkDescriptorPoolSize> poolSizes(2);

        // ubo
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = totalSets;
        // texture sampler
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = totalSets;

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        createInfo.pPoolSizes = poolSizes.data();
        createInfo.maxSets = totalSets;

        if (vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool.");
        };
    }
    void createDescriptorSets() {
#ifdef NDEBUG
    cout << "createDescriptorSets called" << "\n";
#endif
        uint32_t materialCount = static_cast<uint32_t>(materials.size());

        for (uint32_t i = 0; i < materialCount; i++) {
            materials[i].descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
            allocInfo.pSetLayouts = layouts.data();

            if (vkAllocateDescriptorSets(device, &allocInfo, materials[i].descriptorSets.data()) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptor sets for material " + std::to_string(i));
            }

            for (uint32_t j = 0; j < MAX_FRAMES_IN_FLIGHT; j++) {
                // 1. ubo (Binding 0)
                VkDescriptorBufferInfo bufferInfo{};
                bufferInfo.buffer = uniformBuffers[j];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                // 2. texture sampler (Binding 1)
                VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.sampler = textureSampler;

                int texIndex = materials[i].baseColorTextureIndex;
                if (texIndex >= 0 && texIndex < textures.size()) {
                    imageInfo.imageView = textures[texIndex].imageView;
                } else {
                    imageInfo.imageView = dummyTexture.imageView;
                }

                std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstSet = materials[i].descriptorSets[j];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;

                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstSet = materials[i].descriptorSets[j];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pImageInfo = &imageInfo;

                vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }
        }
    }
    void createSyncObjects() {
#ifdef NDEBUG
        cout << "createSyncObjects called" << "\n";
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
    void createTextureSampler() {
#ifdef NDEBUG
        cout << "createBuffers called" << "\n";
#endif
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        
        samplerInfo.anisotropyEnable = VK_TRUE;

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(MAX_MIP_LEVLES);

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create texture sampler.");
        }
    }
    void createBuffers() {
#ifdef NDEBUG
        cout << "createBuffers called" << "\n";
#endif
        // upload a white texture to GPU for those material -1
        createDummyTextureBuffer();
        // upload textures to GPU
        createTextureBuffers();
        // upload vertices to GPU
        createVertexBuffer();
        // upload vertex indices to GPU
        createIndexBuffer();
        // upload MVP mat and some other parameters to GPU
        createUniformBuffers();

        // shader resources ptr
        createDescriptorPool();
        createDescriptorSets();
    }
    void initVulkan() {
#ifdef NDEBUG
        cout << "initVulkan called" << "\n";
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
        createRenderPass();
        
        // pipeline
        createDescriptorSetLayout();
        createGraphicsPipeline();

        // texture sampler
        createTextureSampler();
        
        // load geometry textures materials
        loadModel();

        // command pool
        createCommandPool();

        createColorResources();
        createDepthResources();
        createFrameBuffers();

        allocateCommandBuffers();
                            
        // copy data to GPU
        createBuffers();

        // sync
        createSyncObjects();
    }
    void recreateSwapChain() {
        int width = 0, height = 0;

        glfwGetFramebufferSize(window, &width, &height);

        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);
        
        cleanupSwapChain();
        createSwapChain();
        createColorResources();
        createDepthResources();
        createImageViews();
        createFrameBuffers();
    }
    void updateUniformBuffer(uint32_t currFrame) {
#ifdef NDEBUG
        cout << "updateUniformBuffer called" << "\n";
#endif
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currTime = std::chrono::high_resolution_clock::now();

        float time = 0.2 * std::chrono::duration<float, std::chrono::seconds::period>(currTime - startTime).count();
        
        UniformBufferObject ubo{};
        glm::vec3 diagonalAxis = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));

        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(60.0f), diagonalAxis);

        ubo.view = glm::lookAt(cameraPos, cameraFront, cameraUp);

        ubo.proj = glm::perspective(glm::radians(fov), swapChainExtent.width / (float) swapChainExtent.height, 0.01f, 50.0f);
        ubo.proj[1][1] *= -1;

        ubo.lightPos = glm::vec4(3.0f * sin(time), 3.0f * cos(time), 5.0f, 1.0f); 
        // ubo.lightPos = glm::vec4(0.5f, 0.0f, 0.0f, 1.0f);
        ubo.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        ubo.viewPos = glm::vec4(cameraPos, 1.0f);

        ubo.ka = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
        ubo.kdiff = glm::vec4(0.0f, 0.7f, 0.7f, 1.0f);
        ubo.kspec = glm::vec4(1.0f, 1.0f, 1.0f, 32.0f * (1 + sin(time * 0.5f)));

        memcpy(uniformBuffersMapped[currFrame], &ubo, sizeof(ubo));
    }
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t currFrame) {
#ifdef NDEBUG
        cout << "recordCommandBuffer called" << "\n";
#endif
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // one time command buffer

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) { 
            throw std::runtime_error("Failed to begin recording command buffer"); 
        }; {
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
                for (const auto& part : primitives) {
                    
                    VkDescriptorSet materialSet = materials[part.materialIndex].descriptorSets[currFrame];
                    
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
                        0, 1, &materialSet, 0, nullptr);
                    
                    vkCmdDrawIndexed(commandBuffer, part.indexCount, 1, part.firstIndex, 0, 0);
                }
                // vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currFrame], 0, nullptr);
                // vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(vertIndices.size()), 1, 0, 0, 0);
            } vkCmdEndRenderPass(commandBuffer);
        } if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        };
    }
    void drawFrame() {
#ifdef NDEBUG
        frameRendered++;
        cout << "drawFrame called" << "\n";
        cout << "totally " << frameRendered << " rendered" << "\n";
#endif
        // wait for GPU
        vkWaitForFences(device, 1, &inFlightFences[currFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
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
    void processInput(GLFWwindow *window) {
#ifdef NDEBUG
        cout << "processInput called" << "\n";
#endif
        glm::vec3 normalizedFront = glm::normalize(cameraFront);
        float cameraSpeed = 2.5f * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraPos += normalizedFront * cameraSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraPos -= normalizedFront * cameraSpeed;
        }

        glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraPos -= cameraRight * cameraSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraPos += cameraRight * cameraSpeed;
        }
    }
    void mainLoop() {
#ifdef NDEBUG
        cout << "mainLoop called" << "\n";
#endif
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            
            float currentFrameTime = static_cast<float>(glfwGetTime());
            deltaTime = currentFrameTime - lastFrameTime;
            lastFrameTime = currentFrameTime;

            if (deltaTime > 0.1f) deltaTime = 0.1f;

            processInput(window);

            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }
    void cleanupSwapChain() {
#ifdef NDEBUG
        cout << "cleanupSwapChain called" << "\n";
#endif
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        // swapchain images will be cleaned up by swapchain itself
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
    void cleanup() {
#ifdef NDEBUG
        cout << "cleanup called" << "\n";
#endif
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, vertBuffer, nullptr);
        vkFreeMemory(device, vertBufferMemory, nullptr);

        // textures
        for (auto& texture : textures) {
            vkDestroyImageView(device, texture.imageView, nullptr);
            vkDestroyImage(device, texture.image, nullptr);
            vkFreeMemory(device, texture.imageMemory, nullptr);
        }
        
        // dummy texture
        vkDestroyImageView(device, dummyTexture.imageView, nullptr);
        vkDestroyImage(device, dummyTexture.image, nullptr);
        vkFreeMemory(device, dummyTexture.imageMemory, nullptr);

        // sampler
        vkDestroySampler(device, textureSampler, nullptr);

        // ubo
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        // descriptor
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        // graphics pipeline
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        // sync
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        // infrastructure
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
#ifdef NDEBUG
    cout << "main called" << "\n";
#endif
    auto renderer = std::make_unique<PBRRenderer>(); 
    renderer->run();

    return 0;
}