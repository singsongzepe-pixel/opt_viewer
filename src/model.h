
typedef struct {
    uint32_t firstIndex;
    uint32_t indexCount;
    int materialIndex;
} Primitive;

typedef struct {
    // on CPU
    std::vector<unsigned char> pixels;
    int width, height;
    int components;
    
    // on GPU
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;

    uint32_t mipLevels = 1;
} Texture;

typedef struct {
    // on CPU
    int baseColorTextureIndex = -1;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);

    // for GPU
    std::vector<VkDescriptorSet> descriptorSets;
} Material;
