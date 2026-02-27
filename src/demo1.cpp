#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Helper function to load the compiled SPIR-V binary
std::vector<char> LoadSPIRV(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

int main() {
    // Note: In a real project, you would initialize VkInstance and VkDevice here.
    
    try {
        /* When you run 'xmake', it compiles vector_add.comp to vector_add.comp.spv.
           The output path relative to the executable is usually 'shaders/vector_add.comp.spv'
        */
        auto spirvCode = LoadSPIRV("shaders/vector_add.comp.spv");
        
        std::cout << "Successfully loaded SPIR-V shader module." << std::endl;
        std::cout << "Shader size: " << spirvCode.size() << " bytes." << std::endl;
        std::cout << "Vulkan Compute environment is ready for dispatch." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Hint: Make sure to run 'xmake' to compile shaders first." << std::endl;
        return -1;
    }

    return 0;
}