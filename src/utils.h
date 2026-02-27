#include <vector>
#include <fstream>
#include <string>

namespace utils {
static std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to load file: " + filename);       
    }

    size_t file_size = file.tellg();
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);

    file.close();
    return buffer;
}   
} // namespace utils