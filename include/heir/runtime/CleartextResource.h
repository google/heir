#ifndef INCLUDE_HEIR_RUNTIME_CLEARTEXTRESOURCE_H_
#define INCLUDE_HEIR_RUNTIME_CLEARTEXTRESOURCE_H_

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace heir {

// Loads the little-endian raw representation written by
// --externalize-constants into caller-owned, statically shaped storage.
template <typename T>
void loadResource(const std::string& path, T* data, std::size_t size) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open resource: " << path << '\n';
    std::abort();
  }
  file.read(reinterpret_cast<char*>(data), size * sizeof(T));
  if (!file) {
    std::cerr << "Failed to read expected bytes from resource: " << path
              << '\n';
    std::abort();
  }
}

template <typename T>
void loadResource(std::string_view directory, const std::string& path, T* data,
                  std::size_t size) {
  if (directory.empty() || (!path.empty() && path.front() == '/')) {
    loadResource(path, data, size);
    return;
  }
  std::string fullPath(directory);
  if (fullPath.back() != '/') fullPath.push_back('/');
  fullPath.append(path);
  loadResource(fullPath, data, size);
}

}  // namespace heir

#endif  // INCLUDE_HEIR_RUNTIME_CLEARTEXTRESOURCE_H_
