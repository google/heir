// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

// CHECK: template <typename T>
// CHECK: std::vector<T> load_resource(const std::string& path, size_t size) {
// CHECK:   std::ifstream file(path, std::ios::binary);
// CHECK:   if (!file.is_open()) {
// CHECK:     std::cerr << "Failed to open file: " << path << std::endl;
// CHECK:     std::abort();
// CHECK:   }
// CHECK:   std::vector<T> data(size);
// CHECK:   file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
// CHECK:   if (!file) {
// CHECK:     std::cerr << "Failed to read expected number of bytes from: " << path << std::endl;
// CHECK:     std::abort();
// CHECK:   }
// CHECK:   return data;
// CHECK: }

// CHECK: template <>
// CHECK: inline std::vector<bool> load_resource<bool>(const std::string& path, size_t size) {
// CHECK:   std::ifstream file(path, std::ios::binary);
// CHECK:   if (!file.is_open()) {
// CHECK:     std::cerr << "Failed to open file: " << path << std::endl;
// CHECK:     std::abort();
// CHECK:   }
// CHECK:   std::vector<char> temp(size);
// CHECK:   file.read(temp.data(), size);
// CHECK:   if (!file) {
// CHECK:     std::cerr << "Failed to read expected number of bytes from: " << path << std::endl;
// CHECK:     std::abort();
// CHECK:   }
// CHECK:   std::vector<bool> data(size);
// CHECK:   for (size_t i = 0; i < size; ++i) {
// CHECK:     data[i] = temp[i] != 0;
// CHECK:   }
// CHECK:   return data;
// CHECK: }

// CHECK: std::vector<int32_t> test_external_constant(const std::vector<int32_t>& [[arg0:[^ ]*]])
// CHECK:   static const std::vector<int32_t> [[v0:[^ ]*]] = load_resource<int32_t>("some/path/constant_1.bin", 4);
// CHECK:   std::vector<int32_t> [[v1:[^ ]*]](4);
// CHECK:   for (auto [[i:[^ ]*]] = 0; [[i]] < 4; ++[[i]]) {
// CHECK:     int32_t [[val0:[^ ]*]] = [[arg0]][[[i]]];
// CHECK:     int32_t [[val1:[^ ]*]] = [[v0]][[[i]]];
// CHECK:     int32_t [[add:[^ ]*]] = [[val0]] + [[val1]];
// CHECK:     [[v1]][[[i]]] = [[add]];
// CHECK:   }

// CHECK: std::vector<bool> test_external_constant_i1(const std::vector<bool>& [[arg0:[^ ]*]])
// CHECK:   static const std::vector<bool> [[v0:[^ ]*]] = load_resource<bool>("some/path/constant_i1.bin", 4);
// CHECK:   std::vector<bool> [[v1:[^ ]*]](4);
// CHECK:   for (auto [[i:[^ ]*]] = 0; [[i]] < 4; ++[[i]]) {
// CHECK:     bool [[val0:[^ ]*]] = [[arg0]][[[i]]];
// CHECK:     bool [[val1:[^ ]*]] = [[v0]][[[i]]];
// CHECK:     bool [[and:[^ ]*]] = [[val0]] && [[val1]];
// CHECK:     [[v1]][[[i]]] = [[and]];
// CHECK:   }

// CHECK: std::vector<int32_t> test_tensor_external_constant()
// CHECK-NOT: std::vector<int32_t> {{[^ ]*}}(4);
// CHECK: static const std::vector<int32_t> [[resource:[^ ]*]] = load_resource<int32_t>("some/path/tensor_constant.bin", 4);
// CHECK: return [[resource]];

module attributes {scheme.bgv} {
  func.func @test_external_constant(%arg0: memref<4xi32>) -> memref<4xi32> {
    %0 = memref.alloc() : memref<4xi32>
    preprocessing.load_resource "some/path/constant_1.bin" into %0
        : (memref<4xi32>) -> ()
    %res = memref.alloc() : memref<4xi32>
    affine.for %i = 0 to 4 {
      %val0 = memref.load %arg0[%i] : memref<4xi32>
      %val1 = memref.load %0[%i] : memref<4xi32>
      %add = arith.addi %val0, %val1 : i32
      memref.store %add, %res[%i] : memref<4xi32>
    }
    return %res : memref<4xi32>
  }

  func.func @test_external_constant_i1(%arg0: memref<4xi1>) -> memref<4xi1> {
    %0 = memref.alloc() : memref<4xi1>
    preprocessing.load_resource "some/path/constant_i1.bin" into %0
        : (memref<4xi1>) -> ()
    %res = memref.alloc() : memref<4xi1>
    affine.for %i = 0 to 4 {
      %val0 = memref.load %arg0[%i] : memref<4xi1>
      %val1 = memref.load %0[%i] : memref<4xi1>
      %and = arith.andi %val0, %val1 : i1
      memref.store %and, %res[%i] : memref<4xi1>
    }
    return %res : memref<4xi1>
  }

  func.func @test_tensor_external_constant() -> tensor<4xi32> {
    %destination = tensor.empty() : tensor<4xi32>
    %resource = preprocessing.load_resource
        "some/path/tensor_constant.bin" into %destination
        : (tensor<4xi32>) -> tensor<4xi32>
    return %resource : tensor<4xi32>
  }
}
