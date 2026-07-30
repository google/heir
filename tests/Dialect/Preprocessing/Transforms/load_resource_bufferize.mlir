// RUN: heir-opt --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s

// CHECK: func.func @test_load_resource() -> memref<4xi32> {
// CHECK: %[[RES:.*]] = preprocessing.load_resource "some/path.bin" : memref<4xi32>
// CHECK-NOT: memref.dealloc
// CHECK: return %[[RES]] : memref<4xi32>
func.func @test_load_resource() -> tensor<4xi32> {
  %0 = preprocessing.load_resource "some/path.bin" : tensor<4xi32>
  return %0 : tensor<4xi32>
}
