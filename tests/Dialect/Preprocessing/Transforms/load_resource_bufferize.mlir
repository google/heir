// RUN: heir-opt --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s

// CHECK: func.func @test_load_resource() -> memref<4xi32> {
// CHECK-NEXT: %[[RES:[^ ]+]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
// CHECK-NEXT: preprocessing.load_resource "some/path.bin" into %[[RES]] : (memref<4xi32>) -> ()
// CHECK: return %[[RES]] : memref<4xi32>
func.func @test_load_resource() -> tensor<4xi32> {
  %destination = tensor.empty() : tensor<4xi32>
  %0 = preprocessing.load_resource "some/path.bin" into %destination
      : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK: func.func @test_write_after_load
// CHECK: %[[RESOURCE:[^ ]+]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
// CHECK-NEXT: preprocessing.load_resource "some/path.bin" into %[[RESOURCE]] : (memref<4xi32>) -> ()
// CHECK: %[[COPY:[^ ]+]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
// CHECK: memref.copy %[[RESOURCE]], %[[COPY]] : memref<4xi32> to memref<4xi32>
// CHECK: memref.store %arg0, %[[COPY]]
// CHECK: return %[[COPY]] : memref<4xi32>
func.func @test_write_after_load(%value: i32) -> tensor<4xi32> {
  %destination = tensor.empty() : tensor<4xi32>
  %loaded = preprocessing.load_resource "some/path.bin" into %destination
      : (tensor<4xi32>) -> tensor<4xi32>
  %index = arith.constant 0 : index
  %updated = tensor.insert %value into %loaded[%index] : tensor<4xi32>
  return %updated : tensor<4xi32>
}
