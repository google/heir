// RUN: heir-opt --tensor-ext-to-tensor %s | FileCheck %s

// CHECK: @test_rotate
// CHECK-NOT: tensor_ext.rotate
func.func @test_rotate(%0: tensor<16xi32>) -> tensor<16xi32> {
  %c5 = arith.constant 5 : i32
  %1 = tensor_ext.rotate %0, %c5 : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}

// CHECK: @test_rotate_dynamic_multidim
// CHECK-NOT: tensor_ext.rotate
func.func @test_rotate_dynamic_multidim(%0: tensor<3x4x16xi32>, %shift: index) -> tensor<3x4x16xi32> {
  %1 = tensor_ext.rotate %0, %shift : tensor<3x4x16xi32>, index
  return %1 : tensor<3x4x16xi32>
}
