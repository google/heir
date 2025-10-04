// RUN: heir-opt --tensor-ext-to-tensor %s | FileCheck %s

// CHECK: @test_rotate
func.func @test_rotate(%0: tensor<16xi32>) -> tensor<16xi32> {
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0] [1] [1]
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [1] [15] [1]

  // CHECK: tensor.empty

  // CHECK: tensor.insert_slice
  // CHECK-SAME: [15] [1] [1]
  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0] [15] [1]
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}


// CHECK: @test_rotate_multidim
func.func @test_rotate_multidim(%0: tensor<3x4x16xi32>) -> tensor<3x4x16xi32> {
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0, 0, 0] [3, 4, 3] [1, 1, 1]
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0, 0, 3] [3, 4, 13] [1, 1, 1]

  // CHECK: tensor.empty

  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0, 0, 13] [3, 4, 3] [1, 1, 1]
  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0, 0, 0] [3, 4, 13] [1, 1, 1]
  %c3 = arith.constant 3 : i32
  %1 = tensor_ext.rotate %0, %c3 : tensor<3x4x16xi32>, i32
  return %1 : tensor<3x4x16xi32>
}
