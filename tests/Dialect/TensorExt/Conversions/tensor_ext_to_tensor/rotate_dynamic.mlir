// RUN: heir-opt --tensor-ext-to-tensor %s | FileCheck %s

// CHECK: @test_rotate_dynamic
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[shift:.*]]: index) -> tensor<16xi32>
func.func @test_rotate_dynamic(%0: tensor<16xi32>, %shift: index) -> tensor<16xi32> {
  // CHECK: %[[c16:.*]] = arith.constant 16
  // CHECK: %[[v0:.*]] = arith.remsi %[[shift]], %[[c16]]
  // CHECK: %[[v1:.*]] = arith.addi %[[v0]], %[[c16]]
  // CHECK: %[[v2:.*]] = arith.remsi %[[v1]], %[[c16]]
  // CHECK: %[[v3:.*]] = arith.subi %[[c16]], %[[v2]]
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0] [%[[v2]]] [1]
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [%[[v2]]] [%[[v3]]] [1]

  // CHECK: tensor.empty

  // CHECK: tensor.insert_slice
  // CHECK-SAME: [%[[v3]]] [%[[v2]]] [1]
  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0] [%[[v3]]] [1]
  %1 = tensor_ext.rotate %0, %shift : tensor<16xi32>, index
  return %1 : tensor<16xi32>
}


// CHECK: @test_rotate_dynamic_multidim
func.func @test_rotate_dynamic_multidim(%0: tensor<3x4x16xi32>, %shift: index) -> tensor<3x4x16xi32> {
  // CHECK: %[[c16:.*]] = arith.constant 16
  // CHECK: %[[v0:.*]] = arith.remsi %[[shift]], %[[c16]]
  // CHECK: %[[v1:.*]] = arith.addi %[[v0]], %[[c16]]
  // CHECK: %[[v2:.*]] = arith.remsi %[[v1]], %[[c16]]
  // CHECK: %[[v3:.*]] = arith.subi %[[c16]], %[[v2]]

  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0, 0, 0] [3, 4, %[[v2]]] [1, 1, 1]
  // CHECK: tensor.extract_slice
  // CHECK-SAME: [0, 0, %[[v2]]] [3, 4, %[[v3]]] [1, 1, 1]

  // CHECK: tensor.empty

  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0, 0, %[[v3]]] [3, 4, %[[v2]]] [1, 1, 1]
  // CHECK: tensor.insert_slice
  // CHECK-SAME: [0, 0, 0] [3, 4, %[[v3]]] [1, 1, 1]
  %1 = tensor_ext.rotate %0, %shift : tensor<3x4x16xi32>, index
  return %1 : tensor<3x4x16xi32>
}

// Convert an i32 shift to an index
// CHECK: test_rotate_dynamic_cast
func.func @test_rotate_dynamic_cast(%0: tensor<16xi32>, %shift: i32) -> tensor<16xi32> {
  // CHECK: arith.index_cast
  %1 = tensor_ext.rotate %0, %shift : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}
