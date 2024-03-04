// RUN: heir-opt %s

// Test for syntax

func.func @test_rotate(%0: tensor<16xi32>) -> tensor<16xi32> {
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}
