// RUN: heir-opt --split-input-file --polynomial-approximation %s | FileCheck %s

// CHECK: @test_fpowi_four
func.func @test_fpowi_four(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-COUNT-2: arith.mulf
  // CHECK: return
  %c2 = arith.constant dense<4> : tensor<4xi64>
  %0 = math.fpowi %x, %c2 : tensor<4xf32>, tensor<4xi64>
  return %0 : tensor<4xf32>
}

// -----

// CHECK: @test_fpowi_zero
func.func @test_fpowi_zero(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.constant dense<1
  %c0 = arith.constant dense<0> : tensor<4xi64>
  %0 = math.fpowi %x, %c0 : tensor<4xf32>, tensor<4xi64>
  return %0 : tensor<4xf32>
}

// -----

// CHECK: @test_fpowi_one
// CHECK-SAME: %[[arg:.*]]: tensor<4xf32>
func.func @test_fpowi_one(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %[[arg]]
  %c1 = arith.constant dense<1> : tensor<4xi64>
  %0 = math.fpowi %x, %c1 : tensor<4xf32>, tensor<4xi64>
  return %0 : tensor<4xf32>
}

// -----

// CHECK: @test_fpowi_five
// CHECK-SAME: %[[arg:.*]]: tensor<4xf32>
func.func @test_fpowi_five(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[arg1:.*]] = arith.mulf %[[arg]], %[[arg]]
  // CHECK: %[[arg2:.*]] = arith.mulf %[[arg1]], %[[arg1]]
  // CHECK: %[[arg3:.*]] = arith.mulf %[[arg2]], %[[arg]]
  // CHECK: return %[[arg3]]
  %c1 = arith.constant dense<5> : tensor<4xi64>
  %0 = math.fpowi %x, %c1 : tensor<4xf32>, tensor<4xi64>
  return %0 : tensor<4xf32>
}
