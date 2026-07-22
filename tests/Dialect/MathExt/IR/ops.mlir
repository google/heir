// RUN: heir-opt %s | FileCheck %s

// CHECK: test_sign
func.func @test_sign(%arg0: i32, %arg1: f32, %arg2: tensor<4xi32>) -> (i32, f32, tensor<4xi32>) {
  // CHECK: math_ext.sign
  %0 = math_ext.sign %arg0 : i32
  // CHECK: math_ext.sign
  %1 = math_ext.sign %arg1 : f32
  // CHECK: math_ext.sign
  %2 = math_ext.sign %arg2 : tensor<4xi32>
  return %0, %1, %2 : i32, f32, tensor<4xi32>
}

// CHECK: test_sigmoid
func.func @test_sigmoid(%arg0: f32, %arg1: tensor<4xf32>) -> (f32, tensor<4xf32>) {
  // CHECK: math_ext.sigmoid
  %0 = math_ext.sigmoid %arg0 : f32
  // CHECK: math_ext.sigmoid
  %1 = math_ext.sigmoid %arg1 : tensor<4xf32>
  return %0, %1 : f32, tensor<4xf32>
}

// CHECK: test_softmax
func.func @test_softmax(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: math_ext.softmax
  %0 = math_ext.softmax %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: test_softmax_2d
func.func @test_softmax_2d(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: math_ext.softmax
  %0 = math_ext.softmax %arg0 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
