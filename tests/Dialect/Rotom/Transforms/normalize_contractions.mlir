// RUN: heir-opt %s --rotom-normalize-contractions | FileCheck %s

module {
  // A matvec against a bias accumulator becomes: vector and bias expanded to
  // Kx1 / Mx1, a zero-filled matmul, an explicit add of the bias, and a
  // collapse back to the vector shape.
  // CHECK: func.func @matvec_bias
  // CHECK-DAG: tensor.expand_shape
  // CHECK: linalg.fill
  // CHECK: %[[MM:.*]] = linalg.matmul
  // CHECK: arith.addf
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: linalg.matvec
  func.func @matvec_bias(%A: tensor<8x4xf32>, %x: tensor<4xf32>, %bias: tensor<8xf32>) -> tensor<8xf32> {
    %0 = linalg.matvec ins(%A, %x : tensor<8x4xf32>, tensor<4xf32>) outs(%bias : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

module {
  // A matmul already accumulating into a zero fill is left alone.
  // CHECK: func.func @zero_init_untouched
  // CHECK: linalg.fill
  // CHECK-NEXT: linalg.matmul
  // CHECK-NOT: arith.addf
  func.func @zero_init_untouched(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
