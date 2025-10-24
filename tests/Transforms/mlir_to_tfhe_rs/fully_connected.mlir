// RUN: heir-opt --mlir-to-cggi=abc-fast=true --scheme-to-tfhe-rs %s | FileCheck %s

#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module
module {
  // CHECK: @main([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: tensor<1x1x8x!tfhe_rust.eui3>) -> tensor<1x1x32x!tfhe_rust.eui3>
  func.func @main(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x1xi32> {
    %cst = arith.constant dense<1> : tensor<1x1xi8>
    %cst_0 = arith.constant dense<1> : tensor<1x1xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK-NOT: comb
    // CHECK-NOT: tensor.insert
    // CHECK: [[ALLOC:%.*]] = tensor.from_elements
    // CHECK-SAME: tensor<1x1x32x!tfhe_rust.eui3>
    // CHECK: return [[ALLOC]] : tensor<1x1x32x!tfhe_rust.eui3>
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c-128_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x1xi8>, i32, i32) outs(%cst_0 : tensor<1x1xi32>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
}
