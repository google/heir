// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: module
module {
  // CHECK: @main([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: memref<1x1x8x!tfhe_rust.eui3>) -> memref<1x1x32x!tfhe_rust.eui3>
  func.func @main(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x1xi32> {
    %cst = arith.constant dense<1> : tensor<1x1xi8>
    %cst_0 = arith.constant dense<1> : tensor<1x1xi32>
    %c-128_i32 = arith.constant -128 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[ALLOC:%.*]] = memref.alloc()
    // CHECK-SAME: memref<1x1x32x!tfhe_rust.eui3>
    // CHECK-NOT: comb
    // CHECK-NOT: memref.subview
    // CHECK-NOT: affine.load
    // CEHCK-NOT: affine.store
    // CHECK: return [[ALLOC]] : memref<1x1x32x!tfhe_rust.eui3>
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c-128_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x1xi8>, i32, i32) outs(%cst_0 : tensor<1x1xi32>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
}
