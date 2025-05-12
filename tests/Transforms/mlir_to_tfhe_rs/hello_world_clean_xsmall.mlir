// RUN: heir-opt --mlir-to-cggi=abc-fast=true --scheme-to-tfhe-rs %s | FileCheck %s

// A further reduced dimension version of hello world to speed Yosys up.

// CHECK: module
module {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> tensor<1x3xi32> {
    %cst = arith.constant dense<[[0, 1, 0]]> : tensor<1x3xi8>
    %cst_0 = arith.constant dense<[[1, 2, 0]]> : tensor<1x3xi32>
    %c0_i32 = arith.constant 0 : i32
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x3xi8>, i32, i32) outs(%cst_0 : tensor<1x3xi32>) -> tensor<1x3xi32>
    // CHECK: return
    return %1 : tensor<1x3xi32>
  }
}
