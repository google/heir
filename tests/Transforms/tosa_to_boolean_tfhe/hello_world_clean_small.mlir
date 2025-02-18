// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// A reduced dimension version of hello world to speed Yosys up.

// CHECK-LABEL: module
module {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x1xi32> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<[[9, 54, 57]]> : tensor<1x3xi8>
    %cst_0 = arith.constant dense<[[0, 0, 5438]]> : tensor<1x3xi32>
    %cst_1 = arith.constant dense<[[729, 1954, 610]]> : tensor<3xi32>
    %cst_2 = arith.constant dense<429> : tensor<1x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<[[12, 9, 12], [26, 25, 36], [19, 33, 32]]> : tensor<3x3xi8>
    %cst_4 = arith.constant dense<[[39], [59], [39]]> : tensor<3x1xi8>
    %2 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x3xi8>, i32, i32) outs(%cst_0 : tensor<1x3xi32>) -> tensor<1x3xi32>
    %4 = linalg.quantized_matmul ins(%2, %cst_3, %c0_i32, %c0_i32 : tensor<1x3xi32>, tensor<3x3xi8>, i32, i32) outs(%cst_1 : tensor<1x3xi32>) -> tensor<1x3xi32>
    %7 = linalg.quantized_matmul ins(%4, %cst_4, %c0_i32, %c0_i32 : tensor<1x3xi32>, tensor<3x1xi8>, i32, i32) outs(%cst_2 : tensor<1x1xi32>) -> tensor<1x1xi32>
    // CHECK: return
    return %7 : tensor<1x1xi32>
  }
}
