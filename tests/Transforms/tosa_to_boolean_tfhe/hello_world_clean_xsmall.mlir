// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// A further reduced dimension version of hello world to speed Yosys up.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x3xi32>) {
    %4 = "tosa.const"() {value = dense<[1, 2, 5438]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = "tosa.const"() {value = dense<[[9], [54], [57]]> : tensor<3x1xi8>} : () -> tensor<3x1xi8>
    %6 = "tosa.fully_connected"(%arg0, %5, %4) {input_zp = 0 : i32, weight_zp = 0 : i32} : (tensor<1x1xi8>, tensor<3x1xi8>, tensor<3xi32>) -> tensor<1x3xi32>
    // CHECK: return
    return %6 : tensor<1x3xi32>
  }
}
