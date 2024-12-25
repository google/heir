// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// A reduced dimension version of hello world to speed Yosys up.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {

  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x3xi32> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %4 = "tosa.const"() {value = dense<[0, 0, 5438]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = "tosa.const"() {value = dense<[[9], [54], [57]]> : tensor<3x1xi8>} : () -> tensor<3x1xi8>
    %6 = "tosa.fully_connected"(%arg0, %5, %4) {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>} : (tensor<1x1xi8>, tensor<3x1xi8>, tensor<3xi32>) -> tensor<1x3xi32>
    // CHECK: return
    return %6 : tensor<1x3xi32>
  }
}
