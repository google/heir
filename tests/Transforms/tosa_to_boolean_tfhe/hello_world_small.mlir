// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// A reduced dimension version of hello world to speed Yosys up.

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {

  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x1xi8> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() {value = dense<429> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tosa.const"() {value = dense<[[-39, 59, 39]]> : tensor<1x3xi8>} : () -> tensor<1x3xi8>
    %2 = "tosa.const"() {value = dense<[-729, 1954, 610]> : tensor<3xi32>} : () -> tensor<3xi32>
    %3 = "tosa.const"() {value = dense<"0xF41AED091921F424E0"> : tensor<3x3xi8>} : () -> tensor<3x3xi8>
    %4 = "tosa.const"() {value = dense<[0, 0, -5438]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = "tosa.const"() {value = dense<[[-9], [-54], [57]]> : tensor<3x1xi8>} : () -> tensor<3x1xi8>
    %6 = "tosa.fully_connected"(%arg0, %5, %4) {input_zp = -128 : i32, weight_zp = 0 : i32} : (tensor<1x1xi8>, tensor<3x1xi8>, tensor<3xi32>) -> tensor<1x3xi32>
    %7 = "tosa.rescale"(%6) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 2039655736>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 38>} : (tensor<1x3xi32>) -> tensor<1x3xi8>
    %8 = "tosa.clamp"(%7) {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x3xi8>) -> tensor<1x3xi8>
    %9 = "tosa.fully_connected"(%8, %3, %2) {input_zp = -128 : i32, weight_zp = 0 : i32} : (tensor<1x3xi8>, tensor<3x3xi8>, tensor<3xi32>) -> tensor<1x3xi32>
    %10 = "tosa.rescale"(%9) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1561796795>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 37>} : (tensor<1x3xi32>) -> tensor<1x3xi8>
    %11 = "tosa.clamp"(%10) {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x3xi8>) -> tensor<1x3xi8>
    %12 = "tosa.fully_connected"(%11, %1, %0) {input_zp = -128 : i32, weight_zp = 0 : i32} : (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1xi32>) -> tensor<1x1xi32>
    %13 = "tosa.rescale"(%12) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1630361836>, output_zp = 5 : i32, per_channel = false, scale32 = true, shift = array<i8: 36>} : (tensor<1x1xi32>) -> tensor<1x1xi8>
    // CHECK: return
    return %13 : tensor<1x1xi8>
  }
}
