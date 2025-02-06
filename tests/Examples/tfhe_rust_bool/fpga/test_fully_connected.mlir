// RUN: heir-opt --tosa-to-boolean-fpga-tfhe="abc-fast=true entry-function=fn_under_test" %s | heir-translate --emit-tfhe-rust-bool-packed > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_fully_connected -- 2 | FileCheck %s

// This takes takes the input x and outputs a FC layer operation.
// CHECK: 00000111
module attributes {tf_saved_model.semantics} {
  func.func @fn_under_test(%11: tensor<1x3xi8>) -> tensor<1x3xi32> {
    %0 = "tosa.const"() {value = dense<[3, 1, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = "tosa.const"() {value = dense<[[2, 7, 1], [8,2,8], [1,8,2]]> : tensor<3x3xi8>} : () -> tensor<3x3xi8>
    %2 = "tosa.fully_connected"(%11, %1, %0) {input_zp = 0 : i32, weight_zp = 0 : i32} : (tensor<1x3xi8>, tensor<3x3xi8>, tensor<3xi32>) -> tensor<1x3xi32>
    return %2 : tensor<1x3xi32>
  }
}
