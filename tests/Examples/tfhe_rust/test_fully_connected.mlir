// RUN: heir-opt --tosa-to-boolean-tfhe="abc-fast=true entry-function=fn_under_test" %s | heir-translate --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_fully_connected -- 2 --message_bits=3 | FileCheck %s

// This takes takes the input x and outputs 2 \cdot x + 1.
// CHECK: 00000101
module attributes {tf_saved_model.semantics} {
  func.func @fn_under_test(%11: tensor<1x1xi8>) -> tensor<1x1xi32> {
    %0 = "tosa.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tosa.const"() {value = dense<[[2]]> : tensor<1x1xi8>} : () -> tensor<1x1xi8>
    %2 = "tosa.fully_connected"(%11, %1, %0) {input_zp = 0 : i32, weight_zp = 0 : i32} : (tensor<1x1xi8>, tensor<1x1xi8>, tensor<1xi32>) -> tensor<1x1xi32>
    return %2 : tensor<1x1xi32>
  }
}
