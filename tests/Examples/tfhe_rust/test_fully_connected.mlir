// RUN: heir-opt --tosa-to-boolean-tfhe="abc-fast=true entry-function=fn_under_test" %s | heir-translate --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_fully_connected -- 2 --message_bits=3 | FileCheck %s

// This takes takes the input x and outputs 2 \cdot x + 1.
// CHECK: 00000101
#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fn_under_test(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x1xi32> {
    %cst = arith.constant dense<2> : tensor<1x1xi8>
    %cst_0 = arith.constant dense<1> : tensor<1x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x1xi8>, i32, i32) outs(%cst_0 : tensor<1x1xi32>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
}
