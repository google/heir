// heir-opt --mlir-to-cggi=abc-fast=true --scheme-to-fpt %s | heir-translate --emit-tfhe-rust-bool-packed > %S/src/fn_under_test.rs
// cargo run --release --manifest-path %S/Cargo.toml --bin main_fully_connected -- 2 | FileCheck %s

// This takes takes the input x and outputs a FC layer operation.
module {
  func.func @fn_under_test(%arg0: tensor<1x3xi8> {secret.secret}) -> tensor<1x3xi32> {
    %cst = arith.constant dense<[[2, 8, 1], [7, 2, 8], [1, 8, 2]]> : tensor<3x3xi8>
    %cst_0 = arith.constant dense<[[3, 1, 4]]> : tensor<1x3xi32>
    %c0_i32 = arith.constant 0 : i32
    %2 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i32, %c0_i32 : tensor<1x3xi8>, tensor<3x3xi8>, i32, i32) outs(%cst_0 : tensor<1x3xi32>) -> tensor<1x3xi32>
    return %2 : tensor<1x3xi32>
  }
}
