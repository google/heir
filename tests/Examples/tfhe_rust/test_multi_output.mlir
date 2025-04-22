// RUN: heir-opt --mlir-to-cggi --scheme-to-tfhe-rs %s | heir-translate --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_multi_output -- 2 | FileCheck %s

// First output is 2 + 3 = 5, second output is 2 * 2 = 4
// CHECK: 00000101 00000100
module {
  func.func @multi_output(%arg0: i8 {secret.secret}) -> (i8, i8) {
    %c2_i8 = arith.constant 2 : i8
    %c3_i8 = arith.constant 3 : i8
    %0 = arith.addi %arg0, %c3_i8 : i8
    %1 = arith.muli %arg0, %c2_i8 : i8
    return %0, %1 : i8, i8
  }
}
