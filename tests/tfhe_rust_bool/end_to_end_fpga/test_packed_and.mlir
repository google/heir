// This test ensures the testing harness is working properly with minimal codegen.

// RUN: heir-translate %s --emit-tfhe-rust-bool > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml -- 1 1 | FileCheck %s

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

// CHECK: 1
func.func @fn_under_test(%bsks : !bsks, %a: tensor<8x!eb>, %b: tensor<8x!eb>) -> tensor<8x!eb> {
  %res = tfhe_rust_bool.and_packed %bsks, %a, %b: (!bsks, tensor<8x!eb>, tensor<8x!eb>) -> tensor<8x!eb>
  return %res : tensor<8x!eb>
}
