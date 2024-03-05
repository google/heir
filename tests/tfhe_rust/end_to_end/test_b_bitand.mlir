// This test ensures the testing harness is working properly with minimal codegen.

// RUN: heir-translate %s --emit-tfhe-rust > %S/src/boolean/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_bool -- 1 0 | FileCheck %s

!bsks = !tfhe_rust.bool_server_key
!eb = !tfhe_rust.eb

// CHECK: 0
func.func @fn_under_test(%bsks : !bsks, %a: !eb, %b: !eb) -> !eb {
  %res = tfhe_rust.bitand %bsks, %a, %b: (!bsks, !eb, !eb) -> !eb
  return %res : !eb
}
