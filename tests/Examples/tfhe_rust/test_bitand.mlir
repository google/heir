// This test ensures the testing harness is working properly with minimal codegen.

// RUN: heir-translate %s --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main -- 2 3 --message_bits=3 | FileCheck %s

!sks = !tfhe_rust.server_key
!eui3 = !tfhe_rust.eui3

// CHECK: 0
func.func @fn_under_test(%sks : !sks, %a: !eui3, %b: !eui3) -> !eui3 {
  %res = tfhe_rust.bitand %sks, %a, %b: (!sks, !eui3, !eui3) -> !eui3
  return %res : !eui3
}
