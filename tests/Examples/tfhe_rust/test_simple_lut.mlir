// This test ensures the testing harness is working properly with minimal codegen.

// RUN: heir-translate %s --emit-tfhe-rust > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main -- 1 0 --message_bits=3 | FileCheck %s

!sks = !tfhe_rust.server_key
!lut = !tfhe_rust.lookup_table
!eui3 = !tfhe_rust.eui3

// We're computing, effectively (0b00000111 >> (1 << 1)) & 1, i.e., 0b111 >> 2
// CHECK: 1
func.func @fn_under_test(%sks : !sks, %a: !eui3, %b: !eui3) -> !eui3 {
  %lut = tfhe_rust.generate_lookup_table %sks {truthTable = 7 : ui8} : (!sks) -> !lut
  %c1 = arith.constant 1 : i8
  %0 = tfhe_rust.scalar_left_shift %sks, %a, %c1 : (!sks, !eui3, i8) -> !eui3
  %1 = tfhe_rust.add %sks, %0, %b : (!sks, !eui3, !eui3) -> !eui3
  %2 = tfhe_rust.apply_lookup_table %sks, %1, %lut : (!sks, !eui3, !lut) -> !eui3
  return %2 : !eui3
}
