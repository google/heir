// heir-translate %s --emit-tfhe-rust-bool-packed > %S/src/fn_under_test.rs
// cargo run --release --manifest-path %S/Cargo.toml -- 1 1 | FileCheck %s

!bsks = !tfhe_rust_bool.server_key_enum
!eb = !tfhe_rust_bool.eb

func.func @fn_under_test(%bsks : !bsks, %a: tensor<8x!eb>, %b: tensor<8x!eb>) -> tensor<8x!eb> {
  %res = tfhe_rust_bool.and %bsks, %a, %b: (!bsks, tensor<8x!eb>, tensor<8x!eb>) -> tensor<8x!eb>
  return %res : tensor<8x!eb>
}
