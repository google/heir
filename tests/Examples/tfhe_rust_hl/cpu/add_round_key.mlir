// RUN: heir-opt --mlir-to-cggi=data-type=Integer --scheme-to-tfhe-rs %s | heir-translate --emit-tfhe-rust-hl > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml --bin main_add_round_key

// This function adds (xors) the two blocks
// CHECK: Success
#map = affine_map<(d0) -> (d0)>
func.func @add_round_key(%arg0: tensor<16xi8> {secret.secret}, %arg1: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
  %8 = tensor.empty() : tensor<16xi8>
  %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1: tensor<16xi8>, tensor<16xi8>) outs(%8 : tensor<16xi8>) {
  ^bb0(%in1: i8, %in2: i8, %out: i8):
    %res = arith.xori %in1, %in2 : i8
    linalg.yield %res : i8
  } -> tensor<16xi8>
  return %9 : tensor<16xi8>
}
