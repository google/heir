// RUN: heir-translate %s --emit-tfhe-rust-bool > %S/src/fn_under_test.rs
// RUN: cargo run --release --manifest-path %S/Cargo.toml -- 1 1 | FileCheck %s

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

// CHECK-LABEL: pub fn fn_under_test(
// CHECK-NEXT:   [[bsks:v[0-9]+]]: &ServerKey,
// CHECK-NEXT:   [[input1:v[0-9]+]]: &Vec<Ciphertext>,
// CHECK-NEXT:   [[input2:v[0-9]+]]: &Vec<Ciphertext>,
// CHECK-NEXT: ) -> Vec<Ciphertext> {
func.func @fn_under_test(%bsks : !bsks,  %arg0: tensor<8x!eb>, %arg1: tensor<8x!eb>) -> tensor<8x!eb> {
  %c7 = arith.constant 7 : index
  %c6 = arith.constant 6 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted_00 = tensor.extract_slice %arg0 [0][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_01 = tensor.extract_slice %arg0 [1][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_02 = tensor.extract_slice %arg0 [2][1][1]: tensor<8x!eb> to tensor<1x!eb>
  %extracted_03 = tensor.extract_slice %arg0 [3][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_04 = tensor.extract_slice %arg0 [4][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_05 = tensor.extract_slice %arg0 [5][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_06 = tensor.extract_slice %arg0 [6][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_07 = tensor.extract_slice %arg0 [7][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_10 = tensor.extract_slice %arg1 [0][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_11 = tensor.extract_slice %arg1 [1][1][1]: tensor<8x!eb> to tensor<1x!eb>
  %extracted_12 = tensor.extract_slice %arg1 [2][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_13 = tensor.extract_slice %arg1 [3][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_14 = tensor.extract_slice %arg1 [4][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_15 = tensor.extract_slice %arg1 [5][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_16 = tensor.extract_slice %arg1 [6][1][1] : tensor<8x!eb> to tensor<1x!eb>
  %extracted_17 = tensor.extract_slice %arg1 [7][1][1]: tensor<8x!eb> to tensor<1x!eb>
  %ha_s = tfhe_rust_bool.xor_packed %bsks, %extracted_00, %extracted_10 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %ha_c = tfhe_rust_bool.and_packed %bsks, %extracted_00, %extracted_10: (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa0_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_01, %extracted_11 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa0_2 = tfhe_rust_bool.and_packed %bsks, %extracted_01, %extracted_11 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa0_3 = tfhe_rust_bool.and_packed %bsks, %fa0_1, %ha_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa0_s = tfhe_rust_bool.xor_packed %bsks, %fa0_1, %ha_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa0_c = tfhe_rust_bool.xor_packed %bsks, %fa0_2, %fa0_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa1_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_02, %extracted_12 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa1_2 = tfhe_rust_bool.and_packed %bsks, %extracted_02, %extracted_12 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa1_3 = tfhe_rust_bool.and_packed %bsks, %fa1_1, %fa0_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa1_s = tfhe_rust_bool.xor_packed %bsks, %fa1_1, %fa0_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa1_c = tfhe_rust_bool.xor_packed %bsks, %fa1_2, %fa1_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa2_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_03, %extracted_13 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa2_2 = tfhe_rust_bool.and_packed %bsks, %extracted_03, %extracted_13 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa2_3 = tfhe_rust_bool.and_packed %bsks, %fa2_1, %fa1_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa2_s = tfhe_rust_bool.xor_packed %bsks, %fa2_1, %fa1_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa2_c = tfhe_rust_bool.xor_packed %bsks, %fa2_2, %fa2_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa3_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_04, %extracted_14 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa3_2 = tfhe_rust_bool.and_packed %bsks, %extracted_04, %extracted_14 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa3_3 = tfhe_rust_bool.and_packed %bsks, %fa3_1, %fa2_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa3_s = tfhe_rust_bool.xor_packed %bsks, %fa3_1, %fa2_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa3_c = tfhe_rust_bool.xor_packed %bsks, %fa3_2, %fa3_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa4_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_05, %extracted_15 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa4_2 = tfhe_rust_bool.and_packed %bsks, %extracted_05, %extracted_15 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa4_3 = tfhe_rust_bool.and_packed %bsks, %fa4_1, %fa3_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa4_s = tfhe_rust_bool.xor_packed %bsks, %fa4_1, %fa3_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa4_c = tfhe_rust_bool.xor_packed %bsks, %fa4_2, %fa4_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa5_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_06, %extracted_16 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa5_2 = tfhe_rust_bool.and_packed %bsks, %extracted_06, %extracted_16 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa5_3 = tfhe_rust_bool.and_packed %bsks, %fa5_1, %fa4_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa5_s = tfhe_rust_bool.xor_packed %bsks, %fa5_1, %fa4_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa5_c = tfhe_rust_bool.xor_packed %bsks, %fa5_2, %fa5_3 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa6_1 = tfhe_rust_bool.xor_packed %bsks, %extracted_07, %extracted_17 : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %fa6_s = tfhe_rust_bool.xor_packed %bsks, %fa6_1, %fa5_c : (!bsks, tensor<1x!eb>, tensor<1x!eb>) -> tensor<1x!eb>
  %from_elements = tensor.concat dim(0) %fa6_s, %fa5_s, %fa4_s, %fa3_s, %fa2_s, %fa1_s, %fa0_s, %ha_s
    : (tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>, tensor<1x!eb>) -> tensor<8x!eb>
  return %from_elements : tensor<8x!eb>
}
