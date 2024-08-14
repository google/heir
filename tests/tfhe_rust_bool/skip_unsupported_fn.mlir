// RUN: heir-translate %s --emit-tfhe-rust-bool 2>&1 | FileCheck %s

!bsks = !tfhe_rust_bool.server_key
!eb = !tfhe_rust_bool.eb

module {
  // CHECK-NOT: func @_Z3fooi
  // CHECK: Skipping function _Z3fooi which cannot be emitted because it has an unsupported
  func.func @_Z3fooi(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }

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
    %extracted_00 = tensor.extract %arg0[%c0] : tensor<8x!eb>
    %extracted_01 = tensor.extract %arg0[%c1] : tensor<8x!eb>
    %extracted_02 = tensor.extract %arg0[%c2] : tensor<8x!eb>
    %extracted_03 = tensor.extract %arg0[%c3] : tensor<8x!eb>
    %extracted_04 = tensor.extract %arg0[%c4] : tensor<8x!eb>
    %extracted_05 = tensor.extract %arg0[%c5] : tensor<8x!eb>
    %extracted_06 = tensor.extract %arg0[%c6] : tensor<8x!eb>
    %extracted_07 = tensor.extract %arg0[%c7] : tensor<8x!eb>
    %extracted_10 = tensor.extract %arg1[%c0] : tensor<8x!eb>
    %extracted_11 = tensor.extract %arg1[%c1] : tensor<8x!eb>
    %extracted_12 = tensor.extract %arg1[%c2] : tensor<8x!eb>
    %extracted_13 = tensor.extract %arg1[%c3] : tensor<8x!eb>
    %extracted_14 = tensor.extract %arg1[%c4] : tensor<8x!eb>
    %extracted_15 = tensor.extract %arg1[%c5] : tensor<8x!eb>
    %extracted_16 = tensor.extract %arg1[%c6] : tensor<8x!eb>
    %extracted_17 = tensor.extract %arg1[%c7] : tensor<8x!eb>
    %ha_s = tfhe_rust_bool.xor %bsks, %extracted_00, %extracted_10 : (!bsks, !eb, !eb) -> !eb
    %ha_c = tfhe_rust_bool.and %bsks, %extracted_00, %extracted_10 : (!bsks, !eb, !eb) -> !eb
    %fa0_1 = tfhe_rust_bool.xor %bsks, %extracted_01, %extracted_11 : (!bsks, !eb, !eb) -> !eb
    %fa0_2 = tfhe_rust_bool.and %bsks, %extracted_01, %extracted_11 : (!bsks, !eb, !eb) -> !eb
    %fa0_3 = tfhe_rust_bool.and %bsks, %fa0_1, %ha_c : (!bsks, !eb, !eb) -> !eb
    %fa0_s = tfhe_rust_bool.xor %bsks, %fa0_1, %ha_c : (!bsks, !eb, !eb) -> !eb
    %fa0_c = tfhe_rust_bool.xor %bsks, %fa0_2, %fa0_3 : (!bsks, !eb, !eb) -> !eb
    %fa1_1 = tfhe_rust_bool.xor %bsks, %extracted_02, %extracted_12 : (!bsks, !eb, !eb) -> !eb
    %fa1_2 = tfhe_rust_bool.and %bsks, %extracted_02, %extracted_12 : (!bsks, !eb, !eb) -> !eb
    %fa1_3 = tfhe_rust_bool.and %bsks, %fa1_1, %fa0_c : (!bsks, !eb, !eb) -> !eb
    %fa1_s = tfhe_rust_bool.xor %bsks, %fa1_1, %fa0_c : (!bsks, !eb, !eb) -> !eb
    %fa1_c = tfhe_rust_bool.xor %bsks, %fa1_2, %fa1_3 : (!bsks, !eb, !eb) -> !eb
    %fa2_1 = tfhe_rust_bool.xor %bsks, %extracted_03, %extracted_13 : (!bsks, !eb, !eb) -> !eb
    %fa2_2 = tfhe_rust_bool.and %bsks, %extracted_03, %extracted_13 : (!bsks, !eb, !eb) -> !eb
    %fa2_3 = tfhe_rust_bool.and %bsks, %fa2_1, %fa1_c : (!bsks, !eb, !eb) -> !eb
    %fa2_s = tfhe_rust_bool.xor %bsks, %fa2_1, %fa1_c : (!bsks, !eb, !eb) -> !eb
    %fa2_c = tfhe_rust_bool.xor %bsks, %fa2_2, %fa2_3 : (!bsks, !eb, !eb) -> !eb
    %fa3_1 = tfhe_rust_bool.xor %bsks, %extracted_04, %extracted_14 : (!bsks, !eb, !eb) -> !eb
    %fa3_2 = tfhe_rust_bool.and %bsks, %extracted_04, %extracted_14 : (!bsks, !eb, !eb) -> !eb
    %fa3_3 = tfhe_rust_bool.and %bsks, %fa3_1, %fa2_c : (!bsks, !eb, !eb) -> !eb
    %fa3_s = tfhe_rust_bool.xor %bsks, %fa3_1, %fa2_c : (!bsks, !eb, !eb) -> !eb
    %fa3_c = tfhe_rust_bool.xor %bsks, %fa3_2, %fa3_3 : (!bsks, !eb, !eb) -> !eb
    %fa4_1 = tfhe_rust_bool.xor %bsks, %extracted_05, %extracted_15 : (!bsks, !eb, !eb) -> !eb
    %fa4_2 = tfhe_rust_bool.and %bsks, %extracted_05, %extracted_15 : (!bsks, !eb, !eb) -> !eb
    %fa4_3 = tfhe_rust_bool.and %bsks, %fa4_1, %fa3_c : (!bsks, !eb, !eb) -> !eb
    %fa4_s = tfhe_rust_bool.xor %bsks, %fa4_1, %fa3_c : (!bsks, !eb, !eb) -> !eb
    %fa4_c = tfhe_rust_bool.xor %bsks, %fa4_2, %fa4_3 : (!bsks, !eb, !eb) -> !eb
    %fa5_1 = tfhe_rust_bool.xor %bsks, %extracted_06, %extracted_16 : (!bsks, !eb, !eb) -> !eb
    %fa5_2 = tfhe_rust_bool.and %bsks, %extracted_06, %extracted_16 : (!bsks, !eb, !eb) -> !eb
    %fa5_3 = tfhe_rust_bool.and %bsks, %fa5_1, %fa4_c : (!bsks, !eb, !eb) -> !eb
    %fa5_s = tfhe_rust_bool.xor %bsks, %fa5_1, %fa4_c : (!bsks, !eb, !eb) -> !eb
    %fa5_c = tfhe_rust_bool.xor %bsks, %fa5_2, %fa5_3 : (!bsks, !eb, !eb) -> !eb
    %fa6_1 = tfhe_rust_bool.xor %bsks, %extracted_07, %extracted_17 : (!bsks, !eb, !eb) -> !eb
    %fa6_s = tfhe_rust_bool.xor %bsks, %fa6_1, %fa5_c : (!bsks, !eb, !eb) -> !eb
    %from_elements = tensor.from_elements %fa6_s, %fa5_s, %fa4_s, %fa3_s, %fa2_s, %fa1_s, %fa0_s, %ha_s : tensor<8x!eb>
    return %from_elements : tensor<8x!eb>
  }
}
