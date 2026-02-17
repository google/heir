// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// alloc-to-place should work with multiple functions

!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.bgv.encoder
!evaluator = !lattigo.bgv.evaluator
!param = !lattigo.bgv.parameter
!pt = !lattigo.rlwe.plaintext

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [33832961, 34062337, 281474977349633], P = [281474977595393, 281474978185217], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: func.func @dot_product
  func.func @dot_product(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %ct_0: !ct) -> !ct attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
    // no new allocation found as the two ciphertexts in function argument are enough to store the imtermediate results
    // CHECK-NOT: _new
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %ct_1 = lattigo.bgv.mul_new %evaluator, %ct, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    %ct_2 = lattigo.bgv.relinearize_new %evaluator, %ct_1 : (!evaluator, !ct) -> !ct
    %ct_3 = lattigo.bgv.rotate_columns_new %evaluator, %ct_2 {static_shift = 4 : index} : (!evaluator, !ct) -> !ct
    %ct_4 = lattigo.bgv.add_new %evaluator, %ct_2, %ct_3 : (!evaluator, !ct, !ct) -> !ct
    %ct_5 = lattigo.bgv.rotate_columns_new %evaluator, %ct_4 {static_shift = 2 : index} : (!evaluator, !ct) -> !ct
    %ct_6 = lattigo.bgv.add_new %evaluator, %ct_4, %ct_5 : (!evaluator, !ct, !ct) -> !ct
    %ct_7 = lattigo.bgv.rotate_columns_new %evaluator, %ct_6 {static_shift = 1 : index} : (!evaluator, !ct) -> !ct
    %ct_8 = lattigo.bgv.add_new %evaluator, %ct_6, %ct_7 : (!evaluator, !ct, !ct) -> !ct
    %ct_9 = lattigo.bgv.rescale_new %evaluator, %ct_8 : (!evaluator, !ct) -> !ct
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
    %pt = lattigo.bgv.new_plaintext %param : (!param) -> !pt
    %pt_10 = lattigo.bgv.encode %encoder, %cst, %pt : (!encoder, tensor<8xi16>, !pt) -> !pt
    %ct_11 = lattigo.bgv.mul_new %evaluator, %ct_9, %pt_10 : (!evaluator, !ct, !pt) -> !ct
    %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %ct_11 {static_shift = 7 : index} : (!evaluator, !ct) -> !ct
    %ct_13 = lattigo.bgv.rescale_new %evaluator, %ct_12 : (!evaluator, !ct) -> !ct
    return %ct_13 : !ct
  }
  // CHECK: func.func @dot_product23
  func.func @dot_product23(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %ct_0: !ct) -> !ct attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
    // no new allocation found as the two ciphertexts in function argument are enough to store the imtermediate results
    // CHECK-NOT: _new
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %ct_1 = lattigo.bgv.mul_new %evaluator, %ct, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    %ct_2 = lattigo.bgv.relinearize_new %evaluator, %ct_1 : (!evaluator, !ct) -> !ct
    %ct_3 = lattigo.bgv.rotate_columns_new %evaluator, %ct_2 {static_shift = 4 : index} : (!evaluator, !ct) -> !ct
    %ct_4 = lattigo.bgv.add_new %evaluator, %ct_2, %ct_3 : (!evaluator, !ct, !ct) -> !ct
    %ct_5 = lattigo.bgv.rotate_columns_new %evaluator, %ct_4 {static_shift = 2 : index} : (!evaluator, !ct) -> !ct
    %ct_6 = lattigo.bgv.add_new %evaluator, %ct_4, %ct_5 : (!evaluator, !ct, !ct) -> !ct
    %ct_7 = lattigo.bgv.rotate_columns_new %evaluator, %ct_6 {static_shift = 1 : index} : (!evaluator, !ct) -> !ct
    %ct_8 = lattigo.bgv.add_new %evaluator, %ct_6, %ct_7 : (!evaluator, !ct, !ct) -> !ct
    %ct_9 = lattigo.bgv.rescale_new %evaluator, %ct_8 : (!evaluator, !ct) -> !ct
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
    %pt = lattigo.bgv.new_plaintext %param : (!param) -> !pt
    %pt_10 = lattigo.bgv.encode %encoder, %cst, %pt : (!encoder, tensor<8xi16>, !pt) -> !pt
    %ct_11 = lattigo.bgv.mul_new %evaluator, %ct_9, %pt_10 : (!evaluator, !ct, !pt) -> !ct
    %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %ct_11 {static_shift = 7 : index} : (!evaluator, !ct) -> !ct
    %ct_13 = lattigo.bgv.rescale_new %evaluator, %ct_12 : (!evaluator, !ct) -> !ct
    return %ct_13 : !ct
  }
}
