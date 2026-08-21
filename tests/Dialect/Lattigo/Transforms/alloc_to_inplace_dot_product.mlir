// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// CHECK: func.func @dot_product
func.func @dot_product(%evaluator: !lattigo.bgv.evaluator, %param: !lattigo.bgv.parameter, %encoder: !lattigo.bgv.encoder, %ct: !lattigo.rlwe.ciphertext, %ct_0: !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
  // The two ciphertext arguments are enough to store every intermediate
  // result, so nothing but the rescales allocates. A rescale's result sits
  // one level below every buffer that is live where it runs, and reuse
  // requires a buffer at the result's own level, so each rescale keeps its
  // fresh allocation.
  // CHECK: lattigo.bgv.mul_new
  // CHECK-NOT: relinearize_new
  // CHECK: lattigo.bgv.rotate_columns_new
  // CHECK-NOT: add_new
  // CHECK: lattigo.bgv.rescale_new
  // CHECK-NOT: mul_new
  // CHECK-NOT: rotate_columns_new
  // CHECK: lattigo.bgv.rescale_new
  // CHECK: return
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %ct_1 = lattigo.bgv.mul_new %evaluator, %ct, %ct_0 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_2 = lattigo.bgv.relinearize_new %evaluator, %ct_1 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_3 = lattigo.bgv.rotate_columns_new %evaluator, %ct_2 {static_shift = 4 : index} : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_4 = lattigo.bgv.add_new %evaluator, %ct_2, %ct_3 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_5 = lattigo.bgv.rotate_columns_new %evaluator, %ct_4 {static_shift = 2 : index} : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_6 = lattigo.bgv.add_new %evaluator, %ct_4, %ct_5 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_7 = lattigo.bgv.rotate_columns_new %evaluator, %ct_6 {static_shift = 1 : index} : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_8 = lattigo.bgv.add_new %evaluator, %ct_6, %ct_7 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_9 = lattigo.bgv.rescale_new %evaluator, %ct_8 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
  %pt = lattigo.bgv.new_plaintext %param : (!lattigo.bgv.parameter) -> !lattigo.rlwe.plaintext
  %pt_10 = lattigo.bgv.encode %encoder, %cst, %pt : (!lattigo.bgv.encoder, tensor<8xi16>, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.plaintext
  %ct_11 = lattigo.bgv.mul_new %evaluator, %ct_9, %pt_10 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.ciphertext
  %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %ct_11 {static_shift = 7 : index} : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  %ct_13 = lattigo.bgv.rescale_new %evaluator, %ct_12 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
  return %ct_13 : !lattigo.rlwe.ciphertext
}
