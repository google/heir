// RUN: heir-opt --openfhe-alloc-to-inplace %s | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

// CHECK: func.func @dot_product
func.func @dot_product(%cc: !cc, %ct: !ct, %ct_0: !ct) -> !ct {
  // No new allocation found as the two ciphertexts in function argument are
  // enough to store the imtermediate results. However, openfhe doesn't have an
  // in-place rotate, so those have new allocations.
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  // CHECK: mul_no_relin
  // CHECK: relin_inplace
  // CHECK: rot
  // CHECK: add_inplace
  // CHECK: rot
  // CHECK: add_inplace
  // CHECK: rot
  // CHECK: add_inplace
  // CHECK: mod_reduce_inplace
  // CHECK: make_packed_plaintext
  // CHECK: mul_plain
  // CHECK: rot
  // CHECK: mod_reduce_inplace
  // CHECK: return
  %ct_1 = openfhe.mul_no_relin %cc, %ct, %ct_0 : (!cc, !ct, !ct) -> !ct
  %ct_2 = openfhe.relin %cc, %ct_1 : (!cc, !ct) -> !ct
  %ct_3 = openfhe.rot %cc, %ct_2 {static_shift = 4 : index} : (!cc, !ct) -> !ct
  %ct_4 = openfhe.add %cc, %ct_2, %ct_3 : (!cc, !ct, !ct) -> !ct
  %ct_5 = openfhe.rot %cc, %ct_4 {static_shift = 2 : index} : (!cc, !ct) -> !ct
  %ct_6 = openfhe.add %cc, %ct_4, %ct_5 : (!cc, !ct, !ct) -> !ct
  %ct_7 = openfhe.rot %cc, %ct_6 {static_shift = 1 : index} : (!cc, !ct) -> !ct
  %ct_8 = openfhe.add %cc, %ct_6, %ct_7 : (!cc, !ct, !ct) -> !ct
  %ct_9 = openfhe.mod_reduce %cc, %ct_8 : (!cc, !ct) -> !ct
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
  %pt_10 = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<8xi16>) -> !pt
  %ct_11 = openfhe.mul_plain %cc, %ct_9, %pt_10 : (!cc, !ct, !pt) -> !ct
  %ct_12 = openfhe.rot %cc, %ct_11 {static_shift = 7 : index} : (!cc, !ct) -> !ct
  %ct_13 = openfhe.mod_reduce %cc, %ct_12 : (!cc, !ct) -> !ct
  return %ct_13 : !ct
}
