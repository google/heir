// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext

module attributes {scheme.bgv} {
  // CHECK: EvalMultNoRelin
  // CHECK: RelinearizeInPlace
  // CHECK: EvalRotate
  // CHECK: EvalAddInPlace
  // CHECK: EvalRotate
  // CHECK: EvalAddInPlace
  // CHECK: EvalRotate
  // CHECK: EvalAddInPlace
  // CHECK: ModReduceInPlace
  // CHECK: MakePackedPlaintext
  // CHECK: EvalRotate
  // CHECK: ModReduceInPlace
  // CHECK: return
  func.func @dot_product(%cc: !cc, %ct: !ct, %ct_0: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %ct_1 = openfhe.mul_no_relin %cc, %ct, %ct_0 : (!cc, !ct, !ct) -> !ct
    %ct_2 = openfhe.relin_inplace %cc, %ct_1 : (!cc, !ct) -> !ct
    %ct_3 = openfhe.rot %cc, %ct_2 {static_shift = 4 : index} : (!cc, !ct) -> !ct
    %ct_4 = openfhe.add_inplace %cc, %ct_2, %ct_3 : (!cc, !ct, !ct) -> !ct
    %ct_5 = openfhe.rot %cc, %ct_4 {static_shift = 2 : index} : (!cc, !ct) -> !ct
    %ct_6 = openfhe.add_inplace %cc, %ct_4, %ct_5 : (!cc, !ct, !ct) -> !ct
    %ct_7 = openfhe.rot %cc, %ct_6 {static_shift = 1 : index} : (!cc, !ct) -> !ct
    %ct_8 = openfhe.add_inplace %cc, %ct_6, %ct_7 : (!cc, !ct, !ct) -> !ct
    %ct_9 = openfhe.mod_reduce_inplace %cc, %ct_8 : (!cc, !ct) -> !ct
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 1]> : tensor<8xi16>
    %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<8xi16>) -> !pt
    %ct_10 = openfhe.mul_plain %cc, %ct_9, %pt : (!cc, !ct, !pt) -> !ct
    %ct_11 = openfhe.rot %cc, %ct_10 {static_shift = 7 : index} : (!cc, !ct) -> !ct
    %ct_12 = openfhe.mod_reduce_inplace %cc, %ct_11 : (!cc, !ct) -> !ct
    return %ct_12 : !ct
  }
}
