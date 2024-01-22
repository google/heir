// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#params = #lwe.rlwe_params<cmod=7917, dimension=1, polyDegree=16384>
!cc = !openfhe.crypto_context
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params>

// CHECK-LABEL: CiphertextT test_binops(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG1:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG2:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      auto [[v3:.*]] = [[CC]]->EvalAdd([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      auto [[v4:.*]] = [[CC]]->EvalSub([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      auto [[v5:.*]] = [[CC]]->EvalMul([[v3]], [[v4]]);
// CHECK-NEXT:      return [[v5]];
// CHECK-NEXT:  }
func.func @test_binops(%cc : !cc, %input1 : !ct, %input2 : !ct) -> !ct {
  %add_res = openfhe.add %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
  %sub_res = openfhe.sub %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
  %mul_res = openfhe.mul%cc, %add_res, %sub_res : (!cc, !ct, !ct) -> !ct
  return %mul_res : !ct
}
