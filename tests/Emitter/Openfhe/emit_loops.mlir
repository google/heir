// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {scheme.ckks} {
  // CHECK: test_affine_for
  // CHECK-SAME: CryptoContextT [[cc:.*]], CiphertextT [[ct:.*]]) {
  // CHECK: MutableCiphertextT [[ct1:.*]] = [[ct]]->Clone();
  // CHECK: for (auto [[v0:.*]] = 1; [[v0]] < 2; ++[[v0]]) {
  // CHECK:   [[ct1]] = [[cc]]->EvalRotate([[ct1]], 1);
  // CHECK: }
  // CHECK: return [[ct1]];
  func.func @test_affine_for(%cc: !openfhe.crypto_context, %ct: !ct) -> !ct {
    %1 = affine.for %arg0 = 1 to 2 iter_args(%arg1 = %ct) -> (!ct) {
      %ct_12 = openfhe.rot %cc, %arg1 {index = 1 : index} : (!openfhe.crypto_context, !ct) -> !ct
      affine.yield %ct_12 : !ct
    }
    return %1 : !ct
  }
}
