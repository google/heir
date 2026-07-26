// RUN: heir-translate %s --emit-openfhe-emitc | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {scheme.ckks} {
  // CHECK: CiphertextT test_add_sub_rot(CryptoContextT [[v1:[a-zA-Z0-9_]+]], CiphertextT [[v2:[a-zA-Z0-9_]+]], CiphertextT [[v3:[a-zA-Z0-9_]+]]) {
  // CHECK-NEXT: CiphertextT [[v4:[a-zA-Z0-9_]+]] = [[v1]].EvalAdd([[v2]], [[v3]]);
  // CHECK-NEXT: CiphertextT [[v5:[a-zA-Z0-9_]+]] = [[v1]].EvalSub([[v4]], [[v2]]);
  // CHECK-NEXT: int64_t [[v6:[a-zA-Z0-9_]+]] = 2;
  // CHECK-NEXT: CiphertextT [[v7:[a-zA-Z0-9_]+]] = [[v1]].EvalRotate([[v5]], [[v6]]);
  // CHECK-NEXT: return [[v7]];
  // CHECK-NEXT: }
  func.func @test_add_sub_rot(%cc : !cc, %c1 : !ct, %c2 : !ct) -> !ct {
    %add = openfhe.add %cc, %c1, %c2 : (!cc, !ct, !ct) -> !ct
    %sub = openfhe.sub %cc, %add, %c1 : (!cc, !ct, !ct) -> !ct
    %rot = openfhe.rot %cc, %sub {static_shift = 2 : index} : (!cc, !ct) -> !ct
    return %rot : !ct
  }
}
