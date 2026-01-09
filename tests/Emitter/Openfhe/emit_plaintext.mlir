// Regression test for https://github.com/google/heir/issues/1621
// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!params = !openfhe.cc_params
!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!pk = !openfhe.public_key
!sk = !openfhe.private_key
!ct = !openfhe.ciphertext

module attributes {scheme.bgv} {
  // CHECK: CiphertextT cond
  // CHECK-SAME: CryptoContextT [[cc:.*]], int64_t [[v0:.*]], int64_t [[v1:.*]], CiphertextT [[ct:.*]]
  func.func @cond(%cc: !cc, %arg0: i64, %arg1: i64, %ct: !ct) -> !ct {
    // CHECK: std::vector<int64_t> [[v2:.*]](1024, 1);
    // CHECK-NEXT: auto [[pt:.*]]_filled_n = [[cc]]->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    // CHECK-NEXT: auto [[pt]]_filled = [[v2]]
    // CHECK: auto [[pt]] = [[cc]]->MakePackedPlaintext
    %cst = arith.constant dense<1> : tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<1024xi64>) -> !pt
    %ct_0 = openfhe.negate %cc, %ct : (!cc, !ct) -> !ct
    %ct_1 = openfhe.add_plain %cc, %ct_0, %pt : (!cc, !ct, !pt) -> !ct
    %splat = tensor.splat %arg0 : tensor<1024xi64>
    // CHECK: [[cc]]->MakePackedPlaintext
    %pt_3 = openfhe.make_packed_plaintext %cc, %splat : (!cc, tensor<1024xi64>) -> !pt
    %ct_4 = openfhe.mul_plain %cc, %ct_1, %pt_3 : (!cc, !ct, !pt) -> !ct
    %splat_6 = tensor.splat %arg1 : tensor<1024xi64>
    // CHECK: [[cc]]->MakePackedPlaintext
    %pt_7 = openfhe.make_packed_plaintext %cc, %splat_6 : (!cc, tensor<1024xi64>) -> !pt
    %ct_8 = openfhe.mul_plain %cc, %ct_4, %pt_7 : (!cc, !ct, !pt) -> !ct
    %ct_9 = openfhe.add %cc, %ct_4, %ct_8 : (!cc, !ct, !ct) -> !ct
    // CHECK-NOT: [[pt]]_filled_n
    %pt_10 = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<1024xi64>) -> !pt
    %ct_11 = openfhe.add_plain %cc, %ct_9, %pt_10 : (!cc, !ct, !pt) -> !ct
    %ct_12 = openfhe.mod_reduce %cc, %ct_11 : (!cc, !ct) -> !ct
    // CHECK: return
    return %ct_12 : !ct
  }
  func.func @cond__encrypt__arg2(%cc: !cc, %arg0: i1, %pk: !pk) -> !ct {
    %splat = tensor.splat %arg0 : tensor<1024xi1>
    %0 = arith.extui %splat : tensor<1024xi1> to tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %0 : (!cc, tensor<1024xi64>) -> !pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    return %ct : !ct
  }
  func.func @cond__decrypt__result0(%cc: !cc, %ct: !ct, %sk: !sk) -> i64 {
    %pt = openfhe.decrypt %cc, %ct, %sk : (!cc, !ct, !sk) -> !pt
    %0 = openfhe.decode %pt : !pt -> i64
    return %0 : i64
  }
  func.func @cond__generate_crypto_context() -> !cc {
    %params = openfhe.gen_params  {encryptionTechniqueExtended = false, evalAddCount = 2 : i64, insecure = false, keySwitchCount = 0 : i64, mulDepth = 1 : i64, plainMod = 65537 : i64} : () -> !params
    %cc = openfhe.gen_context %params {supportFHE = false} : (!params) -> !cc
    return %cc : !cc
  }
  func.func @cond__configure_crypto_context(%cc: !cc, %sk: !sk) -> !cc {
    return %cc : !cc
  }
}
