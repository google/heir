// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!pk = !openfhe.public_key
!ct = !openfhe.ciphertext

module attributes {scheme.bgv} {
  // CHECK: CiphertextT emit_bool
  // CHECK-SAME: CryptoContextT [[cc:.*]], bool [[v0:.*]], PublicKeyT [[pk:.*]]) {
  func.func @emit_bool(%cc: !cc, %arg0: i1, %pk: !pk) -> !ct {
    // CHECK: std::vector<bool> [[v1:.*]](1024, [[v0]]);
    // CHECK-NEXT: std::vector<int64_t> [[v2:.*]](std::begin([[v1]]), std::end([[v1]]))
    %splat = tensor.splat %arg0 : tensor<1024xi1>
    %0 = arith.extui %splat : tensor<1024xi1> to tensor<1024xi64>
    %pt = openfhe.make_packed_plaintext %cc, %0 : (!cc, tensor<1024xi64>) -> !pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    return %ct : !ct
  }
}
