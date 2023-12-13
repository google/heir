// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#params = #lwe.rlwe_params<cmod=7917, dimension=1, polyDegree=16384>
!pk = !openfhe.public_key
!cc = !openfhe.crypto_context
!pt = !lwe.rlwe_plaintext<encoding = #encoding>
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params>

module {
  // CHECK-LABEL: func @test_encrypt
  func.func @test_encrypt(%cc: !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    return
  }

  // CHECK-LABEL: func @test_negate
  func.func @test_negate(%cc : !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.negate %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_add
  func.func @test_add(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.add %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_sub
  func.func @test_sub(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.sub %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_mul
  func.func @test_mul(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_mulplain
  func.func @test_mulplain(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 5 : i64
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mulplain %cc, %c1, %pt: (!cc, !ct, !pt) -> !ct
    return
  }

  // CHECK-LABEL: func @test_mulconst
  func.func @test_mulconst(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 5 : i64
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mulconst %cc, %c1, %0: (!cc, !ct, i64) -> !ct
    return
  }

  // CHECK-LABEL: func @test_rot
  func.func @test_rot(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 2 : i32
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.rot %cc, %ct, %0: (!cc, !ct, i32) -> !ct
    return
  }
}
