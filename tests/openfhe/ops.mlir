// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#my_poly = #polynomial.int_polynomial<1 + x**16384>
#ring= #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=1, ring=#ring>

!pk = !openfhe.public_key
!ek = !openfhe.eval_key
!cc = !openfhe.crypto_context
!pt = !lwe.rlwe_plaintext<encoding = #encoding, ring=#ring, underlying_type=i3>
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type=i3>

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

  // CHECK-LABEL: func @test_mul_plain
  func.func @test_mul_plain(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 5 : i64
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul_plain %cc, %c1, %pt: (!cc, !ct, !pt) -> !ct
    return
  }

  // CHECK-LABEL: func @test_mul_no_relin
  func.func @test_mul_no_relin(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul_no_relin %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_square
  func.func @test_square(%cc : !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.square %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_rot
  func.func @test_rot(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 2 : i64
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.rot %cc, %ct, %0: (!cc, !ct, i64) -> !ct
    return
  }

  // CHECK-LABEL: func @test_automorph
  func.func @test_automorph(%cc : !cc, %pt : !pt, %ek: !ek, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.automorph %cc, %ct, %ek : (!cc, !ct, !ek) -> !ct
    return
  }

  // CHECK-LABEL: func @test_key_switch
  func.func @test_key_switch(%cc : !cc, %pt : !pt, %pk: !pk, %ek : !ek) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.key_switch %cc, %ct, %ek: (!cc, !ct, !ek) -> !ct
    return
  }

  // CHECK-LABEL: func @test_relin
  func.func @test_relin(%cc : !cc, %pt : !pt, %pk1 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk1 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.relin %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_mod_reduce
  func.func @test_mod_reduce(%cc : !cc, %pt : !pt, %pk2 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk2 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mod_reduce %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_level_reduce
  func.func @test_level_reduce(%cc : !cc, %pt : !pt, %pk3 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk3 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.level_reduce %cc, %ct: (!cc, !ct) -> !ct
    return
  }
}
