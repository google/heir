// RUN: heir-opt %s | FileCheck %s

// This tests for syntax of in-place ops.

!ek = !openfhe.eval_key
!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

module {
  // CHECK: func @test_negate_inplace
  func.func @test_negate_inplace(%cc : !cc, %ct : !ct) {
    openfhe.negate_inplace %cc, %ct : (!cc, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_add_inplace
  func.func @test_add_inplace(%cc : !cc, %c1 : !ct, %c2: !ct) {
    openfhe.add_inplace %cc, %c1, %c2 : (!cc, !ct, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_sub_inplace
  func.func @test_sub_inplace(%cc : !cc, %c1 : !ct, %c2: !ct) {
    openfhe.sub_inplace %cc, %c1, %c2 : (!cc, !ct, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_add_plain_inplace
  func.func @test_add_plain_inplace(%cc : !cc, %c1 : !ct, %c2: !pt) {
    openfhe.add_plain_inplace %cc, %c1, %c2: (!cc, !ct, !pt) -> (!ct)
    openfhe.add_plain_inplace %cc, %c2, %c1: (!cc, !pt, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_sub_plain_inplace
  func.func @test_sub_plain_inplace(%cc : !cc, %c1 : !ct, %c2: !pt) {
    openfhe.sub_plain_inplace %cc, %c1, %c2: (!cc, !ct, !pt) -> (!ct)
    openfhe.sub_plain_inplace %cc, %c2, %c1: (!cc, !pt, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_square_inplace
  func.func @test_square_inplace(%cc : !cc, %ct : !ct) {
    openfhe.square_inplace %cc, %ct: (!cc, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_key_switch_inplace
  func.func @test_key_switch_inplace(%cc : !cc, %ct : !ct, %ek : !ek) {
    openfhe.key_switch_inplace %cc, %ct, %ek: (!cc, !ct, !ek) -> (!ct)
    return
  }

  // CHECK: func @test_relin_inplace
  func.func @test_relin_inplace(%cc : !cc, %ct : !ct) {
    openfhe.relin_inplace %cc, %ct: (!cc, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_mod_reduce_inplace
  func.func @test_mod_reduce_inplace(%cc : !cc, %ct : !ct) {
    openfhe.mod_reduce_inplace %cc, %ct: (!cc, !ct) -> (!ct)
    return
  }

  // CHECK: func @test_level_reduce_inplace
  func.func @test_level_reduce_inplace(%cc : !cc, %ct : !ct) {
    openfhe.level_reduce_inplace %cc, %ct: (!cc, !ct) -> (!ct)
    return
  }
}
