// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!pk = !openfhe.public_key
!sk = !openfhe.private_key
!ek = !openfhe.eval_key
!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

module {
  // CHECK: func @test_make_packed_plaintext
  func.func @test_make_packed_plaintext(%cc: !cc, %arg0 : tensor<32xi3>) -> !pt {
    %pt = openfhe.make_packed_plaintext %cc, %arg0 : (!cc, tensor<32xi3>) -> !pt
    return %pt : !pt
  }

  // CHECK: func @test_make_ckks_packed_plaintext
  func.func @test_make_ckks_packed_plaintext(%cc: !cc, %arg0 : tensor<32xf16>) -> !pt {
    %pt = openfhe.make_ckks_packed_plaintext %cc, %arg0 : (!cc, tensor<32xf16>) -> !pt
    return %pt : !pt
  }

  // CHECK: func @test_encrypt
  func.func @test_encrypt(%cc: !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    return
  }

  // CHECK: func @test_encrypt_sk
  func.func @test_encrypt_sk(%cc: !cc, %pt : !pt, %sk: !sk) {
    %ct = openfhe.encrypt %cc, %pt, %sk : (!cc, !pt, !sk) -> !ct
    return
  }

  // CHECK: func @test_negate
  func.func @test_negate(%cc : !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.negate %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_add
  func.func @test_add(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.add %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_sub
  func.func @test_sub(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.sub %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_sub_plain
  func.func @test_sub_plain(%cc : !cc, %pt : !pt, %ct: !ct) {
    %out = openfhe.sub_plain %cc, %ct, %pt: (!cc, !ct, !pt) -> !ct
    return
  }

  // CHECK: func @test_mul
  func.func @test_mul(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_mul_plain
  func.func @test_mul_plain(%cc : !cc, %pt : !pt, %pk: !pk) {
    %0 = arith.constant 5 : i64
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul_plain %cc, %c1, %pt: (!cc, !ct, !pt) -> !ct
    return
  }

  // CHECK: func @test_mul_no_relin
  func.func @test_mul_no_relin(%cc : !cc, %pt : !pt, %pk: !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mul_no_relin %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_square
  func.func @test_square(%cc : !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.square %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_rot
  func.func @test_rot(%cc : !cc, %pt : !pt, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.rot %cc, %ct { index = 2 }: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_automorph
  func.func @test_automorph(%cc : !cc, %pt : !pt, %ek: !ek, %pk: !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.automorph %cc, %ct, %ek : (!cc, !ct, !ek) -> !ct
    return
  }

  // CHECK: func @test_key_switch
  func.func @test_key_switch(%cc : !cc, %pt : !pt, %pk: !pk, %ek : !ek) {
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %out = openfhe.key_switch %cc, %ct, %ek: (!cc, !ct, !ek) -> !ct
    return
  }

  // CHECK: func @test_relin
  func.func @test_relin(%cc : !cc, %pt : !pt, %pk1 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk1 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.relin %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_mod_reduce
  func.func @test_mod_reduce(%cc : !cc, %pt : !pt, %pk2 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk2 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.mod_reduce %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_level_reduce
  func.func @test_level_reduce(%cc : !cc, %pt : !pt, %pk3 : !pk) {
    %ct = openfhe.encrypt %cc, %pt, %pk3 : (!cc, !pt, !pk) -> !ct
    %out = openfhe.level_reduce %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_bootstrap
  func.func @test_bootstrap(%cc : !cc, %ct : !ct) {
    %out = openfhe.bootstrap %cc, %ct: (!cc, !ct) -> !ct
    return
  }

  // CHECK: func @test_gen_bootstrap_key
  func.func @test_gen_bootstrap_key(%cc : !cc, %sk : !sk) {
    openfhe.gen_bootstrapkey %cc, %sk: (!cc, !sk) -> ()
    return
  }

  // CHECK: func @test_setup_bootstrap
  func.func @test_setup_bootstrap(%cc : !cc) {
    openfhe.setup_bootstrap %cc {levelBudgetEncode = 3, levelBudgetDecode = 3}: (!cc) -> ()
    return
  }
}
