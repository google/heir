// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#plaintext_space_f16 = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

!pt = !lwe.lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>
!ptf16 = !lwe.lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space_f16>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L0_D3 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!pk = !openfhe.public_key
!sk = !openfhe.private_key
!ek = !openfhe.eval_key
!cc = !openfhe.crypto_context
!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_D3 = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module {
  // CHECK: func @test_make_packed_plaintext
  func.func @test_make_packed_plaintext(%cc: !cc, %arg0 : tensor<32xi3>) -> !pt {
    %pt = openfhe.make_packed_plaintext %cc, %arg0 : (!cc, tensor<32xi3>) -> !pt
    return %pt : !pt
  }

  // CHECK: func @test_make_ckks_packed_plaintext
  func.func @test_make_ckks_packed_plaintext(%cc: !cc, %arg0 : tensor<32xf16>) -> !ptf16 {
    %pt = openfhe.make_ckks_packed_plaintext %cc, %arg0 : (!cc, tensor<32xf16>) -> !ptf16
    return %pt : !ptf16
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

  // CHECK: func @test_encode
  func.func @test_encode(%arg0: tensor<32xi3>, %pt : !pt, %pk: !pk) {
    %0 = arith.extsi %arg0 : tensor<32xi3> to tensor<32xi64>
    %out = lwe.rlwe_encode %0 {encoding=#full_crt_packing_encoding, ring=#ring_Z65537_i64_1_x1024_} : tensor<32xi64> -> !pt
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
  // CHECK: func @test_inplace_add
  func.func @test_inplace_add(%cc: !cc, %pt : !pt, %pk : !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    openfhe.add_inplace %cc, %c1, %c2: (!cc, !ct, !ct) -> ()
    return
  }
  // CHECK: func @test_inplace_sub
  func.func @test_inplace_sub(%cc: !cc, %pt : !pt, %pk : !pk) {
    %c1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %c2 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    openfhe.sub_inplace %cc, %c1, %c2: (!cc, !ct, !ct) -> ()
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
    %out = openfhe.mul_no_relin %cc, %c1, %c2: (!cc, !ct, !ct) -> !ct_D3
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
