// RUN: heir-opt --lwe-to-cheddar %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
!rns_L0 = !rns.rns<!Z36028797018652673_i64>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb, size = 3>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = lsb>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>

#inverse_canonical_encoding_lo = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding_hi = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#plaintext_space_lo = #lwe.plaintext_space<ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding_lo>
#plaintext_space_hi = #lwe.plaintext_space<ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding_hi>
!ct_L1_exact = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_lo, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_exact_1 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_hi, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L0_exact_1 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_hi, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>

// Verify the pass threads cheddar context, encoder, and UI args.
// CHECK-DAG: ![[CTX_T:.*]] = !cheddar.context
// CHECK-DAG: ![[ENC_T:.*]] = !cheddar.encoder
// CHECK-DAG: ![[UI_T:.*]] = !cheddar.user_interface
// CHECK-DAG: ![[CT_T:.*]] = !cheddar.ciphertext
// CHECK: func.func @test_add(%[[CTX:.*]]: ![[CTX_T]], %[[ENC:.*]]: ![[ENC_T]], %[[UI:.*]]: ![[UI_T]], %[[CT0:.*]]: ![[CT_T]], %[[CT1:.*]]: ![[CT_T]])
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  func.func @test_add(%ct0: !ct_L1, %ct1: !ct_L1) -> !ct_L1 {
    // CHECK: cheddar.add %[[CTX]], %[[CT0]], %[[CT1]]
    %result = lwe.radd %ct0, %ct1 : (!ct_L1, !ct_L1) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_sub
  func.func @test_sub(%ct0: !ct_L1, %ct1: !ct_L1) -> !ct_L1 {
    // CHECK: cheddar.sub
    %result = lwe.rsub %ct0, %ct1 : (!ct_L1, !ct_L1) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_negate
  func.func @test_negate(%ct: !ct_L1) -> !ct_L1 {
    // CHECK: cheddar.neg
    %result = ckks.negate %ct : !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_relin
  func.func @test_relin(%ct: !ct_L1_D3) -> !ct_L1 {
    // CHECK: cheddar.get_mult_key
    // CHECK: cheddar.relinearize
    %result = ckks.relinearize %ct {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_rotate
  func.func @test_rotate(%ct: !ct_L1) -> !ct_L1 {
    %c5 = arith.constant 5 : i32
    // CHECK: cheddar.get_rot_key
    // CHECK: cheddar.hrot
    %result = ckks.rotate %ct, %c5 : i32 : !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_add_plain
  func.func @test_add_plain(%ct: !ct_L1, %pt: !pt) -> !ct_L1 {
    // CHECK: cheddar.add_plain
    %result = lwe.radd_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_sub_plain_ct_first
  func.func @test_sub_plain_ct_first(%ct: !ct_L1, %pt: !pt) -> !ct_L1 {
    // CHECK: cheddar.sub_plain
    %result = lwe.rsub_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_sub_plain_pt_first
  func.func @test_sub_plain_pt_first(%pt: !pt, %ct: !ct_L1) -> !ct_L1 {
    // CHECK: cheddar.neg
    // CHECK: cheddar.add_plain
    %result = lwe.rsub_plain %pt, %ct : (!pt, !ct_L1) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_encode
  func.func @test_encode(%ct: !ct_L1) -> !ct_L1 {
    %cst = arith.constant dense<1.0> : tensor<1024xf64>
    // CHECK: cheddar.encode
    %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf64> -> !pt
    // CHECK: cheddar.add_plain
    %result = lwe.radd_plain %ct, %pt : (!ct_L1, !pt) -> !ct_L1
    return %result : !ct_L1
  }

  // CHECK: func.func @test_encrypt
  func.func @test_encrypt(%pt: !pt, %pk: !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>) -> !ct_L1 {
    // CHECK: cheddar.encrypt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>) -> !ct_L1
    return %ct : !ct_L1
  }

  // CHECK: func.func @test_decrypt
  func.func @test_decrypt(%ct: !ct_L1, %sk: !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>) -> !pt {
    // CHECK: cheddar.decrypt
    %pt = lwe.rlwe_decrypt %ct, %sk : (!ct_L1, !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x1024>) -> !pt
    return %pt : !pt
  }

  // CHECK: func.func @test_level_reduce_scaled
  // CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
  // CHECK: %[[RED:.*]] = cheddar.rescale %[[CTX]], %{{.*}}
  // CHECK: %[[CONST:.*]] = cheddar.encode_constant %[[ENC]], %[[ONE]] {level = 0 : i64, scale = 45 : i64}
  // CHECK: cheddar.mult_const %[[CTX]], %[[RED]], %[[CONST]]
  func.func @test_level_reduce_scaled(%ct: !ct_L1_exact_1) -> !ct_L0_exact_1 {
    %result = ckks.level_reduce %ct {levelToDrop = 1 : i64} : !ct_L1_exact_1 -> !ct_L0_exact_1
    return %result : !ct_L0_exact_1
  }

}
