// RUN: heir-opt --jaxiteword-configure-crypto-context=entry-function=simple_mul %s | FileCheck %s

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!Z36028797018652673_i64>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix, size = 3>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L1_C1>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  func.func @simple_mul(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: !ct_L1, %arg3: !ct_L1) -> !ct_L0 {
    %ct = jaxiteword.mul %arg0, %arg2, %arg3, %arg1 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1_D3
    %ct_1 = jaxiteword.relin %arg0, %ct, %arg1 : (!jaxiteword.crypto_context<>, !ct_L1_D3, !jaxiteword.eval_key<>) -> !ct_L1_1
    %ct_2 = jaxiteword.mod_reduce %arg0, %ct_1 : (!jaxiteword.crypto_context<>, !ct_L1_1) -> !ct_L0
    %ct_3 = jaxiteword.rot %arg0, %ct_2, %arg1 {index = 4 : i64} : (!jaxiteword.crypto_context<>, !ct_L0, !jaxiteword.eval_key<>) -> !ct_L0
    %ct_4 = jaxiteword.rot %arg0, %ct_3, %arg1 {index = 8 : i64} : (!jaxiteword.crypto_context<>, !ct_L0, !jaxiteword.eval_key<>) -> !ct_L0
    return %ct_4 : !ct_L0
  }
}

// CHECK: @simple_mul
// CHECK: @simple_mul__generate_crypto_context
// CHECK-SAME: !jaxiteword.public_key
// CHECK-SAME: !jaxiteword.private_key
// CHECK-SAME: !jaxiteword.eval_key
// CHECK: jaxiteword.gen_params
// CHECK-SAME: degree = 8192
// CHECK-SAME: numSlots = 4096

// CHECK: @simple_mul__configure_crypto_context
// CHECK-SAME: !jaxiteword.crypto_context
// CHECK-NOT: !jaxiteword.private_key
// CHECK: jaxiteword.program_initialization
// CHECK-SAME: totalRotationIndices = array<i64: 4, 8>
