// RUN: heir-opt %s | FileCheck %s

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 4 = 0 and 0 <= i1 <= 3 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!Z36028797018652673_i64>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<1x4xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
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
  // CHECK: func.func @simple_mul(
  func.func @simple_mul(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}, %arg3: tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L0> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %extracted_0 = tensor.extract %arg3[%c0] : tensor<1x!ct_L1>
    // CHECK: jaxiteword.mul
    %ct = jaxiteword.mul %arg0, %extracted, %extracted_0, %arg1 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1_D3
    // CHECK: jaxiteword.relin
    %ct_1 = jaxiteword.relin %arg0, %ct, %arg1 : (!jaxiteword.crypto_context<>, !ct_L1_D3, !jaxiteword.eval_key<>) -> !ct_L1_1
    %0 = tensor.empty() : tensor<1x!ct_L0>
    // CHECK: jaxiteword.mod_reduce
    %ct_2 = jaxiteword.mod_reduce %arg0, %ct_1 : (!jaxiteword.crypto_context<>, !ct_L1_1) -> !ct_L0
    %inserted = tensor.insert %ct_2 into %0[%c0] : tensor<1x!ct_L0>
    return %inserted : tensor<1x!ct_L0>
  }
}
