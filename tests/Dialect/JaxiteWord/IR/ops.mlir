// RUN: heir-opt %s | FileCheck %s

!Z1073741441_i64 = !mod_arith.int<1073741441 : i64>
!Z536870273_i64 = !mod_arith.int<536870273 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 29>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 58>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 4 = 0 and 0 <= i1 <= 3 and 0 <= slot <= 7 }">
#modulus_chain_L7_C0 = #lwe.modulus_chain<elements = <1073741441 : i64, 536870273 : i64, 536870401 : i64, 536870497 : i64, 536871233 : i64, 536870561 : i64, 536870657 : i64, 536870849 : i64>, current = 0>
#modulus_chain_L7_C1 = #lwe.modulus_chain<elements = <1073741441 : i64, 536870273 : i64, 536870401 : i64, 536870497 : i64, 536871233 : i64, 536870561 : i64, 536870657 : i64, 536870849 : i64>, current = 1>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L0 = !rns.rns<!Z1073741441_i64>
!rns_L1 = !rns.rns<!Z1073741441_i64, !Z536870273_i64>
#original_type = #tensor_ext.original_type<originalType = tensor<1x4xf32>, layout = #layout>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>>
#ring_rns_L0_1_x8 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8, encryption_type = mix>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8, encryption_type = mix, size = 3>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L7_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L7_C1>
!ct_L1_1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L7_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L7_C1>

module attributes {scheme.ckks} {
  // CHECK: func.func @simple_mul(
  func.func @simple_mul(%arg0: !jaxiteword.crypto_context<>, %arg1: !jaxiteword.eval_key<>, %arg2: tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}, %arg3: tensor<1x!ct_L1> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ct_L0> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg2[%c0] : tensor<1x!ct_L1>
    %extracted_0 = tensor.extract %arg3[%c0] : tensor<1x!ct_L1>
    // CHECK: jaxiteword.mul_no_relin
    %ct = jaxiteword.mul_no_relin %arg0, %extracted, %extracted_0 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1_D3
    // CHECK: jaxiteword.relin
    %ct_1 = jaxiteword.relin %arg0, %ct, %arg1 : (!jaxiteword.crypto_context<>, !ct_L1_D3, !jaxiteword.eval_key<>) -> !ct_L1_1
    %0 = tensor.empty() : tensor<1x!ct_L0>
    // CHECK: jaxiteword.mod_reduce
    %ct_2 = jaxiteword.mod_reduce %arg0, %ct_1 : (!jaxiteword.crypto_context<>, !ct_L1_1) -> !ct_L0
    %inserted = tensor.insert %ct_2 into %0[%c0] : tensor<1x!ct_L0>
    return %inserted : tensor<1x!ct_L0>
  }
}
