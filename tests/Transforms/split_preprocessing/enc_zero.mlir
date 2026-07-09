// RUN: heir-opt --split-preprocessing %s | FileCheck %s

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 1023 }">
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
!pkey_L1 = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x1024>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

// CHECK-NOT: matvec__encrypt__zero__0__preprocessing
// CHECK-NOT: matvec__encrypt__zero__0__preprocessed

// CHECK: func.func @matvec__encrypt__zero__0
// CHECK-SAME: attributes {client.enc_zero_func}
func.func @matvec__encrypt__zero__0(%pk: !pkey_L1) -> !ct_L1 attributes {client.enc_zero_func} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf64>
  %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf64> -> !pt
  %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1) -> !ct_L1
  return %ct : !ct_L1
}
