// RUN: heir-opt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>

!rns_Q = !rns.rns<!Zq0, !Zq1>
!rns_QP = !rns.rns<!Zq0, !Zq1, !Zp0>

#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#space_QP = #lwe.ciphertext_space<ring = #ring_QP, encryption_type = ckks>

!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #space_QP, key = #key>

// CHECK: @test_keygen
func.func @test_keygen() -> (!sk, !pk, !pk, tensor<2x!ksk_ct>) {
  // CHECK: lwe.gen_secret_key : () -> ![[SK:.*]]
  %sk = lwe.gen_secret_key : () -> !sk
  // CHECK: lwe.gen_public_key %{{.*}} : (![[SK]]) -> ![[PK:.*]]
  %pk = lwe.gen_public_key %sk : (!sk) -> !pk
  // CHECK: lwe.gen_public_key %{{.*}} <plaintext_modulus = 65537> : (![[SK]]) -> ![[PK]]
  %pk_bgv = lwe.gen_public_key %sk <plaintext_modulus = 65537> : (!sk) -> !pk
  // CHECK: lwe.gen_relin_key %{{.*}} : (![[SK]]) -> tensor<2x![[KSK:.*]]>
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<2x!ksk_ct>
  return %sk, %pk, %pk_bgv, %rlk : !sk, !pk, !pk, tensor<2x!ksk_ct>
}
