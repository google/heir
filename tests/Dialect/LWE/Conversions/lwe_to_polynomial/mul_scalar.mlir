// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z998595133441_i64 = !mod_arith.int<998595133441 : i64>

!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
!rns_L2 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64, !Z998595133441_i64>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_rns_L1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_rns_L2_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_x1024, encryption_type = lsb>

!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_rns_L2_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key>

// CHECK: func.func @test_mul_scalar
// CHECK-SAME: tensor<2x!poly
// CHECK: polynomial.sub
// CHECK: polynomial.mul_scalar
// CHECK-NOT: lwe.mul_scalar
func.func @test_mul_scalar(%lhs: !ct_L1, %rhs: !ct_L1, %scalar: !rns_L1) -> !ct_L1 {
  %diff = lwe.rsub %lhs, %rhs : (!ct_L1, !ct_L1) -> !ct_L1
  %scaled = lwe.mul_scalar %diff, %scalar : (!ct_L1, !rns_L1) -> !ct_L1
  return %scaled : !ct_L1
}
