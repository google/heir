// RUN: heir-opt %s | FileCheck %s

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!cmod = !mod_arith.int<7917 : i32>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = !cmod, polynomialModulus = #poly>,
  encryption_type = msb, size = 10>
!ct = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: test_add
func.func @test_add(%0: !ct, %1: !ct) -> !ct {
  // CHECK: lwe.add
  %2 = lwe.add %0, %1 : !ct
  return %2 : !ct
}

// CHECK: test_mul_scalar
func.func @test_mul_scalar(%0: !ct, %1: i3) -> !ct {
  // CHECK: lwe.mul_scalar
  %2 = lwe.mul_scalar %0, %1 : (!ct, i3) -> !ct
  return %2 : !ct
}
