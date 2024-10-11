// RUN: heir-opt %s | FileCheck %s

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod=7917, dimension=10>
!ct = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

// CHECK-LABEL: test_add
func.func @test_add(%0: !ct, %1: !ct) -> !ct {
  // CHECK: lwe.add
  %2 = lwe.add %0, %1 : !ct
  return %2 : !ct
}

// CHECK-LABEL: test_mul_scalar
func.func @test_mul_scalar(%0: !ct, %1: i3) -> !ct {
  // CHECK: lwe.mul_scalar
  %2 = lwe.mul_scalar %0, %1 : (!ct, i3) -> !ct
  return %2 : !ct
}
