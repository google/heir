// RUN: heir-opt --canonicalize %s | FileCheck %s

#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#poly_1024 = #polynomial.int_polynomial<1 + x**1024>
!plaintext_coefficient_modulus = !mod_arith.int<65537 : i64>
#plaintext_ring = #polynomial.ring<
  coefficientType=!plaintext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#plaintext_space = #lwe.plaintext_space<
  ring=#plaintext_ring,
  encoding=#encoding>
!ciphertext_coefficient_modulus = !mod_arith.int<1032955396097 : i64>
#ciphertext_ring = #polynomial.ring<
  coefficientType=!ciphertext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#ciphertext_space = #lwe.ciphertext_space<
  ring=#ciphertext_ring,
  encryption_type=lsb,
  size=2>

!ct = !lwe.lwe_ciphertext<
  application_data=<message_type=i3>,
  plaintext_space=#plaintext_space,
  ciphertext_space=#ciphertext_space,
  key=#lwe.key<>>
!pt = !lwe.lwe_plaintext<
  application_data=<message_type=i3>,
  plaintext_space=#plaintext_space>

// CHECK: @test_ct_pt_canonicalize(
// CHECK-SAME: [[X:%.[^:]+]]:
// CHECK-SAME: , [[Y:%.[^:]+]]:
func.func @test_ct_pt_canonicalize(%x: !ct, %y: !pt) -> (!ct, !ct) {
    // CHECK: lwe.radd_plain [[X]], [[Y]]
    %add_ct_rhs = lwe.radd_plain %y, %x : (!pt, !ct) -> !ct
    // CHECK: lwe.rmul_plain [[X]], [[Y]]
    %mul_ct_rhs = lwe.rmul_plain %y, %x : (!pt, !ct) -> !ct
    return %add_ct_rhs, %mul_ct_rhs : !ct, !ct
}
