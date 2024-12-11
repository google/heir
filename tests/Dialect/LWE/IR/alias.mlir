// RUN: heir-opt %s | FileCheck %s

#encoding = #lwe.bit_field_encoding<
  cleartext_start=14,
  cleartext_bitwidth=3>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<7917:i32>, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=10, ring=#ring>
#rlwe_params1 = #lwe.rlwe_params<dimension=3, ring=#ring>
// CHECK: [[TY:!rlwe_ct_D10_[0-9]*]]
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
// CHECK: [[TY1:!rlwe_ct_D3_[0-9]*]]
!ciphertext_rlwe1 = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params1, underlying_type=i4>

// CHECK: @test_alias(%[[ARG0:.*]]: [[TY]], %[[ARG1:.*]]: [[TY1]]) -> [[TY]]
func.func @test_alias(%0 : !ciphertext_rlwe, %1 : !ciphertext_rlwe1) -> !ciphertext_rlwe {
    return %0 : !ciphertext_rlwe
}
