// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=2, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
!rlwe_key = !lwe.rlwe_secret_key<rlwe_params=#rlwe_params>

// CHECK-LABEL: test_encrypt
// CHECK-SAME:  !polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus = <1 + x**1024>>>
// CHECK-SAME:  tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus = <1 + x**1024>>>>
func.func @test_encrypt(%arg0: !plaintext_rlwe, %arg1: !ciphertext_rlwe) {
  return
}
