// RUN: heir-opt %s | FileCheck %s

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=10, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
!rlwe_key = !lwe.rlwe_secret_key<rlwe_params=#rlwe_params>

func.func @test_encrypt(%arg0: tensor<32xi3>, %arg1: !rlwe_key) -> !ciphertext_rlwe {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding, ring = #ring} : tensor<32xi3> -> !plaintext_rlwe
  // CHECK: lwe.rlwe_encrypt
  %1 = lwe.rlwe_encrypt %0, %arg1 : (!plaintext_rlwe, !rlwe_key) -> !ciphertext_rlwe
  return %1 : !ciphertext_rlwe
}
