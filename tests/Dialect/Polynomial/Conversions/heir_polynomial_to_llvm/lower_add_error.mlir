// RUN: heir-opt --lwe-to-polynomial --polynomial-to-mod-arith --verify-diagnostics --split-input-file %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring1 = #polynomial.ring<coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params1 = #lwe.rlwe_params<dimension=3, ring=#ring1>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>

module {
  func.func @simple_sum(%arg0: !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus = <1 + x**1024>>>, underlying_type = tensor<1024xi16>>, %arg1: !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus = <1 + x**1024>>>, underlying_type = tensor<1024xi16>>) -> !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus = <1 + x**1024>>>, underlying_type = tensor<1024xi16>> {
    // expected-error@below {{Can't directly lower for a tensor of polynomials.}}
    // expected-error@below {{failed to legalize operation 'polynomial.add' that was explicitly marked illegal}}
    %0 = lwe.radd  %arg0, %arg1 : !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus = <1 + x**1024>>>, underlying_type = tensor<1024xi16>>
    return %0 : !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus = <1 + x**1024>>>, underlying_type = tensor<1024xi16>>
  }
}
