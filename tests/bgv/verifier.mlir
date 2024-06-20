// RUN: heir-opt --verify-diagnostics --split-input-file %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#my_poly>

#params = #lwe.rlwe_params<dimension=3, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=2, ring=#ring>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>

func.func @test_input_dimension_error(%input: !ct) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = bgv.rotate  %input { offset = 4 }  : !ct
  return
}

// -----

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#my_poly>

#params = #lwe.rlwe_params<dimension=3, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=2, ring=#ring>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>


func.func @test_extract_verification_failure(%input: !ct) -> !ct {
  %offset = arith.constant 4 : index
  // expected-error@+1 {{'bgv.extract' op input RLWE ciphertext type must have a ranked tensor as its underlying_type, but found 'i3'}}
  %ext = bgv.extract %input, %offset : (!ct, index) -> !ct
  return %ext : !ct
}

// -----

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=3, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=2, ring=#ring>

!ct_tensor = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=tensor<64xi16>>
!ct_scalar = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>


func.func @test_extract_verification_failure(%input: !ct_tensor) -> !ct_scalar {
  %offset = arith.constant 4 : index
  // expected-error@+1 {{'bgv.extract' op output RLWE ciphertext's underlying_type must be the element type of the input ciphertext's underlying tensor type, but found tensor type 'tensor<64xi16>' and output type 'i3'}}
  %ext = bgv.extract %input, %offset : (!ct_tensor, index) -> !ct_scalar
  return %ext : !ct_scalar
}
