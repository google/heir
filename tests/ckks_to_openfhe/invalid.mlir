// RUN: heir-opt --ckks-to-openfhe --split-input-file --verify-diagnostics %s 2>&1

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 161729713 : i32, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 2521 : i32, polynomialModulus=#my_poly>

#params = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params1 = #lwe.rlwe_params<dimension=4, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=2, ring=#ring2>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2, underlying_type=i3>

func.func @test_relin_to_basis_error(%x: !ct1) {
  // expected-error@+2 {{toBasis must be [0, 1], got [0, 2]}}
  // expected-error@+1 {{failed to legalize operation 'ckks.relinearize'}}
  %relin_error = ckks.relinearize %x  { from_basis = array<i32: 0, 1, 2, 3>, to_basis = array<i32: 0, 2> }: !ct1 -> !ct
  return
}

// -----
#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 33538049 : i32, polynomialModulus=#my_poly>

#params = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params1 = #lwe.rlwe_params<dimension=4, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=2, ring=#ring2>

!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2, underlying_type=i3>

func.func @test_modswitch_level_error(%x: !ct2) {
  // expected-error@+1 {{output ring should match to_ring}}
  %relin_error = bgv.modulus_switch %x  {to_ring=#ring2}: !ct2 -> !ct1
  return
}
