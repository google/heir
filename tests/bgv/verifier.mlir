// RUN: heir-opt --verify-diagnostics %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<cmod=463187969, ideal=#my_poly>

#params = #lwe.rlwe_params<dimension=3, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=2, ring=#ring>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1>

func.func @test_input_dimension_error(%input: !ct) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = bgv.rotate (%input) {offset = 4} : (!ct) -> !ct1

  return
}
