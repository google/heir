// RUN: heir-opt --lazy-relin %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 161729713 : i32, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 2521 : i32, polynomialModulus=#my_poly>

#params = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params1 = #lwe.rlwe_params<dimension=3, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=2, ring=#ring2>

!pt = !lwe.rlwe_plaintext<encoding=#encoding, ring=#ring1, underlying_type=i3>
!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct_tensor = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=tensor<32xi16>>
!ct_scalar = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i16>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2, underlying_type=i3>


// CHECK-LABEL: test_lazy_relin
// CHECK-SAME: %[[ARG0:.*]] : !ct,  %[[ARG1:.*]] : !ct) -> !ct{
// [[V0.*]] = bgv.mul %[[ARG0:.*]],  %[[ARG1:.*]]
// CHECK: bgv.mul
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.add
// CHECK-NEXT: bgv.relinearize

func.func @test_lazy_relin(%arg0 : !ct, %arg1 : !ct) -> !ct { 
    %x = bgv.mul %arg0, %arg1 : (!ct, !ct) -> !ct1
    %x_relin = bgv.relinearize %x  { from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 2> } :!ct1 -> !ct
    %y = bgv.mul %arg0, %arg1 : (!ct, !ct) -> !ct1
    %y_relin = bgv.relinearize %y  { from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 2> } :!ct1 -> !ct
    %z = bgv.add %x_relin, %y_relin :!ct
    return %z : !ct
}