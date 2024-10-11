// RUN: heir-opt --color %s | FileCheck %s

// This simply tests for syntax.

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

// CHECK: module
module {
  // CHECK-LABEL: @test_multiply
  func.func @test_multiply(%arg0 : !ct, %arg1: !ct) -> !ct {
    %add = ckks.add %arg0, %arg1 : !ct
    %sub = ckks.sub %arg0, %arg1 : !ct
    %neg = ckks.negate %arg0 : !ct

    %0 = ckks.mul %arg0, %arg1  : (!ct, !ct) -> !ct1
    %1 = ckks.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct
    %2 = ckks.rescale %1  {to_ring = #ring2} : !ct -> !ct2
    // CHECK: rlwe_params = <dimension = 3, ring = <coefficientType = i32, coefficientModulus = 161729713 : i32, polynomialModulus = <1 + x**1024>>>
    return %arg0 : !ct
  }

  // CHECK-LABEL: @test_ciphertext_plaintext
  func.func @test_ciphertext_plaintext(%arg0: !pt, %arg1: !pt, %arg2: !pt, %arg3: !ct) -> !ct {
    %add = ckks.add_plain %arg3, %arg0 : (!ct, !pt) -> !ct
    %sub = ckks.sub_plain %add, %arg1 : (!ct, !pt) -> !ct
    %mul = ckks.mul_plain %sub, %arg2 : (!ct, !pt) -> !ct
    // CHECK: rlwe_params = <ring = <coefficientType = i32, coefficientModulus = 161729713 : i32, polynomialModulus = <1 + x**1024>>>
    return %mul : !ct
  }

  // CHECK-LABEL: @test_rotate_extract
  func.func @test_rotate_extract(%arg3: !ct_tensor) -> !ct_scalar {
    %c0 = arith.constant 0 : index
    %add = ckks.rotate %arg3 { offset = 1 } : !ct_tensor
    %ext = ckks.extract %add, %c0 : (!ct_tensor, index) -> !ct_scalar
    // CHECK: rlwe_params = <ring = <coefficientType = i32, coefficientModulus = 161729713 : i32, polynomialModulus = <1 + x**1024>>>
    return %ext : !ct_scalar
  }
}
