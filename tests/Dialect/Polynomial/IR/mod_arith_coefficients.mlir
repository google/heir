// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#my_poly_2 = #polynomial.int_polynomial<2>
#my_poly_3 = #polynomial.int_polynomial<3x>
#my_poly_4 = #polynomial.int_polynomial<t**3 + 4t + 2>
!coeff_ty1 = !mod_arith.int<2837465:i32>
#ring1 = #polynomial.ring<coefficientType=!coeff_ty1, polynomialModulus=#my_poly>
#one_plus_x_squared = #polynomial.int_polynomial<1 + x**2>

!coeff_ty2 = !mod_arith.int<256:i32>
#ideal = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=!coeff_ty2, polynomialModulus=#ideal>

#ntt_poly = #polynomial.int_polynomial<-1 + x**8>
#ntt_ring = #polynomial.ring<coefficientType=!coeff_ty2, polynomialModulus=#ntt_poly>
!poly_ty = !polynomial.polynomial<ring=#ntt_ring>
!ntt_poly_ty = !polynomial.polynomial<ring=#ntt_ring, form=eval>
#ntt_ring_1_root_val = #mod_arith.value<31:!coeff_ty2>
#ntt_ring_1_root = #polynomial.primitive_root<value=#ntt_ring_1_root_val, degree=8:i32>

!coeff_ty3 = !mod_arith.int<786433:i32>
#ntt_poly_2 = #polynomial.int_polynomial<1 + x**65536>
#ntt_ring_2 = #polynomial.ring<coefficientType=!coeff_ty3, polynomialModulus=#ntt_poly_2>
#ntt_ring_2_root_val = #mod_arith.value<283965:!coeff_ty3>
#ntt_ring_2_root = #polynomial.primitive_root<value=#ntt_ring_2_root_val, degree=131072:i32>
!poly_ty_2 = !polynomial.polynomial<ring=#ntt_ring_2>

module {
  func.func @test_multiply() -> !polynomial.polynomial<ring=#ring1> {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i32
    %five = arith.constant 5 : i32
    %coeffs1_ = tensor.from_elements %two, %two, %five : tensor<3xi32>
    %coeffs2_ = tensor.from_elements %five, %five, %two : tensor<3xi32>
    %coeffs1 = mod_arith.encapsulate %coeffs1_ : tensor<3xi32> -> tensor<3x!coeff_ty1>
    %coeffs2 = mod_arith.encapsulate %coeffs2_ : tensor<3xi32> -> tensor<3x!coeff_ty1>

    %poly1 = polynomial.from_tensor %coeffs1 : tensor<3x!coeff_ty1> -> !polynomial.polynomial<ring=#ring1>
    %poly2 = polynomial.from_tensor %coeffs2 : tensor<3x!coeff_ty1> -> !polynomial.polynomial<ring=#ring1>

    %3 = polynomial.mul %poly1, %poly2 : !polynomial.polynomial<ring=#ring1>

    return %3 : !polynomial.polynomial<ring=#ring1>
  }

  func.func @test_elementwise(%p0 : !polynomial.polynomial<ring=#ring1>, %p1: !polynomial.polynomial<ring=#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %tp1 = tensor.from_elements %p1, %p0 : tensor<2x!polynomial.polynomial<ring=#ring1>>

    %c = arith.constant 2 : i32
    %c_mod_arith = mod_arith.encapsulate %c : i32 -> !coeff_ty1
    %mul_const_sclr = polynomial.mul_scalar %tp0, %c_mod_arith : tensor<2x!polynomial.polynomial<ring=#ring1>>, !coeff_ty1

    %add = polynomial.add %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %sub = polynomial.sub %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %mul = polynomial.mul %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>

    return
  }

  func.func @test_to_from_tensor(%p0 : !polynomial.polynomial<ring=#ring1>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i32
    %coeffs1_ = tensor.from_elements %two, %two : tensor<2xi32>
    %coeffs1 = mod_arith.encapsulate %coeffs1_ : tensor<2xi32> -> tensor<2x!coeff_ty1>
    // CHECK: from_tensor
    %poly = polynomial.from_tensor %coeffs1 : tensor<2x!coeff_ty1> -> !polynomial.polynomial<ring=#ring1>
    // CHECK: to_tensor
    %tensor = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring1> -> tensor<1024x!coeff_ty1>

    return
  }

  func.func @test_degree(%p0 : !polynomial.polynomial<ring=#ring1>) {
    %0, %1 = polynomial.leading_term %p0 : !polynomial.polynomial<ring=#ring1> -> (index, !coeff_ty1)
    return
  }

  func.func @test_monomial() {
    %deg = arith.constant 1023 : index
    %five = mod_arith.constant 5 : !coeff_ty1
    %0 = polynomial.monomial %five, %deg : (!coeff_ty1, index) -> !polynomial.polynomial<ring=#ring1>
    return
  }

  func.func @test_monic_monomial_mul() {
    %five = arith.constant 5 : index
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %1 = polynomial.monic_monomial_mul %0, %five : (!polynomial.polynomial<ring=#ring1>, index) -> !polynomial.polynomial<ring=#ring1>
    return
  }

  func.func @test_constant() {
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %1 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>

    // Test verbose fallbacks
    %verb0 = polynomial.constant #polynomial.typed_int_polynomial<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    return
  }

  func.func @test_ntt(%0 : !poly_ty) {
    %1 = polynomial.ntt %0 {root=#ntt_ring_1_root} : !poly_ty
    return
  }

  func.func @test_ntt_with_overflowing_root(%0 : !poly_ty_2) {
    %1 = polynomial.ntt %0 {root=#ntt_ring_2_root} : !poly_ty_2
    return
  }

  func.func @test_intt(%0 : !ntt_poly_ty) {
    %1 = polynomial.intt %0 {root=#ntt_ring_1_root} : !ntt_poly_ty
    return
  }
}
