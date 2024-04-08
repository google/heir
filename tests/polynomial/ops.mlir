// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

#my_poly = #_polynomial.polynomial<1 + x**1024>
#my_poly_2 = #_polynomial.polynomial<2>
#my_poly_3 = #_polynomial.polynomial<3x>
#my_poly_4 = #_polynomial.polynomial<t**3 + 4t + 2>
#ring1 = #_polynomial.ring<cmod=2837465, ideal=#my_poly>
#one_plus_x_squared = #_polynomial.polynomial<1 + x**2>

#ring = #_polynomial.ring<cmod=256, ideal=#_polynomial.polynomial<x**4 + 1>, root=31>
!poly_ty = !_polynomial.polynomial<#ring>

module {
  func.func @test_multiply() -> !_polynomial.polynomial<#ring1> {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %five = arith.constant 5 : i16
    %coeffs1 = tensor.from_elements %two, %two, %five : tensor<3xi16>
    %coeffs2 = tensor.from_elements %five, %five, %two : tensor<3xi16>

    %poly1 = _polynomial.from_tensor %coeffs1 : tensor<3xi16> -> !_polynomial.polynomial<#ring1>
    %poly2 = _polynomial.from_tensor %coeffs2 : tensor<3xi16> -> !_polynomial.polynomial<#ring1>

    // CHECK: #_polynomial.ring<cmod=2837465, ideal=#_polynomial.polynomial<1 + x**1024>>
    %3 = _polynomial.mul(%poly1, %poly2) {ring = #ring1} : !_polynomial.polynomial<#ring1>

    return %3 : !_polynomial.polynomial<#ring1>
  }

  func.func @test_elementwise(%p0 : !_polynomial.polynomial<#ring1>, %p1: !_polynomial.polynomial<#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!_polynomial.polynomial<#ring1>>
    %tp1 = tensor.from_elements %p1, %p0 : tensor<2x!_polynomial.polynomial<#ring1>>

    %c = arith.constant 2 : i32
    %mul_const_sclr = _polynomial.mul_scalar %tp0, %c : tensor<2x!_polynomial.polynomial<#ring1>>, i32

    %add = _polynomial.add(%tp0, %tp1) : tensor<2x!_polynomial.polynomial<#ring1>>
    %sub = _polynomial.sub(%tp0, %tp1) : tensor<2x!_polynomial.polynomial<#ring1>>
    %mul = _polynomial.mul(%tp0, %tp1) : tensor<2x!_polynomial.polynomial<#ring1>>

    return
  }

  func.func @test_to_from_tensor(%p0 : !_polynomial.polynomial<#ring1>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16>
    // CHECK: from_tensor
    %poly = _polynomial.from_tensor %coeffs1 : tensor<2xi16> -> !_polynomial.polynomial<#ring1>
    // CHECK: to_tensor
    %tensor = _polynomial.to_tensor %poly : !_polynomial.polynomial<#ring1> -> tensor<1024xi16>

    return
  }

  func.func @test_degree(%p0 : !_polynomial.polynomial<#ring1>) {
    %0, %1 = _polynomial.leading_term %p0 : !_polynomial.polynomial<#ring1> -> (index, i32)
    return
  }

  func.func @test_monomial() {
    %deg = arith.constant 1023 : index
    %five = arith.constant 5 : i16
    %0 = _polynomial.monomial %five, %deg : (i16, index) -> !_polynomial.polynomial<#ring1>
    return
  }

  func.func @test_constant() {
    %0 = _polynomial.constant #one_plus_x_squared : !_polynomial.polynomial<#ring1>
    %1 = _polynomial.constant <1 + x**2> : !_polynomial.polynomial<#ring1>
    return
  }

  func.func @test_ntt(%0 : !poly_ty) {
    %1 = _polynomial.ntt %0 : !poly_ty -> tensor<4xi32, #ring>
    return
  }

  func.func @test_intt(%0 : tensor<4xi32, #ring>) {
    %1 = _polynomial.intt %0 : tensor<4xi32, #ring> -> !poly_ty
    return
  }
}
