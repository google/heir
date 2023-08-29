// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #poly.polynomial<1 + x**1024>
#my_poly_2 = #poly.polynomial<2>
#my_poly_3 = #poly.polynomial<3x>
#my_poly_4 = #poly.polynomial<t**3 + 4t + 2>
#ring1 = #poly.ring<cmod=2837465, ideal=#my_poly>
module {
  func.func @test_multiply() -> i32 {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i32
    %five = arith.constant 5 : i32
    %coeffs1 = tensor.from_elements %two, %two, %five : tensor<3xi32>
    %coeffs2 = tensor.from_elements %five, %five, %two : tensor<3xi32>

    %poly1 = poly.from_coeffs(%coeffs1) : (tensor<3xi32>) -> !poly.poly<#ring1>
    %poly2 = poly.from_coeffs(%coeffs2) : (tensor<3xi32>) -> !poly.poly<#ring1>

    // CHECK: #poly.ring<cmod=2837465, ideal=#poly.polynomial<1 + x**1024>>
    %3 = poly.mul(%poly1, %poly2) {ring = #ring1} : !poly.poly<#ring1>
    %4 = poly.get_coeff(%3, %c0) : (!poly.poly<#ring1>, index) -> i32

    return %4 : i32
  }

  func.func @test_elementwise(%p0 : !poly.poly<#ring1>, %p1: !poly.poly<#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!poly.poly<#ring1>>
    %tp1 = tensor.from_elements %p1, %p0 : tensor<2x!poly.poly<#ring1>>

    %c = arith.constant 2 : i32
    %mul_const_sclr = poly.mul_constant(%tp0, %c) : (tensor<2x!poly.poly<#ring1>>, i32) -> tensor<2x!poly.poly<#ring1>>

    %add = poly.add(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>
    %sub = poly.sub(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>
    %mul = poly.mul(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>

    return
  }
}
