// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #poly.polynomial<1 + x**1024>
#my_poly_2 = #poly.polynomial<2>
#my_poly_3 = #poly.polynomial<3x>
#my_poly_4 = #poly.polynomial<t**3 + 4t + 2>
#ring1 = #poly.ring<cmod=2837465, ideal=#my_poly>
module {
  func.func @test_multiply() -> !poly.poly<#ring1> {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %five = arith.constant 5 : i16
    %coeffs1 = tensor.from_elements %two, %two, %five : tensor<3xi16>
    %coeffs2 = tensor.from_elements %five, %five, %two : tensor<3xi16>

    %poly1 = poly.from_tensor %coeffs1 : tensor<3xi16> -> !poly.poly<#ring1>
    %poly2 = poly.from_tensor %coeffs2 : tensor<3xi16> -> !poly.poly<#ring1>

    // CHECK: #poly.ring<cmod=2837465, ideal=#poly.polynomial<1 + x**1024>>
    %3 = poly.mul(%poly1, %poly2) {ring = #ring1} : !poly.poly<#ring1>

    return %3 : !poly.poly<#ring1>
  }

  func.func @test_elementwise(%p0 : !poly.poly<#ring1>, %p1: !poly.poly<#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!poly.poly<#ring1>>
    %tp1 = tensor.from_elements %p1, %p0 : tensor<2x!poly.poly<#ring1>>

    %c = arith.constant 2 : i32
    %mul_const_sclr = poly.mul_constant(%tp0, %c) : tensor<2x!poly.poly<#ring1>>, i32

    %add = poly.add(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>
    %sub = poly.sub(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>
    %mul = poly.mul(%tp0, %tp1) : tensor<2x!poly.poly<#ring1>>

    return
  }

  func.func @test_to_from_tensor(%p0 : !poly.poly<#ring1>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16>
    // CHECK: from_tensor
    %poly = poly.from_tensor %coeffs1 : tensor<2xi16> -> !poly.poly<#ring1>
    // CHECK: to_tensor
    %tensor = poly.to_tensor %poly : !poly.poly<#ring1> -> tensor<1024xi16>

    return
  }

  func.func @test_degree(%p0 : !poly.poly<#ring1>) {
    %0 = poly.degree %p0 : !poly.poly<#ring1>
    return
  }

  func.func @test_monomial() {
    %deg = arith.constant 1023 : index
    %five = arith.constant 5 : i16
    %0 = poly.monomial %five, %deg : (i16, index) -> !poly.poly<#ring1>
    return
  }
}
