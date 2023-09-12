// RUN: heir-opt %s 2>%t; FileCheck %s < %t

// Test for errors in syntax, the CHECK directives look for error messages
// output to stderr.

#my_poly = #poly.polynomial<1 + x**1024>
#ring1 = #poly.ring<cmod=256, ideal=#my_poly>
module {
  func.func @test_from_tensor_too_large_coeffs() {
    %two = arith.constant 2 : i32
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi32>
    // CHECK: is too large to fit in the coefficients
    %poly = poly.from_tensor %coeffs1 : tensor<2xi32> -> !poly.poly<#ring1>
    return
  }
}
