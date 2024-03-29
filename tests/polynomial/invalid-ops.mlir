// RUN: heir-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// This tests for invalid syntax from operation verifications.

#ideal = #polynomial.polynomial<-1 + x**1024>
#ring = #polynomial.ring<cmod=18, ideal=#ideal>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK-NOT: @test_invalid_ntt
// CHECK-NOT: polynomial.ntt
func.func @test_invalid_ntt(%0 : !poly_ty) {
  // expected-error@+1 {{a ring encoding was not provided}}
  %1 = polynomial.ntt %0 : !poly_ty -> tensor<1024xi32>
  return
}

// -----

#ideal = #polynomial.polynomial<-1 + x**1024>
#ring = #polynomial.ring<cmod=18, ideal=#ideal>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK-NOT: @test_invalid_ntt
// CHECK-NOT: polynomial.ntt
func.func @test_invalid_ntt(%0 : !poly_ty) {
  // expected-error@+1 {{tensor encoding is not a ring attribute}}
  %1 = polynomial.ntt %0 : !poly_ty -> tensor<1024xi32, #ideal>
  return
}

// -----

#ideal = #polynomial.polynomial<-1 + x**1024>
#ring = #polynomial.ring<cmod=18, ideal=#ideal>
#ring1 = #polynomial.ring<cmod=29, ideal=#ideal>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK-NOT: @test_invalid_intt
// CHECK-NOT: polynomial.intt
func.func @test_invalid_intt(%0 : tensor<1024xi32, #ring1>) {
  // expected-error@+1 {{not equivalent to the polynomial ring}}
  %1 = polynomial.intt %0 : tensor<1024xi32, #ring1> -> !poly_ty
  return
}

// -----

#ideal = #polynomial.polynomial<-1 + x**1024>
#ring = #polynomial.ring<cmod=18, ideal=#ideal>
!poly_ty = !polynomial.polynomial<#ring>

// CHECK-NOT: @test_invalid_intt
// CHECK-NOT: polynomial.intt
func.func @test_invalid_intt(%0 : tensor<1025xi32, #ring>) {
  // expected-error@+1 {{must be a tensor of shape [d]}}
  %1 = polynomial.intt %0 : tensor<1025xi32, #ring> -> !poly_ty
  return
}

// -----
