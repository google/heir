// RUN: heir-opt --materialize-roots %s | FileCheck %s

#cycl_1024 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=3758161921, ideal=#cycl_1024>

!tensor_ty = tensor<1024xi32, #ring>
!poly_ty = !_polynomial.polynomial<#ring>

// CHECK: func.func @unmaterialized_root(%[[POLY:.*]]:
// CHECK-SAME: [[POLY_TY:!_polynomial.polynomial<<cmod=3758161921, ideal=.*, root=376008217>>]])
// CHECK-SAME: -> [[POLY_TY]] {
func.func @unmaterialized_root(%poly: !poly_ty) -> !poly_ty {
  // CHECK: %[[NTT:.*]] = _polynomial.ntt %[[POLY]] : [[POLY_TY]] -> [[TENSOR_TY:tensor<1024xi10, #_polynomial.ring<cmod=3758161921, ideal=.*, root=376008217>>]]
  // CHECK: %[[INTT:.*]] = _polynomial.intt %[[NTT]] : [[TENSOR_TY]]
  // CHECK: return %[[INTT]] : [[POLY_TY]]
  %ntt = _polynomial.ntt %poly : !poly_ty -> tensor<1024xi10, #ring>
  %intt = _polynomial.intt %ntt : tensor<1024xi10, #ring> -> !poly_ty
  return %intt : !poly_ty
}
